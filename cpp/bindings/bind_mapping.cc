// Bindings of the map-level geometry: levels and ramps (gr/mapping/levels.h).

#include <algorithm>
#include <exception>
#include <utility>
#include <vector>

#include "bind_util.h"
#include "gr/mapping/bev.h"
#include "gr/mapping/levels.h"

namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(build/namespaces)

namespace gr::bindings {
namespace {

using FloorArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

py::dict GridToDict(const FloorGrid& grid) {
  FloorArray z({grid.height, grid.width});
  std::copy(grid.z.begin(), grid.z.end(), z.mutable_data());
  return py::dict("x0"_a = grid.x0, "y0"_a = grid.y0, "res"_a = grid.res,
                  "z"_a = z);
}

py::dict RampToDict(const Ramp& ramp) {
  RowMatX2d polygon(static_cast<Eigen::Index>(ramp.polygon.size()), 2);
  for (size_t k = 0; k < ramp.polygon.size(); ++k) {
    polygon.row(static_cast<Eigen::Index>(k)) = ramp.polygon[k].transpose();
  }
  return py::dict("level_from"_a = ramp.level_from,
                  "level_to"_a = ramp.level_to, "plane"_a = ramp.plane,
                  "polygon"_a = polygon, "slope"_a = ramp.slope);
}


// The point_map.npz arrays BevPoints refers to (kept alive by the caller).
struct BevArrays {
  Points xyz;
  std::vector<float> intensity;
  std::vector<int16_t> level;
  std::vector<uint8_t> dynamic;
  std::vector<float> height;
  Points normals;

  BevPoints View() const {
    BevPoints view;
    view.xyz = &xyz;
    view.intensity = &intensity;
    view.level = &level;
    view.dynamic = &dynamic;
    view.height = &height;
    view.normals = &normals;
    return view;
  }
};

BevArrays BevArraysFrom(const py::object& xyz, const py::object& intensity,
                        const py::object& level, const py::object& dynamic,
                        const py::object& height, const py::object& normals) {
  BevArrays out;
  out.xyz = PointsFrom(xyz);
  out.intensity = VectorOf<float>(intensity);
  out.level = VectorOf<int16_t>(level);
  out.dynamic = VectorOf<uint8_t>(dynamic);
  out.height = VectorOf<float>(height);
  out.normals = PointsFrom(normals);
  const size_t n = out.xyz.size();
  Require(out.level.size() == n && out.height.size() == n &&
              (out.intensity.empty() || out.intensity.size() == n) &&
              (out.dynamic.empty() || out.dynamic.size() == n) &&
              (out.normals.empty() || out.normals.size() == n),
          "one level, height (and intensity, dynamic, normal) per point");
  return out;
}

Grid2D GridOf(double x0, double y0, double res, int width, int height) {
  Require(res > 0.0 && width >= 0 && height >= 0, "invalid grid");
  Grid2D grid;
  grid.x0 = x0;
  grid.y0 = y0;
  grid.res = res;
  grid.width = width;
  grid.height = height;
  return grid;
}

py::array FloorHeightsArray(const FloorHeights& heights, const Grid2D& grid) {
  if (!heights.single_precision) {
    return ToImage(heights.z, grid.height, grid.width);
  }
  return ToImage(std::vector<float>(heights.z.begin(), heights.z.end()),
                 grid.height, grid.width);
}

}  // namespace

void BindMapping(py::module_& m) {
  m.def(
      "level_grid",
      [](const py::object& xyz, const py::object& level,
         const py::object& dynamic, int lv, double res, double margin) {
        Require(res > 0.0, "res must be positive");
        const BevArrays arrays =
            BevArraysFrom(xyz, py::none(), level, dynamic,
                          py::array_t<float>(py::len(level)), py::none());
        const Grid2D grid = LevelGrid(arrays.View(), lv, res, margin);
        Require(grid.width > 0, "no static points on this level");
        return py::make_tuple(grid.x0, grid.y0, grid.width, grid.height);
      },
      "xyz"_a, "level"_a, "dynamic"_a, "lv"_a, "res"_a, "margin"_a,
      "bev.level_grid: (x0, y0, width, height) of the static points of a "
      "level plus the margin.");

  m.def(
      "floor_height_raster",
      [](const std::vector<double>& z, const py::list& grids,
         const py::list& ramps, int lv, double x0, double y0, double res,
         int width, int height) {
        const LevelModel model = LevelModelFrom(z, grids, ramps);
        RequireIndex(lv >= 0 && lv < model.num_levels(),
                     "level out of range");
        const Grid2D grid = GridOf(x0, y0, res, width, height);
        return FloorHeightsArray(FloorHeightRaster(model, lv, grid), grid);
      },
      "z"_a, "grids"_a, "ramps"_a, "lv"_a, "x0"_a, "y0"_a, "res"_a,
      "width"_a, "height"_a,
      "bev.floor_height_raster: (height, width) floor height (float32 with "
      "a floor grid).");

  m.def(
      "bev_rasters",
      [](const py::object& xyz, const py::object& intensity,
         const py::object& level, const py::object& dynamic,
         const py::object& height, const py::object& normals,
         const std::vector<double>& z, const py::list& grids,
         const py::list& ramps, int lv, double x0, double y0, double res,
         int width, int height_cells, double floor_tolerance,
         double object_max_height) {
        const BevArrays arrays =
            BevArraysFrom(xyz, intensity, level, dynamic, height, normals);
        Require(arrays.intensity.size() == arrays.xyz.size(),
                "one intensity per point");
        const LevelModel model = LevelModelFrom(z, grids, ramps);
        RequireIndex(lv >= 0 && lv < model.num_levels(),
                     "level out of range");
        const Grid2D grid = GridOf(x0, y0, res, width, height_cells);
        BevConfig config;
        config.floor_tolerance = floor_tolerance;
        config.object_max_height = object_max_height;
        BevRasters rasters;
        {
          py::gil_scoped_release release;
          rasters = LevelRasters(arrays.View(), model, lv, grid, config);
        }
        return py::dict("intensity"_a = rasters.intensity,
                        "floor_count"_a = rasters.floor_count,
                        "height_max"_a = rasters.height_max,
                        "floor_z"_a = FloorHeightsArray(rasters.floor_z, grid));
      },
      "xyz"_a, "intensity"_a, "level"_a, "dynamic"_a, "height"_a,
      "normals"_a, "z"_a, "grids"_a, "ramps"_a, "lv"_a, "x0"_a, "y0"_a,
      "res"_a, "width"_a, "height_cells"_a, "floor_tolerance"_a,
      "object_max_height"_a,
      "bev.level_rasters: dict of intensity, floor_count, height_max and "
      "floor_z rasters (height, width).");

  py::class_<LevelConfig>(m, "LevelConfig")
      .def(py::init<>())
      .def_readwrite("min_separation", &LevelConfig::min_separation)
      .def_readwrite("flat_slope", &LevelConfig::flat_slope)
      .def_readwrite("min_samples", &LevelConfig::min_samples)
      .def_readwrite("grid_res", &LevelConfig::grid_res)
      .def_readwrite("floor_band", &LevelConfig::floor_band);

  m.def(
      "detect_levels",
      [](const py::list& paths, const PointsIn& xyz, const PointsIn& normals,
         const LevelConfig& config, const py::object& log) {
        Require(xyz.rows() == normals.rows(),
                "xyz and normals must have the same length");
        Require(config.grid_res > 0.0, "grid_res must be positive");
        std::vector<Points> vehicle_paths;
        for (const py::handle& path : paths) {
          vehicle_paths.push_back(
              PointsFrom(py::reinterpret_borrow<py::object>(path)));
        }
        const Points p = ToPoints(xyz);
        const Points n = ToPoints(normals);
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        LevelModel model;
        {
          py::gil_scoped_release release;
          model = DetectLevels(vehicle_paths, p, n, config, logger);
        }
        RethrowIfSet(log_error);
        py::list grids;
        for (const FloorGrid& g : model.grids) grids.append(GridToDict(g));
        py::list ramps;
        for (const Ramp& r : model.ramps) ramps.append(RampToDict(r));
        return py::dict("z"_a = model.z, "ceiling"_a = model.ceiling,
                        "grids"_a = grids, "ramps"_a = ramps);
      },
      "paths"_a, "xyz"_a, "normals"_a, "config"_a = LevelConfig(),
      "log"_a = py::none(),
      "Levels and ramps from the vehicle paths ((N, 3) per trip) and the map "
      "points: dict of z, ceiling, grids (x0, y0, res, z (H, W)), ramps "
      "(level_from, level_to, plane, polygon, slope).");

  m.def(
      "assign_levels",
      [](const std::vector<double>& z, const py::list& grids,
         const py::list& ramps, const PointsIn& xyz) {
        const LevelModel model = LevelModelFrom(z, grids, ramps);
        Require(model.grids.empty() || model.grids.size() == model.z.size(),
                "one grid per level (or none)");
        const Points p = ToPoints(xyz);
        LevelAssignment out;
        {
          py::gil_scoped_release release;
          out = AssignLevels(model, p);
        }
        py::array_t<bool> on_ramp(
            static_cast<py::ssize_t>(out.on_ramp.size()));
        for (size_t i = 0; i < out.on_ramp.size(); ++i) {
          on_ramp.mutable_data()[i] = out.on_ramp[i] != 0;
        }
        return py::make_tuple(ToArray(out.level), ToArray(out.height),
                              on_ramp);
      },
      "z"_a, "grids"_a, "ramps"_a, "xyz"_a,
      "Level index (-1 below every floor), height above it (NaN) and the "
      "on-ramp flag of every point.");
}

}  // namespace gr::bindings
