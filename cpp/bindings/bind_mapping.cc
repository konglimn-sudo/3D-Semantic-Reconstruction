// Bindings of the map-level geometry: levels and ramps (gr/mapping/levels.h).

#include <algorithm>
#include <exception>
#include <utility>
#include <vector>

#include "bind_util.h"
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

// The level model from the Python LevelModel's parts.
LevelModel ModelFrom(const std::vector<double>& z, const py::list& grids,
                     const py::list& ramps) {
  LevelModel model;
  model.z = z;
  for (const py::handle& item : grids) {
    const py::tuple t = item.cast<py::tuple>();
    Require(t.size() == 4, "grids are (x0, y0, res, z) tuples");
    FloorGrid grid;
    grid.x0 = t[0].cast<double>();
    grid.y0 = t[1].cast<double>();
    grid.res = t[2].cast<double>();
    const FloorArray values = FloorArray::ensure(t[3]);
    Require(values && values.ndim() == 2, "a grid is an (H, W) array");
    Require(grid.res > 0.0, "grid res must be positive");
    grid.height = static_cast<int>(values.shape(0));
    grid.width = static_cast<int>(values.shape(1));
    grid.z.assign(values.data(), values.data() + values.size());
    model.grids.push_back(std::move(grid));
  }
  for (const py::handle& item : ramps) {
    const py::tuple t = item.cast<py::tuple>();
    Require(t.size() == 5,
            "ramps are (level_from, level_to, plane, polygon, slope) tuples");
    Ramp ramp;
    ramp.level_from = t[0].cast<int>();
    ramp.level_to = t[1].cast<int>();
    ramp.plane = t[2].cast<Vec3>();
    const RowMatX2d polygon = t[3].cast<RowMatX2d>();
    for (Eigen::Index k = 0; k < polygon.rows(); ++k) {
      ramp.polygon.push_back(polygon.row(k).transpose());
    }
    ramp.slope = t[4].cast<double>();
    model.ramps.push_back(std::move(ramp));
  }
  return model;
}

}  // namespace

void BindMapping(py::module_& m) {
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
        const LevelModel model = ModelFrom(z, grids, ramps);
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
