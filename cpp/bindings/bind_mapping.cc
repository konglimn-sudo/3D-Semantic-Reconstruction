// Bindings of the map-level geometry: levels and ramps (gr/mapping/levels.h).

#include <algorithm>
#include <exception>
#include <utility>
#include <vector>

#include "bind_util.h"
#include "gr/mapping/bev.h"
#include "gr/mapping/levels.h"
#include "gr/mapping/tiled_map.h"

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
  RowMatX2d polygon(static_cast<Eigen::Index>(ramp.polygon.size()),
                    kPlaneDim);
  for (size_t vertex = 0; vertex < ramp.polygon.size(); ++vertex) {
    polygon.row(static_cast<Eigen::Index>(vertex)) =
        ramp.polygon[vertex].transpose();
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
  const size_t count = out.xyz.size();
  Require(out.level.size() == count && out.height.size() == count &&
              (out.intensity.empty() || out.intensity.size() == count) &&
              (out.dynamic.empty() || out.dynamic.size() == count) &&
              (out.normals.empty() || out.normals.size() == count),
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

py::dict LevelsToDict(const LevelModel& model) {
  py::list grids;
  for (const FloorGrid& grid : model.grids) grids.append(GridToDict(grid));
  py::list ramps;
  for (const Ramp& ramp : model.ramps) ramps.append(RampToDict(ramp));
  return py::dict("z"_a = model.z, "ceiling"_a = model.ceiling,
                  "grids"_a = grids, "ramps"_a = ramps);
}

// The point_map.npz arrays of a chunk (static_map.PointMap dtypes), plus
// its order key.
py::dict ChunkToDict(const MapChunk& chunk) {
  const py::ssize_t count = static_cast<py::ssize_t>(chunk.size());
  py::array_t<float> normals({count, static_cast<py::ssize_t>(kSpaceDim)});
  float* data = normals.mutable_data();
  for (size_t i = 0; i < chunk.size(); ++i) {
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      data[kSpaceDim * i + static_cast<size_t>(axis)] =
          static_cast<float>(chunk.normals[i][axis]);
    }
  }
  std::vector<bool> dynamic(chunk.size());
  for (size_t i = 0; i < chunk.size(); ++i) {
    dynamic[i] =
        chunk.dynamic_kind[i] != static_cast<uint8_t>(DynamicKind::kStatic);
  }
  std::vector<bool> on_ramp(chunk.on_ramp.begin(), chunk.on_ramp.end());
  py::dict out("xyz"_a = PointsToArray(chunk.xyz),
               "intensity"_a = ToArray(chunk.intensity),
               "kf"_a = ToArray(chunk.keyframe), "trip"_a = ToArray(chunk.trip),
               "normals"_a = normals, "planarity"_a = ToArray(chunk.planarity),
               "observations"_a = ToArray(chunk.observations),
               "dynamic"_a = ToBoolArray(dynamic),
               "dynamic_kind"_a = ToArray(chunk.dynamic_kind),
               "level"_a = ToArray(chunk.level),
               "height"_a = ToArray(chunk.height),
               "on_ramp"_a = ToBoolArray(on_ramp),
               "range"_a = ToArray(chunk.range),
               "sequence"_a = ToArray(chunk.sequence));
  if (!chunk.sigma.empty()) out["sigma"] = ToArray(chunk.sigma);
  return out;
}

}  // namespace

void BindMapping(py::module_& m) {
  m.def(
      "level_grid",
      [](const py::object& xyz, const py::object& level,
         const py::object& dynamic, int selected_level, double res,
         double margin) {
        Require(res > 0.0, "res must be positive");
        const BevArrays arrays =
            BevArraysFrom(xyz, py::none(), level, dynamic,
                          py::array_t<float>(py::len(level)), py::none());
        const Grid2D grid =
            LevelGrid(arrays.View(), selected_level, res, margin);
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
        const Points cloud = ToPoints(xyz);
        const Points point_normals = ToPoints(normals);
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        LevelModel model;
        {
          py::gil_scoped_release release;
          model =
              DetectLevels(vehicle_paths, cloud, point_normals, config, logger);
        }
        RethrowIfSet(log_error);
        return LevelsToDict(model);
      },
      "paths"_a, "xyz"_a, "normals"_a, "config"_a = LevelConfig(),
      "log"_a = py::none(),
      "Levels and ramps from the vehicle paths ((N, 3) per trip) and the map "
      "points: dict of z, ceiling, grids (x0, y0, res, z (H, W)), ramps "
      "(level_from, level_to, plane, polygon, slope).");

  py::class_<TiledMapConfig>(m, "TiledMapConfig")
      .def(py::init<>())
      .def_readwrite("voxel", &TiledMapConfig::voxel)
      .def_readwrite("max_range", &TiledMapConfig::max_range)
      .def_readwrite("normal_radius", &TiledMapConfig::normal_radius)
      .def_readwrite("tile_size", &TiledMapConfig::tile_size)
      .def_readwrite("fusion", &TiledMapConfig::fusion)
      .def_readwrite("levels", &TiledMapConfig::levels)
      .def_readwrite("dynamic", &TiledMapConfig::dynamic)
      .def_readwrite("dynamic_config", &TiledMapConfig::dynamic_config);

  m.def(
      "build_tiled_map",
      [](const py::object& world_from_lidar, const py::object& keyframe_trip,
         const py::list& paths, const py::object& load,
         const TiledMapConfig& config, const py::object& sink,
         const py::object& log) {
        Require(config.voxel > 0.0, "voxel must be positive");
        const std::vector<Pose> poses = PosesFrom(world_from_lidar);
        const std::vector<int> trips = ToIntVector(keyframe_trip);
        Require(trips.size() == poses.size(), "one trip per keyframe");
        std::vector<Points> vehicle_paths;
        for (const py::handle& path : paths) {
          vehicle_paths.push_back(
              PointsFrom(py::reinterpret_borrow<py::object>(path)));
        }
        // Python errors stay out of the exception-free library: the first
        // one is kept, later callbacks do nothing, and it is raised after.
        std::exception_ptr error;
        const SweepLoader loader = [&](int k, Points* points,
                                       std::vector<float>* intensity) {
          points->clear();
          intensity->clear();
          py::gil_scoped_acquire acquire;
          if (error) return;
          try {
            const py::tuple sweep = load(k);
            *points = PointsFrom(sweep[0]);
            *intensity = ToFloatVector(sweep[1]);
          } catch (...) {
            error = std::current_exception();
          }
        };
        const ChunkSink writer = [&](const MapTile& tile, MapChunk&& chunk) {
          py::gil_scoped_acquire acquire;
          if (error) return;
          try {
            sink(py::dict("ix"_a = tile.ix, "iy"_a = tile.iy,
                          "whole"_a = tile.whole),
                 ChunkToDict(chunk));
          } catch (...) {
            error = std::current_exception();
          }
        };
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        TiledMapSummary summary;
        {
          py::gil_scoped_release release;
          summary = BuildTiledMap(poses, trips, vehicle_paths, loader, config,
                                  writer, logger);
        }
        RethrowIfSet(error);
        RethrowIfSet(log_error);
        return py::dict("levels"_a = LevelsToDict(summary.levels),
                        "tiles"_a = summary.tiles, "points"_a = summary.points,
                        "moved"_a = summary.moved,
                        "changed"_a = summary.changed,
                        "elevation_resolution"_a =
                            summary.elevation_resolution);
      },
      "T_world_lidar"_a, "trip"_a, "paths"_a, "load"_a, "config"_a, "sink"_a,
      "log"_a = py::none(),
      "The map stage tile by tile (gr/mapping/tiled_map.h): load(k) -> "
      "(points (N, 3), intensity (N,)) of keyframe k; sink(tile, arrays) "
      "gets each non-empty tile ({ix, iy, whole}) with its point_map arrays "
      "plus range / sequence (the map order). Returns the levels (as "
      "detect_levels), tiles, points, moved, changed and the dynamic "
      "elevation_resolution.");

  m.def(
      "assign_levels",
      [](const std::vector<double>& z, const py::list& grids,
         const py::list& ramps, const PointsIn& xyz) {
        const LevelModel model = LevelModelFrom(z, grids, ramps);
        Require(model.grids.empty() || model.grids.size() == model.z.size(),
                "one grid per level (or none)");
        const Points cloud = ToPoints(xyz);
        LevelAssignment out;
        {
          py::gil_scoped_release release;
          out = AssignLevels(model, cloud);
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
