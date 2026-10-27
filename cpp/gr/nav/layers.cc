#include "gr/nav/layers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "gr/core/check.h"

namespace gr {
namespace {

bool In(const std::vector<int>& ids, int id) {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

// Rasterises the selected points (xy, value) with a reducer.
class CellRaster {
 public:
  explicit CellRaster(const Grid2D& grid) : grid_(grid) {}

  void Add(const Vec2& xy, double value) {
    xy_.push_back(xy);
    values_.push_back(value);
  }

  Eigen::MatrixXd Reduce(Reducer reducer, double fill) const {
    RowMatX2d xy(static_cast<Eigen::Index>(xy_.size()), 2);
    for (size_t k = 0; k < xy_.size(); ++k) {
      xy.row(static_cast<Eigen::Index>(k)) = xy_[k].transpose();
    }
    const Eigen::Map<const Eigen::VectorXd> values(
        values_.data(), static_cast<Eigen::Index>(values_.size()));
    return Rasterize(xy, values, grid_, reducer, fill);
  }

 private:
  Grid2D grid_;
  std::vector<Vec2> xy_;
  std::vector<double> values_;
};

// Row-major cells of a (height x width) raster with `test`.
template <typename Test>
Mask Cells(const Eigen::MatrixXd& raster, Test test) {
  Mask out(static_cast<size_t>(raster.size()));
  size_t k = 0;
  for (Eigen::Index r = 0; r < raster.rows(); ++r) {
    for (Eigen::Index c = 0; c < raster.cols(); ++c) {
      out[k++] = test(raster(r, c));
    }
  }
  return out;
}

}  // namespace

void FillWorldPolygon(const std::vector<Vec2>& polygon, const Grid2D& grid,
                      int32_t value, std::vector<int32_t>* image) {
  std::vector<std::pair<int64_t, int64_t>> vertices;
  vertices.reserve(polygon.size());
  for (const Vec2& p : polygon) {
    // np.round(((p - origin) / res - 0.5) * 16): half to even.
    vertices.emplace_back(
        static_cast<int64_t>(
            std::nearbyint(((p.x() - grid.x0) / grid.res - 0.5) * 16.0)),
        static_cast<int64_t>(
            std::nearbyint(((p.y() - grid.y0) / grid.res - 0.5) * 16.0)));
  }
  FillPoly8(vertices, 4, grid.width, grid.height, value, image);
}

Mask FillHoles(const Mask& known, const Mask& obstacles, int width,
               int height, double res, double max_area) {
  Mask unknown(known.size());
  for (size_t k = 0; k < known.size(); ++k) {
    unknown[k] = !(known[k] || obstacles[k]);
  }
  const Components components = Label8(unknown, width, height);
  Mask out = known;
  for (const std::vector<int>& cells : components.cells) {
    bool touches = false;
    for (int c : cells) {
      const int x = c % width;
      const int y = c / width;
      touches |= x == 0 || y == 0 || x == width - 1 || y == height - 1;
    }
    // area * res ** 2 < max_area.
    const double area = static_cast<double>(cells.size()) * (res * res);
    if (touches || area >= max_area) continue;
    for (int c : cells) out[static_cast<size_t>(c)] = 1;
  }
  return out;
}

LevelLayers ComputeLevelLayers(const LayerPoints& points,
                               const LayerClasses& classes,
                               const LevelModel& levels, int level,
                               const std::vector<LayerSlot>& slots,
                               const std::vector<Vec2>& path_xy,
                               const Grid2D& grid, const LayerConfig& config) {
  const BevPoints& map = points.map;
  const size_t n = map.xyz->size();
  const size_t cells = static_cast<size_t>(grid.width) * grid.height;
  const bool with_normals =
      map.normals != nullptr && map.normals->size() == n;
  const bool with_observations =
      points.observations != nullptr && points.observations->size() == n;
  // numpy compares the float32 heights / normals in float32.
  const float floor_tolerance = static_cast<float>(config.floor_tolerance);
  const float min_height = static_cast<float>(config.obstacle_min_height);
  const float top =
      static_cast<float>(config.vehicle_height + config.vehicle_top_margin);
  const float max_headroom = static_cast<float>(config.max_headroom);

  CellRaster floor_points(grid), structure_known(grid), unclassified(grid);
  CellRaster overhead_points(grid), vehicles_in_band(grid), low_points(grid);
  CellRaster above_known(grid), above_unclassified(grid),
      above_unclassified_height(grid);
  for (size_t i = 0; i < n; ++i) {
    const bool dynamic = map.dynamic != nullptr && !map.dynamic->empty() &&
                         (*map.dynamic)[i];
    if ((*map.level)[i] != level || dynamic) continue;
    const float raw = (*map.height)[i];
    const float h = std::isnan(raw) ? -10.0f : raw;
    const float nz = with_normals
                         ? std::abs(static_cast<float>((*map.normals)[i].z()))
                         : 1.0f;
    const Vec2 xy = (*map.xyz)[i].head<2>();
    const int label = (*points.labels)[i];
    const bool vehicle = In(classes.vehicle, label);
    const bool low = In(classes.low, label);
    const double observations =
        with_observations ? (*points.observations)[i]
                          : config.min_cell_observations;
    if (std::abs(h) < floor_tolerance && nz > 0.7f) floor_points.Add(xy, 1.0);
    const bool band = h > min_height && h < top;
    if (low) low_points.Add(xy, 1.0);
    if (band && vehicle) vehicles_in_band.Add(xy, 1.0);
    if (band && !vehicle && !low) {  // Structure points.
      if (label == classes.unlabeled) {
        unclassified.Add(xy, observations);
      } else {
        structure_known.Add(xy, 1.0);
      }
      if (In(classes.overhead, label) && h > 0.5f) overhead_points.Add(xy, 1.0);
    }
    if (h >= top && h < max_headroom && !vehicle) {  // Above the band.
      if (label == classes.unlabeled) {
        above_unclassified.Add(xy, observations);
        above_unclassified_height.Add(xy, h);
      } else {
        above_known.Add(xy, h);
      }
    }
  }

  const double min_points = config.obstacle_min_points;
  const double min_observations = config.min_cell_observations;
  const auto at_least = [](double threshold) {
    return [threshold](double v) { return v >= threshold; };
  };
  LevelLayers out;
  out.structure = Cells(structure_known.Reduce(Reducer::kCount, 0.0),
                        at_least(min_points));
  const Mask seen_unclassified =
      Cells(unclassified.Reduce(Reducer::kSum, 0.0),
            at_least(min_observations));
  for (size_t k = 0; k < cells; ++k) out.structure[k] |= seen_unclassified[k];
  out.overhead = Cells(overhead_points.Reduce(Reducer::kCount, 0.0),
                       at_least(min_points));
  out.parked = Cells(vehicles_in_band.Reduce(Reducer::kCount, 0.0),
                     at_least(min_points));
  out.low =
      Cells(low_points.Reduce(Reducer::kCount, 0.0), at_least(min_points));
  for (size_t k = 0; k < cells; ++k) {
    out.overhead[k] &= out.structure[k];
    out.parked[k] &= !out.structure[k];
  }

  // Headroom: the lowest static object above the band, unclassified points
  // filtered by their observations.
  const double inf = std::numeric_limits<double>::infinity();
  const Eigen::MatrixXd lowest_known = above_known.Reduce(Reducer::kMin, inf);
  const Eigen::MatrixXd lowest_unclassified =
      above_unclassified_height.Reduce(Reducer::kMin, inf);
  const Mask seen = Cells(above_unclassified.Reduce(Reducer::kSum, 0.0),
                          at_least(min_observations));
  out.headroom.resize(cells);
  for (int r = 0; r < grid.height; ++r) {
    for (int c = 0; c < grid.width; ++c) {
      const size_t k = static_cast<size_t>(r) * grid.width + c;
      const double known = lowest_known(r, c);
      out.headroom[k] = static_cast<float>(
          seen[k] ? std::min(known, lowest_unclassified(r, c)) : known);
    }
  }

  const int k = config.floor_closing;
  const Mask observed = Cells(floor_points.Reduce(Reducer::kCount, 0.0),
                              [](double v) { return v > 0.0; });
  out.floor = CloseKernel(observed, grid.width, grid.height,
                          EllipseKernel(2 * k + 1), 2 * k + 1);

  out.driven.assign(cells, 0);
  if (!path_xy.empty()) {
    CellRaster path(grid);
    for (const Vec2& xy : path_xy) path.Add(xy, 1.0);
    const Mask hits = Cells(path.Reduce(Reducer::kCount, 0.0),
                            [](double v) { return v > 0.0; });
    const int r = std::max(
        1, static_cast<int>(std::ceil(0.5 * config.vehicle_width / grid.res)));
    const std::vector<uint8_t> dilated = DilateKernel(
        hits, grid.width, grid.height, EllipseKernel(2 * r + 1), 2 * r + 1);
    for (size_t c = 0; c < cells; ++c) out.driven[c] = dilated[c] != 0;
  }

  out.slot.assign(cells, 0);
  for (const LayerSlot& slot : slots) {
    FillWorldPolygon(slot.corners, grid, slot.index + 1, &out.slot);
  }
  std::vector<int32_t> ramp(cells, 0);
  for (const Ramp& r : levels.ramps) {
    if (r.level_from == level || r.level_to == level) {
      FillWorldPolygon(r.polygon, grid, 1, &ramp);
    }
  }
  out.ramp.resize(cells);
  for (size_t c = 0; c < cells; ++c) out.ramp[c] = ramp[c] > 0;

  Mask known(cells);
  Mask obstacles(cells);
  for (size_t c = 0; c < cells; ++c) {
    known[c] = out.floor[c] || out.driven[c] || out.ramp[c];
    obstacles[c] = out.structure[c] || out.parked[c];
  }
  known = FillHoles(known, obstacles, grid.width, grid.height, grid.res,
                    config.max_hole_area);
  Mask blocked(cells);
  for (size_t c = 0; c < cells; ++c) {
    blocked[c] = obstacles[c] || !known[c];
    out.floor[c] = known[c] && !(out.driven[c] || out.ramp[c]);
  }
  const std::vector<float> distance =
      DistanceTransform(blocked, grid.width, grid.height);
  const float res = static_cast<float>(grid.res);
  out.clearance.resize(cells);
  for (size_t c = 0; c < cells; ++c) out.clearance[c] = distance[c] * res;
  out.floor_z = FloorHeightRaster(levels, level, grid);
  return out;
}

}  // namespace gr
