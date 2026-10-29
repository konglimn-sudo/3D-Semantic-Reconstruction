#include "gr/nav/layers.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "gr/core/check.h"

namespace gr {
namespace {

// Polygon vertices go to FillPoly8 in fixed point with this many fraction
// bits (cv2.fillPoly's shift).
constexpr int kPolygonFractionBits = 4;
constexpr double kPolygonScale = 1 << kPolygonFractionBits;
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;
// Points without a level height count as this far below the floor (m).
constexpr float kNoHeight = -10.0f;
// Floor points face up at least this much (|normal z|, float32 as numpy).
constexpr float kLayerFloorMinNormalZ = 0.7f;
// Overhead structure (ducts, beams) is at least this high (m).
constexpr float kOverheadMinHeight = 0.5f;

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
    RowMatX2d xy(static_cast<Eigen::Index>(xy_.size()), kPlaneDim);
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
  size_t cell = 0;
  for (Eigen::Index row = 0; row < raster.rows(); ++row) {
    for (Eigen::Index column = 0; column < raster.cols(); ++column) {
      out[cell++] = test(raster(row, column));
    }
  }
  return out;
}

}  // namespace

void FillWorldPolygon(const std::vector<Vec2>& polygon, const Grid2D& grid,
                      int32_t value, std::vector<int32_t>* image) {
  std::vector<std::pair<int64_t, int64_t>> vertices;
  vertices.reserve(polygon.size());
  for (const Vec2& vertex : polygon) {
    // np.round(((vertex - origin) / res - 0.5) * 16): half to even.
    vertices.emplace_back(
        static_cast<int64_t>(std::nearbyint(
            ((vertex.x() - grid.x0) / grid.res - kCellCentre) *
            kPolygonScale)),
        static_cast<int64_t>(std::nearbyint(
            ((vertex.y() - grid.y0) / grid.res - kCellCentre) *
            kPolygonScale)));
  }
  FillPoly8(vertices, kPolygonFractionBits, grid.width, grid.height, value,
            image);
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
    bool touches_border = false;
    for (int cell : cells) {
      const int x = cell % width;
      const int y = cell / width;
      touches_border |= x == 0 || y == 0 || x == width - 1 || y == height - 1;
    }
    // area * res ** 2 < max_area.
    const double area = static_cast<double>(cells.size()) * (res * res);
    if (touches_border || area >= max_area) continue;
    for (int cell : cells) out[static_cast<size_t>(cell)] = 1;
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
  const size_t count = map.xyz->size();
  const size_t cells = static_cast<size_t>(grid.width) * grid.height;
  const bool with_normals =
      map.normals != nullptr && map.normals->size() == count;
  const bool with_observations =
      points.observations != nullptr && points.observations->size() == count;
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
  for (size_t i = 0; i < count; ++i) {
    const bool dynamic = map.dynamic != nullptr && !map.dynamic->empty() &&
                         (*map.dynamic)[i];
    if ((*map.level)[i] != level || dynamic) continue;
    const float raw_height = (*map.height)[i];
    const float height = std::isnan(raw_height) ? kNoHeight : raw_height;
    const float normal_z =
        with_normals ? std::abs(static_cast<float>((*map.normals)[i].z()))
                     : 1.0f;
    const Vec2 xy = (*map.xyz)[i].head<2>();
    const int label = (*points.labels)[i];
    const bool vehicle = In(classes.vehicle, label);
    const bool low = In(classes.low, label);
    const double observations =
        with_observations ? (*points.observations)[i]
                          : config.min_cell_observations;
    if (std::abs(height) < floor_tolerance &&
        normal_z > kLayerFloorMinNormalZ) {
      floor_points.Add(xy, 1.0);
    }
    const bool band = height > min_height && height < top;
    if (low) low_points.Add(xy, 1.0);
    if (band && vehicle) vehicles_in_band.Add(xy, 1.0);
    if (band && !vehicle && !low) {  // Structure points.
      if (label == classes.unlabeled) {
        unclassified.Add(xy, observations);
      } else {
        structure_known.Add(xy, 1.0);
      }
      if (In(classes.overhead, label) && height > kOverheadMinHeight) {
        overhead_points.Add(xy, 1.0);
      }
    }
    if (height >= top && height < max_headroom && !vehicle) {  // Above.
      if (label == classes.unlabeled) {
        above_unclassified.Add(xy, observations);
        above_unclassified_height.Add(xy, height);
      } else {
        above_known.Add(xy, height);
      }
    }
  }

  const double min_points = config.obstacle_min_points;
  const double min_observations = config.min_cell_observations;
  const auto at_least = [](double threshold) {
    return [threshold](double value) { return value >= threshold; };
  };
  const auto positive = [](double value) { return value > 0.0; };
  LevelLayers out;
  out.structure = Cells(structure_known.Reduce(Reducer::kCount, 0.0),
                        at_least(min_points));
  const Mask seen_unclassified =
      Cells(unclassified.Reduce(Reducer::kSum, 0.0),
            at_least(min_observations));
  for (size_t cell = 0; cell < cells; ++cell) {
    out.structure[cell] |= seen_unclassified[cell];
  }
  out.overhead = Cells(overhead_points.Reduce(Reducer::kCount, 0.0),
                       at_least(min_points));
  out.parked = Cells(vehicles_in_band.Reduce(Reducer::kCount, 0.0),
                     at_least(min_points));
  out.low =
      Cells(low_points.Reduce(Reducer::kCount, 0.0), at_least(min_points));
  for (size_t cell = 0; cell < cells; ++cell) {
    out.overhead[cell] &= out.structure[cell];
    out.parked[cell] &= !out.structure[cell];
  }

  // Headroom: the lowest static object above the band, unclassified points
  // filtered by their observations.
  constexpr double kInf = std::numeric_limits<double>::infinity();
  const Eigen::MatrixXd lowest_known = above_known.Reduce(Reducer::kMin, kInf);
  const Eigen::MatrixXd lowest_unclassified =
      above_unclassified_height.Reduce(Reducer::kMin, kInf);
  const Mask seen = Cells(above_unclassified.Reduce(Reducer::kSum, 0.0),
                          at_least(min_observations));
  out.headroom.resize(cells);
  for (int row = 0; row < grid.height; ++row) {
    for (int column = 0; column < grid.width; ++column) {
      const size_t cell = static_cast<size_t>(row) * grid.width + column;
      const double known = lowest_known(row, column);
      out.headroom[cell] = static_cast<float>(
          seen[cell] ? std::min(known, lowest_unclassified(row, column))
                     : known);
    }
  }

  const int closing_size = 2 * config.floor_closing + 1;
  const Mask observed =
      Cells(floor_points.Reduce(Reducer::kCount, 0.0), positive);
  out.floor = CloseKernel(observed, grid.width, grid.height,
                          EllipseKernel(closing_size), closing_size);

  out.driven.assign(cells, 0);
  if (!path_xy.empty()) {
    CellRaster path(grid);
    for (const Vec2& xy : path_xy) path.Add(xy, 1.0);
    const Mask hits = Cells(path.Reduce(Reducer::kCount, 0.0), positive);
    const int half_width_cells = std::max(
        1, static_cast<int>(std::ceil(0.5 * config.vehicle_width / grid.res)));
    const int kernel_size = 2 * half_width_cells + 1;
    const std::vector<uint8_t> dilated =
        DilateKernel(hits, grid.width, grid.height,
                     EllipseKernel(kernel_size), kernel_size);
    for (size_t cell = 0; cell < cells; ++cell) {
      out.driven[cell] = dilated[cell] != 0;
    }
  }

  out.slot.assign(cells, 0);
  for (const LayerSlot& slot : slots) {
    FillWorldPolygon(slot.corners, grid, slot.index + 1, &out.slot);
  }
  std::vector<int32_t> ramp(cells, 0);
  for (const Ramp& level_ramp : levels.ramps) {
    if (level_ramp.level_from == level || level_ramp.level_to == level) {
      FillWorldPolygon(level_ramp.polygon, grid, 1, &ramp);
    }
  }
  out.ramp.resize(cells);
  for (size_t cell = 0; cell < cells; ++cell) out.ramp[cell] = ramp[cell] > 0;

  Mask known(cells);
  Mask obstacles(cells);
  for (size_t cell = 0; cell < cells; ++cell) {
    known[cell] = out.floor[cell] || out.driven[cell] || out.ramp[cell];
    obstacles[cell] = out.structure[cell] || out.parked[cell];
  }
  known = FillHoles(known, obstacles, grid.width, grid.height, grid.res,
                    config.max_hole_area);
  Mask blocked(cells);
  for (size_t cell = 0; cell < cells; ++cell) {
    blocked[cell] = obstacles[cell] || !known[cell];
    out.floor[cell] = known[cell] && !(out.driven[cell] || out.ramp[cell]);
  }
  const std::vector<float> distance =
      DistanceTransform(blocked, grid.width, grid.height);
  const float res = static_cast<float>(grid.res);
  out.clearance.resize(cells);
  for (size_t cell = 0; cell < cells; ++cell) {
    out.clearance[cell] = distance[cell] * res;
  }
  out.floor_z = FloorHeightRaster(levels, level, grid);
  return out;
}

}  // namespace gr
