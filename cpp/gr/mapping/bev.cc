#include "gr/mapping/bev.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gr/core/check.h"

namespace gr {
namespace {

// Floor points have a normal at most ~45 deg from vertical (its z above
// this).
constexpr float kMinFloorNormalZ = 0.7f;

// Whether point `index` is a static point of `level`.
bool IsStaticOnLevel(const BevPoints& points, size_t index, int level) {
  if ((*points.level)[index] != level) return false;
  return points.dynamic == nullptr || points.dynamic->empty() ||
         !(*points.dynamic)[index];
}

}  // namespace

Grid2D LevelGrid(const BevPoints& points, int level, double res,
                 double margin) {
  GR_CHECK(res > 0.0);
  Vec2 lower = Vec2::Constant(std::numeric_limits<double>::infinity());
  Vec2 upper = -lower;
  for (size_t index = 0; index < points.xyz->size(); ++index) {
    if (!IsStaticOnLevel(points, index, level)) continue;
    lower = lower.cwiseMin((*points.xyz)[index].head<kPlaneDim>());
    upper = upper.cwiseMax((*points.xyz)[index].head<kPlaneDim>());
  }
  Grid2D grid;
  grid.res = res;
  if (!std::isfinite(lower.x())) return grid;
  lower -= Vec2::Constant(margin);
  upper += Vec2::Constant(margin);
  grid.x0 = lower.x();
  grid.y0 = lower.y();
  grid.width = static_cast<int>(std::ceil((upper.x() - lower.x()) / res));
  grid.height = static_cast<int>(std::ceil((upper.y() - lower.y()) / res));
  return grid;
}

FloorHeights FloorHeightRaster(const LevelModel& levels, int level,
                               const Grid2D& grid) {
  GR_CHECK(level >= 0 && level < levels.num_levels());
  const size_t level_index = static_cast<size_t>(level);
  FloorHeights heights;
  heights.single_precision = level_index < levels.grids.size() &&
                             levels.grids[level_index].width > 0;
  heights.z.resize(static_cast<size_t>(grid.width) * grid.height);
  for (int row = 0; row < grid.height; ++row) {
    for (int col = 0; col < grid.width; ++col) {
      const Vec2 center(grid.x0 + (col + 0.5) * grid.res,
                        grid.y0 + (row + 0.5) * grid.res);
      double z = levels.FloorZ(center, level);
      for (const Ramp& ramp : levels.ramps) {
        if (ramp.level_from == level && ramp.Contains(center)) {
          z = ramp.ZAt(center);
        }
      }
      if (heights.single_precision) z = static_cast<float>(z);
      heights.z[static_cast<size_t>(row) * grid.width + col] = z;
    }
  }
  return heights;
}

BevRasters LevelRasters(const BevPoints& points, const LevelModel& levels,
                        int level, const Grid2D& grid,
                        const BevConfig& config) {
  const float tolerance = static_cast<float>(config.floor_tolerance);
  const float top = static_cast<float>(config.object_max_height);
  const bool with_normals = points.normals != nullptr &&
                            points.normals->size() == points.xyz->size();
  std::vector<Vec2> floor_xy;
  std::vector<double> floor_intensity;
  std::vector<Vec2> object_xy;
  std::vector<double> object_height;
  for (size_t index = 0; index < points.xyz->size(); ++index) {
    if (!IsStaticOnLevel(points, index, level)) continue;
    const float height = (*points.height)[index];
    const Vec2 xy = (*points.xyz)[index].head<kPlaneDim>();
    const float normal_z =
        with_normals ? static_cast<float>((*points.normals)[index].z()) : 1.0f;
    if (std::abs(height) < tolerance && normal_z > kMinFloorNormalZ) {
      floor_xy.push_back(xy);
      floor_intensity.push_back((*points.intensity)[index]);
    }
    if (height > tolerance && height < top) {
      object_xy.push_back(xy);
      object_height.push_back(height);
    }
  }
  const auto to_rows = [](const std::vector<Vec2>& xy) {
    RowMatX2d rows(static_cast<Eigen::Index>(xy.size()), kPlaneDim);
    for (size_t point = 0; point < xy.size(); ++point) {
      rows.row(static_cast<Eigen::Index>(point)) = xy[point].transpose();
    }
    return rows;
  };
  const auto as_vector = [](const std::vector<double>& values) {
    return Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  BevRasters rasters;
  const RowMatX2d floor = to_rows(floor_xy);
  rasters.intensity =
      Rasterize(floor, as_vector(floor_intensity), grid, Reducer::kMean, nan);
  rasters.floor_count = Rasterize(
      floor, Eigen::VectorXd::Ones(static_cast<Eigen::Index>(floor_xy.size())),
      grid, Reducer::kCount, 0.0);
  rasters.height_max = Rasterize(to_rows(object_xy), as_vector(object_height),
                                 grid, Reducer::kMax, nan);
  rasters.floor_z = FloorHeightRaster(levels, level, grid);
  return rasters;
}

}  // namespace gr
