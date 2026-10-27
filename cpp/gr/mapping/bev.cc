#include "gr/mapping/bev.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gr/core/check.h"

namespace gr {
namespace {

bool Static(const BevPoints& points, size_t i, int level) {
  if ((*points.level)[i] != level) return false;
  return points.dynamic == nullptr || points.dynamic->empty() ||
         !(*points.dynamic)[i];
}

}  // namespace

Grid2D LevelGrid(const BevPoints& points, int level, double res,
                 double margin) {
  GR_CHECK(res > 0.0);
  Vec2 lo = Vec2::Constant(std::numeric_limits<double>::infinity());
  Vec2 hi = -lo;
  for (size_t i = 0; i < points.xyz->size(); ++i) {
    if (!Static(points, i, level)) continue;
    lo = lo.cwiseMin((*points.xyz)[i].head<2>());
    hi = hi.cwiseMax((*points.xyz)[i].head<2>());
  }
  Grid2D grid;
  grid.res = res;
  if (!std::isfinite(lo.x())) return grid;
  lo -= Vec2::Constant(margin);
  hi += Vec2::Constant(margin);
  grid.x0 = lo.x();
  grid.y0 = lo.y();
  grid.width = static_cast<int>(std::ceil((hi.x() - lo.x()) / res));
  grid.height = static_cast<int>(std::ceil((hi.y() - lo.y()) / res));
  return grid;
}

FloorHeights FloorHeightRaster(const LevelModel& levels, int level,
                               const Grid2D& grid) {
  GR_CHECK(level >= 0 && level < levels.num_levels());
  const size_t l = static_cast<size_t>(level);
  FloorHeights out;
  out.single_precision = l < levels.grids.size() && levels.grids[l].width > 0;
  out.z.resize(static_cast<size_t>(grid.width) * grid.height);
  for (int j = 0; j < grid.height; ++j) {
    for (int i = 0; i < grid.width; ++i) {
      const Vec2 c(grid.x0 + (i + 0.5) * grid.res,
                   grid.y0 + (j + 0.5) * grid.res);
      double z = levels.FloorZ(c, level);
      for (const Ramp& ramp : levels.ramps) {
        if (ramp.level_from == level && ramp.Contains(c)) z = ramp.ZAt(c);
      }
      if (out.single_precision) z = static_cast<float>(z);
      out.z[static_cast<size_t>(j) * grid.width + i] = z;
    }
  }
  return out;
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
  for (size_t i = 0; i < points.xyz->size(); ++i) {
    if (!Static(points, i, level)) continue;
    const float h = (*points.height)[i];
    const Vec2 xy = (*points.xyz)[i].head<2>();
    const float nz =
        with_normals ? static_cast<float>((*points.normals)[i].z()) : 1.0f;
    if (std::abs(h) < tolerance && nz > 0.7f) {
      floor_xy.push_back(xy);
      floor_intensity.push_back((*points.intensity)[i]);
    }
    if (h > tolerance && h < top) {
      object_xy.push_back(xy);
      object_height.push_back(h);
    }
  }
  const auto rows = [](const std::vector<Vec2>& xy) {
    RowMatX2d out(static_cast<Eigen::Index>(xy.size()), 2);
    for (size_t k = 0; k < xy.size(); ++k) {
      out.row(static_cast<Eigen::Index>(k)) = xy[k].transpose();
    }
    return out;
  };
  const auto vector = [](const std::vector<double>& values) {
    return Eigen::Map<const Eigen::VectorXd>(
        values.data(), static_cast<Eigen::Index>(values.size()));
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  BevRasters out;
  const RowMatX2d floor = rows(floor_xy);
  out.intensity =
      Rasterize(floor, vector(floor_intensity), grid, Reducer::kMean, nan);
  out.floor_count = Rasterize(
      floor, Eigen::VectorXd::Ones(static_cast<Eigen::Index>(floor_xy.size())),
      grid, Reducer::kCount, 0.0);
  out.height_max = Rasterize(rows(object_xy), vector(object_height), grid,
                             Reducer::kMax, nan);
  out.floor_z = FloorHeightRaster(levels, level, grid);
  return out;
}

}  // namespace gr
