#include "gr/mapping/levels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/geometry.h"
#include "gr/core/kdtree.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// No such cell / row / parabola.
constexpr int kNone = -1;

// Path slopes are measured over this arc length (m) either side of a sample,
// on a base of at least kMinSlopeBase (m).
constexpr double kSlopeHalfWindow = 1.0;
constexpr double kMinSlopeBase = 0.2;
// A stretch is a ramp where the slope is this many times the flat slope,
// over at least kMinRampLength (m).
constexpr double kRampSlopeFactor = 2.0;
constexpr double kMinRampLength = 3.0;
// Flat heights farther apart than this share of the level separation start
// a new level.
constexpr double kLevelGapShare = 0.5;

// The part of a ramp stretch (shares of its arc length) away from the
// levels it joins, and the samples it needs.
constexpr double kRampCoreStart = 0.15;
constexpr double kRampCoreEnd = 0.85;
constexpr size_t kMinRampCoreSamples = 3;
// Ramp floor points: within this (m, horizontal) of the core path and this
// (m, vertical) of its height.
constexpr double kRampSearchRadius = 3.5;
constexpr double kRampFloorBand = 0.2;
// Points needed for a ramp plane and for its outline.
constexpr size_t kMinRampPoints = 50;
// The plane is refitted on the points within kRampFitTolerance (m), this
// many times.
constexpr int kRampTrimRounds = 2;
constexpr double kRampFitTolerance = 0.05;
// Two ramp planes closer than this (coefficients) are the same ramp.
constexpr double kDuplicateRampTolerance = 0.05;
// Outline points: farther than this (m) from both levels' floors, and within
// kRampOutlineRadius (m) of the ramp stretch.
constexpr double kRampLevelClearance = 0.1;
constexpr double kRampOutlineRadius = 4.5;
// Points this close (m) to a ramp surface are ramp, not level floor.
constexpr double kRampSurfaceTolerance = 0.1;

// The floor grids extend this far (m) beyond the map.
constexpr double kGridMargin = 1.0;
// The floor grid is smoothed by a median over this many cells per side.
constexpr int kMedianWindowSide = 3;

// Ceilings: downward-facing surfaces at least this high (m) above the
// floor; the top level looks up to kTopLevelHeadroom (m).
constexpr double kMinCeilingClearance = 2.0;
constexpr double kTopLevelHeadroom = 6.0;
// The ceiling height is this percentile of their clearance, given more
// than kMinCeilingPoints; otherwise kDefaultCeilingHeight (m).
constexpr double kCeilingPercentile = 90.0;
constexpr size_t kMinCeilingPoints = 50;
constexpr double kDefaultCeilingHeight = 3.0;

// Arc length along a path in the xy plane.
std::vector<double> ArcLength(const Points& path) {
  std::vector<double> arc_length(path.size(), 0.0);
  for (size_t k = 1; k < path.size(); ++k) {
    arc_length[k] =
        arc_length[k - 1] + (path[k] - path[k - 1]).head<2>().norm();
  }
  return arc_length;
}

// numpy.searchsorted(values, query) (side "left").
size_t SearchSorted(const std::vector<double>& values, double query) {
  return static_cast<size_t>(
      std::lower_bound(values.begin(), values.end(), query) - values.begin());
}

// [begin, end) runs of true flags.
std::vector<std::pair<size_t, size_t>> Runs(const std::vector<uint8_t>& flags) {
  std::vector<std::pair<size_t, size_t>> out;
  const size_t no_run = flags.size();
  size_t start = no_run;
  for (size_t k = 0; k < flags.size(); ++k) {
    if (flags[k] && start == no_run) start = k;
    if (!flags[k] && start != no_run) {
      out.emplace_back(start, k);
      start = no_run;
    }
  }
  if (start != no_run) out.emplace_back(start, flags.size());
  return out;
}

int NearestLevel(const std::vector<double>& level_z, double z) {
  int best = 0;
  for (size_t level = 1; level < level_z.size(); ++level) {
    if (std::abs(level_z[level] - z) <
        std::abs(level_z[static_cast<size_t>(best)] - z)) {
      best = static_cast<int>(level);
    }
  }
  return best;
}

// z = a x + b y + c by least squares on the points flagged in `use`;
// returns (a, b, c).
Vec3 FitPlane(const Points& points, const std::vector<uint8_t>& use) {
  Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
  Vec3 right_side = Vec3::Zero();
  for (size_t i = 0; i < points.size(); ++i) {
    if (!use[i]) continue;
    const Vec3 row(points[i].x(), points[i].y(), 1.0);
    normal_matrix += row * row.transpose();
    right_side += row * points[i].z();
  }
  return normal_matrix.ldlt().solve(right_side);
}

// The height of plane (a, b, c) at `point`'s xy.
double PlaneHeight(const Vec3& plane, const Vec3& point) {
  return plane.x() * point.x() + plane.y() * point.y() + plane.z();
}

// Per-cell median (the lower middle element for an even count, as the
// Python tooling did) of `z` at `xy`; NaN for empty cells.
std::vector<float> CellMedian(const std::vector<Vec2>& xy,
                              const std::vector<double>& z, double x0,
                              double y0, double res, int width, int height) {
  std::vector<std::vector<double>> cells(static_cast<size_t>(width) *
                                         static_cast<size_t>(height));
  for (size_t k = 0; k < xy.size(); ++k) {
    const int column = static_cast<int>((xy[k].x() - x0) / res);
    const int row = static_cast<int>((xy[k].y() - y0) / res);
    if (column < 0 || row < 0 || column >= width || row >= height) continue;
    cells[static_cast<size_t>(row) * static_cast<size_t>(width) +
          static_cast<size_t>(column)]
        .push_back(z[k]);
  }
  std::vector<float> out(cells.size(), std::numeric_limits<float>::quiet_NaN());
  for (size_t cell = 0; cell < cells.size(); ++cell) {
    std::vector<double>& values = cells[cell];
    if (values.empty()) continue;
    const size_t middle = (values.size() - 1) / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<long>(middle), values.end());
    out[cell] = static_cast<float>(values[middle]);
  }
  return out;
}

// Exact Euclidean nearest-feature transform (Felzenszwalb & Huttenlocher):
// the index of the nearest finite cell of every cell.
std::vector<int> NearestFinite(const std::vector<float>& grid, int width,
                               int height) {
  const size_t cells = grid.size();
  // Along columns: nearest finite row per cell.
  std::vector<int> nearest_row(cells, kNone);
  for (int column = 0; column < width; ++column) {
    int last = kNone;
    for (int row = 0; row < height; ++row) {
      const size_t cell = static_cast<size_t>(row * width + column);
      if (std::isfinite(grid[cell])) last = row;
      nearest_row[cell] = last;
    }
    last = kNone;
    for (int row = height - 1; row >= 0; --row) {
      const size_t cell = static_cast<size_t>(row * width + column);
      if (std::isfinite(grid[cell])) last = row;
      if (last != kNone && (nearest_row[cell] == kNone ||
                            last - row < row - nearest_row[cell])) {
        nearest_row[cell] = last;
      }
    }
  }
  // Along rows: lower envelope of the parabolas (x - apex)^2 + dy(apex)^2.
  std::vector<int> out(cells, kNone);
  std::vector<int> apexes(static_cast<size_t>(width));
  std::vector<double> boundary(static_cast<size_t>(width) + 1);
  for (int row = 0; row < height; ++row) {
    // Squared row distance from column `apex` to its nearest finite row.
    const auto row_distance = [&](int apex) {
      const int finite_row =
          nearest_row[static_cast<size_t>(row * width + apex)];
      return finite_row == kNone
                 ? kInf
                 : static_cast<double>((finite_row - row) * (finite_row - row));
    };
    int last = kNone;  // The rightmost parabola of the envelope.
    for (int column = 0; column < width; ++column) {
      const double offset = row_distance(column);
      if (!std::isfinite(offset)) continue;
      double crossing = -kInf;
      while (last != kNone) {
        const int apex = apexes[static_cast<size_t>(last)];
        crossing = ((offset + column * column) -
                    (row_distance(apex) + apex * apex)) /
                   (2.0 * (column - apex));
        if (crossing > boundary[static_cast<size_t>(last)]) break;
        --last;
      }
      ++last;
      apexes[static_cast<size_t>(last)] = column;
      boundary[static_cast<size_t>(last)] = last == 0 ? -kInf : crossing;
      boundary[static_cast<size_t>(last) + 1] = kInf;
    }
    if (last == kNone) continue;
    int parabola = 0;
    for (int column = 0; column < width; ++column) {
      while (boundary[static_cast<size_t>(parabola) + 1] < column) ++parabola;
      const int apex = apexes[static_cast<size_t>(parabola)];
      out[static_cast<size_t>(row * width + column)] =
          nearest_row[static_cast<size_t>(row * width + apex)] * width + apex;
    }
  }
  return out;
}

// Median filter over kMedianWindowSide x kMedianWindowSide cells; the
// border repeats the edge cells.
std::vector<float> MedianFilter(const std::vector<float>& grid, int width,
                                int height) {
  constexpr int kReach = kMedianWindowSide / 2;
  constexpr int kWindow = kMedianWindowSide * kMedianWindowSide;
  constexpr int kMiddle = kWindow / 2;
  std::vector<float> out(grid.size());
  std::array<float, kWindow> window;
  for (int row = 0; row < height; ++row) {
    for (int column = 0; column < width; ++column) {
      int filled = 0;
      for (int dy = -kReach; dy <= kReach; ++dy) {
        for (int dx = -kReach; dx <= kReach; ++dx) {
          const int y = std::clamp(row + dy, 0, height - 1);
          const int x = std::clamp(column + dx, 0, width - 1);
          window[static_cast<size_t>(filled++)] =
              grid[static_cast<size_t>(y * width + x)];
        }
      }
      std::nth_element(window.begin(), window.begin() + kMiddle,
                       window.end());
      out[static_cast<size_t>(row * width + column)] =
          window[static_cast<size_t>(kMiddle)];
    }
  }
  return out;
}

// Ramp planes from the sloped path stretches: fitted on floor points beside
// the middle part of the stretch (its ends blend into the level floors),
// outlined by the points that fit the plane but lie on neither level.
std::vector<Ramp> FitRamps(const std::vector<Points>& segments,
                           const std::vector<double>& level_z,
                           const Points& xyz,
                           const std::vector<double>& normal_z) {
  std::vector<Ramp> ramps;
  const size_t count = xyz.size();
  for (const Points& segment : segments) {
    double z_top = -kInf;
    double z_bottom = kInf;
    for (const Vec3& sample : segment) {
      z_top = std::max(z_top, sample.z());
      z_bottom = std::min(z_bottom, sample.z());
    }
    const int from = NearestLevel(level_z, z_top);
    const int to = NearestLevel(level_z, z_bottom);
    if (from == to) continue;
    const std::vector<double> arc_length = ArcLength(segment);
    Points core;
    for (size_t k = 0; k < segment.size(); ++k) {
      if (arc_length[k] > kRampCoreStart * arc_length.back() &&
          arc_length[k] < kRampCoreEnd * arc_length.back()) {
        core.push_back(segment[k]);
      }
    }
    if (core.size() < kMinRampCoreSamples) continue;
    std::vector<Vec2> core_xy;
    for (const Vec3& sample : core) core_xy.push_back(sample.head<2>());
    const KdTree2 core_tree(core_xy);
    std::vector<uint8_t> flags(count, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
      const size_t i = static_cast<size_t>(index);
      if (normal_z[i] <= kRampMinNormalZ) continue;
      const KdTree2::Hit hit =
          core_tree.Nearest(xyz[i].head<2>(), kRampSearchRadius);
      flags[i] = hit.index >= 0 &&
                 std::abs(xyz[i].z() -
                          core[static_cast<size_t>(hit.index)].z()) <
                     kRampFloorBand;
    }
    Points near;
    for (size_t i = 0; i < count; ++i) {
      if (flags[i]) near.push_back(xyz[i]);
    }
    if (near.size() < kMinRampPoints) continue;
    std::vector<uint8_t> use(near.size(), 1);
    Vec3 plane = FitPlane(near, use);
    for (int round = 0; round < kRampTrimRounds; ++round) {
      for (size_t k = 0; k < near.size(); ++k) {
        use[k] = std::abs(PlaneHeight(plane, near[k]) - near[k].z()) <
                 kRampFitTolerance;
      }
      plane = FitPlane(near, use);
    }
    bool duplicate = false;  // The same ramp driven twice.
    for (const Ramp& ramp : ramps) {
      duplicate |= (ramp.plane - plane).cwiseAbs().maxCoeff() <
                       kDuplicateRampTolerance &&
                   ramp.level_from == from;
    }
    if (duplicate) continue;
    std::vector<Vec2> segment_xy;
    for (const Vec3& sample : segment) segment_xy.push_back(sample.head<2>());
    const KdTree2 segment_tree(segment_xy);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
      const size_t i = static_cast<size_t>(index);
      flags[i] =
          normal_z[i] > kRampMinNormalZ &&
          std::abs(xyz[i].z() - PlaneHeight(plane, xyz[i])) <
              kRampFitTolerance &&
          std::abs(xyz[i].z() - level_z[static_cast<size_t>(from)]) >
              kRampLevelClearance &&
          std::abs(xyz[i].z() - level_z[static_cast<size_t>(to)]) >
              kRampLevelClearance &&
          segment_tree.Nearest(xyz[i].head<2>(), kRampOutlineRadius).index >=
              0;
    }
    std::vector<Vec2> outline_points;
    for (size_t i = 0; i < count; ++i) {
      if (flags[i]) outline_points.push_back(xyz[i].head<2>());
    }
    if (outline_points.size() < kMinRampPoints) continue;
    Ramp ramp;
    ramp.level_from = from;
    ramp.level_to = to;
    ramp.plane = plane;
    ramp.polygon = ConvexHull2D(outline_points);
    ramp.slope = std::hypot(plane.x(), plane.y());
    ramps.push_back(ramp);
  }
  return ramps;
}

}  // namespace

double FloorGrid::Sample(const Vec2& xy) const {
  GR_CHECK(width > 0 && height > 0);
  const int column =
      std::clamp(static_cast<int>((xy.x() - x0) / res), 0, width - 1);
  const int row =
      std::clamp(static_cast<int>((xy.y() - y0) / res), 0, height - 1);
  return z[static_cast<size_t>(row * width + column)];
}

bool Ramp::Contains(const Vec2& xy) const {
  return PointInPolygon(polygon, xy);
}

double LevelModel::FloorZ(const Vec2& xy, int level) const {
  GR_CHECK(level >= 0 && level < num_levels());
  const size_t index = static_cast<size_t>(level);
  if (index < grids.size() && grids[index].width > 0) {
    return grids[index].Sample(xy);
  }
  return z[index];
}

LevelAssignment AssignLevels(const LevelModel& model, const Points& xyz) {
  const size_t count = xyz.size();
  LevelAssignment out;
  out.level.assign(count, kNoLevel);
  out.height.assign(count, std::numeric_limits<float>::quiet_NaN());
  out.on_ramp.assign(count, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
    const size_t i = static_cast<size_t>(index);
    const Vec2 xy = xyz[i].head<2>();
    double best = kInf;
    for (int level = 0; level < model.num_levels(); ++level) {
      const double height = xyz[i].z() - model.FloorZ(xy, level);
      if (height >= -kBelowFloorTolerance && height < best) {
        best = height;
        out.level[i] = static_cast<int16_t>(level);
      }
    }
    for (const Ramp& ramp : model.ramps) {
      if (!ramp.Contains(xy)) continue;
      const double height = xyz[i].z() - ramp.ZAt(xy);
      if (height >= -kBelowFloorTolerance && height < best) {
        best = height;
        out.level[i] = static_cast<int16_t>(ramp.level_from);
        out.on_ramp[i] = 1;
      }
    }
    if (std::isfinite(best)) out.height[i] = static_cast<float>(best);
  }
  return out;
}

LevelModel DetectLevels(const std::vector<Points>& vehicle_paths,
                        const Points& xyz, const Points& normals,
                        const LevelConfig& config, const Logger& log,
                        const std::optional<MapBounds>& bounds) {
  GR_CHECK(normals.size() == xyz.size());
  GR_CHECK(config.grid_res > 0.0);
  // Slope along every path over +-kSlopeHalfWindow of arc length.
  std::vector<double> flat_z;
  Points flat_path;
  std::vector<Points> ramp_segments;
  for (const Points& path : vehicle_paths) {
    const std::vector<double> arc_length = ArcLength(path);
    std::vector<double> slope(path.size(), 0.0);
    for (size_t k = 0; k < path.size(); ++k) {
      const size_t behind =
          SearchSorted(arc_length, arc_length[k] - kSlopeHalfWindow);
      const size_t ahead =
          std::min(path.size() - 1,
                   SearchSorted(arc_length, arc_length[k] + kSlopeHalfWindow));
      const double base = arc_length[ahead] - arc_length[behind];
      slope[k] = base > kMinSlopeBase
                     ? (path[ahead].z() - path[behind].z()) / base
                     : 0.0;
    }
    std::vector<uint8_t> steep(path.size());
    for (size_t k = 0; k < path.size(); ++k) {
      if (std::abs(slope[k]) < config.flat_slope) {
        flat_z.push_back(path[k].z());
        flat_path.push_back(path[k]);
      }
      steep[k] = std::abs(slope[k]) >= kRampSlopeFactor * config.flat_slope;
    }
    for (const auto& [begin, end] : Runs(steep)) {
      if (arc_length[end - 1] - arc_length[begin] > kMinRampLength) {
        ramp_segments.emplace_back(path.begin() + static_cast<long>(begin),
                                   path.begin() + static_cast<long>(end));
      }
    }
  }
  // Levels: clusters of the flat heights.
  std::vector<double> sorted_z = flat_z;
  std::sort(sorted_z.begin(), sorted_z.end());
  std::vector<std::vector<double>> clusters;
  for (double z : sorted_z) {
    if (clusters.empty() ||
        z - clusters.back().back() > kLevelGapShare * config.min_separation) {
      clusters.emplace_back();
    }
    clusters.back().push_back(z);
  }
  std::vector<double> level_z;
  for (const std::vector<double>& cluster : clusters) {
    if (static_cast<int>(cluster.size()) >= config.min_samples) {
      level_z.push_back(Median(cluster));
    }
  }
  std::sort(level_z.rbegin(), level_z.rend());
  if (level_z.empty()) {
    level_z.push_back(sorted_z.empty() ? 0.0 : Median(sorted_z));
  }

  const size_t count = xyz.size();
  std::vector<double> normal_z(count);
  for (size_t i = 0; i < count; ++i) normal_z[i] = normals[i].z();
  LevelModel model;
  model.z = level_z;
  model.ramps = FitRamps(ramp_segments, level_z, xyz, normal_z);

  // Floor grid per level: per-cell median of upward-facing points near the
  // level's own path height, ramp surfaces excluded.
  std::vector<uint8_t> on_ramp(count, 0);
  for (const Ramp& ramp : model.ramps) {
    for (size_t i = 0; i < count; ++i) {
      if (std::abs(xyz[i].z() - ramp.ZAt(xyz[i].head<2>())) <
              kRampSurfaceTolerance &&
          ramp.Contains(xyz[i].head<2>())) {
        on_ramp[i] = 1;
      }
    }
  }
  Vec2 lo = Vec2::Constant(kInf);
  Vec2 hi = Vec2::Constant(-kInf);
  if (bounds) {
    lo = bounds->lo;
    hi = bounds->hi;
  } else {
    for (const Vec3& point : xyz) {
      lo = lo.cwiseMin(point.head<2>());
      hi = hi.cwiseMax(point.head<2>());
    }
  }
  lo -= Vec2::Constant(kGridMargin);
  hi += Vec2::Constant(kGridMargin);
  const double res = config.grid_res;
  const int width = static_cast<int>(std::ceil((hi.x() - lo.x()) / res));
  const int height = static_cast<int>(std::ceil((hi.y() - lo.y()) / res));
  std::vector<int> owner(flat_path.size());
  for (size_t k = 0; k < flat_path.size(); ++k) {
    owner[k] = NearestLevel(level_z, flat_path[k].z());
  }
  for (size_t level = 0; level < level_z.size(); ++level) {
    Points own_path;
    for (size_t k = 0; k < flat_path.size(); ++k) {
      if (owner[k] == static_cast<int>(level)) own_path.push_back(flat_path[k]);
    }
    // The height of the level's own path nearest to each point.
    std::vector<double> path_height(count, level_z[level]);
    if (static_cast<int>(own_path.size()) >= config.min_samples) {
      std::vector<Vec2> own_path_xy;
      for (const Vec3& sample : own_path) {
        own_path_xy.push_back(sample.head<2>());
      }
      const KdTree2 tree(own_path_xy);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
        const size_t i = static_cast<size_t>(index);
        path_height[i] =
            own_path[static_cast<size_t>(tree.Nearest(xyz[i].head<2>()).index)]
                .z();
      }
    }
    std::vector<Vec2> cell_xy;
    std::vector<double> cell_z;
    for (size_t i = 0; i < count; ++i) {
      if (normal_z[i] > kFloorMinNormalZ && !on_ramp[i] &&
          std::abs(xyz[i].z() - path_height[i]) < config.floor_band) {
        cell_xy.push_back(xyz[i].head<2>());
        cell_z.push_back(xyz[i].z());
      }
    }
    FloorGrid grid;
    grid.x0 = lo.x();
    grid.y0 = lo.y();
    grid.res = res;
    grid.width = width;
    grid.height = height;
    grid.z = CellMedian(cell_xy, cell_z, lo.x(), lo.y(), res, width, height);
    if (std::any_of(grid.z.begin(), grid.z.end(),
                    [](float value) { return std::isfinite(value); })) {
      const std::vector<int> nearest = NearestFinite(grid.z, width, height);
      std::vector<float> filled(grid.z.size());
      for (size_t cell = 0; cell < filled.size(); ++cell) {
        filled[cell] = grid.z[static_cast<size_t>(nearest[cell])];
      }
      grid.z = MedianFilter(filled, width, height);
    } else {
      grid.z.assign(grid.z.size(), static_cast<float>(level_z[level]));
    }
    model.grids.push_back(std::move(grid));
  }

  // Ceiling per level: downward-facing surfaces well above its floor (below
  // the next level up).
  for (size_t level = 0; level < level_z.size(); ++level) {
    const double top =
        level > 0 ? level_z[level - 1] - level_z[level] : kTopLevelHeadroom;
    std::vector<double> clearance;
    for (size_t i = 0; i < count; ++i) {
      if (normal_z[i] >= kCeilingMaxNormalZ) continue;
      const double above =
          xyz[i].z() - model.grids[level].Sample(xyz[i].head<2>());
      if (above > kMinCeilingClearance && above < top) {
        clearance.push_back(above);
      }
    }
    model.ceiling.push_back(
        clearance.size() > kMinCeilingPoints
            ? level_z[level] + Percentile(clearance, kCeilingPercentile)
            : level_z[level] + kDefaultCeilingHeight);
  }
  if (log) {
    std::string heights;
    std::string ceilings;
    std::string slopes;
    for (size_t level = 0; level < level_z.size(); ++level) {
      const char* separator = level ? ", " : "";
      heights += StringPrintf("%s%.2f", separator, level_z[level]);
      ceilings += StringPrintf("%s%.2f", separator, model.ceiling[level]);
    }
    for (size_t ramp = 0; ramp < model.ramps.size(); ++ramp) {
      slopes += StringPrintf("%s%.3f", ramp ? ", " : "",
                             model.ramps[ramp].slope);
    }
    LogLine(log, StringPrintf("[levels] %zu levels at z=[%s], ceilings [%s], "
                              "%zu ramps [%s]",
                              level_z.size(), heights.c_str(),
                              ceilings.c_str(), model.ramps.size(),
                              slopes.c_str()));
  }
  return model;
}

}  // namespace gr
