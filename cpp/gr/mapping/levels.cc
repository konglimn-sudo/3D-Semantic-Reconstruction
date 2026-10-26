#include "gr/mapping/levels.h"

#include <algorithm>
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

// Arc length along a path in the xy plane.
std::vector<double> ArcLength(const Points& path) {
  std::vector<double> s(path.size(), 0.0);
  for (size_t k = 1; k < path.size(); ++k) {
    s[k] = s[k - 1] + (path[k] - path[k - 1]).head<2>().norm();
  }
  return s;
}

// numpy.searchsorted(values, query) (side "left").
size_t SearchSorted(const std::vector<double>& values, double query) {
  return static_cast<size_t>(
      std::lower_bound(values.begin(), values.end(), query) - values.begin());
}

// [begin, end) runs of true flags.
std::vector<std::pair<size_t, size_t>> Runs(const std::vector<uint8_t>& flags) {
  std::vector<std::pair<size_t, size_t>> out;
  size_t start = flags.size();
  for (size_t k = 0; k < flags.size(); ++k) {
    if (flags[k] && start == flags.size()) start = k;
    if (!flags[k] && start != flags.size()) {
      out.emplace_back(start, k);
      start = flags.size();
    }
  }
  if (start != flags.size()) out.emplace_back(start, flags.size());
  return out;
}

int NearestLevel(const std::vector<double>& level_z, double z) {
  int best = 0;
  for (size_t i = 1; i < level_z.size(); ++i) {
    if (std::abs(level_z[i] - z) <
        std::abs(level_z[static_cast<size_t>(best)] - z)) {
      best = static_cast<int>(i);
    }
  }
  return best;
}

// z = a x + b y + c by least squares on the points flagged in `use`.
Vec3 FitPlane(const Points& points, const std::vector<uint8_t>& use) {
  Eigen::Matrix3d ata = Eigen::Matrix3d::Zero();
  Vec3 atb = Vec3::Zero();
  for (size_t i = 0; i < points.size(); ++i) {
    if (!use[i]) continue;
    const Vec3 row(points[i].x(), points[i].y(), 1.0);
    ata += row * row.transpose();
    atb += row * points[i].z();
  }
  return ata.ldlt().solve(atb);
}

// Per-cell median (the lower middle element for an even count, as the
// Python tooling did) of `z` at `xy`; NaN for empty cells.
std::vector<float> CellMedian(const std::vector<Vec2>& xy,
                              const std::vector<double>& z, double x0,
                              double y0, double res, int width, int height) {
  std::vector<std::vector<double>> cells(static_cast<size_t>(width) *
                                         static_cast<size_t>(height));
  for (size_t k = 0; k < xy.size(); ++k) {
    const int i = static_cast<int>((xy[k].x() - x0) / res);
    const int j = static_cast<int>((xy[k].y() - y0) / res);
    if (i < 0 || j < 0 || i >= width || j >= height) continue;
    cells[static_cast<size_t>(j) * static_cast<size_t>(width) +
          static_cast<size_t>(i)]
        .push_back(z[k]);
  }
  std::vector<float> out(cells.size(), std::numeric_limits<float>::quiet_NaN());
  for (size_t c = 0; c < cells.size(); ++c) {
    std::vector<double>& v = cells[c];
    if (v.empty()) continue;
    const size_t middle = (v.size() - 1) / 2;
    std::nth_element(v.begin(), v.begin() + static_cast<long>(middle), v.end());
    out[c] = static_cast<float>(v[middle]);
  }
  return out;
}

// Exact Euclidean nearest-feature transform (Felzenszwalb & Huttenlocher):
// the index of the nearest finite cell of every cell.
std::vector<int> NearestFinite(const std::vector<float>& grid, int width,
                               int height) {
  const size_t n = grid.size();
  // Along columns: nearest finite row per cell.
  std::vector<int> row_of(n, -1);
  for (int i = 0; i < width; ++i) {
    int last = -1;
    for (int j = 0; j < height; ++j) {
      const size_t c = static_cast<size_t>(j * width + i);
      if (std::isfinite(grid[c])) last = j;
      row_of[c] = last;
    }
    last = -1;
    for (int j = height - 1; j >= 0; --j) {
      const size_t c = static_cast<size_t>(j * width + i);
      if (std::isfinite(grid[c])) last = j;
      if (last >= 0 && (row_of[c] < 0 || last - j < j - row_of[c])) {
        row_of[c] = last;
      }
    }
  }
  // Along rows: lower envelope of the parabolas (i - q)^2 + dy(q)^2.
  std::vector<int> out(n, -1);
  std::vector<int> v(static_cast<size_t>(width));
  std::vector<double> boundary(static_cast<size_t>(width) + 1);
  for (int j = 0; j < height; ++j) {
    auto f = [&](int q) {
      const int r = row_of[static_cast<size_t>(j * width + q)];
      return r < 0 ? kInf : static_cast<double>((r - j) * (r - j));
    };
    int k = -1;
    for (int q = 0; q < width; ++q) {
      const double fq = f(q);
      if (!std::isfinite(fq)) continue;
      double s = -kInf;
      while (k >= 0) {
        const int p = v[static_cast<size_t>(k)];
        s = ((fq + q * q) - (f(p) + p * p)) / (2.0 * (q - p));
        if (s > boundary[static_cast<size_t>(k)]) break;
        --k;
      }
      ++k;
      v[static_cast<size_t>(k)] = q;
      boundary[static_cast<size_t>(k)] = k == 0 ? -kInf : s;
      boundary[static_cast<size_t>(k) + 1] = kInf;
    }
    if (k < 0) continue;
    int m = 0;
    for (int q = 0; q < width; ++q) {
      while (boundary[static_cast<size_t>(m) + 1] < q) ++m;
      const int p = v[static_cast<size_t>(m)];
      out[static_cast<size_t>(j * width + q)] =
          row_of[static_cast<size_t>(j * width + p)] * width + p;
    }
  }
  return out;
}

// 3 x 3 median filter; the border repeats the edge cells.
std::vector<float> Median3x3(const std::vector<float>& grid, int width,
                             int height) {
  std::vector<float> out(grid.size());
  float window[9];
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      int n = 0;
      for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          const int jj = std::clamp(j + dj, 0, height - 1);
          const int ii = std::clamp(i + di, 0, width - 1);
          window[n++] = grid[static_cast<size_t>(jj * width + ii)];
        }
      }
      std::nth_element(window, window + 4, window + 9);
      out[static_cast<size_t>(j * width + i)] = window[4];
    }
  }
  return out;
}

// Ramp planes from the sloped path stretches: fitted on floor points beside
// the middle part of the stretch (its ends blend into the level floors),
// outlined by the points that fit the plane but lie on neither level.
std::vector<Ramp> FitRamps(const std::vector<Points>& segments,
                           const std::vector<double>& level_z,
                           const Points& xyz, const std::vector<double>& nz) {
  std::vector<Ramp> ramps;
  const size_t n = xyz.size();
  for (const Points& seg : segments) {
    double z_top = -kInf;
    double z_bottom = kInf;
    for (const Vec3& p : seg) {
      z_top = std::max(z_top, p.z());
      z_bottom = std::min(z_bottom, p.z());
    }
    const int from = NearestLevel(level_z, z_top);
    const int to = NearestLevel(level_z, z_bottom);
    if (from == to) continue;
    const std::vector<double> s = ArcLength(seg);
    Points core;
    for (size_t k = 0; k < seg.size(); ++k) {
      if (s[k] > 0.15 * s.back() && s[k] < 0.85 * s.back()) {
        core.push_back(seg[k]);
      }
    }
    if (core.size() < 3) continue;
    std::vector<Vec2> core_xy;
    for (const Vec3& p : core) core_xy.push_back(p.head<2>());
    const KdTree2 core_tree(core_xy);
    std::vector<uint8_t> flags(n, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      if (nz[i] <= 0.8) continue;
      const KdTree2::Hit hit = core_tree.Nearest(xyz[i].head<2>(), 3.5);
      flags[i] = hit.index >= 0 &&
                 std::abs(xyz[i].z() -
                          core[static_cast<size_t>(hit.index)].z()) < 0.2;
    }
    Points near;
    for (size_t i = 0; i < n; ++i) {
      if (flags[i]) near.push_back(xyz[i]);
    }
    if (near.size() < 50) continue;
    std::vector<uint8_t> use(near.size(), 1);
    Vec3 plane = FitPlane(near, use);
    for (int round = 0; round < 2; ++round) {  // Trim outliers and refit.
      for (size_t k = 0; k < near.size(); ++k) {
        const double zp = plane.x() * near[k].x() + plane.y() * near[k].y() +
                          plane.z();
        use[k] = std::abs(zp - near[k].z()) < 0.05;
      }
      plane = FitPlane(near, use);
    }
    bool duplicate = false;  // The same ramp driven twice.
    for (const Ramp& r : ramps) {
      duplicate |= (r.plane - plane).cwiseAbs().maxCoeff() < 0.05 &&
                   r.level_from == from;
    }
    if (duplicate) continue;
    std::vector<Vec2> seg_xy;
    for (const Vec3& p : seg) seg_xy.push_back(p.head<2>());
    const KdTree2 seg_tree(seg_xy);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
      const size_t i = static_cast<size_t>(ii);
      const double zp =
          plane.x() * xyz[i].x() + plane.y() * xyz[i].y() + plane.z();
      flags[i] =
          nz[i] > 0.8 && std::abs(xyz[i].z() - zp) < 0.05 &&
          std::abs(xyz[i].z() - level_z[static_cast<size_t>(from)]) > 0.1 &&
          std::abs(xyz[i].z() - level_z[static_cast<size_t>(to)]) > 0.1 &&
          seg_tree.Nearest(xyz[i].head<2>(), 4.5).index >= 0;
    }
    std::vector<Vec2> on;
    for (size_t i = 0; i < n; ++i) {
      if (flags[i]) on.push_back(xyz[i].head<2>());
    }
    if (on.size() < 50) continue;
    Ramp ramp;
    ramp.level_from = from;
    ramp.level_to = to;
    ramp.plane = plane;
    ramp.polygon = ConvexHull2D(on);
    ramp.slope = std::hypot(plane.x(), plane.y());
    ramps.push_back(ramp);
  }
  return ramps;
}

}  // namespace

double FloorGrid::Sample(const Vec2& xy) const {
  GR_CHECK(width > 0 && height > 0);
  const int i = std::clamp(static_cast<int>((xy.x() - x0) / res), 0, width - 1);
  const int j =
      std::clamp(static_cast<int>((xy.y() - y0) / res), 0, height - 1);
  return z[static_cast<size_t>(j * width + i)];
}

bool Ramp::Contains(const Vec2& xy) const {
  return PointInPolygon(polygon, xy);
}

double LevelModel::FloorZ(const Vec2& xy, int level) const {
  GR_CHECK(level >= 0 && level < num_levels());
  const size_t l = static_cast<size_t>(level);
  if (l < grids.size() && grids[l].width > 0) return grids[l].Sample(xy);
  return z[l];
}

LevelAssignment AssignLevels(const LevelModel& model, const Points& xyz) {
  const size_t n = xyz.size();
  LevelAssignment out;
  out.level.assign(n, -1);
  out.height.assign(n, std::numeric_limits<float>::quiet_NaN());
  out.on_ramp.assign(n, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    const Vec2 xy = xyz[i].head<2>();
    double best = kInf;
    for (int l = 0; l < model.num_levels(); ++l) {
      const double h = xyz[i].z() - model.FloorZ(xy, l);
      if (h >= -0.3 && h < best) {
        best = h;
        out.level[i] = static_cast<int16_t>(l);
      }
    }
    for (const Ramp& ramp : model.ramps) {
      if (!ramp.Contains(xy)) continue;
      const double h = xyz[i].z() - ramp.ZAt(xy);
      if (h >= -0.3 && h < best) {
        best = h;
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
                        const LevelConfig& config, const Logger& log) {
  GR_CHECK(normals.size() == xyz.size());
  GR_CHECK(config.grid_res > 0.0);
  // Slope along every path over +-1 m of arc length.
  std::vector<double> flat_z;
  Points flat_path;
  std::vector<Points> ramp_segments;
  for (const Points& p : vehicle_paths) {
    const std::vector<double> s = ArcLength(p);
    std::vector<double> slope(p.size(), 0.0);
    for (size_t k = 0; k < p.size(); ++k) {
      const size_t a = SearchSorted(s, s[k] - 1.0);
      const size_t b = std::min(p.size() - 1, SearchSorted(s, s[k] + 1.0));
      const double ds = s[b] - s[a];
      slope[k] = ds > 0.2 ? (p[b].z() - p[a].z()) / ds : 0.0;
    }
    std::vector<uint8_t> steep(p.size());
    for (size_t k = 0; k < p.size(); ++k) {
      if (std::abs(slope[k]) < config.flat_slope) {
        flat_z.push_back(p[k].z());
        flat_path.push_back(p[k]);
      }
      steep[k] = std::abs(slope[k]) >= 2.0 * config.flat_slope;
    }
    for (const auto& [a, b] : Runs(steep)) {
      if (s[b - 1] - s[a] > 3.0) {
        ramp_segments.emplace_back(p.begin() + static_cast<long>(a),
                                   p.begin() + static_cast<long>(b));
      }
    }
  }
  // Levels: clusters of the flat heights.
  std::vector<double> zs = flat_z;
  std::sort(zs.begin(), zs.end());
  std::vector<std::vector<double>> clusters;
  for (double z : zs) {
    if (clusters.empty() ||
        z - clusters.back().back() > 0.5 * config.min_separation) {
      clusters.emplace_back();
    }
    clusters.back().push_back(z);
  }
  std::vector<double> level_z;
  for (const std::vector<double>& c : clusters) {
    if (static_cast<int>(c.size()) >= config.min_samples) {
      level_z.push_back(Median(c));
    }
  }
  std::sort(level_z.rbegin(), level_z.rend());
  if (level_z.empty()) level_z.push_back(zs.empty() ? 0.0 : Median(zs));

  const size_t n = xyz.size();
  std::vector<double> nz(n);
  for (size_t i = 0; i < n; ++i) nz[i] = normals[i].z();
  LevelModel model;
  model.z = level_z;
  model.ramps = FitRamps(ramp_segments, level_z, xyz, nz);

  // Floor grid per level: per-cell median of upward-facing points near the
  // level's own path height, ramp surfaces excluded.
  std::vector<uint8_t> on_ramp(n, 0);
  for (const Ramp& r : model.ramps) {
    for (size_t i = 0; i < n; ++i) {
      if (std::abs(xyz[i].z() - r.ZAt(xyz[i].head<2>())) < 0.1 &&
          r.Contains(xyz[i].head<2>())) {
        on_ramp[i] = 1;
      }
    }
  }
  Vec2 lo = Vec2::Constant(kInf);
  Vec2 hi = Vec2::Constant(-kInf);
  for (const Vec3& p : xyz) {
    lo = lo.cwiseMin(p.head<2>());
    hi = hi.cwiseMax(p.head<2>());
  }
  lo -= Vec2::Ones();
  hi += Vec2::Ones();
  const double res = config.grid_res;
  const int width = static_cast<int>(std::ceil((hi.x() - lo.x()) / res));
  const int height = static_cast<int>(std::ceil((hi.y() - lo.y()) / res));
  std::vector<int> owner(flat_path.size());
  for (size_t k = 0; k < flat_path.size(); ++k) {
    owner[k] = NearestLevel(level_z, flat_path[k].z());
  }
  for (size_t l = 0; l < level_z.size(); ++l) {
    Points mine;
    for (size_t k = 0; k < flat_path.size(); ++k) {
      if (owner[k] == static_cast<int>(l)) mine.push_back(flat_path[k]);
    }
    std::vector<double> reference(n, level_z[l]);
    if (static_cast<int>(mine.size()) >= config.min_samples) {
      std::vector<Vec2> mine_xy;
      for (const Vec3& p : mine) mine_xy.push_back(p.head<2>());
      const KdTree2 tree(mine_xy);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        reference[i] =
            mine[static_cast<size_t>(tree.Nearest(xyz[i].head<2>()).index)]
                .z();
      }
    }
    std::vector<Vec2> cell_xy;
    std::vector<double> cell_z;
    for (size_t i = 0; i < n; ++i) {
      if (nz[i] > 0.85 && !on_ramp[i] &&
          std::abs(xyz[i].z() - reference[i]) < config.floor_band) {
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
                    [](float v) { return std::isfinite(v); })) {
      const std::vector<int> nearest = NearestFinite(grid.z, width, height);
      std::vector<float> filled(grid.z.size());
      for (size_t c = 0; c < filled.size(); ++c) {
        filled[c] = grid.z[static_cast<size_t>(nearest[c])];
      }
      grid.z = Median3x3(filled, width, height);
    } else {
      grid.z.assign(grid.z.size(), static_cast<float>(level_z[l]));
    }
    model.grids.push_back(std::move(grid));
  }

  // Ceiling per level: downward-facing surfaces 2 m or more above its floor
  // (below the next level up), the 90th percentile of their clearance.
  for (size_t l = 0; l < level_z.size(); ++l) {
    const double top = l > 0 ? level_z[l - 1] - level_z[l] : 6.0;
    std::vector<double> clearance;
    for (size_t i = 0; i < n; ++i) {
      if (nz[i] >= -0.85) continue;
      const double h = xyz[i].z() - model.grids[l].Sample(xyz[i].head<2>());
      if (h > 2.0 && h < top) clearance.push_back(h);
    }
    model.ceiling.push_back(clearance.size() > 50
                                ? level_z[l] + Percentile(clearance, 90.0)
                                : level_z[l] + 3.0);
  }
  if (log) {
    std::string zs_text;
    std::string ceilings;
    std::string slopes;
    for (size_t l = 0; l < level_z.size(); ++l) {
      zs_text += StringPrintf("%s%.2f", l ? ", " : "", level_z[l]);
      ceilings += StringPrintf("%s%.2f", l ? ", " : "", model.ceiling[l]);
    }
    for (size_t r = 0; r < model.ramps.size(); ++r) {
      slopes += StringPrintf("%s%.3f", r ? ", " : "", model.ramps[r].slope);
    }
    LogLine(log, StringPrintf("[levels] %zu levels at z=[%s], ceilings [%s], "
                              "%zu ramps [%s]",
                              level_z.size(), zs_text.c_str(),
                              ceilings.c_str(), model.ramps.size(),
                              slopes.c_str()));
  }
  return model;
}

}  // namespace gr
