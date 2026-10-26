#include "gr/splat/gaussians.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/voxel.h"

namespace gr {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// Rotation whose third column is the unit vector `z`.
Mat3 FrameFromNormal(const Vec3& z) {
  const Vec3 helper = std::abs(z.z()) < 0.9 ? Vec3::UnitZ() : Vec3::UnitX();
  const Vec3 x = helper.cross(z).normalized();
  Mat3 frame;
  frame.col(0) = x;
  frame.col(1) = z.cross(x);
  frame.col(2) = z;
  return frame;
}

// Mean distance to the `k` nearest neighbours within the grid radius, or NaN
// with fewer than three neighbours.
double LocalSpacing(
    const Points& points, size_t i, int k, double inv_cell, double radius,
    const std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>& grid) {
  const VoxelKey center = VoxelOf(points[i], inv_cell);
  const double r2 = radius * radius;
  std::vector<double> d2;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        const auto it =
            grid.find({center.x + dx, center.y + dy, center.z + dz});
        if (it == grid.end()) continue;
        for (int j : it->second) {
          if (static_cast<size_t>(j) == i) continue;
          const double d = (points[static_cast<size_t>(j)] - points[i])
                               .squaredNorm();
          if (d < r2 && d > 0.0) d2.push_back(d);
        }
      }
    }
  }
  if (d2.size() < 3) return kNaN;
  const size_t m = std::min(d2.size(), static_cast<size_t>(k));
  std::partial_sort(d2.begin(), d2.begin() + static_cast<long>(m), d2.end());
  double sum = 0.0;
  for (size_t a = 0; a < m; ++a) sum += std::sqrt(d2[a]);
  return sum / static_cast<double>(m);
}

}  // namespace

void GaussianSet::FillDefaults() {
  const size_t n = means.size();
  rotations.resize(n, Eigen::Quaterniond::Identity());
  scales.resize(n, Vec3::Constant(0.01));
  opacity.resize(n, 1.0f);
  intensity.resize(n, 0.0f);
  trip.resize(n, -1);
}

GaussianSet InitGaussians(const Points& points, const Points& normals,
                          const std::vector<double>& sigma,
                          const GaussianInitConfig& config) {
  GR_CHECK(normals.empty() || normals.size() == points.size());
  GR_CHECK(sigma.empty() || sigma.size() == points.size());
  GR_CHECK(config.search_radius > 0.0 && config.neighbors > 0);
  GaussianSet set;
  set.means = points;
  set.FillDefaults();
  const size_t n = points.size();
  const double inv_cell = 1.0 / config.search_radius;
  std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> grid;
  grid.reserve(n / 8 + 1);
  for (size_t i = 0; i < n; ++i) {
    grid[VoxelOf(points[i], inv_cell)].push_back(static_cast<int>(i));
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    double spacing = LocalSpacing(points, i, config.neighbors, inv_cell,
                                  config.search_radius, grid);
    if (!std::isfinite(spacing)) spacing = config.default_spacing;
    const double tangent = std::clamp(config.tangent_factor * spacing,
                                      config.min_tangent, config.max_tangent);
    const Vec3 normal = normals.empty() ? Vec3::Zero() : normals[i];
    if (normal.norm() > 0.5) {
      const double along = sigma.empty() || !std::isfinite(sigma[i])
                               ? config.normal_floor
                               : std::max(sigma[i], config.normal_floor);
      set.rotations[i] =
          Eigen::Quaterniond(FrameFromNormal(normal.normalized()));
      set.scales[i] = Vec3(tangent, tangent, std::min(along, tangent));
    } else {
      set.scales[i] = Vec3::Constant(tangent);
    }
    set.opacity[i] = static_cast<float>(config.opacity);
  }
  return set;
}

GaussianRayRenderer::GaussianRayRenderer(GaussianSet gaussians,
                                         const RayRenderConfig& config)
    : set_(std::move(gaussians)), config_(config) {
  set_.FillDefaults();
  const size_t n = set_.size();
  axes_.resize(n);
  std::vector<Vec3> extents(n);
  Vec3 lo = Vec3::Constant(kInf);
  Vec3 hi = Vec3::Constant(-kInf);
  for (size_t i = 0; i < n; ++i) {
    const Mat3 r = set_.rotations[i].normalized().toRotationMatrix();
    const Vec3 s = set_.scales[i].cwiseMax(1e-6);
    Eigen::Matrix3f axes;
    for (int a = 0; a < 3; ++a) {
      axes.row(a) = (r.col(a) / s[a]).transpose().cast<float>();
    }
    axes_[i] = axes;
    // Half-extent of the cutoff ellipsoid along each world axis.
    for (int w = 0; w < 3; ++w) {
      double variance = 0.0;
      for (int a = 0; a < 3; ++a) variance += std::pow(r(w, a) * s[a], 2);
      extents[i][w] = config_.cutoff * std::sqrt(variance);
    }
    lo = lo.cwiseMin(set_.means[i] - extents[i]);
    hi = hi.cwiseMax(set_.means[i] + extents[i]);
    max_extent_ = std::max(max_extent_, extents[i].norm());
  }
  // Attributes the renderer does not need any more.
  set_.rotations = {};
  set_.scales = {};
  if (n == 0) return;
  min_bound_ = lo;
  cell_ = config_.cell;
  const Vec3 span = hi - lo;
  auto cells_along = [&](double extent) {
    return std::max(1, static_cast<int>(std::ceil(extent / cell_)));
  };
  while (true) {
    nx_ = cells_along(span.x());
    ny_ = cells_along(span.y());
    nz_ = cells_along(span.z());
    if (static_cast<int64_t>(nx_) * ny_ * nz_ <= config_.max_cells) break;
    cell_ *= 1.25;
  }
  const int64_t num_cells = static_cast<int64_t>(nx_) * ny_ * nz_;
  auto cell_range = [&](size_t i, Eigen::Vector3i& a, Eigen::Vector3i& b) {
    const Vec3 g0 = (set_.means[i] - extents[i] - min_bound_) / cell_;
    const Vec3 g1 = (set_.means[i] + extents[i] - min_bound_) / cell_;
    const int dims[3] = {nx_, ny_, nz_};
    for (int w = 0; w < 3; ++w) {
      a[w] = std::clamp(static_cast<int>(std::floor(g0[w])), 0, dims[w] - 1);
      b[w] = std::clamp(static_cast<int>(std::floor(g1[w])), 0, dims[w] - 1);
    }
  };
  auto linear = [&](int i, int j, int k) {
    return (static_cast<int64_t>(i) * ny_ + j) * nz_ + k;
  };
  offsets_.assign(static_cast<size_t>(num_cells + 1), 0);
  Eigen::Vector3i a, b;
  for (size_t i = 0; i < n; ++i) {
    cell_range(i, a, b);
    for (int x = a.x(); x <= b.x(); ++x) {
      for (int y = a.y(); y <= b.y(); ++y) {
        for (int z = a.z(); z <= b.z(); ++z) ++offsets_[linear(x, y, z) + 1];
      }
    }
  }
  for (int64_t c = 0; c < num_cells; ++c) offsets_[c + 1] += offsets_[c];
  entries_.resize(static_cast<size_t>(offsets_.back()));
  std::vector<int64_t> cursor(offsets_.begin(), offsets_.end() - 1);
  for (size_t i = 0; i < n; ++i) {
    cell_range(i, a, b);
    for (int x = a.x(); x <= b.x(); ++x) {
      for (int y = a.y(); y <= b.y(); ++y) {
        for (int z = a.z(); z <= b.z(); ++z) {
          entries_[static_cast<size_t>(cursor[linear(x, y, z)]++)] =
              static_cast<int>(i);
        }
      }
    }
  }
}

bool GaussianRayRenderer::Evaluate(int index, const Vec3& origin,
                                   const Vec3& direction, Hit& hit) const {
  const size_t i = static_cast<size_t>(index);
  const Eigen::Matrix3f& axes = axes_[i];
  const Eigen::Vector3f o = axes * (origin - set_.means[i]).cast<float>();
  const Eigen::Vector3f d = axes * direction.cast<float>();
  const float dd = d.squaredNorm();
  if (dd < 1e-12f) return false;
  const float t = -o.dot(d) / dd;
  if (t <= 0.0f) return false;
  const float m2 = (o + t * d).squaredNorm();
  if (m2 > config_.cutoff * config_.cutoff) return false;
  const double alpha = std::min(
      static_cast<double>(set_.opacity[i]) * std::exp(-0.5 * m2),
      config_.max_alpha);
  if (alpha < config_.min_alpha) return false;
  hit = {static_cast<double>(t), alpha, index};
  return true;
}

void GaussianRayRenderer::CollectHits(const Vec3& origin,
                                      const Vec3& direction, int trip,
                                      std::vector<Hit>& hits) const {
  hits.clear();
  if (entries_.empty()) return;
  // Clip the ray to the grid box (slab test, metres).
  const Vec3 hi = min_bound_ + cell_ * Vec3(nx_, ny_, nz_);
  double t0 = 0.0;
  double t1 = config_.max_range;
  for (int a = 0; a < 3; ++a) {
    if (std::abs(direction[a]) < 1e-12) {
      if (origin[a] < min_bound_[a] || origin[a] > hi[a]) return;
      continue;
    }
    double near = (min_bound_[a] - origin[a]) / direction[a];
    double far = (hi[a] - origin[a]) / direction[a];
    if (near > far) std::swap(near, far);
    t0 = std::max(t0, near);
    t1 = std::min(t1, far);
  }
  if (t0 > t1) return;
  const Vec3 start = (origin + (t0 + 1e-9) * direction - min_bound_) / cell_;
  const double t_end = (t1 - t0) / cell_;
  const double min_t = config_.min_transmittance;
  double saturated = kInf;  // Range where the ray became opaque.
  size_t sorted = 0;
  int ijk[3] = {static_cast<int>(std::floor(start.x())),
                static_cast<int>(std::floor(start.y())),
                static_cast<int>(std::floor(start.z()))};
  const int dims[3] = {nx_, ny_, nz_};
  int step[3];
  double t_max[3];
  double t_delta[3];
  for (int a = 0; a < 3; ++a) {
    ijk[a] = std::clamp(ijk[a], 0, dims[a] - 1);
    if (direction[a] > 0.0) {
      step[a] = 1;
      t_max[a] = (ijk[a] + 1 - start[a]) / direction[a];
      t_delta[a] = 1.0 / direction[a];
    } else if (direction[a] < 0.0) {
      step[a] = -1;
      t_max[a] = (ijk[a] - start[a]) / direction[a];
      t_delta[a] = -1.0 / direction[a];
    } else {
      step[a] = 0;
      t_max[a] = kInf;
      t_delta[a] = kInf;
    }
  }
  double t_cell = 0.0;  // Entry of the current cell (grid units from start).
  while (true) {
    if (t0 + t_cell * cell_ > saturated + max_extent_) break;
    const int64_t c = (static_cast<int64_t>(ijk[0]) * ny_ + ijk[1]) * nz_ +
                      ijk[2];
    for (int64_t e = offsets_[static_cast<size_t>(c)];
         e < offsets_[static_cast<size_t>(c) + 1]; ++e) {
      const int g = entries_[static_cast<size_t>(e)];
      const int g_trip = set_.trip[static_cast<size_t>(g)];
      if (g_trip >= 0 && g_trip != trip) continue;
      Hit hit;
      if (Evaluate(g, origin, direction, hit)) hits.push_back(hit);
    }
    if (hits.size() > sorted) {
      std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
        return a.t < b.t || (a.t == b.t && a.index < b.index);
      });
      hits.erase(std::unique(hits.begin(), hits.end(),
                             [](const Hit& a, const Hit& b) {
                               return a.index == b.index;
                             }),
                 hits.end());
      sorted = hits.size();
      double transmittance = 1.0;
      saturated = kInf;
      for (const Hit& h : hits) {
        transmittance *= 1.0 - h.alpha;
        if (transmittance < min_t) {
          saturated = h.t;
          break;
        }
      }
    }
    int axis = 0;
    if (t_max[1] < t_max[axis]) axis = 1;
    if (t_max[2] < t_max[axis]) axis = 2;
    if (t_max[axis] > t_end) break;
    t_cell = t_max[axis];
    ijk[axis] += step[axis];
    if (ijk[axis] < 0 || ijk[axis] >= dims[axis]) break;
    t_max[axis] += t_delta[axis];
  }
}

RayRender GaussianRayRenderer::Render(const Points& origins,
                                      const Points& directions,
                                      int trip) const {
  GR_CHECK(origins.size() == directions.size());
  const size_t n = origins.size();
  RayRender out;
  out.depth.assign(n, kNaN);
  out.mean_depth.assign(n, kNaN);
  out.intensity.assign(n, std::numeric_limits<float>::quiet_NaN());
  out.opacity.assign(n, 0.0f);
  out.dominant.assign(n, -1);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<Hit> hits;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (int64_t rr = 0; rr < static_cast<int64_t>(n); ++rr) {
      const size_t r = static_cast<size_t>(rr);
      const Vec3 direction = directions[r].normalized();
      CollectHits(origins[r], direction, trip, hits);
      double transmittance = 1.0;
      double sum_w = 0.0;
      double sum_wt = 0.0;
      double sum_wi = 0.0;
      double best = 0.0;
      for (const Hit& h : hits) {
        const double w = transmittance * h.alpha;
        sum_w += w;
        sum_wt += w * h.t;
        sum_wi += w * set_.intensity[static_cast<size_t>(h.index)];
        if (w > best) {
          best = w;
          out.dominant[r] = h.index;
        }
        transmittance *= 1.0 - h.alpha;
        if (std::isnan(out.depth[r]) && transmittance < 0.5) {
          out.depth[r] = h.t;
        }
        if (transmittance < config_.min_transmittance) break;
      }
      out.opacity[r] = static_cast<float>(1.0 - transmittance);
      if (sum_w > 0.0) {
        out.mean_depth[r] = sum_wt / sum_w;
        out.intensity[r] = static_cast<float>(sum_wi / sum_w);
      }
    }
  }
  return out;
}

}  // namespace gr
