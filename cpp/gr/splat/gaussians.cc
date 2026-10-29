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
// Normals with a z component up to this take world z as the helper axis of
// their frame, others world x.
constexpr double kMaxHelperCos = 0.9;
// The spacing comes from the voxels this many steps around a point and needs
// at least kMinSpacingNeighbours neighbours.
constexpr int kNeighbourhoodRadius = 1;
constexpr size_t kMinSpacingNeighbours = 3;
// The neighbour grid is reserved for one cell per kPointsPerCellEstimate
// points.
constexpr size_t kPointsPerCellEstimate = 8;
// Points per OpenMP chunk of the initialisation, rays per chunk of the
// renderer.
constexpr int kPointsPerChunk = 1024;
constexpr int kRaysPerChunk = 64;
// Normals shorter than this are missing.
constexpr double kMinNormalNorm = 0.5;
// Defaults of FillDefaults: sigma (m) and opacity.
constexpr double kDefaultScale = 0.01;
constexpr float kDefaultOpacity = 1.0f;
// Sigmas are at least this (m) when inverted.
constexpr double kMinScale = 1e-6;
// The grid cell grows by this factor until the grid fits `max_cells`.
constexpr double kCellGrowth = 1.25;
// Rays with a smaller direction component are parallel to that axis; local
// directions with a smaller squared norm are degenerate.
constexpr double kParallelDirection = 1e-12;
constexpr float kMinLocalDirection = 1e-12f;
// The grid walk starts this far (m) inside the clipped ray.
constexpr double kEntryNudge = 1e-9;

// Rotation whose third column is the unit vector `z`.
Mat3 FrameFromNormal(const Vec3& z) {
  const Vec3 helper =
      std::abs(z.z()) < kMaxHelperCos ? Vec3::UnitZ() : Vec3::UnitX();
  const Vec3 x = helper.cross(z).normalized();
  Mat3 frame;
  frame.col(0) = x;
  frame.col(1) = z.cross(x);
  frame.col(2) = z;
  return frame;
}

// Mean distance to the `neighbors` nearest neighbours within the grid
// radius, or NaN with fewer than kMinSpacingNeighbours neighbours.
double LocalSpacing(
    const Points& points, size_t point, int neighbors, double inverse_cell,
    double radius,
    const std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>& grid) {
  const VoxelKey center = VoxelOf(points[point], inverse_cell);
  const double squared_radius = radius * radius;
  std::vector<double> squared_distances;
  for (int dx = -kNeighbourhoodRadius; dx <= kNeighbourhoodRadius; ++dx) {
    for (int dy = -kNeighbourhoodRadius; dy <= kNeighbourhoodRadius; ++dy) {
      for (int dz = -kNeighbourhoodRadius; dz <= kNeighbourhoodRadius; ++dz) {
        const auto found =
            grid.find({center.x + dx, center.y + dy, center.z + dz});
        if (found == grid.end()) continue;
        for (int other : found->second) {
          if (static_cast<size_t>(other) == point) continue;
          const double squared_distance =
              (points[static_cast<size_t>(other)] - points[point])
                  .squaredNorm();
          if (squared_distance < squared_radius && squared_distance > 0.0) {
            squared_distances.push_back(squared_distance);
          }
        }
      }
    }
  }
  if (squared_distances.size() < kMinSpacingNeighbours) return kNaN;
  const size_t nearest =
      std::min(squared_distances.size(), static_cast<size_t>(neighbors));
  std::partial_sort(squared_distances.begin(),
                    squared_distances.begin() + static_cast<long>(nearest),
                    squared_distances.end());
  double sum = 0.0;
  for (size_t k = 0; k < nearest; ++k) sum += std::sqrt(squared_distances[k]);
  return sum / static_cast<double>(nearest);
}

}  // namespace

void GaussianSet::FillDefaults() {
  const size_t count = means.size();
  rotations.resize(count, Eigen::Quaterniond::Identity());
  scales.resize(count, Vec3::Constant(kDefaultScale));
  opacity.resize(count, kDefaultOpacity);
  intensity.resize(count, 0.0f);
  trip.resize(count, kSharedTrip);
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
  const size_t count = points.size();
  const double inverse_cell = 1.0 / config.search_radius;
  std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> grid;
  grid.reserve(count / kPointsPerCellEstimate + 1);
  for (size_t point = 0; point < count; ++point) {
    grid[VoxelOf(points[point], inverse_cell)].push_back(
        static_cast<int>(point));
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, kPointsPerChunk)
#endif
  for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
    const size_t point = static_cast<size_t>(index);
    double spacing = LocalSpacing(points, point, config.neighbors,
                                  inverse_cell, config.search_radius, grid);
    if (!std::isfinite(spacing)) spacing = config.default_spacing;
    const double tangent = std::clamp(config.tangent_factor * spacing,
                                      config.min_tangent, config.max_tangent);
    const Vec3 normal = normals.empty() ? Vec3::Zero() : normals[point];
    if (normal.norm() > kMinNormalNorm) {
      const double along = sigma.empty() || !std::isfinite(sigma[point])
                               ? config.normal_floor
                               : std::max(sigma[point], config.normal_floor);
      set.rotations[point] =
          Eigen::Quaterniond(FrameFromNormal(normal.normalized()));
      set.scales[point] = Vec3(tangent, tangent, std::min(along, tangent));
    } else {
      set.scales[point] = Vec3::Constant(tangent);
    }
    set.opacity[point] = static_cast<float>(config.opacity);
  }
  return set;
}

GaussianRayRenderer::GaussianRayRenderer(GaussianSet gaussians,
                                         const RayRenderConfig& config)
    : set_(std::move(gaussians)), config_(config) {
  set_.FillDefaults();
  const size_t count = set_.size();
  axes_.resize(count);
  std::vector<Vec3> extents(count);
  Vec3 lower = Vec3::Constant(kInf);
  Vec3 upper = Vec3::Constant(-kInf);
  for (size_t gaussian = 0; gaussian < count; ++gaussian) {
    const Mat3 rotation =
        set_.rotations[gaussian].normalized().toRotationMatrix();
    const Vec3 sigma = set_.scales[gaussian].cwiseMax(kMinScale);
    Eigen::Matrix3f axes;
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      axes.row(axis) =
          (rotation.col(axis) / sigma[axis]).transpose().cast<float>();
    }
    axes_[gaussian] = axes;
    // Half-extent of the cutoff ellipsoid along each world axis.
    for (int world_axis = 0; world_axis < kSpaceDim; ++world_axis) {
      double variance = 0.0;
      for (int axis = 0; axis < kSpaceDim; ++axis) {
        variance += std::pow(rotation(world_axis, axis) * sigma[axis], 2);
      }
      extents[gaussian][world_axis] = config_.cutoff * std::sqrt(variance);
    }
    lower = lower.cwiseMin(set_.means[gaussian] - extents[gaussian]);
    upper = upper.cwiseMax(set_.means[gaussian] + extents[gaussian]);
    max_extent_ = std::max(max_extent_, extents[gaussian].norm());
  }
  // Attributes the renderer does not need any more.
  set_.rotations = {};
  set_.scales = {};
  if (count == 0) return;
  min_bound_ = lower;
  cell_ = config_.cell;
  const Vec3 span = upper - lower;
  const auto cells_along = [&](double extent) {
    return std::max(1, static_cast<int>(std::ceil(extent / cell_)));
  };
  while (true) {
    nx_ = cells_along(span.x());
    ny_ = cells_along(span.y());
    nz_ = cells_along(span.z());
    if (static_cast<int64_t>(nx_) * ny_ * nz_ <= config_.max_cells) break;
    cell_ *= kCellGrowth;
  }
  const int64_t num_cells = static_cast<int64_t>(nx_) * ny_ * nz_;
  // The first and last cell of a Gaussian's bounding box along each axis.
  const auto cell_range = [&](size_t gaussian, Eigen::Vector3i& first,
                              Eigen::Vector3i& last) {
    const Vec3 grid_first =
        (set_.means[gaussian] - extents[gaussian] - min_bound_) / cell_;
    const Vec3 grid_last =
        (set_.means[gaussian] + extents[gaussian] - min_bound_) / cell_;
    const int dims[kSpaceDim] = {nx_, ny_, nz_};
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      first[axis] = std::clamp(static_cast<int>(std::floor(grid_first[axis])),
                               0, dims[axis] - 1);
      last[axis] = std::clamp(static_cast<int>(std::floor(grid_last[axis])),
                              0, dims[axis] - 1);
    }
  };
  const auto linear = [&](int x, int y, int z) {
    return (static_cast<int64_t>(x) * ny_ + y) * nz_ + z;
  };
  offsets_.assign(static_cast<size_t>(num_cells + 1), 0);
  Eigen::Vector3i first;
  Eigen::Vector3i last;
  for (size_t gaussian = 0; gaussian < count; ++gaussian) {
    cell_range(gaussian, first, last);
    for (int x = first.x(); x <= last.x(); ++x) {
      for (int y = first.y(); y <= last.y(); ++y) {
        for (int z = first.z(); z <= last.z(); ++z) {
          ++offsets_[linear(x, y, z) + 1];
        }
      }
    }
  }
  for (int64_t cell = 0; cell < num_cells; ++cell) {
    offsets_[cell + 1] += offsets_[cell];
  }
  entries_.resize(static_cast<size_t>(offsets_.back()));
  std::vector<int64_t> cursor(offsets_.begin(), offsets_.end() - 1);
  for (size_t gaussian = 0; gaussian < count; ++gaussian) {
    cell_range(gaussian, first, last);
    for (int x = first.x(); x <= last.x(); ++x) {
      for (int y = first.y(); y <= last.y(); ++y) {
        for (int z = first.z(); z <= last.z(); ++z) {
          entries_[static_cast<size_t>(cursor[linear(x, y, z)]++)] =
              static_cast<int>(gaussian);
        }
      }
    }
  }
}

bool GaussianRayRenderer::Evaluate(int index, const Vec3& origin,
                                   const Vec3& direction, Hit& hit) const {
  const size_t gaussian = static_cast<size_t>(index);
  const Eigen::Matrix3f& axes = axes_[gaussian];
  // The ray in the Gaussian's whitened frame.
  const Eigen::Vector3f local_origin =
      axes * (origin - set_.means[gaussian]).cast<float>();
  const Eigen::Vector3f local_direction = axes * direction.cast<float>();
  const float squared_length = local_direction.squaredNorm();
  if (squared_length < kMinLocalDirection) return false;
  const float peak = -local_origin.dot(local_direction) / squared_length;
  if (peak <= 0.0f) return false;
  const float squared_distance =
      (local_origin + peak * local_direction).squaredNorm();
  if (squared_distance > config_.cutoff * config_.cutoff) return false;
  const double alpha = std::min(
      static_cast<double>(set_.opacity[gaussian]) *
          std::exp(-0.5 * squared_distance),
      config_.max_alpha);
  if (alpha < config_.min_alpha) return false;
  hit = {static_cast<double>(peak), alpha, index};
  return true;
}

void GaussianRayRenderer::CollectHits(const Vec3& origin,
                                      const Vec3& direction, int trip,
                                      std::vector<Hit>& hits) const {
  hits.clear();
  if (entries_.empty()) return;
  // Clip the ray to the grid box (slab test, metres).
  const Vec3 max_bound = min_bound_ + cell_ * Vec3(nx_, ny_, nz_);
  double t_enter = 0.0;
  double t_exit = config_.max_range;
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    if (std::abs(direction[axis]) < kParallelDirection) {
      if (origin[axis] < min_bound_[axis] || origin[axis] > max_bound[axis]) {
        return;
      }
      continue;
    }
    double near = (min_bound_[axis] - origin[axis]) / direction[axis];
    double far = (max_bound[axis] - origin[axis]) / direction[axis];
    if (near > far) std::swap(near, far);
    t_enter = std::max(t_enter, near);
    t_exit = std::min(t_exit, far);
  }
  if (t_enter > t_exit) return;
  const Vec3 start =
      (origin + (t_enter + kEntryNudge) * direction - min_bound_) / cell_;
  const double t_end = (t_exit - t_enter) / cell_;
  const double min_transmittance = config_.min_transmittance;
  double saturated = kInf;  // Range where the ray became opaque.
  size_t sorted = 0;
  int cell_index[kSpaceDim] = {static_cast<int>(std::floor(start.x())),
                               static_cast<int>(std::floor(start.y())),
                               static_cast<int>(std::floor(start.z()))};
  const int dims[kSpaceDim] = {nx_, ny_, nz_};
  int step[kSpaceDim];
  double t_max[kSpaceDim];    // Ray parameter of the next cell boundary.
  double t_delta[kSpaceDim];  // Ray parameter between two boundaries.
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    cell_index[axis] = std::clamp(cell_index[axis], 0, dims[axis] - 1);
    if (direction[axis] > 0.0) {
      step[axis] = 1;
      t_max[axis] = (cell_index[axis] + 1 - start[axis]) / direction[axis];
      t_delta[axis] = 1.0 / direction[axis];
    } else if (direction[axis] < 0.0) {
      step[axis] = -1;
      t_max[axis] = (cell_index[axis] - start[axis]) / direction[axis];
      t_delta[axis] = -1.0 / direction[axis];
    } else {
      step[axis] = 0;
      t_max[axis] = kInf;
      t_delta[axis] = kInf;
    }
  }
  double t_cell = 0.0;  // Entry of the current cell (grid units from start).
  while (true) {
    if (t_enter + t_cell * cell_ > saturated + max_extent_) break;
    const int64_t cell =
        (static_cast<int64_t>(cell_index[0]) * ny_ + cell_index[1]) * nz_ +
        cell_index[2];
    for (int64_t entry = offsets_[static_cast<size_t>(cell)];
         entry < offsets_[static_cast<size_t>(cell) + 1]; ++entry) {
      const int gaussian = entries_[static_cast<size_t>(entry)];
      const int gaussian_trip = set_.trip[static_cast<size_t>(gaussian)];
      const bool trip_specific = gaussian_trip >= 0;
      if (trip_specific && gaussian_trip != trip) continue;
      Hit hit;
      if (Evaluate(gaussian, origin, direction, hit)) hits.push_back(hit);
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
      for (const Hit& hit : hits) {
        transmittance *= 1.0 - hit.alpha;
        if (transmittance < min_transmittance) {
          saturated = hit.t;
          break;
        }
      }
    }
    int axis = 0;  // The axis whose boundary comes first.
    for (int other = 1; other < kSpaceDim; ++other) {
      if (t_max[other] < t_max[axis]) axis = other;
    }
    if (t_max[axis] > t_end) break;
    t_cell = t_max[axis];
    cell_index[axis] += step[axis];
    if (cell_index[axis] < 0 || cell_index[axis] >= dims[axis]) break;
    t_max[axis] += t_delta[axis];
  }
}

RayRender GaussianRayRenderer::Render(const Points& origins,
                                      const Points& directions,
                                      int trip) const {
  GR_CHECK(origins.size() == directions.size());
  const size_t count = origins.size();
  RayRender out;
  out.depth.assign(count, kNaN);
  out.mean_depth.assign(count, kNaN);
  out.intensity.assign(count, std::numeric_limits<float>::quiet_NaN());
  out.opacity.assign(count, 0.0f);
  out.dominant.assign(count, RayRender::kNoGaussian);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<Hit> hits;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, kRaysPerChunk)
#endif
    for (int64_t index = 0; index < static_cast<int64_t>(count); ++index) {
      const size_t ray = static_cast<size_t>(index);
      const Vec3 direction = directions[ray].normalized();
      CollectHits(origins[ray], direction, trip, hits);
      double transmittance = 1.0;
      double weight_sum = 0.0;
      double weighted_range = 0.0;
      double weighted_intensity = 0.0;
      double best = 0.0;
      for (const Hit& hit : hits) {
        const double weight = transmittance * hit.alpha;
        weight_sum += weight;
        weighted_range += weight * hit.t;
        weighted_intensity +=
            weight * set_.intensity[static_cast<size_t>(hit.index)];
        if (weight > best) {
          best = weight;
          out.dominant[ray] = hit.index;
        }
        transmittance *= 1.0 - hit.alpha;
        if (std::isnan(out.depth[ray]) &&
            transmittance < RayRender::kDepthTransmittance) {
          out.depth[ray] = hit.t;
        }
        if (transmittance < config_.min_transmittance) break;
      }
      out.opacity[ray] = static_cast<float>(1.0 - transmittance);
      if (weight_sum > 0.0) {
        out.mean_depth[ray] = weighted_range / weight_sum;
        out.intensity[ray] =
            static_cast<float>(weighted_intensity / weight_sum);
      }
    }
  }
  return out;
}

}  // namespace gr
