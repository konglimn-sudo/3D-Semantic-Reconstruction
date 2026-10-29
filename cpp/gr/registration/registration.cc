#include "gr/registration/registration.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Eigen/Eigenvalues"

namespace gr {
namespace {

// Gauss-Newton normal equations accumulated over correspondences: J^T J,
// J^T r, the sum of squared residuals and the number of residuals.
struct NormalEquations {
  Mat6 jtj = Mat6::Zero();
  Vec6 jtr = Vec6::Zero();
  double sse = 0.0;
  int count = 0;

  void Add(const NormalEquations& other) {
    jtj += other.jtj;
    jtr += other.jtr;
    sse += other.sse;
    count += other.count;
  }
};

enum class ResidualType { kPointToPoint, kPointToPlane, kSkip };

struct CachedPlane {
  ResidualType type = ResidualType::kPointToPoint;
  Vec3 centroid = Vec3::Zero();
  Vec3 normal = Vec3::UnitZ();
};

using PlaneCache = std::unordered_map<VoxelKey, CachedPlane, VoxelKeyHash>;

// Radii up to this much beyond the largest one are still tried (m).
constexpr double kRadiusTolerance = 1e-9;
// Residual models are cached per cell of half the plane radius.
constexpr double kCacheCellsPerRadius = 2.0;
// Positions of SelfAdjointEigenSolver's eigenvalues (ascending).
constexpr int kSmallest = 0;
constexpr int kMiddle = 1;
constexpr int kLargest = 2;
// Neighbourhoods with a smaller largest eigenvalue (m^2) have no spread.
constexpr double kMinSpread = 1e-12;
// AlignMultiScale: the source is downsampled to half the gate, the robust
// kernel is a third of it, and the final statistics use a gate of two
// voxels when the schedule is empty. Its maps keep 30 points per voxel.
constexpr double kSourceVoxelDivisor = 2.0;
constexpr double kKernelDivisor = 3.0;
constexpr double kDefaultFinestGateVoxels = 2.0;
constexpr int kMultiScalePointsPerVoxel = 30;

// Residual model of the map around `point`: grows the neighbourhood until it
// is planar, because sparse sweeps sample flat surfaces as rings that are far
// apart and small radii see only lines.
CachedPlane ClassifyNeighbourhood(const VoxelHashMap& map, const Vec3& point,
                                  double plane_radius,
                                  const IcpOptions& options) {
  CachedPlane plane;
  const double max_radius = std::max(plane_radius, options.max_plane_radius);
  for (double radius = plane_radius; radius <= max_radius + kRadiusTolerance;
       radius *= 2.0) {
    const std::optional<VoxelHashMap::LocalPlane> local = map.FitLocalPlane(
        point, radius, options.min_plane_points, options.exclude_frame);
    if (!local) continue;
    const ResidualType fallback = local->linearity > options.max_linearity
                                      ? ResidualType::kSkip
                                      : ResidualType::kPointToPoint;
    if (local->thickness > options.max_plane_thickness) {
      plane.type = fallback;  // A larger radius will not get thinner.
      break;
    }
    if (local->planarity >= options.min_planarity) {
      plane.type = ResidualType::kPointToPlane;
      plane.centroid = local->centroid;
      plane.normal = local->normal;
      break;
    }
    plane.type = fallback;
  }
  return plane;
}

// Correspondences are summed in blocks of a fixed number of source points,
// added in block order, so that the normal equations do not depend on the
// number of threads.
constexpr int kBlockSize = 256;

// One linearisation pass. `robust` toggles the Geman-McClure weights; the
// unweighted pass gives the final statistics and Hessian. Residual models
// are cached per cell of half the plane radius: a cell as large as the
// radius shares one model between the faces of a pillar or the two sides of
// an edge, and coarse-to-fine alignment then stalls along a corridor. A
// cell not in `cache` yet is classified around the first nearest map point
// (in source order) inside it and kept for the later passes, so that the
// cache, too, is the same for any number of threads.
NormalEquations Linearize(const Points& source, const VoxelHashMap& map,
                          const Pose& pose, const IcpOptions& options,
                          bool robust, PlaneCache* cache) {
  const int point_count = static_cast<int>(source.size());
  const double kernel = options.kernel;
  const double plane_radius =
      options.plane_radius > 0.0 ? options.plane_radius : map.voxel_size();
  const double inverse_cache_cell = kCacheCellsPerRadius / plane_radius;
  [[maybe_unused]] const bool parallel = options.parallel;
  // Geman-McClure: k^2 / (k + r^2)^2.
  auto weight = [&](double squared_residual) {
    if (!robust) return 1.0;
    const double denominator = kernel + squared_residual;
    return (kernel * kernel) / (denominator * denominator);
  };
  std::vector<Vec3> world(static_cast<size_t>(point_count));
  std::vector<std::optional<VoxelHashMap::Neighbor>> nearest(
      static_cast<size_t>(point_count));
  std::vector<VoxelKey> keys(options.point_to_plane ? world.size() : 0);
  std::vector<uint8_t> cached(keys.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
  for (int signed_index = 0; signed_index < point_count; ++signed_index) {
    const size_t index = static_cast<size_t>(signed_index);
    world[index] = pose * source[index];
    nearest[index] =
        map.Nearest(world[index], options.max_corr_dist, options.exclude_frame,
                    options.max_search_voxels);
    if (!nearest[index] || keys.empty()) continue;
    keys[index] = VoxelOf(nearest[index]->point, inverse_cache_cell);
    cached[index] = cache->count(keys[index]) > 0;
  }
  if (!keys.empty()) {
    std::vector<std::pair<VoxelKey, Vec3>> fresh;
    std::unordered_set<VoxelKey, VoxelKeyHash> seen;
    for (size_t index = 0; index < keys.size(); ++index) {
      if (!nearest[index] || cached[index]) continue;
      if (seen.insert(keys[index]).second) {
        fresh.emplace_back(keys[index], nearest[index]->point);
      }
    }
    std::vector<CachedPlane> planes(fresh.size());
    const int num_fresh = static_cast<int>(fresh.size());
#ifdef _OPENMP
    constexpr int kClassifyChunk = 16;  // Neighbourhoods per work chunk.
#pragma omp parallel for schedule(dynamic, kClassifyChunk) if (parallel)
#endif
    for (int signed_cell = 0; signed_cell < num_fresh; ++signed_cell) {
      const size_t cell = static_cast<size_t>(signed_cell);
      planes[cell] =
          ClassifyNeighbourhood(map, fresh[cell].second, plane_radius, options);
    }
    for (size_t cell = 0; cell < fresh.size(); ++cell) {
      cache->emplace(fresh[cell].first, planes[cell]);
    }
  }
  auto accumulate = [&](size_t index, NormalEquations& sums) {
    if (!nearest[index]) return;
    const Vec3& world_point = world[index];
    // d(world point) / d(left perturbation [t, phi]).
    Eigen::Matrix<double, kSpaceDim, kPoseDim> jacobian;
    jacobian.block<kSpaceDim, kSpaceDim>(0, kTranslationOffset) =
        Mat3::Identity();
    jacobian.block<kSpaceDim, kSpaceDim>(0, kRotationOffset) =
        -Hat(world_point);
    if (options.point_to_plane) {
      const CachedPlane& plane = cache->at(keys[index]);
      if (plane.type == ResidualType::kSkip) return;
      if (plane.type == ResidualType::kPointToPlane) {
        const double residual = plane.normal.dot(world_point - plane.centroid);
        const Eigen::Matrix<double, 1, kPoseDim> plane_jacobian =
            plane.normal.transpose() * jacobian;
        const double residual_weight = weight(residual * residual);
        sums.jtj.noalias() +=
            residual_weight * plane_jacobian.transpose() * plane_jacobian;
        sums.jtr.noalias() +=
            residual_weight * plane_jacobian.transpose() * residual;
        sums.sse += residual * residual;
        sums.count += 1;
        return;
      }
    }
    const double point_scale =
        options.point_to_plane ? options.point_weight : 1.0;
    if (point_scale <= 0.0) return;
    const Vec3 residual = world_point - nearest[index]->point;
    const double residual_weight = point_scale * weight(residual.squaredNorm());
    sums.jtj.noalias() += residual_weight * jacobian.transpose() * jacobian;
    sums.jtr.noalias() += residual_weight * jacobian.transpose() * residual;
    sums.sse += residual.squaredNorm();
    sums.count += 1;
  };
  const int num_blocks = (point_count + kBlockSize - 1) / kBlockSize;
  std::vector<NormalEquations> blocks(static_cast<size_t>(num_blocks));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
  for (int block = 0; block < num_blocks; ++block) {
    const int end = std::min(point_count, (block + 1) * kBlockSize);
    for (int index = block * kBlockSize; index < end; ++index) {
      accumulate(static_cast<size_t>(index),
                 blocks[static_cast<size_t>(block)]);
    }
  }
  NormalEquations total;
  for (const NormalEquations& block : blocks) total.Add(block);
  return total;
}

}  // namespace

VoxelHashMap::VoxelHashMap(double voxel_size, double max_distance,
                           int max_points_per_voxel)
    : voxel_size_(voxel_size),
      inverse_voxel_size_(1.0 / voxel_size),
      max_distance_(max_distance),
      max_points_per_voxel_(max_points_per_voxel) {}

size_t VoxelHashMap::CountPoints() const {
  size_t count = 0;
  for (const auto& [key, block] : map_) count += block.points.size();
  return count;
}

void VoxelHashMap::AddPoints(const Points& points, int frame_id) {
  for (const Vec3& point : points) {
    Block& block = map_[VoxelOf(point, inverse_voxel_size_)];
    if (static_cast<int>(block.points.size()) < max_points_per_voxel_) {
      block.points.push_back(point);
      block.frames.push_back(frame_id);
    }
  }
}

void VoxelHashMap::RemoveFar(const Vec3& origin) {
  const double squared_max_distance = max_distance_ * max_distance_;
  for (auto voxel = map_.begin(); voxel != map_.end();) {
    if (voxel->second.points.empty() ||
        (voxel->second.points.front() - origin).squaredNorm() >
            squared_max_distance) {
      voxel = map_.erase(voxel);
    } else {
      ++voxel;
    }
  }
}

Points VoxelHashMap::PointCloud() const {
  Points out;
  out.reserve(CountPoints());
  for (const auto& [key, block] : map_) {
    out.insert(out.end(), block.points.begin(), block.points.end());
  }
  return out;
}

std::optional<VoxelHashMap::Neighbor> VoxelHashMap::Nearest(
    const Vec3& query, double max_distance, int exclude_frame,
    int max_search_voxels) const {
  const VoxelKey center = VoxelOf(query, inverse_voxel_size_);
  const int reach = std::clamp(
      static_cast<int>(std::ceil(max_distance * inverse_voxel_size_)), 1,
      std::max(1, max_search_voxels));
  double best = max_distance * max_distance;
  std::optional<Neighbor> nearest;
  for (int dx = -reach; dx <= reach; ++dx) {
    for (int dy = -reach; dy <= reach; ++dy) {
      for (int dz = -reach; dz <= reach; ++dz) {
        const auto voxel =
            map_.find({center.x + dx, center.y + dy, center.z + dz});
        if (voxel == map_.end()) continue;
        const Block& block = voxel->second;
        for (size_t k = 0; k < block.points.size(); ++k) {
          if (exclude_frame >= 0 && block.frames[k] == exclude_frame) continue;
          const double squared_distance =
              (block.points[k] - query).squaredNorm();
          if (squared_distance < best) {
            best = squared_distance;
            nearest = Neighbor{block.points[k], squared_distance};
          }
        }
      }
    }
  }
  return nearest;
}

std::optional<VoxelHashMap::LocalPlane> VoxelHashMap::FitLocalPlane(
    const Vec3& query, double radius, int min_points, int exclude_frame) const {
  const VoxelKey center = VoxelOf(query, inverse_voxel_size_);
  const int reach =
      std::max(1, static_cast<int>(std::ceil(radius * inverse_voxel_size_)));
  const double squared_radius = radius * radius;
  Vec3 sum = Vec3::Zero();
  Mat3 outer = Mat3::Zero();
  int count = 0;
  for (int dx = -reach; dx <= reach; ++dx) {
    for (int dy = -reach; dy <= reach; ++dy) {
      for (int dz = -reach; dz <= reach; ++dz) {
        const auto voxel =
            map_.find({center.x + dx, center.y + dy, center.z + dz});
        if (voxel == map_.end()) continue;
        const Block& block = voxel->second;
        for (size_t k = 0; k < block.points.size(); ++k) {
          if (exclude_frame >= 0 && block.frames[k] == exclude_frame) continue;
          const Vec3& point = block.points[k];
          if ((point - query).squaredNorm() > squared_radius) continue;
          sum += point;
          outer.noalias() += point * point.transpose();
          ++count;
        }
      }
    }
  }
  if (count < min_points) return std::nullopt;
  LocalPlane plane;
  plane.centroid = sum / count;
  const Mat3 covariance =
      outer / count - plane.centroid * plane.centroid.transpose();
  Eigen::SelfAdjointEigenSolver<Mat3> solver(covariance);
  if (solver.info() != Eigen::Success) return std::nullopt;
  const Vec3 eigenvalues = solver.eigenvalues();
  if (eigenvalues[kLargest] <= kMinSpread) return std::nullopt;
  plane.normal = solver.eigenvectors().col(kSmallest).normalized();
  plane.planarity =
      (eigenvalues[kMiddle] - std::max(eigenvalues[kSmallest], 0.0)) /
      eigenvalues[kLargest];
  plane.linearity =
      (eigenvalues[kLargest] - eigenvalues[kMiddle]) / eigenvalues[kLargest];
  plane.thickness = std::sqrt(std::max(eigenvalues[kSmallest], 0.0));
  return plane;
}

IcpResult AlignToMap(const Points& source, const VoxelHashMap& map,
                     const Pose& initial_guess, const IcpOptions& options) {
  IcpResult result;
  result.pose = initial_guess;
  if (source.empty() || map.empty()) return result;
  PlaneCache cache;
  for (int iteration = 0; iteration < options.max_iters; ++iteration) {
    const NormalEquations equations =
        Linearize(source, map, result.pose, options, true, &cache);
    result.iterations = iteration + 1;
    // Fewer residuals than pose degrees of freedom constrain nothing.
    if (equations.count < kPoseDim) break;
    const Vec6 step = equations.jtj.ldlt().solve(-equations.jtr);
    if (!step.allFinite()) break;
    result.pose = Pose(So3Exp(step.segment<kSpaceDim>(kRotationOffset)),
                       step.segment<kSpaceDim>(kTranslationOffset)) *
                  result.pose;
    result.pose.Normalize();
    if (step.norm() < options.conv_eps) {
      result.converged = true;
      break;
    }
  }
  const NormalEquations final_equations =
      Linearize(source, map, result.pose, options, false, &cache);
  result.inliers = final_equations.count;
  result.fitness = static_cast<double>(final_equations.count) /
                   static_cast<double>(source.size());
  result.rmse = final_equations.count > 0
                    ? std::sqrt(final_equations.sse / final_equations.count)
                    : 0.0;
  result.hessian = final_equations.jtj;
  return result;
}

IcpResult AlignMultiScale(const Points& source, const Points& target,
                          const Pose& initial_guess,
                          const std::vector<double>& schedule,
                          double base_voxel, int max_iterations,
                          bool parallel) {
  IcpOptions options;
  options.parallel = parallel;
  Pose pose = initial_guess;
  for (double gate : schedule) {
    VoxelHashMap map(std::max(base_voxel, gate), VoxelHashMap::kUnbounded,
                     kMultiScalePointsPerVoxel);
    map.AddPoints(target);
    const Points coarse = VoxelDownsample(
        source, std::max(base_voxel, gate / kSourceVoxelDivisor));
    options.max_corr_dist = gate;
    options.kernel = gate / kKernelDivisor;
    options.max_iters = max_iterations;
    pose = AlignToMap(coarse, map, pose, options).pose;
  }
  const double finest = schedule.empty() ? kDefaultFinestGateVoxels * base_voxel
                                         : schedule.back();
  VoxelHashMap map(base_voxel, VoxelHashMap::kUnbounded,
                   kMultiScalePointsPerVoxel);
  map.AddPoints(target);
  options.max_corr_dist = finest;
  options.kernel = finest / kKernelDivisor;
  options.max_iters = 0;  // Statistics only.
  return AlignToMap(source, map, pose, options);
}

}  // namespace gr
