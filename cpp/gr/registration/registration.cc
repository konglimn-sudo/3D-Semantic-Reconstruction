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

// Gauss-Newton normal equations accumulated over correspondences.
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

// Residual model of the map around `point`: grows the neighbourhood until it
// is planar, because sparse sweeps sample flat surfaces as rings that are far
// apart and small radii see only lines.
CachedPlane ClassifyNeighbourhood(const VoxelHashMap& map, const Vec3& point,
                                  double plane_radius,
                                  const IcpOptions& options) {
  CachedPlane plane;
  const double max_radius = std::max(plane_radius, options.max_plane_radius);
  for (double radius = plane_radius; radius <= max_radius + 1e-9;
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
                          bool robust, PlaneCache& cache) {
  const int n = static_cast<int>(source.size());
  const double k = options.kernel;
  const double plane_radius =
      options.plane_radius > 0.0 ? options.plane_radius : map.voxel_size();
  const double inv_cache_cell = 2.0 / plane_radius;
  const bool parallel = options.parallel;
  auto weight = [&](double r2) {
    if (!robust) return 1.0;
    const double s = k + r2;
    return (k * k) / (s * s);
  };
  std::vector<Vec3> world(static_cast<size_t>(n));
  std::vector<std::optional<VoxelHashMap::Neighbor>> nearest(
      static_cast<size_t>(n));
  std::vector<VoxelKey> keys(options.point_to_plane ? world.size() : 0);
  std::vector<uint8_t> cached(keys.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
  for (int i = 0; i < n; ++i) {
    const size_t u = static_cast<size_t>(i);
    world[u] = pose * source[u];
    nearest[u] = map.Nearest(world[u], options.max_corr_dist,
                             options.exclude_frame, options.max_search_voxels);
    if (!nearest[u] || keys.empty()) continue;
    keys[u] = VoxelOf(nearest[u]->point, inv_cache_cell);
    cached[u] = cache.count(keys[u]) > 0;
  }
  if (!keys.empty()) {
    std::vector<std::pair<VoxelKey, Vec3>> fresh;
    std::unordered_set<VoxelKey, VoxelKeyHash> seen;
    for (size_t u = 0; u < keys.size(); ++u) {
      if (!nearest[u] || cached[u]) continue;
      if (seen.insert(keys[u]).second) {
        fresh.emplace_back(keys[u], nearest[u]->point);
      }
    }
    std::vector<CachedPlane> planes(fresh.size());
    const int num_fresh = static_cast<int>(fresh.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) if (parallel)
#endif
    for (int f = 0; f < num_fresh; ++f) {
      const size_t v = static_cast<size_t>(f);
      planes[v] =
          ClassifyNeighbourhood(map, fresh[v].second, plane_radius, options);
    }
    for (size_t v = 0; v < fresh.size(); ++v) {
      cache.emplace(fresh[v].first, planes[v]);
    }
  }
  auto accumulate = [&](size_t u, NormalEquations& acc) {
    if (!nearest[u]) return;
    const Vec3& world_point = world[u];
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.block<3, 3>(0, 0) = Mat3::Identity();
    jacobian.block<3, 3>(0, 3) = -Hat(world_point);
    if (options.point_to_plane) {
      const CachedPlane& plane = cache.at(keys[u]);
      if (plane.type == ResidualType::kSkip) return;
      if (plane.type == ResidualType::kPointToPlane) {
        const double r = plane.normal.dot(world_point - plane.centroid);
        const Eigen::Matrix<double, 1, 6> jp =
            plane.normal.transpose() * jacobian;
        const double w = weight(r * r);
        acc.jtj.noalias() += w * jp.transpose() * jp;
        acc.jtr.noalias() += w * jp.transpose() * r;
        acc.sse += r * r;
        acc.count += 1;
        return;
      }
    }
    const double point_scale =
        options.point_to_plane ? options.point_weight : 1.0;
    if (point_scale <= 0.0) return;
    const Vec3 r = world_point - nearest[u]->point;
    const double w = point_scale * weight(r.squaredNorm());
    acc.jtj.noalias() += w * jacobian.transpose() * jacobian;
    acc.jtr.noalias() += w * jacobian.transpose() * r;
    acc.sse += r.squaredNorm();
    acc.count += 1;
  };
  const int num_blocks = (n + kBlockSize - 1) / kBlockSize;
  std::vector<NormalEquations> blocks(static_cast<size_t>(num_blocks));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (parallel)
#endif
  for (int b = 0; b < num_blocks; ++b) {
    const int end = std::min(n, (b + 1) * kBlockSize);
    for (int i = b * kBlockSize; i < end; ++i) {
      accumulate(static_cast<size_t>(i), blocks[static_cast<size_t>(b)]);
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
      inv_voxel_size_(1.0 / voxel_size),
      max_distance_(max_distance),
      max_points_per_voxel_(max_points_per_voxel) {}

size_t VoxelHashMap::CountPoints() const {
  size_t n = 0;
  for (const auto& [key, block] : map_) n += block.points.size();
  return n;
}

void VoxelHashMap::AddPoints(const Points& points, int frame_id) {
  for (const Vec3& p : points) {
    Block& block = map_[VoxelOf(p, inv_voxel_size_)];
    if (static_cast<int>(block.points.size()) < max_points_per_voxel_) {
      block.points.push_back(p);
      block.frames.push_back(frame_id);
    }
  }
}

void VoxelHashMap::RemoveFar(const Vec3& origin) {
  const double max2 = max_distance_ * max_distance_;
  for (auto it = map_.begin(); it != map_.end();) {
    if (it->second.points.empty() ||
        (it->second.points.front() - origin).squaredNorm() > max2) {
      it = map_.erase(it);
    } else {
      ++it;
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
  const VoxelKey center = VoxelOf(query, inv_voxel_size_);
  const int reach =
      std::clamp(static_cast<int>(std::ceil(max_distance * inv_voxel_size_)), 1,
                 std::max(1, max_search_voxels));
  double best = max_distance * max_distance;
  std::optional<Neighbor> nearest;
  for (int dx = -reach; dx <= reach; ++dx) {
    for (int dy = -reach; dy <= reach; ++dy) {
      for (int dz = -reach; dz <= reach; ++dz) {
        const auto it =
            map_.find({center.x + dx, center.y + dy, center.z + dz});
        if (it == map_.end()) continue;
        const Block& block = it->second;
        for (size_t k = 0; k < block.points.size(); ++k) {
          if (exclude_frame >= 0 && block.frames[k] == exclude_frame) continue;
          const double d2 = (block.points[k] - query).squaredNorm();
          if (d2 < best) {
            best = d2;
            nearest = Neighbor{block.points[k], d2};
          }
        }
      }
    }
  }
  return nearest;
}

std::optional<VoxelHashMap::LocalPlane> VoxelHashMap::FitLocalPlane(
    const Vec3& query, double radius, int min_points, int exclude_frame) const {
  const VoxelKey center = VoxelOf(query, inv_voxel_size_);
  const int reach =
      std::max(1, static_cast<int>(std::ceil(radius * inv_voxel_size_)));
  const double r2 = radius * radius;
  Vec3 sum = Vec3::Zero();
  Mat3 outer = Mat3::Zero();
  int n = 0;
  for (int dx = -reach; dx <= reach; ++dx) {
    for (int dy = -reach; dy <= reach; ++dy) {
      for (int dz = -reach; dz <= reach; ++dz) {
        const auto it =
            map_.find({center.x + dx, center.y + dy, center.z + dz});
        if (it == map_.end()) continue;
        const Block& block = it->second;
        for (size_t k = 0; k < block.points.size(); ++k) {
          if (exclude_frame >= 0 && block.frames[k] == exclude_frame) continue;
          const Vec3& p = block.points[k];
          if ((p - query).squaredNorm() > r2) continue;
          sum += p;
          outer.noalias() += p * p.transpose();
          ++n;
        }
      }
    }
  }
  if (n < min_points) return std::nullopt;
  LocalPlane plane;
  plane.centroid = sum / n;
  const Mat3 covariance =
      outer / n - plane.centroid * plane.centroid.transpose();
  Eigen::SelfAdjointEigenSolver<Mat3> solver(covariance);
  if (solver.info() != Eigen::Success) return std::nullopt;
  const Vec3 eigenvalues = solver.eigenvalues();  // Ascending.
  if (eigenvalues[2] <= 1e-12) return std::nullopt;
  plane.normal = solver.eigenvectors().col(0).normalized();
  plane.planarity =
      (eigenvalues[1] - std::max(eigenvalues[0], 0.0)) / eigenvalues[2];
  plane.linearity = (eigenvalues[2] - eigenvalues[1]) / eigenvalues[2];
  plane.thickness = std::sqrt(std::max(eigenvalues[0], 0.0));
  return plane;
}

IcpResult AlignToMap(const Points& source, const VoxelHashMap& map,
                     const Pose& initial_guess, const IcpOptions& options) {
  IcpResult result;
  result.pose = initial_guess;
  if (source.empty() || map.empty()) return result;
  PlaneCache cache;
  for (int iteration = 0; iteration < options.max_iters; ++iteration) {
    const NormalEquations eq =
        Linearize(source, map, result.pose, options, true, cache);
    result.iterations = iteration + 1;
    if (eq.count < 6) break;
    const Vec6 dx = eq.jtj.ldlt().solve(-eq.jtr);
    if (!dx.allFinite()) break;
    result.pose = Pose(So3Exp(dx.tail<3>()), dx.head<3>()) * result.pose;
    result.pose.Normalize();
    if (dx.norm() < options.conv_eps) {
      result.converged = true;
      break;
    }
  }
  const NormalEquations final_eq =
      Linearize(source, map, result.pose, options, false, cache);
  result.inliers = final_eq.count;
  result.fitness =
      static_cast<double>(final_eq.count) / static_cast<double>(source.size());
  result.rmse =
      final_eq.count > 0 ? std::sqrt(final_eq.sse / final_eq.count) : 0.0;
  result.hessian = final_eq.jtj;
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
    VoxelHashMap map(std::max(base_voxel, gate), 1e9, 30);
    map.AddPoints(target);
    const Points coarse =
        VoxelDownsample(source, std::max(base_voxel, gate / 2.0));
    options.max_corr_dist = gate;
    options.kernel = gate / 3.0;
    options.max_iters = max_iterations;
    pose = AlignToMap(coarse, map, pose, options).pose;
  }
  const double finest = schedule.empty() ? 2.0 * base_voxel : schedule.back();
  VoxelHashMap map(base_voxel, 1e9, 30);
  map.AddPoints(target);
  options.max_corr_dist = finest;
  options.kernel = finest / 3.0;
  options.max_iters = 0;
  return AlignToMap(source, map, pose, options);
}

}  // namespace gr
