#include "gr/lidar_ba.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

#include "gr/check.h"
#include "gr/voxel.h"

namespace gr {
namespace {

// A sweep point in a root voxel.
struct Entry {
  int64_t key = 0;
  int32_t keyframe = 0;
  int32_t index = 0;
};

int64_t PackKey(const VoxelKey& key) {
  constexpr int64_t kOffset = int64_t{1} << 20;
  return ((static_cast<int64_t>(key.x) + kOffset) << 42) |
         ((static_cast<int64_t>(key.y) + kOffset) << 21) |
         (static_cast<int64_t>(key.z) + kOffset);
}

// The points of one root voxel, in world and keyframe frames.
struct VoxelPoints {
  std::vector<Vec3> world;
  std::vector<Vec3> local;
  std::vector<int> keyframe;  // Non-decreasing.
};

class PlaneExtractor {
 public:
  PlaneExtractor(const LidarBaConfig& config, const VoxelPoints& points,
                 const std::vector<Pose>& poses, std::vector<BaPlane>& out)
      : config_(config), points_(points), poses_(poses), out_(out) {}

  void Process(const std::vector<int>& indices, const Vec3& corner, double size,
               int depth) {
    const int min_total = config_.min_points * config_.min_keyframes;
    if (static_cast<int>(indices.size()) < min_total) return;
    PointCluster merged;
    for (int i : indices) merged.Add(points_.world[static_cast<size_t>(i)]);
    const ClusterPlane plane = FitClusterPlane(merged);
    if (std::sqrt(plane.eigenvalues[0]) <= config_.max_thickness &&
        std::sqrt(plane.eigenvalues[1]) >= config_.min_extent * size &&
        TryEmit(indices, plane, size)) {
      return;
    }
    if (depth >= config_.max_depth) return;
    const double half = 0.5 * size;
    std::vector<int> children[8];
    for (int i : indices) {
      const Vec3 offset = points_.world[static_cast<size_t>(i)] - corner;
      const int octant = (offset.x() >= half ? 1 : 0) |
                         (offset.y() >= half ? 2 : 0) |
                         (offset.z() >= half ? 4 : 0);
      children[octant].push_back(i);
    }
    for (int octant = 0; octant < 8; ++octant) {
      const Vec3 child_corner = corner + half * Vec3((octant & 1) ? 1.0 : 0.0,
                                                     (octant & 2) ? 1.0 : 0.0,
                                                     (octant & 4) ? 1.0 : 0.0);
      Process(children[octant], child_corner, half, depth + 1);
    }
  }

 private:
  // Re-fits the plane to its inliers (twice) and stores it if the inliers
  // dominate, are thin and come from enough keyframes. Returns false if the
  // voxel does not hold one clean plane.
  bool TryEmit(const std::vector<int>& indices, ClusterPlane plane,
               double size) {
    std::vector<int> inliers;
    PointCluster merged;
    for (int pass = 0; pass < 2; ++pass) {
      inliers.clear();
      merged = PointCluster();
      for (int i : indices) {
        const Vec3& world = points_.world[static_cast<size_t>(i)];
        if (std::abs(plane.normal.dot(world) + plane.offset) >
            config_.inlier_gate) {
          continue;
        }
        inliers.push_back(i);
        merged.Add(world);
      }
      if (inliers.size() < 3) return false;
      plane = FitClusterPlane(merged);
    }
    if (static_cast<double>(inliers.size()) <
            config_.min_inlier_fraction * static_cast<double>(indices.size()) ||
        std::sqrt(plane.eigenvalues[0]) > config_.max_rms ||
        std::sqrt(plane.eigenvalues[1]) < config_.min_extent * size) {
      return false;
    }
    // Inliers grouped by keyframe (indices are in keyframe order).
    std::vector<PlaneObservation> observations;
    std::vector<int> counts;
    for (int i : inliers) {
      const size_t ii = static_cast<size_t>(i);
      const int keyframe = points_.keyframe[ii];
      if (observations.empty() || observations.back().node != keyframe) {
        observations.push_back({keyframe, PointCluster()});
        counts.push_back(0);
      }
      observations.back().points.Add(points_.local[ii]);
      ++counts.back();
    }
    std::vector<PlaneObservation> kept;
    for (size_t a = 0; a < observations.size(); ++a) {
      if (counts[a] >= config_.min_points) kept.push_back(observations[a]);
    }
    // A clean plane seen by too few keyframes: splitting it cannot help.
    if (static_cast<int>(kept.size()) < config_.min_keyframes) return true;
    const int limit = std::max(config_.max_keyframes, config_.min_keyframes);
    if (static_cast<int>(kept.size()) > limit) {
      // Evenly over the keyframe order, which spreads the choice over the
      // trips and the passes of each trip.
      std::vector<PlaneObservation> subset;
      const double step = static_cast<double>(kept.size() - 1) / (limit - 1);
      for (int s = 0; s < limit; ++s) {
        subset.push_back(kept[static_cast<size_t>(std::lround(s * step))]);
      }
      kept = std::move(subset);
    }
    for (PlaneObservation& observation : kept) {
      const double weight = observation.points.weight();
      if (weight > config_.max_cluster_weight) {
        observation.points.Scale(config_.max_cluster_weight / weight);
      }
      if (config_.angular_sigma > 0.0) {
        // The cluster mean is in the keyframe's sensor frame: the ray.
        const Vec3 ray = observation.points.Mean();
        const double range = ray.norm();
        const Vec3 normal_local =
            poses_[static_cast<size_t>(observation.node)].rotation.transpose() *
            plane.normal;
        const double cos_incidence =
            range > 1e-9 ? std::abs(normal_local.dot(ray)) / range : 1.0;
        const double sin2 = std::max(0.0, 1.0 - cos_incidence * cos_incidence);
        const double angular = range * config_.angular_sigma;
        const double sigma2 = config_.point_sigma * config_.point_sigma;
        observation.points.Scale(sigma2 / (sigma2 + angular * angular * sin2));
      }
    }
    BaPlane out;
    out.observations = std::move(kept);
    out.centroid = plane.centroid;
    out.normal = plane.normal;
    out.rms = std::sqrt(plane.eigenvalues[0]);
    out.size = size;
    out_.push_back(std::move(out));
    return true;
  }

  const LidarBaConfig& config_;
  const VoxelPoints& points_;
  const std::vector<Pose>& poses_;
  std::vector<BaPlane>& out_;
};

}  // namespace

std::vector<BaPlane> ExtractPlanes(const KeyframeSet& keyframes,
                                   const std::vector<Pose>& poses,
                                   const LidarBaConfig& config) {
  GR_CHECK(poses.size() == keyframes.size());
  GR_CHECK(config.voxel > 0.0 && config.max_depth >= 0);
  GR_CHECK(config.min_keyframes >= 2 && config.min_points >= 1);
  GR_CHECK(keyframes.HasVoxel(config.sweep_voxel));
  const int64_t num_keyframes = static_cast<int64_t>(keyframes.size());
  const double inv_voxel = 1.0 / config.voxel;
  std::vector<std::vector<Entry>> per_keyframe(
      static_cast<size_t>(num_keyframes));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
  for (int64_t k = 0; k < num_keyframes; ++k) {
    const size_t kk = static_cast<size_t>(k);
    const Points& sweep = keyframes.Cloud(kk, config.sweep_voxel);
    std::vector<Entry>& entries = per_keyframe[kk];
    entries.reserve(sweep.size());
    for (size_t i = 0; i < sweep.size(); ++i) {
      const double range = sweep[i].norm();
      if (range < config.min_range || range > config.max_range) continue;
      const Vec3 world = poses[kk] * sweep[i];
      entries.push_back({PackKey(VoxelOf(world, inv_voxel)),
                         static_cast<int32_t>(k), static_cast<int32_t>(i)});
    }
  }
  std::vector<Entry> entries;
  for (std::vector<Entry>& part : per_keyframe) {
    entries.insert(entries.end(), part.begin(), part.end());
    std::vector<Entry>().swap(part);
  }
  std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
    return std::tie(a.key, a.keyframe, a.index) <
           std::tie(b.key, b.keyframe, b.index);
  });
  std::vector<size_t> starts;
  for (size_t e = 0; e < entries.size(); ++e) {
    if (e == 0 || entries[e].key != entries[e - 1].key) starts.push_back(e);
  }
  starts.push_back(entries.size());

  const int64_t num_voxels = static_cast<int64_t>(starts.size()) - 1;
  std::vector<std::vector<BaPlane>> found(static_cast<size_t>(num_voxels));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16)
#endif
  for (int64_t v = 0; v < num_voxels; ++v) {
    const size_t first = starts[static_cast<size_t>(v)];
    const size_t last = starts[static_cast<size_t>(v) + 1];
    if (static_cast<int>(last - first) <
        config.min_points * config.min_keyframes) {
      continue;
    }
    VoxelPoints points;
    std::vector<int> indices;
    for (size_t e = first; e < last; ++e) {
      const size_t k = static_cast<size_t>(entries[e].keyframe);
      const Vec3& local = keyframes.Cloud(
          k, config.sweep_voxel)[static_cast<size_t>(entries[e].index)];
      indices.push_back(static_cast<int>(points.world.size()));
      points.local.push_back(local);
      points.world.push_back(poses[k] * local);
      points.keyframe.push_back(entries[e].keyframe);
    }
    const Vec3 corner =
        config.voxel * Vec3(std::floor(points.world[0].x() * inv_voxel),
                            std::floor(points.world[0].y() * inv_voxel),
                            std::floor(points.world[0].z() * inv_voxel));
    PlaneExtractor extractor(config, points, poses,
                             found[static_cast<size_t>(v)]);
    extractor.Process(indices, corner, config.voxel, 0);
  }
  std::vector<BaPlane> planes;
  for (std::vector<BaPlane>& part : found) {
    for (BaPlane& plane : part) planes.push_back(std::move(plane));
  }
  return planes;
}

}  // namespace gr
