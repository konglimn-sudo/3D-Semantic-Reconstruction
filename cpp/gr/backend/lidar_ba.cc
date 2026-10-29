#include "gr/backend/lidar_ba.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/voxel.h"

namespace gr {
namespace {

// A root voxel key packs its x, y and z into 21 bits each, offset by half
// their range.
constexpr int kKeyBits = 21;
constexpr int64_t kKeyOffset = int64_t{1} << (kKeyBits - 1);
// An octree voxel splits into eight children; bit 0 of a child's octant
// says it is in the upper half along x, bit 1 along y, bit 2 along z.
constexpr int kOctants = 8;
constexpr int kUpperX = 1;
constexpr int kUpperY = 2;
constexpr int kUpperZ = 4;
// Positions of ClusterPlane's eigenvalues (ascending): the spread along the
// normal and the smaller in-plane spread.
constexpr int kNormalSpread = 0;
constexpr int kInPlaneSpread = 1;
// A plane is refitted to its inliers twice and needs three of them.
constexpr int kRefitPasses = 2;
constexpr size_t kMinPlanePoints = 3;
// Clusters closer (m) to the sensor have no ray direction.
constexpr double kMinRayLength = 1e-9;

// A sweep point in a root voxel.
struct Entry {
  int64_t key = 0;
  int32_t keyframe = 0;
  int32_t index = 0;
};

int64_t PackKey(const VoxelKey& key) {
  return ((static_cast<int64_t>(key.x) + kKeyOffset) << (2 * kKeyBits)) |
         ((static_cast<int64_t>(key.y) + kKeyOffset) << kKeyBits) |
         (static_cast<int64_t>(key.z) + kKeyOffset);
}

// The points of one root voxel, in world and keyframe frames.
struct VoxelPoints {
  std::vector<Vec3> world;
  std::vector<Vec3> local;
  std::vector<int> keyframe;  // Non-decreasing.
};

// Cuts the points of one root voxel into planes (an adaptive octree),
// appending them to `out`.
class PlaneExtractor {
 public:
  PlaneExtractor(const LidarBaConfig& config, const VoxelPoints& points,
                 const std::vector<Pose>& poses, std::vector<BaPlane>* out)
      : config_(config), points_(points), poses_(poses), out_(out) {}

  void Process(const std::vector<int>& indices, const Vec3& corner, double size,
               int depth) {
    const int min_total = config_.min_points * config_.min_keyframes;
    if (static_cast<int>(indices.size()) < min_total) return;
    PointCluster merged;
    for (int index : indices) {
      merged.Add(points_.world[static_cast<size_t>(index)]);
    }
    const ClusterPlane plane = FitClusterPlane(merged);
    if (std::sqrt(plane.eigenvalues[kNormalSpread]) <= config_.max_thickness &&
        std::sqrt(plane.eigenvalues[kInPlaneSpread]) >=
            config_.min_extent * size &&
        TryEmit(indices, plane, size)) {
      return;
    }
    if (depth >= config_.max_depth) return;
    const double half = 0.5 * size;
    std::vector<int> children[kOctants];
    for (int index : indices) {
      const Vec3 offset = points_.world[static_cast<size_t>(index)] - corner;
      const int octant = (offset.x() >= half ? kUpperX : 0) |
                         (offset.y() >= half ? kUpperY : 0) |
                         (offset.z() >= half ? kUpperZ : 0);
      children[octant].push_back(index);
    }
    for (int octant = 0; octant < kOctants; ++octant) {
      const Vec3 child_corner =
          corner + half * Vec3((octant & kUpperX) ? 1.0 : 0.0,
                               (octant & kUpperY) ? 1.0 : 0.0,
                               (octant & kUpperZ) ? 1.0 : 0.0);
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
    for (int pass = 0; pass < kRefitPasses; ++pass) {
      inliers.clear();
      merged = PointCluster();
      for (int index : indices) {
        const Vec3& world = points_.world[static_cast<size_t>(index)];
        if (std::abs(plane.normal.dot(world) + plane.offset) >
            config_.inlier_gate) {
          continue;
        }
        inliers.push_back(index);
        merged.Add(world);
      }
      if (inliers.size() < kMinPlanePoints) return false;
      plane = FitClusterPlane(merged);
    }
    if (static_cast<double>(inliers.size()) <
            config_.min_inlier_fraction * static_cast<double>(indices.size()) ||
        std::sqrt(plane.eigenvalues[kNormalSpread]) > config_.max_rms ||
        std::sqrt(plane.eigenvalues[kInPlaneSpread]) <
            config_.min_extent * size) {
      return false;
    }
    // Inliers grouped by keyframe (indices are in keyframe order).
    std::vector<PlaneObservation> observations;
    std::vector<int> counts;
    for (int inlier : inliers) {
      const size_t point = static_cast<size_t>(inlier);
      const int keyframe = points_.keyframe[point];
      if (observations.empty() || observations.back().node != keyframe) {
        observations.push_back({keyframe, PointCluster()});
        counts.push_back(0);
      }
      observations.back().points.Add(points_.local[point]);
      ++counts.back();
    }
    std::vector<PlaneObservation> kept;
    for (size_t k = 0; k < observations.size(); ++k) {
      if (counts[k] >= config_.min_points) kept.push_back(observations[k]);
    }
    // A clean plane seen by too few keyframes: splitting it cannot help.
    if (static_cast<int>(kept.size()) < config_.min_keyframes) return true;
    const int limit = std::max(config_.max_keyframes, config_.min_keyframes);
    if (static_cast<int>(kept.size()) > limit) {
      // Evenly over the keyframe order, which spreads the choice over the
      // trips and the passes of each trip.
      std::vector<PlaneObservation> subset;
      const double step = static_cast<double>(kept.size() - 1) / (limit - 1);
      for (int chosen = 0; chosen < limit; ++chosen) {
        subset.push_back(kept[static_cast<size_t>(std::lround(chosen * step))]);
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
            range > kMinRayLength ? std::abs(normal_local.dot(ray)) / range
                                  : 1.0;
        const double squared_sin_incidence =
            std::max(0.0, 1.0 - cos_incidence * cos_incidence);
        const double angular = range * config_.angular_sigma;
        const double variance = config_.point_sigma * config_.point_sigma;
        observation.points.Scale(
            variance / (variance + angular * angular * squared_sin_incidence));
      }
    }
    BaPlane out;
    out.observations = std::move(kept);
    out.centroid = plane.centroid;
    out.normal = plane.normal;
    out.rms = std::sqrt(plane.eigenvalues[kNormalSpread]);
    out.size = size;
    out_->push_back(std::move(out));
    return true;
  }

  const LidarBaConfig& config_;
  const VoxelPoints& points_;
  const std::vector<Pose>& poses_;
  std::vector<BaPlane>* out_;
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
  const double inverse_voxel = 1.0 / config.voxel;
  std::vector<std::vector<Entry>> per_keyframe(
      static_cast<size_t>(num_keyframes));
#ifdef _OPENMP
  constexpr int kKeyframesPerChunk = 4;
#pragma omp parallel for schedule(dynamic, kKeyframesPerChunk)
#endif
  for (int64_t signed_keyframe = 0; signed_keyframe < num_keyframes;
       ++signed_keyframe) {
    const size_t keyframe = static_cast<size_t>(signed_keyframe);
    const Points& sweep = keyframes.Cloud(keyframe, config.sweep_voxel);
    std::vector<Entry>& entries = per_keyframe[keyframe];
    entries.reserve(sweep.size());
    for (size_t i = 0; i < sweep.size(); ++i) {
      const double range = sweep[i].norm();
      if (range < config.min_range || range > config.max_range) continue;
      const Vec3 world = poses[keyframe] * sweep[i];
      entries.push_back({PackKey(VoxelOf(world, inverse_voxel)),
                         static_cast<int32_t>(signed_keyframe),
                         static_cast<int32_t>(i)});
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
  std::vector<size_t> starts;  // Of each root voxel's entries.
  for (size_t entry = 0; entry < entries.size(); ++entry) {
    if (entry == 0 || entries[entry].key != entries[entry - 1].key) {
      starts.push_back(entry);
    }
  }
  starts.push_back(entries.size());

  const int64_t num_voxels = static_cast<int64_t>(starts.size()) - 1;
  std::vector<std::vector<BaPlane>> found(static_cast<size_t>(num_voxels));
#ifdef _OPENMP
  constexpr int kVoxelsPerChunk = 16;
#pragma omp parallel for schedule(dynamic, kVoxelsPerChunk)
#endif
  for (int64_t voxel = 0; voxel < num_voxels; ++voxel) {
    const size_t first = starts[static_cast<size_t>(voxel)];
    const size_t last = starts[static_cast<size_t>(voxel) + 1];
    if (static_cast<int>(last - first) <
        config.min_points * config.min_keyframes) {
      continue;
    }
    VoxelPoints points;
    std::vector<int> indices;
    for (size_t entry = first; entry < last; ++entry) {
      const size_t keyframe = static_cast<size_t>(entries[entry].keyframe);
      const Vec3& local = keyframes.Cloud(
          keyframe,
          config.sweep_voxel)[static_cast<size_t>(entries[entry].index)];
      indices.push_back(static_cast<int>(points.world.size()));
      points.local.push_back(local);
      points.world.push_back(poses[keyframe] * local);
      points.keyframe.push_back(entries[entry].keyframe);
    }
    const Vec3 corner =
        config.voxel * Vec3(std::floor(points.world[0].x() * inverse_voxel),
                            std::floor(points.world[0].y() * inverse_voxel),
                            std::floor(points.world[0].z() * inverse_voxel));
    PlaneExtractor extractor(config, points, poses,
                             &found[static_cast<size_t>(voxel)]);
    extractor.Process(indices, corner, config.voxel, /*depth=*/0);
  }
  std::vector<BaPlane> planes;
  for (std::vector<BaPlane>& part : found) {
    for (BaPlane& plane : part) planes.push_back(std::move(plane));
  }
  return planes;
}

}  // namespace gr
