#include "gr/mapping/change_detection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// Neighbours are the voxels this many steps around (26 of them).
constexpr int kNeighbourhoodRadius = 1;
// A voxel's centre, in voxels from its corner.
constexpr double kVoxelCentre = 0.5;
// Rounding slack (m) of the flat-cluster test.
constexpr double kThicknessTolerance = 1e-9;

using VoxelSet = std::unordered_set<VoxelKey, VoxelKeyHash>;

bool HasNeighbour(const VoxelSet& set, const VoxelKey& key) {
  for (int dx = -kNeighbourhoodRadius; dx <= kNeighbourhoodRadius; ++dx) {
    for (int dy = -kNeighbourhoodRadius; dy <= kNeighbourhoodRadius; ++dy) {
      for (int dz = -kNeighbourhoodRadius; dz <= kNeighbourhoodRadius; ++dz) {
        if (set.count({key.x + dx, key.y + dy, key.z + dz})) return true;
      }
    }
  }
  return false;
}

// The voxel coordinate of `key` along `axis`.
int& Coordinate(VoxelKey& key, int axis) {
  return axis == 0 ? key.x : axis == 1 ? key.y : key.z;
}

}  // namespace

ChangeDetector::ChangeDetector(const Points& map_points,
                               const ChangeConfig& config)
    : config_(config), inverse_voxel_(1.0 / config.voxel) {
  for (const Vec3& point : map_points) {
    map_.insert(VoxelOf(point, inverse_voxel_));
  }
}

void ChangeDetector::AddSweep(const Points& sweep, const Vec3& sensor,
                              double stamp) {
  ++sweeps_;
  VoxelSet hit;
  VoxelSet ray_voxels;  // Ray end voxels.
  const double inverse_ray_voxel = 1.0 / config_.ray_voxel;
  std::vector<Vec3> ends;
  for (const Vec3& point : sweep) {
    if ((point - sensor).norm() > config_.max_range) continue;
    hit.insert(VoxelOf(point, inverse_voxel_));
    if (ray_voxels.insert(VoxelOf(point, inverse_ray_voxel)).second) {
      ends.push_back(point);
    }
  }
  for (const VoxelKey& key : hit) {
    Hits& hits = hits_[key];
    if (hits.sweeps++ == 0) hits.first = stamp;
    hits.last = stamp;
  }
  // Map voxels the rays pass through, once per sweep.
  VoxelSet passed;
  const Vec3 origin = sensor * inverse_voxel_;
  for (const Vec3& end : ends) {
    const Vec3 ray = end - sensor;
    const double length = ray.norm() - config_.free_margin;
    if (length <= 0.0) continue;
    const Vec3 direction = ray / ray.norm();
    // Voxel walk in voxel units (Amanatides & Woo, unbounded).
    VoxelKey key = VoxelOf(sensor, inverse_voxel_);
    int step[kSpaceDim];
    double t_max[kSpaceDim];    // Ray parameter of the next voxel boundary.
    double t_delta[kSpaceDim];  // Ray parameter between two boundaries.
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      const int start = Coordinate(key, axis);
      if (direction[axis] > 0.0) {
        step[axis] = 1;
        t_max[axis] = (start + 1 - origin[axis]) / direction[axis];
        t_delta[axis] = 1.0 / direction[axis];
      } else if (direction[axis] < 0.0) {
        step[axis] = -1;
        t_max[axis] = (start - origin[axis]) / direction[axis];
        t_delta[axis] = -1.0 / direction[axis];
      } else {
        step[axis] = 0;
        t_max[axis] = t_delta[axis] = kInf;
      }
    }
    const double t_end = length * inverse_voxel_;
    while (true) {
      if (map_.count(key)) passed.insert(key);
      int axis = 0;  // The axis whose boundary comes first.
      for (int other = 1; other < kSpaceDim; ++other) {
        if (t_max[other] < t_max[axis]) axis = other;
      }
      if (t_max[axis] > t_end) break;
      Coordinate(key, axis) += step[axis];
      t_max[axis] += t_delta[axis];
    }
  }
  for (const VoxelKey& key : passed) ++free_[key];
}

std::vector<ChangeCluster> ChangeDetector::Clusters() const {
  VoxelSet added;
  for (const auto& [key, hits] : hits_) {
    if (hits.sweeps >= config_.min_hits &&
        hits.last - hits.first >= config_.min_hit_span && !map_.count(key) &&
        !HasNeighbour(map_, key)) {
      added.insert(key);
    }
  }
  // Voxels the new trip hit (any number of times): a map voxel next to one
  // is still there (surface jitter across a voxel boundary).
  VoxelSet hit;
  for (const auto& [key, hits] : hits_) hit.insert(key);
  VoxelSet removed;
  for (const auto& [key, count] : free_) {
    if (count >= config_.min_free && !HasNeighbour(hit, key)) {
      removed.insert(key);
    }
  }
  std::vector<ChangeCluster> out;
  for (bool is_added : {true, false}) {
    VoxelSet unvisited = is_added ? added : removed;
    while (!unvisited.empty()) {
      std::vector<VoxelKey> stack = {*unvisited.begin()};
      unvisited.erase(unvisited.begin());
      ChangeCluster cluster;
      cluster.added = is_added;
      cluster.min = Vec3::Constant(kInf);
      cluster.max = -cluster.min;
      while (!stack.empty()) {
        const VoxelKey key = stack.back();
        stack.pop_back();
        const Vec3 corner = Vec3(key.x, key.y, key.z) * config_.voxel;
        cluster.min = cluster.min.cwiseMin(corner);
        cluster.max =
            cluster.max.cwiseMax(corner + Vec3::Constant(config_.voxel));
        cluster.center +=
            corner + Vec3::Constant(kVoxelCentre * config_.voxel);
        ++cluster.voxels;
        for (int dx = -kNeighbourhoodRadius; dx <= kNeighbourhoodRadius;
             ++dx) {
          for (int dy = -kNeighbourhoodRadius; dy <= kNeighbourhoodRadius;
               ++dy) {
            for (int dz = -kNeighbourhoodRadius; dz <= kNeighbourhoodRadius;
                 ++dz) {
              const VoxelKey neighbour{key.x + dx, key.y + dy, key.z + dz};
              const auto found = unvisited.find(neighbour);
              if (found != unvisited.end()) {
                unvisited.erase(found);
                stack.push_back(neighbour);
              }
            }
          }
        }
      }
      if (cluster.voxels < config_.min_cluster_voxels) continue;
      cluster.center /= cluster.voxels;
      const Vec3 size = cluster.max - cluster.min;
      if (size.z() <= config_.flat_thickness + kThicknessTolerance) continue;
      const double length = std::max(size.x(), size.y());
      const double width = std::min(size.x(), size.y());
      cluster.vehicle_like = length >= config_.vehicle_min_length &&
                             length <= config_.vehicle_max_length &&
                             width >= config_.vehicle_min_width &&
                             width <= config_.vehicle_max_width &&
                             size.z() <= config_.vehicle_max_height;
      out.push_back(cluster);
    }
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const ChangeCluster& left, const ChangeCluster& right) {
                     return left.voxels > right.voxels;
                   });
  return out;
}

}  // namespace gr
