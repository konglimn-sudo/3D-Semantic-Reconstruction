#include "gr/mapping/change_detection.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace gr {
namespace {

bool HasNeighbour(const std::unordered_set<VoxelKey, VoxelKeyHash>& set,
                  const VoxelKey& key) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (set.count({key.x + dx, key.y + dy, key.z + dz})) return true;
      }
    }
  }
  return false;
}

}  // namespace

ChangeDetector::ChangeDetector(const Points& map_points,
                               const ChangeConfig& config)
    : config_(config), inv_(1.0 / config.voxel) {
  for (const Vec3& p : map_points) map_.insert(VoxelOf(p, inv_));
}

void ChangeDetector::AddSweep(const Points& sweep, const Vec3& sensor,
                              double stamp) {
  ++sweeps_;
  std::unordered_set<VoxelKey, VoxelKeyHash> hit;
  std::unordered_set<VoxelKey, VoxelKeyHash> rays;  // Ray end voxels.
  const double ray_inv = 1.0 / config_.ray_voxel;
  std::vector<Vec3> ends;
  for (const Vec3& p : sweep) {
    if ((p - sensor).norm() > config_.max_range) continue;
    hit.insert(VoxelOf(p, inv_));
    if (rays.insert(VoxelOf(p, ray_inv)).second) ends.push_back(p);
  }
  for (const VoxelKey& key : hit) {
    Hits& h = hits_[key];
    if (h.sweeps++ == 0) h.first = stamp;
    h.last = stamp;
  }
  // Map voxels the rays pass through, once per sweep.
  std::unordered_set<VoxelKey, VoxelKeyHash> passed;
  const Vec3 origin = sensor * inv_;
  for (const Vec3& end : ends) {
    const Vec3 d = end - sensor;
    const double length = d.norm() - config_.free_margin;
    if (length <= 0.0) continue;
    const Vec3 u = d / d.norm();
    // Voxel walk in voxel units (Amanatides & Woo, unbounded).
    VoxelKey key = VoxelOf(sensor, inv_);
    int step[3];
    double t_max[3];
    double t_delta[3];
    const int ijk[3] = {key.x, key.y, key.z};
    for (int a = 0; a < 3; ++a) {
      if (u[a] > 0.0) {
        step[a] = 1;
        t_max[a] = (ijk[a] + 1 - origin[a]) / u[a];
        t_delta[a] = 1.0 / u[a];
      } else if (u[a] < 0.0) {
        step[a] = -1;
        t_max[a] = (ijk[a] - origin[a]) / u[a];
        t_delta[a] = -1.0 / u[a];
      } else {
        step[a] = 0;
        t_max[a] = t_delta[a] = std::numeric_limits<double>::infinity();
      }
    }
    const double t_end = length * inv_;
    while (true) {
      if (map_.count(key)) passed.insert(key);
      int axis = 0;
      if (t_max[1] < t_max[axis]) axis = 1;
      if (t_max[2] < t_max[axis]) axis = 2;
      if (t_max[axis] > t_end) break;
      (axis == 0 ? key.x : axis == 1 ? key.y : key.z) += step[axis];
      t_max[axis] += t_delta[axis];
    }
  }
  for (const VoxelKey& key : passed) ++free_[key];
}

std::vector<ChangeCluster> ChangeDetector::Clusters() const {
  std::unordered_set<VoxelKey, VoxelKeyHash> added;
  for (const auto& [key, h] : hits_) {
    if (h.sweeps >= config_.min_hits &&
        h.last - h.first >= config_.min_hit_span && !map_.count(key) &&
        !HasNeighbour(map_, key)) {
      added.insert(key);
    }
  }
  // Voxels the new trip hit (any number of times): a map voxel next to one
  // is still there (surface jitter across a voxel boundary).
  std::unordered_set<VoxelKey, VoxelKeyHash> hit;
  for (const auto& [key, h] : hits_) hit.insert(key);
  std::unordered_set<VoxelKey, VoxelKeyHash> removed;
  for (const auto& [key, count] : free_) {
    if (count >= config_.min_free && !HasNeighbour(hit, key)) {
      removed.insert(key);
    }
  }
  std::vector<ChangeCluster> out;
  for (bool is_added : {true, false}) {
    std::unordered_set<VoxelKey, VoxelKeyHash> todo = is_added ? added
                                                               : removed;
    while (!todo.empty()) {
      std::vector<VoxelKey> stack = {*todo.begin()};
      todo.erase(todo.begin());
      ChangeCluster c;
      c.added = is_added;
      c.min = Vec3::Constant(std::numeric_limits<double>::infinity());
      c.max = -c.min;
      while (!stack.empty()) {
        const VoxelKey key = stack.back();
        stack.pop_back();
        const Vec3 lo = Vec3(key.x, key.y, key.z) * config_.voxel;
        c.min = c.min.cwiseMin(lo);
        c.max = c.max.cwiseMax(lo + Vec3::Constant(config_.voxel));
        c.center += lo + Vec3::Constant(0.5 * config_.voxel);
        ++c.voxels;
        for (int dx = -1; dx <= 1; ++dx) {
          for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
              const VoxelKey n{key.x + dx, key.y + dy, key.z + dz};
              const auto found = todo.find(n);
              if (found != todo.end()) {
                todo.erase(found);
                stack.push_back(n);
              }
            }
          }
        }
      }
      if (c.voxels < config_.min_cluster_voxels) continue;
      c.center /= c.voxels;
      const Vec3 size = c.max - c.min;
      if (size.z() <= config_.flat_thickness + 1e-9) continue;
      const double length = std::max(size.x(), size.y());
      const double width = std::min(size.x(), size.y());
      c.vehicle_like = length >= config_.vehicle_min_length &&
                       length <= config_.vehicle_max_length &&
                       width >= config_.vehicle_min_width &&
                       width <= config_.vehicle_max_width &&
                       size.z() <= config_.vehicle_max_height;
      out.push_back(c);
    }
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const ChangeCluster& a, const ChangeCluster& b) {
                     return a.voxels > b.voxels;
                   });
  return out;
}

}  // namespace gr
