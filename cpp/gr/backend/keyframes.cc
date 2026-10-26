#include "gr/backend/keyframes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/voxel.h"

namespace gr {

KeyframeSet::KeyframeSet(std::vector<double> voxels) {
  for (double v : voxels) {
    GR_CHECK(v > 0.0);
    if (Slot(v) < 0) voxels_.push_back(v);
  }
  clouds_.resize(voxels_.size());
}

int KeyframeSet::Add(const Keyframe& keyframe, const Points& sweep) {
  const int k = static_cast<int>(keyframes_.size());
  keyframes_.push_back(keyframe);
  for (size_t s = 0; s < voxels_.size(); ++s) {
    clouds_[s].push_back(VoxelDownsample(sweep, voxels_[s]));
  }
  by_trip_[keyframe.trip].push_back(k);
  return k;
}

const std::vector<int>& KeyframeSet::TripIndices(int trip) const {
  // Never destroyed (Google style: no non-trivial static destructors).
  static const auto* const kEmpty = new std::vector<int>();
  const auto it = by_trip_.find(trip);
  return it == by_trip_.end() ? *kEmpty : it->second;
}

std::vector<Pose> KeyframeSet::OdomPoses() const {
  std::vector<Pose> out;
  out.reserve(keyframes_.size());
  for (const Keyframe& keyframe : keyframes_) out.push_back(keyframe.odom);
  return out;
}

int KeyframeSet::Slot(double voxel) const {
  for (size_t s = 0; s < voxels_.size(); ++s) {
    if (std::abs(voxels_[s] - voxel) < 1e-6) return static_cast<int>(s);
  }
  return -1;
}

bool KeyframeSet::HasVoxel(double voxel) const { return Slot(voxel) >= 0; }

const Points& KeyframeSet::Cloud(size_t k, double voxel) const {
  const int slot = Slot(voxel);
  GR_CHECK(slot >= 0);
  GR_CHECK(k < keyframes_.size());
  return clouds_[static_cast<size_t>(slot)][k];
}

std::vector<double> KeyframeSet::PathDistance(
    const std::vector<Pose>& poses) const {
  GR_CHECK(poses.size() == keyframes_.size());
  std::vector<double> out(keyframes_.size(), 0.0);
  for (const auto& [trip, indices] : by_trip_) {
    double travelled = 0.0;
    for (size_t a = 1; a < indices.size(); ++a) {
      travelled += (poses[static_cast<size_t>(indices[a])].translation -
                    poses[static_cast<size_t>(indices[a - 1])].translation)
                       .norm();
      out[static_cast<size_t>(indices[a])] = travelled;
    }
  }
  return out;
}

Points KeyframeSet::Submap(size_t k, int half_window, double voxel,
                           const std::vector<Pose>& poses) const {
  GR_CHECK(poses.size() == keyframes_.size());
  const int slot = Slot(voxel);
  GR_CHECK(slot >= 0);
  GR_CHECK(k < keyframes_.size());
  const std::vector<int>& indices = TripIndices(keyframes_[k].trip);
  const int64_t position =
      std::lower_bound(indices.begin(), indices.end(), static_cast<int>(k)) -
      indices.begin();
  const int64_t lo = std::max<int64_t>(0, position - half_window);
  const int64_t hi = std::min<int64_t>(static_cast<int64_t>(indices.size()),
                                       position + half_window + 1);
  const Pose k_from_world = poses[k].Inverse();
  Points points;
  for (int64_t a = lo; a < hi; ++a) {
    const size_t j = static_cast<size_t>(indices[static_cast<size_t>(a)]);
    const Pose k_from_j = k_from_world * poses[j];
    for (const Vec3& p : clouds_[static_cast<size_t>(slot)][j]) {
      points.push_back(k_from_j * p);
    }
  }
  return VoxelDownsample(points, voxel);
}

}  // namespace gr
