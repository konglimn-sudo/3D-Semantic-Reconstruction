#include "gr/backend/keyframes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/voxel.h"

namespace gr {
namespace {

// Resolutions closer than this (m) are the same.
constexpr double kVoxelTolerance = 1e-6;

}  // namespace

KeyframeSet::KeyframeSet(std::vector<double> voxels) {
  for (double voxel : voxels) {
    GR_CHECK(voxel > 0.0);
    if (Slot(voxel) == kNoSlot) voxels_.push_back(voxel);
  }
  clouds_.resize(voxels_.size());
}

int KeyframeSet::Add(const Keyframe& keyframe, const Points& sweep) {
  const int index = static_cast<int>(keyframes_.size());
  keyframes_.push_back(keyframe);
  for (size_t slot = 0; slot < voxels_.size(); ++slot) {
    clouds_[slot].push_back(VoxelDownsample(sweep, voxels_[slot]));
  }
  by_trip_[keyframe.trip].push_back(index);
  return index;
}

const std::vector<int>& KeyframeSet::TripIndices(int trip) const {
  // Never destroyed (Google style: no non-trivial static destructors).
  static const auto* const kEmpty = new std::vector<int>();
  const auto found = by_trip_.find(trip);
  return found == by_trip_.end() ? *kEmpty : found->second;
}

std::vector<Pose> KeyframeSet::OdomPoses() const {
  std::vector<Pose> out;
  out.reserve(keyframes_.size());
  for (const Keyframe& keyframe : keyframes_) out.push_back(keyframe.odom);
  return out;
}

int KeyframeSet::Slot(double voxel) const {
  for (size_t slot = 0; slot < voxels_.size(); ++slot) {
    if (std::abs(voxels_[slot] - voxel) < kVoxelTolerance) {
      return static_cast<int>(slot);
    }
  }
  return kNoSlot;
}

bool KeyframeSet::HasVoxel(double voxel) const {
  return Slot(voxel) != kNoSlot;
}

const Points& KeyframeSet::Cloud(size_t index, double voxel) const {
  const int slot = Slot(voxel);
  GR_CHECK(slot != kNoSlot);
  GR_CHECK(index < keyframes_.size());
  return clouds_[static_cast<size_t>(slot)][index];
}

std::vector<double> KeyframeSet::PathDistance(
    const std::vector<Pose>& poses) const {
  GR_CHECK(poses.size() == keyframes_.size());
  std::vector<double> out(keyframes_.size(), 0.0);
  for (const auto& [trip, indices] : by_trip_) {
    double travelled = 0.0;
    for (size_t position = 1; position < indices.size(); ++position) {
      travelled +=
          (poses[static_cast<size_t>(indices[position])].translation -
           poses[static_cast<size_t>(indices[position - 1])].translation)
              .norm();
      out[static_cast<size_t>(indices[position])] = travelled;
    }
  }
  return out;
}

Points KeyframeSet::Submap(size_t index, int half_window, double voxel,
                           const std::vector<Pose>& poses) const {
  GR_CHECK(poses.size() == keyframes_.size());
  const int slot = Slot(voxel);
  GR_CHECK(slot != kNoSlot);
  GR_CHECK(index < keyframes_.size());
  const std::vector<int>& indices = TripIndices(keyframes_[index].trip);
  const int64_t position = std::lower_bound(indices.begin(), indices.end(),
                                            static_cast<int>(index)) -
                           indices.begin();
  const int64_t first = std::max<int64_t>(0, position - half_window);
  const int64_t end = std::min<int64_t>(static_cast<int64_t>(indices.size()),
                                        position + half_window + 1);
  const Pose center_from_world = poses[index].Inverse();
  Points points;
  for (int64_t neighbor_position = first; neighbor_position < end;
       ++neighbor_position) {
    const size_t neighbor =
        static_cast<size_t>(indices[static_cast<size_t>(neighbor_position)]);
    const Pose center_from_neighbor = center_from_world * poses[neighbor];
    for (const Vec3& point : clouds_[static_cast<size_t>(slot)][neighbor]) {
      points.push_back(center_from_neighbor * point);
    }
  }
  return VoxelDownsample(points, voxel);
}

}  // namespace gr
