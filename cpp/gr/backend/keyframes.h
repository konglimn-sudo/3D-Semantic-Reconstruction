// Keyframes of all trips in one table: the nodes of the pose graph, with
// their sweeps kept at the voxel resolutions the back-end works at (Scan
// Context, loop verification, scan-to-map refinement).

#ifndef GARAGE_RECON_GR_BACKEND_KEYFRAMES_H_
#define GARAGE_RECON_GR_BACKEND_KEYFRAMES_H_

#include <map>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"

namespace gr {

struct Keyframe {
  int trip = 0;
  int frame = 0;  // LiDAR frame index within the trip.
  double stamp = 0.0;
  Pose odom;  // world_from_lidar from odometry.
};

class KeyframeSet {
 public:
  // `voxels`: the resolutions (m, all positive) at which sweeps are kept.
  explicit KeyframeSet(std::vector<double> voxels = {0.2});
  KeyframeSet(const KeyframeSet&) = default;
  KeyframeSet& operator=(const KeyframeSet&) = default;
  KeyframeSet(KeyframeSet&&) = default;
  KeyframeSet& operator=(KeyframeSet&&) = default;

  // Adds a keyframe; `sweep` is in its LiDAR frame and is stored downsampled
  // at every resolution of the set. Returns the keyframe index.
  int Add(const Keyframe& keyframe, const Points& sweep);

  size_t size() const { return keyframes_.size(); }
  const Keyframe& operator[](size_t k) const { return keyframes_[k]; }
  const std::vector<Keyframe>& keyframes() const { return keyframes_; }
  const std::vector<double>& voxels() const { return voxels_; }
  int num_trips() const {
    return by_trip_.empty() ? 0 : by_trip_.rbegin()->first + 1;
  }

  // Keyframe indices of a trip, ascending (empty for an unknown trip).
  const std::vector<int>& TripIndices(int trip) const;
  std::vector<Pose> OdomPoses() const;
  bool HasVoxel(double voxel) const;

  // The sweep of keyframe k at one of the kept resolutions.
  const Points& Cloud(size_t k, double voxel) const;
  // Cumulative travelled distance along each trip (restarts at 0 per trip).
  std::vector<double> PathDistance(const std::vector<Pose>& poses) const;
  // Sweeps of the keyframes within `half_window` of k on its trip, in k's
  // frame, merged and voxel-downsampled.
  Points Submap(size_t k, int half_window, double voxel,
                const std::vector<Pose>& poses) const;

 private:
  int Slot(double voxel) const;  // -1 if the resolution is not kept.

  std::vector<double> voxels_;
  std::vector<Keyframe> keyframes_;
  std::vector<std::vector<Points>> clouds_;  // [resolution][keyframe]
  std::map<int, std::vector<int>> by_trip_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_BACKEND_KEYFRAMES_H_
