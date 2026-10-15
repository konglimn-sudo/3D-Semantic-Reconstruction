// Plane extraction for the LiDAR bundle adjustment (plan section 3.2).
//
// The keyframe sweeps are placed with the current poses and cut into an
// adaptive voxel octree (BALM / VoxelMap): a voxel whose points from all
// keyframes are planar becomes a plane, otherwise it is split into eight
// children down to `max_depth`. Each plane keeps, per keyframe, the moments
// of that keyframe's points in its own frame (plane_factor.h), so the pose
// graph can move the keyframes while the plane is re-fitted. The octree
// separates surfaces that meet at a kink (floor / ramp, wall / column),
// which a fixed-radius plane fit would blend into one biased plane.

#ifndef GARAGE_RECON_GR_LIDAR_BA_H_
#define GARAGE_RECON_GR_LIDAR_BA_H_

#include <vector>

#include "gr/keyframes.h"
#include "gr/plane_factor.h"
#include "gr/pose.h"
#include "gr/types.h"

namespace gr {

struct LidarBaConfig {
  bool enabled = true;
  double sweep_voxel = 0.1;  // Keyframe sweep resolution used (m).
  double min_range = 1.0;    // Returns used (m); distant structure
  double max_range = 30.0;   // constrains corridors and ramps.
  double voxel = 1.0;        // Root voxel of the octree (m).
  int max_depth = 2;         // Subdivisions below the root (1 m -> 0.25 m).
  // A voxel is a plane candidate if sqrt(l_min) of all its points is below
  // `max_thickness` (m) and sqrt(l_mid) / voxel size above `min_extent` (the
  // points spread across the voxel, not along a line). The plane is then
  // re-fitted to the points within `inlier_gate` (m) of it and kept if at
  // least `min_inlier_fraction` of the points are inliers with an RMS below
  // `max_rms` (m); otherwise the voxel is split. This rejects voxels at
  // edges (wall / floor, column corners) where two surfaces mix.
  double max_thickness = 0.04;
  double min_extent = 0.1;
  double inlier_gate = 0.05;
  double min_inlier_fraction = 0.9;
  double max_rms = 0.02;
  int min_points = 6;  // Per keyframe and plane.
  int min_keyframes = 2;
  int max_keyframes = 20;     // Per plane; evenly subsampled beyond this.
  double point_sigma = 0.02;  // Point-to-plane noise (m).
  // Beam direction error (rad): it moves a point by range * angle across
  // the ray, i.e. range * angle * sin(incidence) along the plane normal. A
  // keyframe's cluster on a plane is weighted by point_sigma^2 / (that^2 +
  // point_sigma^2): distant floor seen at grazing incidence, where a beam
  // elevation bias of 0.1-0.2 deg moves the floor by centimetres, then
  // cannot bend a long corridor. 0 weights all clusters alike.
  double angular_sigma = 0.0;
  // Points of one keyframe on one plane share calibration and deskew
  // errors, so their information is capped at this many points.
  double max_cluster_weight = 30.0;
  double robust_scale = 3.0;  // Cauchy scale on the mean point chi2.
  int rounds = 2;             // Plane extraction + optimisation.
  int iterations = 10;        // Levenberg-Marquardt iterations per round.
};

struct BaPlane {
  std::vector<PlaneObservation> observations;  // Ascending keyframe.
  Vec3 centroid = Vec3::Zero();                // World frame.
  Vec3 normal = Vec3::UnitZ();
  double rms = 0.0;   // Point-to-plane RMS at extraction (m).
  double size = 0.0;  // Edge of the octree voxel (m).
};

// Planes of the keyframe sweeps at `poses` (world_from_lidar per keyframe).
// The KeyframeSet must keep sweeps at config.sweep_voxel.
std::vector<BaPlane> ExtractPlanes(const KeyframeSet& keyframes,
                                   const std::vector<Pose>& poses,
                                   const LidarBaConfig& config);

struct LidarBaRound {
  int planes = 0;
  int observations = 0;
  double rms_median_before = 0.0;  // Plane RMS distribution (m).
  double rms_p90_before = 0.0;
  double rms_median_after = 0.0;
  double rms_p90_after = 0.0;
  double shift_median = 0.0;  // Keyframe translation change (m).
  double shift_max = 0.0;
  int iterations = 0;
};

struct LidarBaSummary {
  std::vector<LidarBaRound> rounds;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_LIDAR_BA_H_
