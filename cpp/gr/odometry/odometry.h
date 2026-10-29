// LiDAR odometry front-end in the spirit of KISS-ICP (Vizzo et al., RA-L
// 2023): constant-velocity prediction, sweep deskewing, adaptive
// correspondence threshold and a local voxel-hash map. `OdometryPipeline`
// wraps it into the per-trip front-end: wheel-speed + gyro motion prior, IMU
// in-sweep deskew, keyframe selection and keyframe sweeps for the back-end.
// Rigs with an IMU can run the LiDAR-inertial odometry of lio.h instead, or
// feed externally computed LIO trajectories.

#ifndef GARAGE_RECON_GR_ODOMETRY_ODOMETRY_H_
#define GARAGE_RECON_GR_ODOMETRY_ODOMETRY_H_

#include <optional>
#include <vector>

#include "gr/core/pose.h"
#include "gr/odometry/lio.h"
#include "gr/registration/registration.h"
#include "gr/sensors/imu.h"

namespace gr {

// Motion-compensates one sweep. `times` are seconds relative to the frame
// stamp (<= 0; the frame pose refers to the end of the sweep). The body moves
// with constant angular and linear velocity (body frame).
Points DeskewConstantVelocity(const Points& points,
                              const std::vector<double>& times,
                              const Vec3& angular_velocity,
                              const Vec3& linear_velocity);

struct OdometryConfig {
  double max_range = 60.0;  // m; returns beyond are dropped.
  double min_range = 1.0;   // m; returns closer are dropped (the vehicle).
  double voxel_size = 0.5;  // m, of the local map.
  int max_points_per_voxel = 20;
  // m: the correspondence threshold before the motion model has samples.
  double initial_threshold = 1.0;
  // m: deviations from the motion model below this do not adapt it.
  double min_motion_th = 0.1;
  int max_icp_iters = 50;
  double icp_conv = 1e-4;  // ICP stops when an update moves less (m / rad).
  bool deskew = true;
  bool use_wheel_prior = true;  // Wheel speed + gyro prior when available.
  // Without wheel speed: gyro rotation and the velocity of the last two
  // registered sweeps (handheld and backpack rigs).
  bool use_gyro_prior = true;
  // LiDAR-inertial odometry (lio.h) instead of the ICP front-end, for rigs
  // with an IMU: needed where the geometry degenerates (open decks seen by a
  // short-range LiDAR, long corridors). Ranges come from min / max_range.
  bool use_lio = false;
  LioConfig lio;
};

class Odometry {
 public:
  explicit Odometry(const OdometryConfig& config,
                    const Pose& initial_pose = Pose());
  Odometry(const Odometry&) = default;
  Odometry& operator=(const Odometry&) = default;
  Odometry(Odometry&&) = default;
  Odometry& operator=(Odometry&&) = default;

  // Registers one sweep (sensor frame) and returns world_from_sensor.
  // `times` are relative to `stamp` (may be empty); `predicted_motion` is an
  // optional previous -> current motion (wheel / IMU).
  Pose RegisterFrame(
      const Points& points, const std::vector<double>& times, double stamp,
      const std::optional<Pose>& predicted_motion = std::nullopt);

  const std::vector<Pose>& poses() const { return poses_; }
  const std::vector<double>& stamps() const { return stamps_; }
  Points LocalMap() const { return map_.PointCloud(); }
  const Points& last_deskewed() const { return last_deskewed_; }
  const IcpResult& last_icp() const { return last_icp_; }
  double last_sigma() const { return last_sigma_; }

 private:
  double AdaptiveThreshold() const;

  OdometryConfig config_;
  Pose initial_pose_;
  VoxelHashMap map_;
  std::vector<Pose> poses_;
  std::vector<double> stamps_;
  double model_sse_ = 0.0;
  int num_samples_ = 0;
  Points last_deskewed_;
  IcpResult last_icp_;
  double last_sigma_ = 0.0;
};

struct KeyframeConfig {
  double distance = 1.0;  // m travelled since the last keyframe.
  double angle = 10.0;    // deg rotated since the last keyframe.
  double voxel = 0.05;    // Storage resolution of keyframe sweeps.
};

struct OdometryFrame {
  Pose pose;  // world_from_lidar
  bool is_keyframe = false;
  // Keyframes (every frame with fill_sweep): the sweep in the LiDAR frame,
  // motion-compensated, cropped to [min_range, max_range] and
  // voxel-downsampled.
  Points points;
  std::vector<float> intensity;
};

class OdometryPipeline {
 public:
  OdometryPipeline(const OdometryConfig& config,
                   const KeyframeConfig& keyframe_config,
                   const Pose& vehicle_from_lidar,
                   const Pose& world_from_vehicle0 = Pose(),
                   MotionPrior prior = MotionPrior());
  OdometryPipeline(const OdometryPipeline&) = default;
  OdometryPipeline& operator=(const OdometryPipeline&) = default;
  OdometryPipeline(OdometryPipeline&&) = default;
  OdometryPipeline& operator=(OdometryPipeline&&) = default;

  // One sweep (sensor frame); `times` are relative to `stamp`. The first
  // sweep and sweeps with `force_keyframe` (e.g. the last one) are always
  // keyframes. `external_pose` (world_from_lidar) replaces the registration.
  // `fill_sweep` returns the motion-compensated sweep of every frame, not
  // only of keyframes (for map localization).
  OdometryFrame Process(const Points& points,
                        const std::vector<float>& intensity,
                        const std::vector<double>& times, double stamp,
                        bool force_keyframe = false,
                        const std::optional<Pose>& external_pose = std::nullopt,
                        bool fill_sweep = false);

  const std::vector<Pose>& poses() const { return poses_; }
  const std::vector<double>& stamps() const { return stamps_; }
  const std::vector<int>& keyframes() const { return keyframes_; }
  bool uses_prior() const { return use_prior_; }
  bool uses_gyro_prior() const { return use_gyro_; }
  bool uses_lio() const { return lio_.has_value(); }
  // Statistics of the last LIO update (uses_lio only).
  const LioStats& lio_stats() const { return lio_->last_stats(); }

 private:
  bool IsKeyframe(const Pose& pose) const;
  // The keyframe sweep: motion-compensated, cropped and downsampled.
  void FillKeyframe(const Points& points, const std::vector<float>& intensity,
                    const std::vector<double>& times, double stamp,
                    std::optional<Points> imu_deskewed,
                    OdometryFrame* frame) const;

  OdometryConfig config_;
  KeyframeConfig keyframe_config_;
  Pose vehicle_from_lidar_;
  Pose lidar_from_vehicle_;
  MotionPrior prior_;
  bool use_prior_ = false;
  bool use_gyro_ = false;
  Odometry odometry_;
  std::optional<LioOdometry> lio_;
  std::vector<Pose> poses_;
  std::vector<double> stamps_;
  std::vector<int> keyframes_;
  Pose last_keyframe_pose_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_ODOMETRY_ODOMETRY_H_
