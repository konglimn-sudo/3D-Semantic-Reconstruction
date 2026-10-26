// LiDAR-inertial odometry: an iterated error-state Kalman filter in the
// spirit of FAST-LIO2 (Xu et al., T-RO 2022), on the vehicle frame.
//
// State: world <- body rotation and position, world velocity, gyro and
// accelerometer biases (15 error states); gravity along world -z (the vehicle
// frame is levelled at conversion). The body is the vehicle axes at the IMU
// (ImuSamples::position), where the accelerometer measures; the vehicle
// origin of a handheld rig lies on the ground, 1.8 m away, where rotation
// would read as acceleration. Between sweeps the IMU propagates the
// state and its covariance; the propagated poses deskew the sweep; then
// point-to-plane residuals against a local voxel map update the state in a
// few Gauss-Newton iterations. Directions the geometry does not constrain
// (a ground-dominated scene, a corridor) keep the IMU prediction instead of
// drifting freely, which the constant-velocity ICP front-end cannot do.

#ifndef GARAGE_RECON_GR_ODOMETRY_LIO_H_
#define GARAGE_RECON_GR_ODOMETRY_LIO_H_

#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/registration/registration.h"
#include "gr/sensors/imu.h"

namespace gr {

struct LioConfig {
  double min_range = 1.0;  // Returns used (m).
  double max_range = 60.0;
  double map_voxel = 0.5;  // Local map (m); at most this many points each.
  int max_points_per_voxel = 20;
  double map_radius = 100.0;   // Voxels farther are dropped (m).
  double scan_voxel = 0.5;     // Sweep points used in the update (m).
  double insert_voxel = 0.25;  // Sweep points added to the map (m).
  int max_iterations = 6;
  double convergence = 1e-4;  // Update step (rad / m) that ends iterating.
  // Planes of the map around a point: PCA within the radius, doubled up to
  // max_plane_radius until planar.
  double plane_radius = 0.0;  // <= 0: map_voxel.
  double max_plane_radius = 1.2;
  int min_plane_points = 6;
  double min_planarity = 0.5;
  double max_plane_thickness = 0.05;
  double max_residual = 0.5;   // Point-to-plane gate (m).
  double point_sigma = 0.05;   // Point-to-plane noise (m).
  double robust_scale = 0.1;   // Cauchy scale of the residual (m).
  // Where no plane is found (cars, parapets, vegetation): point-to-point
  // residuals to the nearest map point, with this weight relative to a
  // plane residual (0 disables). They constrain the directions a scene of
  // floor and ceiling leaves open.
  double point_weight = 0.05;
  // Zero-velocity update: the rig stands still when, over `zupt_window`
  // seconds before the sweep, the gyro stays below `zupt_max_gyro` (rad/s)
  // and the specific force within `zupt_max_accel` (m/s^2) of its mean
  // (handheld surveys stop at control points). The velocity is then
  // measured as zero with `zupt_sigma` (m/s); 0 disables.
  double zupt_window = 0.3;
  double zupt_max_gyro = 0.03;
  double zupt_max_accel = 0.1;
  double zupt_sigma = 0.02;
  // Speed limit of the carrier (m/s; 0 disables): a velocity above it is
  // pulled back to it with `speed_sigma` (m/s). On a walking rig, a stretch of
  // degenerate geometry otherwise lets the IMU-integrated velocity run away.
  double max_speed = 0.0;
  double speed_sigma = 0.1;
  // Continuous-time IMU noise (effective values: vibration, scale and timing
  // errors included) and bias random walks. Too tight a prediction outweighs
  // the weak LiDAR constraints of a degenerate stretch: on Hilti Parking_1
  // accel 0.05 drifted by 30 m, 0.3 tracked to 1.4 m (FAST-LIO uses ~0.3).
  double gyro_noise = 0.05;        // rad/s/sqrt(Hz)
  double accel_noise = 0.3;        // m/s^2/sqrt(Hz)
  double gyro_bias_walk = 1e-4;    // rad/s^2/sqrt(Hz)
  double accel_bias_walk = 1e-3;   // m/s^3/sqrt(Hz)
  double gravity = 9.80665;        // m/s^2
  // Initial uncertainty: the first sweep may already be in motion.
  double init_velocity_sigma = 1.0;    // m/s
  double init_tilt_sigma_deg = 1.0;    // roll / pitch
  double init_gyro_bias_sigma = 0.01;  // rad/s
  double init_accel_bias_sigma = 0.1;  // m/s^2
};

struct LioStats {
  int correspondences = 0;
  int iterations = 0;
  double rmse = 0.0;  // Point-to-plane, inliers (m).
  // Smallest eigenvalue of the translation block of the LiDAR information
  // (1/m^2): small in degenerate geometry.
  double min_translation_information = 0.0;
  int point_residuals = 0;  // Point-to-point fallbacks among them.
  bool stationary = false;  // A zero-velocity update was applied.
  Vec3 velocity = Vec3::Zero();  // World, of the IMU (m/s).
};

class LioOdometry {
 public:
  // `imu` in the vehicle frame (ImuSamples); starts at world <- vehicle
  // `world_from_vehicle0`.
  LioOdometry(const LioConfig& config, ImuSamples imu,
              const Pose& vehicle_from_lidar,
              const Pose& world_from_vehicle0 = Pose());

  // Registers one sweep (LiDAR frame, `times` relative to `stamp`, <= 0) and
  // returns world <- LiDAR at `stamp`.
  Pose Process(const Points& points, const std::vector<double>& times,
               double stamp);

  // The last sweep deskewed to its stamp (LiDAR frame, all points).
  const Points& last_deskewed() const { return deskewed_; }
  const LioStats& last_stats() const { return stats_; }
  Pose world_from_vehicle() const;
  const Vec3& velocity() const { return velocity_; }
  const Vec3& gyro_bias() const { return gyro_bias_; }
  const Vec3& accel_bias() const { return accel_bias_; }

  using Mat15 = Eigen::Matrix<double, 15, 15>;
  using Vec15 = Eigen::Matrix<double, 15, 1>;
  const Mat15& covariance() const { return covariance_; }

 private:
  struct Sample {
    double t;
    Mat3 rotation;
    Vec3 position;
  };
  // Propagates the state and covariance to `t`, recording the poses.
  void Propagate(double t);
  // World <- body at `t` within the recorded propagation (clamped).
  void PoseAt(double t, Mat3& rotation, Vec3& position) const;
  // Gyro-only poses before the first stamp (static start assumed).
  void BackfillFirstSweep(double t_first, double stamp);
  Points Deskew(const Points& points, const std::vector<double>& times,
                double stamp) const;
  void Update(const Points& sweep_body, bool stationary);
  // Whether the IMU says the rig stood still just before `t`.
  bool Stationary(double t) const;

  LioConfig config_;
  ImuSamples imu_;
  Pose body_from_vehicle_;
  Pose body_from_lidar_;
  Pose lidar_from_body_;
  VoxelHashMap map_;
  bool initialized_ = false;
  double time_ = 0.0;
  size_t imu_index_ = 0;  // First IMU sample after `time_`.
  Mat3 rotation_ = Mat3::Identity();  // World <- body.
  Vec3 position_ = Vec3::Zero();
  Vec3 velocity_ = Vec3::Zero();
  Vec3 gyro_bias_ = Vec3::Zero();
  Vec3 accel_bias_ = Vec3::Zero();
  Mat15 covariance_ = Mat15::Identity();
  std::vector<Sample> samples_;  // Poses since the previous sweep.
  Points deskewed_;
  LioStats stats_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_ODOMETRY_LIO_H_
