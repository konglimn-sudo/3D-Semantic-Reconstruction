// LiDAR-IMU calibration from one trip: the IMU's rotation relative to the
// LiDAR, the gyro and accelerometer biases and the accelerometer's per-axis
// scale, from the IMU against the kinematics of the LiDAR odometry.
//
// At every `stride`-th LiDAR stamp t, over the sweeps about `half_window`
// to either side [t - h0, t + h1] (positions not interpolated), with
// hat-shaped weights peaking at t (zero at the window ends):
//
//   gyro    the weighted mean IMU rate against the weighted mean LiDAR rate
//           (piecewise constant between sweeps);
//   accel   the divided second difference of the LiDAR positions, which is
//           exactly the weighted mean world acceleration, against the IMU
//           specific forces rotated into the world with the LiDAR attitude
//           at each IMU sample, plus gravity.
//
// Model (R = imu_from_lidar, S = diag(scale), IMU time = LiDAR time + dt):
//
//   gyro    w_I = R w_L + b_g
//   accel   a_W = R_WL R^T S^-1 (f_I - b_a) + g_W
//
// The IMU at r (LiDAR axes) also reads the rotation's acceleration there,
// alpha x r + w x (w x r) (rates from the gyro, turned into LiDAR axes with
// the rig's rotation: a correction term); r is estimated with a weak
// prior (`lever_arm_sigma`) on the rig's value, which only bounds it: under
// a steady turn its horizontal part looks like an accelerometer bias, and a
// car turning about one axis does not observe its height. g_W is the gravity of the odometry
// frame (its tilt is
// estimated: the odometry may not start levelled); dt within
// `max_time_offset`. Levenberg-Marquardt with a Huber loss on each window's
// gyro and accelerometer residual (scaled by `gyro_sigma`, `accel_sigma`).
// The standard deviations come from the normal matrix at the solution: a
// car turns about one axis, so the yaw of R is fixed by the accelerations
// alone and the accelerometer scales by the tilts the trip saw; large
// values mean "not observed by this trip".

#ifndef GARAGE_RECON_GR_CALIBRATION_LIDAR_IMU_H_
#define GARAGE_RECON_GR_CALIBRATION_LIDAR_IMU_H_

#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/sensors/imu.h"

namespace gr {

struct LidarImuConfig {
  double half_window = 0.25;  // h (s).
  int stride = 1;
  bool estimate_scale = true;
  bool estimate_time_offset = true;
  double max_time_offset = 0.1;  // s
  bool estimate_lever_arm = true;
  double lever_arm_sigma = 1.0;  // m, prior on the rig's lever arm.
  double gyro_sigma = 0.01;   // rad/s
  double accel_sigma = 0.1;   // m/s^2
  double huber = 3.0;         // In sigmas.
  int max_iterations = 50;
  double tolerance = 1e-7;  // Relative cost decrease that ends the solve.
  double gravity = 9.80665;
};

struct LidarImuResult {
  bool converged = false;
  int samples = 0;
  Mat3 imu_from_lidar = Mat3::Identity();
  Vec3 gyro_bias = Vec3::Zero();
  Vec3 accel_bias = Vec3::Zero();
  Vec3 accel_scale = Vec3::Ones();
  Vec3 gravity_world = Vec3(0.0, 0.0, -9.80665);
  double time_offset = 0.0;  // s: IMU time = LiDAR time + offset.
  Vec3 lever_arm = Vec3::Zero();  // The IMU origin in LiDAR axes (m).
  // Standard deviations: rotation about the LiDAR axes (deg), biases,
  // scales (1).
  Vec3 sigma_rotation_deg = Vec3::Zero();
  Vec3 sigma_gyro_bias = Vec3::Zero();
  Vec3 sigma_accel_bias = Vec3::Zero();
  Vec3 sigma_accel_scale = Vec3::Zero();
  double sigma_time_offset = 0.0;
  Vec3 sigma_lever_arm = Vec3::Zero();
  double gyro_rms = 0.0;   // rad/s
  double accel_rms = 0.0;  // m/s^2
};

// `world_from_lidar` at `stamps` (ascending); `imu` in the IMU's own axes
// (ImuSamples::FromRows with an identity rotation). `initial` is the rig's
// imu_from_lidar.
LidarImuResult CalibrateLidarImu(const std::vector<double>& stamps,
                                 const std::vector<Pose>& world_from_lidar,
                                 const ImuSamples& imu, const Pose& initial,
                                 const LidarImuConfig& config = {});

}  // namespace gr

#endif  // GARAGE_RECON_GR_CALIBRATION_LIDAR_IMU_H_
