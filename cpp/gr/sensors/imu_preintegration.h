// IMU preintegration on the manifold (Forster et al., "On-Manifold
// Preintegration for Real-Time Visual-Inertial Odometry", T-RO 2017).
//
// The gyro and accelerometer samples between two keyframes are summarised as
// rotation, velocity and position increments in the body frame of the first
// keyframe, with their covariance and first-order Jacobians with respect to
// the IMU bias. The pose graph then relates the poses, velocities and biases
// of the two keyframes without re-integrating the samples.
//
// Conventions: the body frame is the vehicle frame (ImuSamples are expressed
// in it), world z points up and gravity is g = (0, 0, -kGravity). Bias
// vectors are [gyro bias (rad/s), accelerometer bias (m/s^2)].

#ifndef GARAGE_RECON_GR_SENSORS_IMU_PREINTEGRATION_H_
#define GARAGE_RECON_GR_SENSORS_IMU_PREINTEGRATION_H_

#include <optional>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/sensors/imu.h"

namespace gr {

inline constexpr double kGravity = 9.81;  // m/s^2, along world -z.

using Vec9 = Eigen::Matrix<double, 9, 1>;
using Mat9 = Eigen::Matrix<double, 9, 9>;

// Continuous-time noise densities of the IMU.
struct ImuNoise {
  double gyro_noise = 2e-4;       // rad/s/sqrt(Hz)
  double accel_noise = 2e-3;      // m/s^2/sqrt(Hz)
  double gyro_bias_walk = 2e-5;   // rad/s^2/sqrt(Hz)
  double accel_bias_walk = 2e-4;  // m/s^3/sqrt(Hz)
};

class ImuPreintegration {
 public:
  // Columns: [phi_i, p_i, v_i, phi_j, p_j, v_j, gyro bias_i, accel bias_i].
  using Jacobian = Eigen::Matrix<double, 9, 24>;

  // `bias` is the linearisation point of the bias Jacobians.
  explicit ImuPreintegration(const Vec6& bias = Vec6::Zero(),
                             const ImuNoise& noise = ImuNoise());

  ImuPreintegration(const ImuPreintegration&) = default;
  ImuPreintegration& operator=(const ImuPreintegration&) = default;
  ImuPreintegration(ImuPreintegration&&) = default;
  ImuPreintegration& operator=(ImuPreintegration&&) = default;

  // Adds `dt` > 0 seconds of constant body rates and specific force.
  void Integrate(const Vec3& gyro, const Vec3& accel, double dt);

  double dt() const { return dt_; }
  const Vec6& bias() const { return bias_; }
  const Mat3& delta_rotation() const { return delta_rotation_; }
  const Vec3& delta_velocity() const { return delta_velocity_; }
  const Vec3& delta_position() const { return delta_position_; }
  // Covariance of the [rotation, velocity, position] residual.
  const Mat9& covariance() const { return covariance_; }

  // Residual [rotation, velocity, position] between the states of keyframes
  // i and j (world_from_body poses, world-frame velocities) for the bias at
  // i. The Jacobian (optional) is for right rotation perturbations and
  // world-frame increments of position and velocity.
  Vec9 Residual(const Pose& world_from_body_i, const Vec3& velocity_i,
                const Pose& world_from_body_j, const Vec3& velocity_j,
                const Vec6& bias_i, Jacobian* jacobian = nullptr) const;

  // The state at j predicted from the state at i.
  void Predict(const Pose& world_from_body_i, const Vec3& velocity_i,
               const Vec6& bias_i, Pose& world_from_body_j,
               Vec3& velocity_j) const;

 private:
  // Increments corrected to first order for `bias` - bias_.
  Mat3 CorrectedRotation(const Vec6& bias) const;
  Vec3 CorrectedVelocity(const Vec6& bias) const;
  Vec3 CorrectedPosition(const Vec6& bias) const;

  Vec6 bias_;
  ImuNoise noise_;
  double dt_ = 0.0;
  Mat3 delta_rotation_ = Mat3::Identity();
  Vec3 delta_velocity_ = Vec3::Zero();
  Vec3 delta_position_ = Vec3::Zero();
  Mat9 covariance_ = Mat9::Zero();
  Mat3 rotation_by_gyro_bias_ = Mat3::Zero();
  Mat3 velocity_by_gyro_bias_ = Mat3::Zero();
  Mat3 velocity_by_accel_bias_ = Mat3::Zero();
  Mat3 position_by_gyro_bias_ = Mat3::Zero();
  Mat3 position_by_accel_bias_ = Mat3::Zero();
};

// Preintegrates the samples over [t0, t1]. The interval is split at every
// sample stamp and each piece uses the (linearly interpolated) rates at its
// midpoint, as MotionPrior does. nullopt unless t1 > t0 and the samples
// cover [t0, t1].
std::optional<ImuPreintegration> PreintegrateImu(const ImuSamples& imu,
                                                 double t0, double t1,
                                                 const Vec6& bias,
                                                 const ImuNoise& noise);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SENSORS_IMU_PREINTEGRATION_H_
