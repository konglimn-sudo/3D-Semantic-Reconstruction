#include "gr/sensors/imu_preintegration.h"

#include <algorithm>
#include <vector>

#include "gr/core/check.h"

namespace gr {
namespace {

// The white noise that drives the increments: gyro noise, then
// accelerometer noise.
constexpr int kGyroNoiseOffset = 0;
constexpr int kAccelNoiseOffset = kGyroNoiseOffset + kSpaceDim;
constexpr int kImuNoiseDim = kAccelNoiseOffset + kSpaceDim;

Vec3 GravityVector() { return Vec3(0.0, 0.0, -kGravity); }

Vec3 GyroBias(const Vec6& bias) {
  return bias.segment<kSpaceDim>(kGyroBiasOffset);
}
Vec3 AccelBias(const Vec6& bias) {
  return bias.segment<kSpaceDim>(kAccelBiasOffset);
}

}  // namespace

ImuPreintegration::ImuPreintegration(const Vec6& bias, const ImuNoise& noise)
    : bias_(bias), noise_(noise) {}

void ImuPreintegration::Integrate(const Vec3& gyro, const Vec3& accel,
                                  double dt) {
  GR_CHECK(dt > 0.0);
  const Vec3 rate = gyro - GyroBias(bias_);
  const Vec3 force = accel - AccelBias(bias_);
  const Mat3 step_rotation = So3Exp(rate * dt);
  const Mat3 step_jacobian = So3RightJacobian(rate * dt);
  const Mat3 rotation = delta_rotation_;  // Delta R from i to this sample.
  const Mat3 rotated_force_hat = rotation * Hat(force);
  const double squared_dt = dt * dt;

  // Covariance of [rotation, velocity, position]: propagated by the state
  // transition, plus the noise that enters through the input matrix.
  Mat9 transition = Mat9::Identity();
  transition.block<kSpaceDim, kSpaceDim>(kRotationRow, kRotationRow) =
      step_rotation.transpose();
  transition.block<kSpaceDim, kSpaceDim>(kVelocityRow, kRotationRow) =
      -rotated_force_hat * dt;
  transition.block<kSpaceDim, kSpaceDim>(kPositionRow, kRotationRow) =
      -0.5 * rotated_force_hat * squared_dt;
  transition.block<kSpaceDim, kSpaceDim>(kPositionRow, kVelocityRow) =
      Mat3::Identity() * dt;
  Eigen::Matrix<double, kImuResidualDim, kImuNoiseDim> noise_input =
      Eigen::Matrix<double, kImuResidualDim, kImuNoiseDim>::Zero();
  noise_input.block<kSpaceDim, kSpaceDim>(kRotationRow, kGyroNoiseOffset) =
      step_jacobian * dt;
  noise_input.block<kSpaceDim, kSpaceDim>(kVelocityRow, kAccelNoiseOffset) =
      rotation * dt;
  noise_input.block<kSpaceDim, kSpaceDim>(kPositionRow, kAccelNoiseOffset) =
      0.5 * rotation * squared_dt;
  Mat6 noise = Mat6::Zero();
  noise.block<kSpaceDim, kSpaceDim>(kGyroNoiseOffset, kGyroNoiseOffset) =
      Mat3::Identity() * (noise_.gyro_noise * noise_.gyro_noise / dt);
  noise.block<kSpaceDim, kSpaceDim>(kAccelNoiseOffset, kAccelNoiseOffset) =
      Mat3::Identity() * (noise_.accel_noise * noise_.accel_noise / dt);
  covariance_ = transition * covariance_ * transition.transpose() +
                noise_input * noise * noise_input.transpose();

  // Bias Jacobians; the order matters (each uses the previous values).
  position_by_accel_bias_ +=
      velocity_by_accel_bias_ * dt - 0.5 * rotation * squared_dt;
  position_by_gyro_bias_ +=
      velocity_by_gyro_bias_ * dt -
      0.5 * rotated_force_hat * rotation_by_gyro_bias_ * squared_dt;
  velocity_by_accel_bias_ -= rotation * dt;
  velocity_by_gyro_bias_ -= rotated_force_hat * rotation_by_gyro_bias_ * dt;
  rotation_by_gyro_bias_ =
      step_rotation.transpose() * rotation_by_gyro_bias_ - step_jacobian * dt;

  // Increments.
  delta_position_ += delta_velocity_ * dt + 0.5 * rotation * force * squared_dt;
  delta_velocity_ += rotation * force * dt;
  delta_rotation_ = rotation * step_rotation;
  Pose normalized(delta_rotation_, Vec3::Zero());
  normalized.Normalize();
  delta_rotation_ = normalized.rotation;
  dt_ += dt;
}

Mat3 ImuPreintegration::CorrectedRotation(const Vec6& bias) const {
  const Vec3 gyro_bias_change = GyroBias(bias - bias_);
  return delta_rotation_ * So3Exp(rotation_by_gyro_bias_ * gyro_bias_change);
}

Vec3 ImuPreintegration::CorrectedVelocity(const Vec6& bias) const {
  const Vec6 change = bias - bias_;
  return delta_velocity_ + velocity_by_gyro_bias_ * GyroBias(change) +
         velocity_by_accel_bias_ * AccelBias(change);
}

Vec3 ImuPreintegration::CorrectedPosition(const Vec6& bias) const {
  const Vec6 change = bias - bias_;
  return delta_position_ + position_by_gyro_bias_ * GyroBias(change) +
         position_by_accel_bias_ * AccelBias(change);
}

Vec9 ImuPreintegration::Residual(const Pose& world_from_body_i,
                                 const Vec3& velocity_i,
                                 const Pose& world_from_body_j,
                                 const Vec3& velocity_j, const Vec6& bias_i,
                                 Jacobian* jacobian) const {
  const Mat3& rotation_i = world_from_body_i.rotation;
  const Mat3& rotation_j = world_from_body_j.rotation;
  const Mat3 rotation_i_inverse = rotation_i.transpose();
  const Vec3 velocity_change = velocity_j - velocity_i - GravityVector() * dt_;
  const Vec3 position_change =
      world_from_body_j.translation - world_from_body_i.translation -
      velocity_i * dt_ - 0.5 * GravityVector() * dt_ * dt_;
  const Vec3 gyro_bias_change = GyroBias(bias_i - bias_);
  const Mat3 corrected_rotation = CorrectedRotation(bias_i);

  Vec9 residual;
  const Vec3 rotation_error =
      So3Log(corrected_rotation.transpose() * rotation_i_inverse * rotation_j);
  residual.segment<kSpaceDim>(kRotationRow) = rotation_error;
  residual.segment<kSpaceDim>(kVelocityRow) =
      rotation_i_inverse * velocity_change - CorrectedVelocity(bias_i);
  residual.segment<kSpaceDim>(kPositionRow) =
      rotation_i_inverse * position_change - CorrectedPosition(bias_i);
  if (jacobian == nullptr) return residual;

  Jacobian& derivative = *jacobian;
  const auto block = [&derivative](int row, int column) {
    return derivative.block<kSpaceDim, kSpaceDim>(row, column);
  };
  derivative.setZero();
  const Mat3 inverse_right_jacobian = So3RightJacobianInverse(rotation_error);
  block(kRotationRow, kRotationICol) =
      -inverse_right_jacobian * rotation_j.transpose() * rotation_i;
  block(kRotationRow, kRotationJCol) = inverse_right_jacobian;
  block(kRotationRow, kGyroBiasCol) =
      -inverse_right_jacobian * So3Exp(rotation_error).transpose() *
      So3RightJacobian(rotation_by_gyro_bias_ * gyro_bias_change) *
      rotation_by_gyro_bias_;
  block(kVelocityRow, kRotationICol) =
      Hat(rotation_i_inverse * velocity_change);
  block(kVelocityRow, kVelocityICol) = -rotation_i_inverse;
  block(kVelocityRow, kVelocityJCol) = rotation_i_inverse;
  block(kVelocityRow, kGyroBiasCol) = -velocity_by_gyro_bias_;
  block(kVelocityRow, kAccelBiasCol) = -velocity_by_accel_bias_;
  block(kPositionRow, kRotationICol) =
      Hat(rotation_i_inverse * position_change);
  block(kPositionRow, kPositionICol) = -rotation_i_inverse;
  block(kPositionRow, kVelocityICol) = -rotation_i_inverse * dt_;
  block(kPositionRow, kPositionJCol) = rotation_i_inverse;
  block(kPositionRow, kGyroBiasCol) = -position_by_gyro_bias_;
  block(kPositionRow, kAccelBiasCol) = -position_by_accel_bias_;
  return residual;
}

void ImuPreintegration::Predict(const Pose& world_from_body_i,
                                const Vec3& velocity_i, const Vec6& bias_i,
                                Pose* world_from_body_j,
                                Vec3* velocity_j) const {
  const Mat3& rotation_i = world_from_body_i.rotation;
  world_from_body_j->rotation = rotation_i * CorrectedRotation(bias_i);
  world_from_body_j->Normalize();
  world_from_body_j->translation = world_from_body_i.translation +
                                   velocity_i * dt_ +
                                   0.5 * GravityVector() * dt_ * dt_ +
                                   rotation_i * CorrectedPosition(bias_i);
  *velocity_j = velocity_i + GravityVector() * dt_ +
                rotation_i * CorrectedVelocity(bias_i);
}

std::optional<ImuPreintegration> PreintegrateImu(const ImuSamples& imu,
                                                 double start_time,
                                                 double end_time,
                                                 const Vec6& bias,
                                                 const ImuNoise& noise) {
  if (imu.size() < 2 || end_time <= start_time ||
      start_time < imu.stamps.front() || end_time > imu.stamps.back()) {
    return std::nullopt;
  }
  GR_CHECK(imu.gyro.size() == imu.size() && imu.accel.size() == imu.size());
  const std::vector<double> knots =
      SplitAtStamps(imu.stamps, start_time, end_time);
  ImuPreintegration preintegration(bias, noise);
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    const double step = knots[k + 1] - knots[k];
    if (step <= 0.0) continue;
    const double midpoint = 0.5 * (knots[k] + knots[k + 1]);
    preintegration.Integrate(InterpolateSeries(imu.stamps, imu.gyro, midpoint),
                             InterpolateSeries(imu.stamps, imu.accel, midpoint),
                             step);
  }
  return preintegration;
}

}  // namespace gr
