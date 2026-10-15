#include "gr/imu_preintegration.h"

#include <algorithm>
#include <vector>

#include "gr/check.h"

namespace gr {
namespace {

Vec3 GravityVector() { return Vec3(0.0, 0.0, -kGravity); }

}  // namespace

ImuPreintegration::ImuPreintegration(const Vec6& bias, const ImuNoise& noise)
    : bias_(bias), noise_(noise) {}

void ImuPreintegration::Integrate(const Vec3& gyro, const Vec3& accel,
                                  double dt) {
  GR_CHECK(dt > 0.0);
  const Vec3 rate = gyro - bias_.head<3>();
  const Vec3 force = accel - bias_.tail<3>();
  const Mat3 step_rotation = So3Exp(rate * dt);
  const Mat3 step_jacobian = So3RightJacobian(rate * dt);
  const Mat3 rotation = delta_rotation_;  // Delta R from i to this sample.
  const Mat3 rotated_force_hat = rotation * Hat(force);
  const double dt2 = dt * dt;

  // Covariance of [rotation, velocity, position].
  Mat9 a = Mat9::Identity();
  a.block<3, 3>(0, 0) = step_rotation.transpose();
  a.block<3, 3>(3, 0) = -rotated_force_hat * dt;
  a.block<3, 3>(6, 0) = -0.5 * rotated_force_hat * dt2;
  a.block<3, 3>(6, 3) = Mat3::Identity() * dt;
  Eigen::Matrix<double, 9, 6> b = Eigen::Matrix<double, 9, 6>::Zero();
  b.block<3, 3>(0, 0) = step_jacobian * dt;
  b.block<3, 3>(3, 3) = rotation * dt;
  b.block<3, 3>(6, 3) = 0.5 * rotation * dt2;
  Mat6 noise = Mat6::Zero();
  noise.block<3, 3>(0, 0) =
      Mat3::Identity() * (noise_.gyro_noise * noise_.gyro_noise / dt);
  noise.block<3, 3>(3, 3) =
      Mat3::Identity() * (noise_.accel_noise * noise_.accel_noise / dt);
  covariance_ = a * covariance_ * a.transpose() + b * noise * b.transpose();

  // Bias Jacobians; the order matters (each uses the previous values).
  position_by_accel_bias_ +=
      velocity_by_accel_bias_ * dt - 0.5 * rotation * dt2;
  position_by_gyro_bias_ +=
      velocity_by_gyro_bias_ * dt -
      0.5 * rotated_force_hat * rotation_by_gyro_bias_ * dt2;
  velocity_by_accel_bias_ -= rotation * dt;
  velocity_by_gyro_bias_ -= rotated_force_hat * rotation_by_gyro_bias_ * dt;
  rotation_by_gyro_bias_ =
      step_rotation.transpose() * rotation_by_gyro_bias_ - step_jacobian * dt;

  // Increments.
  delta_position_ += delta_velocity_ * dt + 0.5 * rotation * force * dt2;
  delta_velocity_ += rotation * force * dt;
  delta_rotation_ = rotation * step_rotation;
  Pose normalized(delta_rotation_, Vec3::Zero());
  normalized.Normalize();
  delta_rotation_ = normalized.rotation;
  dt_ += dt;
}

Mat3 ImuPreintegration::CorrectedRotation(const Vec6& bias) const {
  const Vec3 gyro_bias_change = (bias - bias_).head<3>();
  return delta_rotation_ * So3Exp(rotation_by_gyro_bias_ * gyro_bias_change);
}

Vec3 ImuPreintegration::CorrectedVelocity(const Vec6& bias) const {
  const Vec6 change = bias - bias_;
  return delta_velocity_ + velocity_by_gyro_bias_ * change.head<3>() +
         velocity_by_accel_bias_ * change.tail<3>();
}

Vec3 ImuPreintegration::CorrectedPosition(const Vec6& bias) const {
  const Vec6 change = bias - bias_;
  return delta_position_ + position_by_gyro_bias_ * change.head<3>() +
         position_by_accel_bias_ * change.tail<3>();
}

Vec9 ImuPreintegration::Residual(const Pose& world_from_body_i,
                                 const Vec3& velocity_i,
                                 const Pose& world_from_body_j,
                                 const Vec3& velocity_j, const Vec6& bias_i,
                                 Jacobian* jacobian) const {
  const Mat3& rotation_i = world_from_body_i.rotation;
  const Mat3& rotation_j = world_from_body_j.rotation;
  const Mat3 ri_t = rotation_i.transpose();
  const Vec3 velocity_change = velocity_j - velocity_i - GravityVector() * dt_;
  const Vec3 position_change =
      world_from_body_j.translation - world_from_body_i.translation -
      velocity_i * dt_ - 0.5 * GravityVector() * dt_ * dt_;
  const Vec3 gyro_bias_change = (bias_i - bias_).head<3>();
  const Mat3 corrected_rotation = CorrectedRotation(bias_i);

  Vec9 residual;
  const Vec3 rotation_error =
      So3Log(corrected_rotation.transpose() * ri_t * rotation_j);
  residual.segment<3>(0) = rotation_error;
  residual.segment<3>(3) = ri_t * velocity_change - CorrectedVelocity(bias_i);
  residual.segment<3>(6) = ri_t * position_change - CorrectedPosition(bias_i);
  if (jacobian == nullptr) return residual;

  Jacobian& j = *jacobian;
  j.setZero();
  const Mat3 jr_inv = So3RightJacobianInverse(rotation_error);
  j.block<3, 3>(0, 0) = -jr_inv * rotation_j.transpose() * rotation_i;
  j.block<3, 3>(0, 9) = jr_inv;
  j.block<3, 3>(0, 18) =
      -jr_inv * So3Exp(rotation_error).transpose() *
      So3RightJacobian(rotation_by_gyro_bias_ * gyro_bias_change) *
      rotation_by_gyro_bias_;
  j.block<3, 3>(3, 0) = Hat(ri_t * velocity_change);
  j.block<3, 3>(3, 6) = -ri_t;
  j.block<3, 3>(3, 15) = ri_t;
  j.block<3, 3>(3, 18) = -velocity_by_gyro_bias_;
  j.block<3, 3>(3, 21) = -velocity_by_accel_bias_;
  j.block<3, 3>(6, 0) = Hat(ri_t * position_change);
  j.block<3, 3>(6, 3) = -ri_t;
  j.block<3, 3>(6, 6) = -ri_t * dt_;
  j.block<3, 3>(6, 12) = ri_t;
  j.block<3, 3>(6, 18) = -position_by_gyro_bias_;
  j.block<3, 3>(6, 21) = -position_by_accel_bias_;
  return residual;
}

void ImuPreintegration::Predict(const Pose& world_from_body_i,
                                const Vec3& velocity_i, const Vec6& bias_i,
                                Pose& world_from_body_j,
                                Vec3& velocity_j) const {
  const Mat3& rotation_i = world_from_body_i.rotation;
  world_from_body_j.rotation = rotation_i * CorrectedRotation(bias_i);
  world_from_body_j.Normalize();
  world_from_body_j.translation = world_from_body_i.translation +
                                  velocity_i * dt_ +
                                  0.5 * GravityVector() * dt_ * dt_ +
                                  rotation_i * CorrectedPosition(bias_i);
  velocity_j = velocity_i + GravityVector() * dt_ +
               rotation_i * CorrectedVelocity(bias_i);
}

std::optional<ImuPreintegration> PreintegrateImu(const ImuSamples& imu,
                                                 double t0, double t1,
                                                 const Vec6& bias,
                                                 const ImuNoise& noise) {
  if (imu.size() < 2 || t1 <= t0 || t0 < imu.stamps.front() ||
      t1 > imu.stamps.back()) {
    return std::nullopt;
  }
  GR_CHECK(imu.gyro.size() == imu.size() && imu.accel.size() == imu.size());
  std::vector<double> knots{t0};
  for (auto it = std::upper_bound(imu.stamps.begin(), imu.stamps.end(), t0);
       it != imu.stamps.end() && *it < t1; ++it) {
    knots.push_back(*it);
  }
  knots.push_back(t1);
  ImuPreintegration preintegration(bias, noise);
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    const double dt = knots[k + 1] - knots[k];
    if (dt <= 0.0) continue;
    const double mid = 0.5 * (knots[k] + knots[k + 1]);
    preintegration.Integrate(InterpolateSeries(imu.stamps, imu.gyro, mid),
                             InterpolateSeries(imu.stamps, imu.accel, mid), dt);
  }
  return preintegration;
}

}  // namespace gr
