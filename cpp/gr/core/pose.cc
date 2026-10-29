#include "gr/core/pose.h"

#include <algorithm>
#include <cmath>

#include "gr/core/check.h"

namespace gr {
namespace {

// Below this angle the closed forms of the SO(3) Jacobians lose precision
// and their Taylor series take over.
constexpr double kSeriesAngle = 1e-6;

// Jr(phi) = I - (1 - cos a) / a^2 Hat(phi) + (a - sin a) / a^3 Hat(phi)^2
// (a = |phi|): the coefficients tend to these as a -> 0.
constexpr double kJacobianLinearLimit = 1.0 / 2.0;
constexpr double kJacobianQuadraticLimit = 1.0 / 6.0;

// Jr^-1(phi) = I + Hat(phi) / 2 + c(a) Hat(phi)^2 with
// c(a) = 1 / a^2 - (1 + cos a) / (2 a sin a), which tends to 1 / 12.
constexpr double kInverseJacobianLinear = 1.0 / 2.0;
constexpr double kInverseJacobianQuadraticLimit = 1.0 / 12.0;

}  // namespace

Vec3 So3Log(const Mat3& rotation) {
  const Eigen::AngleAxisd angle_axis(rotation);
  double angle = angle_axis.angle();
  Vec3 axis = angle_axis.axis();
  if (angle > kPi) {  // Keep the angle in [0, pi].
    angle = 2.0 * kPi - angle;
    axis = -axis;
  }
  return angle * axis;
}

Mat3 So3RightJacobian(const Vec3& phi) {
  const double angle = phi.norm();
  const Mat3 skew = Hat(phi);
  if (angle < kSeriesAngle) {
    return Mat3::Identity() - kJacobianLinearLimit * skew +
           kJacobianQuadraticLimit * skew * skew;
  }
  const double squared_angle = angle * angle;
  return Mat3::Identity() - (1.0 - std::cos(angle)) / squared_angle * skew +
         (angle - std::sin(angle)) / (squared_angle * angle) * skew * skew;
}

Mat3 So3RightJacobianInverse(const Vec3& phi) {
  const double angle = phi.norm();
  const Mat3 skew = Hat(phi);
  if (angle < kSeriesAngle) {
    return Mat3::Identity() + kInverseJacobianLinear * skew +
           kInverseJacobianQuadraticLimit * skew * skew;
  }
  const double quadratic =
      1.0 / (angle * angle) -
      (1.0 + std::cos(angle)) / (2.0 * angle * std::sin(angle));
  return Mat3::Identity() + kInverseJacobianLinear * skew +
         quadratic * skew * skew;
}

Mat4 Pose::Matrix() const {
  Mat4 matrix = Mat4::Identity();
  matrix.topLeftCorner<kSpaceDim, kSpaceDim>() = rotation;
  matrix.topRightCorner<kSpaceDim, 1>() = translation;
  return matrix;
}

void Pose::Normalize() {
  Eigen::Quaterniond quaternion(rotation);
  quaternion.normalize();
  rotation = quaternion.toRotationMatrix();
}

Pose Interpolate(const Pose& start, const Pose& end, double fraction) {
  const Eigen::Quaterniond start_rotation(start.rotation);
  const Eigen::Quaterniond end_rotation(end.rotation);
  return Pose(
      start_rotation.slerp(fraction, end_rotation).toRotationMatrix(),
      (1.0 - fraction) * start.translation + fraction * end.translation);
}

Points TransformPoints(const Points& points, const Pose& pose) {
  Points out(points.size());
  for (size_t i = 0; i < points.size(); ++i) out[i] = pose * points[i];
  return out;
}

Mat6 PoseInformation(double sigma_translation, double sigma_rotation_deg) {
  Mat6 information = Mat6::Zero();
  const double translation_weight =
      1.0 / (sigma_translation * sigma_translation);
  const double sigma_rotation = DegToRad(sigma_rotation_deg);
  const double rotation_weight = 1.0 / (sigma_rotation * sigma_rotation);
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    information(kTranslationOffset + axis, kTranslationOffset + axis) =
        translation_weight;
    information(kRotationOffset + axis, kRotationOffset + axis) =
        rotation_weight;
  }
  return information;
}

Pose InterpolatePose(const std::vector<double>& times,
                     const std::vector<Pose>& poses, double time) {
  GR_CHECK(!times.empty() && times.size() == poses.size());
  if (times.size() == 1 || time <= times.front()) return poses.front();
  if (time >= times.back()) return poses.back();
  const size_t after = static_cast<size_t>(
      std::upper_bound(times.begin(), times.end(), time) - times.begin());
  const size_t before = after - 1;
  const double interval = times[after] - times[before];
  return Interpolate(poses[before], poses[after],
                     interval > 0.0 ? (time - times[before]) / interval : 0.0);
}

std::vector<Pose> InterpolatePoses(const std::vector<double>& times,
                                   const std::vector<Pose>& poses,
                                   const std::vector<double>& query_times) {
  std::vector<Pose> out;
  out.reserve(query_times.size());
  for (double time : query_times) {
    out.push_back(InterpolatePose(times, poses, time));
  }
  return out;
}

}  // namespace gr
