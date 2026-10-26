#include "gr/core/pose.h"

#include <algorithm>
#include <cmath>

#include "gr/core/check.h"

namespace gr {

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
  const Mat3 p = Hat(phi);
  if (angle < 1e-6) return Mat3::Identity() - 0.5 * p + (1.0 / 6.0) * p * p;
  const double angle2 = angle * angle;
  return Mat3::Identity() - (1.0 - std::cos(angle)) / angle2 * p +
         (angle - std::sin(angle)) / (angle2 * angle) * p * p;
}

Mat3 So3RightJacobianInverse(const Vec3& phi) {
  const double angle = phi.norm();
  const Mat3 p = Hat(phi);
  if (angle < 1e-6) return Mat3::Identity() + 0.5 * p + (1.0 / 12.0) * p * p;
  const double c = 1.0 / (angle * angle) -
                   (1.0 + std::cos(angle)) / (2.0 * angle * std::sin(angle));
  return Mat3::Identity() + 0.5 * p + c * p * p;
}

Mat4 Pose::Matrix() const {
  Mat4 m = Mat4::Identity();
  m.block<3, 3>(0, 0) = rotation;
  m.block<3, 1>(0, 3) = translation;
  return m;
}

void Pose::Normalize() {
  Eigen::Quaterniond q(rotation);
  q.normalize();
  rotation = q.toRotationMatrix();
}

Pose Interpolate(const Pose& a, const Pose& b, double fraction) {
  const Eigen::Quaterniond qa(a.rotation), qb(b.rotation);
  return Pose(qa.slerp(fraction, qb).toRotationMatrix(),
              (1.0 - fraction) * a.translation + fraction * b.translation);
}

Points TransformPoints(const Points& points, const Pose& pose) {
  Points out(points.size());
  for (size_t i = 0; i < points.size(); ++i) out[i] = pose * points[i];
  return out;
}

Mat6 PoseInformation(double sigma_translation, double sigma_rotation_deg) {
  Mat6 information = Mat6::Zero();
  const double inv_t = 1.0 / (sigma_translation * sigma_translation);
  const double sigma_r = DegToRad(sigma_rotation_deg);
  const double inv_r = 1.0 / (sigma_r * sigma_r);
  for (int a = 0; a < 3; ++a) {
    information(a, a) = inv_t;
    information(a + 3, a + 3) = inv_r;
  }
  return information;
}

Pose InterpolatePose(const std::vector<double>& times,
                     const std::vector<Pose>& poses, double t) {
  GR_CHECK(!times.empty() && times.size() == poses.size());
  if (times.size() == 1 || t <= times.front()) return poses.front();
  if (t >= times.back()) return poses.back();
  const size_t j = static_cast<size_t>(
      std::upper_bound(times.begin(), times.end(), t) - times.begin());
  const double dt = times[j] - times[j - 1];
  return Interpolate(poses[j - 1], poses[j],
                     dt > 0.0 ? (t - times[j - 1]) / dt : 0.0);
}

std::vector<Pose> InterpolatePoses(const std::vector<double>& times,
                                   const std::vector<Pose>& poses,
                                   const std::vector<double>& query) {
  std::vector<Pose> out;
  out.reserve(query.size());
  for (double t : query) out.push_back(InterpolatePose(times, poses, t));
  return out;
}

}  // namespace gr
