// Rigid transforms: SO(3) helpers, the Pose type and pose sequences.
//
// Transform names follow "a_from_b": world_from_lidar maps LiDAR-frame
// coordinates into the world frame. Optimisation code perturbs poses on
// R^3 x SO(3): t <- t + dt, R <- R * Exp(dphi) (right) or Exp(dphi) * R (left).

#ifndef GARAGE_RECON_GR_CORE_POSE_H_
#define GARAGE_RECON_GR_CORE_POSE_H_

#include <cmath>
#include <vector>

#include "gr/core/types.h"

namespace gr {

// Skew-symmetric matrix of v, so that Hat(v) * x == v.cross(x).
inline Mat3 Hat(const Vec3& v) {
  Mat3 m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

// Exponential map of SO(3).
inline Mat3 So3Exp(const Vec3& omega) {
  const double angle = omega.norm();
  if (angle < 1e-12) return Mat3::Identity() + Hat(omega);
  return Eigen::AngleAxisd(angle, omega / angle).toRotationMatrix();
}

// Logarithm map of SO(3); the angle of the result is in [0, pi].
Vec3 So3Log(const Mat3& rotation);
// Right Jacobian of SO(3): Exp(phi + d) ~= Exp(phi) Exp(Jr(phi) d).
Mat3 So3RightJacobian(const Vec3& phi);
// Inverse of the right Jacobian of SO(3).
Mat3 So3RightJacobianInverse(const Vec3& phi);

// Rotation angle (rad, in [0, pi]).
inline double RotationAngle(const Mat3& rotation) {
  return So3Log(rotation).norm();
}
// Yaw of the rotated x axis.
inline double YawOf(const Mat3& rotation) {
  return std::atan2(rotation(1, 0), rotation(0, 0));
}
// Wraps an angle to [-pi, pi).
inline double WrapAngle(double angle) {
  return angle - 2.0 * kPi * std::floor((angle + kPi) / (2.0 * kPi));
}

// Rotation plus translation. A plain value type: all fields are public.
struct Pose {
  Mat3 rotation = Mat3::Identity();
  Vec3 translation = Vec3::Zero();

  Pose() = default;
  Pose(const Mat3& rotation, const Vec3& translation)
      : rotation(rotation), translation(translation) {}

  static Pose FromMatrix(const Mat4& matrix) {
    return Pose(matrix.block<3, 3>(0, 0), matrix.block<3, 1>(0, 3));
  }
  Mat4 Matrix() const;
  Pose Inverse() const {
    return Pose(rotation.transpose(), -rotation.transpose() * translation);
  }
  Pose operator*(const Pose& other) const {
    return Pose(rotation * other.rotation,
                rotation * other.translation + translation);
  }
  Vec3 operator*(const Vec3& point) const {
    return rotation * point + translation;
  }
  // Re-orthonormalises the rotation.
  void Normalize();
};

// Slerp on the rotation and linear interpolation of the translation.
Pose Interpolate(const Pose& a, const Pose& b, double fraction);

Points TransformPoints(const Points& points, const Pose& pose);

// Information matrix of a pose measurement, order [t, phi]: isotropic
// translation sigma (m) and rotation sigma (deg).
Mat6 PoseInformation(double sigma_translation, double sigma_rotation_deg);

// Pose of a time-ordered sequence at time `t`, clamped to the ends.
Pose InterpolatePose(const std::vector<double>& times,
                     const std::vector<Pose>& poses, double t);
std::vector<Pose> InterpolatePoses(const std::vector<double>& times,
                                   const std::vector<Pose>& poses,
                                   const std::vector<double>& query);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_POSE_H_
