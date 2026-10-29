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

// Layout of a pose perturbation [t, phi] (and of the matrices over it): the
// translation first, then the rotation vector.
inline constexpr int kTranslationOffset = 0;
inline constexpr int kRotationOffset = kSpaceDim;

// Skew-symmetric matrix of `vec`, so that Hat(vec) * x == vec.cross(x).
inline Mat3 Hat(const Vec3& vec) {
  Mat3 skew;
  skew << 0.0, -vec.z(), vec.y(), vec.z(), 0.0, -vec.x(), -vec.y(), vec.x(),
      0.0;
  return skew;
}

// Exponential map of SO(3).
inline Mat3 So3Exp(const Vec3& omega) {
  // Below this angle I + Hat(omega) is exact to double precision.
  constexpr double kFirstOrderAngle = 1e-12;
  const double angle = omega.norm();
  if (angle < kFirstOrderAngle) return Mat3::Identity() + Hat(omega);
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

  // The pose of a homogeneous 4 x 4 matrix.
  static Pose FromMatrix(const Mat4& matrix) {
    return Pose(matrix.topLeftCorner<kSpaceDim, kSpaceDim>(),
                matrix.topRightCorner<kSpaceDim, 1>());
  }
  // The homogeneous 4 x 4 matrix of the pose.
  Mat4 Matrix() const;
  Pose Inverse() const {
    return Pose(rotation.transpose(), -rotation.transpose() * translation);
  }
  // Composition: (a * b) * x == a * (b * x).
  Pose operator*(const Pose& other) const {
    return Pose(rotation * other.rotation,
                rotation * other.translation + translation);
  }
  // The pose applied to a point.
  Vec3 operator*(const Vec3& point) const {
    return rotation * point + translation;
  }
  // Re-orthonormalises the rotation.
  void Normalize();
};

// Slerp on the rotation and linear interpolation of the translation, from
// `start` (fraction 0) to `end` (fraction 1).
Pose Interpolate(const Pose& start, const Pose& end, double fraction);

// Every point moved by `pose`.
Points TransformPoints(const Points& points, const Pose& pose);

// Information matrix of a pose measurement, order [t, phi]: isotropic
// translation sigma (m) and rotation sigma (deg).
Mat6 PoseInformation(double sigma_translation, double sigma_rotation_deg);

// Pose of a time-ordered sequence at `time`, clamped to the ends.
Pose InterpolatePose(const std::vector<double>& times,
                     const std::vector<Pose>& poses, double time);
// InterpolatePose at each of `query_times`.
std::vector<Pose> InterpolatePoses(const std::vector<double>& times,
                                   const std::vector<Pose>& poses,
                                   const std::vector<double>& query_times);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_POSE_H_
