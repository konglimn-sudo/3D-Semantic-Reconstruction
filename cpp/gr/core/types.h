// Eigen types and point containers shared by the library.

#ifndef GARAGE_RECON_GR_CORE_TYPES_H_
#define GARAGE_RECON_GR_CORE_TYPES_H_

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace gr {

// Dimensions of the plane, of space and of a pose perturbation (a
// translation and a rotation vector).
inline constexpr int kPlaneDim = 2;
inline constexpr int kSpaceDim = 3;
inline constexpr int kPoseDim = 2 * kSpaceDim;
// Rows and columns of a homogeneous transform (Mat4).
inline constexpr int kHomogeneousDim = kSpaceDim + 1;

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, kPoseDim, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat6 = Eigen::Matrix<double, kPoseDim, kPoseDim>;
using RowMatX3d =
    Eigen::Matrix<double, Eigen::Dynamic, kSpaceDim, Eigen::RowMajor>;
using RowMatX2d =
    Eigen::Matrix<double, Eigen::Dynamic, kPlaneDim, Eigen::RowMajor>;
using Points = std::vector<Vec3>;

inline constexpr double kPi = 3.14159265358979323846;
// Degrees in half a turn (pi radians).
inline constexpr double kDegreesPerHalfTurn = 180.0;
// For reports in centimetres.
inline constexpr double kCentimetresPerMetre = 100.0;

// Conversions between degrees and radians.
inline double DegToRad(double degrees) {
  return degrees * kPi / kDegreesPerHalfTurn;
}
inline double RadToDeg(double radians) {
  return radians * kDegreesPerHalfTurn / kPi;
}

// Conversions between an (N, 3) row-major matrix and a point vector.
Points ToPoints(const Eigen::Ref<const RowMatX3d>& matrix);
RowMatX3d ToMatrix(const Points& points);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_TYPES_H_
