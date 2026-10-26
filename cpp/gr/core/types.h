// Eigen types and point containers shared by the library.

#ifndef GARAGE_RECON_GR_CORE_TYPES_H_
#define GARAGE_RECON_GR_CORE_TYPES_H_

#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace gr {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using RowMatX3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using RowMatX2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using Points = std::vector<Vec3>;

inline constexpr double kPi = 3.14159265358979323846;

inline double DegToRad(double degrees) { return degrees * kPi / 180.0; }
inline double RadToDeg(double radians) { return radians * 180.0 / kPi; }

// Conversions between an (N, 3) row-major matrix and a point vector.
Points ToPoints(const Eigen::Ref<const RowMatX3d>& matrix);
RowMatX3d ToMatrix(const Points& points);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_TYPES_H_
