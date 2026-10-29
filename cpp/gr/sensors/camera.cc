#include "gr/sensors/camera.h"

#include <cmath>

#include "gr/core/check.h"

namespace gr {
namespace {

// Positions in the parameter vectors: the intrinsics every model starts
// with, then each model's own.
constexpr size_t kFxParam = 0;
constexpr size_t kFyParam = 1;
constexpr size_t kCxParam = 2;
constexpr size_t kCyParam = 3;
constexpr size_t kPinholeK1Param = 4;
constexpr size_t kPinholeK2Param = 5;
constexpr size_t kPinholeP1Param = 6;
constexpr size_t kPinholeP2Param = 7;
constexpr size_t kKannalaBrandtFirstKParam = 4;
constexpr size_t kKannalaBrandtMaxThetaParam = 8;
constexpr size_t kDoubleSphereXiParam = 4;
constexpr size_t kDoubleSphereAlphaParam = 5;

// The pinhole model projects points at least this far (m) in front of the
// camera.
constexpr double kMinPinholeDepth = 1e-6;
// Fixed-point iterations that undo the radial-tangential distortion.
constexpr int kUndistortIterations = 20;
// Newton iterations that invert the Kannala-Brandt polynomial.
constexpr int kNewtonIterations = 15;
// Below this, radii, slopes and denominators count as zero.
constexpr double kNearZero = 1e-12;

double ParamOr(const std::vector<double>& params, size_t index,
               double fallback) {
  return index < params.size() ? params[index] : fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pinhole + radial-tangential distortion
// ---------------------------------------------------------------------------
PinholeCamera::PinholeCamera(int width, int height,
                             const std::vector<double>& params)
    : CameraModel(width, height) {
  GR_CHECK(params.size() >= kMinParams);
  fx_ = params[kFxParam];
  fy_ = params[kFyParam];
  cx_ = params[kCxParam];
  cy_ = params[kCyParam];
  k1_ = ParamOr(params, kPinholeK1Param, 0.0);
  k2_ = ParamOr(params, kPinholeK2Param, 0.0);
  p1_ = ParamOr(params, kPinholeP1Param, 0.0);
  p2_ = ParamOr(params, kPinholeP2Param, 0.0);
}

std::optional<Vec2> PinholeCamera::Project(const Vec3& point_camera) const {
  if (point_camera.z() <= kMinPinholeDepth) return std::nullopt;
  const double x = point_camera.x() / point_camera.z();
  const double y = point_camera.y() / point_camera.z();
  const double squared_radius = x * x + y * y;
  const double radial =
      1.0 + k1_ * squared_radius + k2_ * squared_radius * squared_radius;
  const double distorted_x =
      x * radial + 2.0 * p1_ * x * y + p2_ * (squared_radius + 2.0 * x * x);
  const double distorted_y =
      y * radial + p1_ * (squared_radius + 2.0 * y * y) + 2.0 * p2_ * x * y;
  return Vec2(fx_ * distorted_x + cx_, fy_ * distorted_y + cy_);
}

std::optional<Vec3> PinholeCamera::Unproject(const Vec2& uv) const {
  const double distorted_x = (uv.x() - cx_) / fx_;
  const double distorted_y = (uv.y() - cy_) / fy_;
  double x = distorted_x;
  double y = distorted_y;
  for (int iteration = 0; iteration < kUndistortIterations; ++iteration) {
    const double squared_radius = x * x + y * y;
    const double radial =
        1.0 + k1_ * squared_radius + k2_ * squared_radius * squared_radius;
    const double tangential_x =
        2.0 * p1_ * x * y + p2_ * (squared_radius + 2.0 * x * x);
    const double tangential_y =
        p1_ * (squared_radius + 2.0 * y * y) + 2.0 * p2_ * x * y;
    x = (distorted_x - tangential_x) / radial;
    y = (distorted_y - tangential_y) / radial;
  }
  return Vec3(x, y, 1.0).normalized();
}

// ---------------------------------------------------------------------------
// Kannala-Brandt (equidistant polynomial)
// ---------------------------------------------------------------------------
KannalaBrandtCamera::KannalaBrandtCamera(int width, int height,
                                         const std::vector<double>& params)
    : CameraModel(width, height) {
  GR_CHECK(params.size() >= kMinParams);
  fx_ = params[kFxParam];
  fy_ = params[kFyParam];
  cx_ = params[kCxParam];
  cy_ = params[kCyParam];
  for (int term = 0; term < kDistortionTerms; ++term) {
    k_[term] = ParamOr(
        params, kKannalaBrandtFirstKParam + static_cast<size_t>(term), 0.0);
  }
  max_theta_ = ParamOr(params, kKannalaBrandtMaxThetaParam, kDefaultMaxTheta);
}

std::vector<double> KannalaBrandtCamera::params() const {
  std::vector<double> out = {fx_, fy_, cx_, cy_};
  out.insert(out.end(), k_.begin(), k_.end());
  out.push_back(max_theta_);
  return out;
}

double KannalaBrandtCamera::Distort(double theta) const {
  // Horner's scheme: theta * (1 + theta^2 (k1 + theta^2 (k2 + ...))).
  const double theta_squared = theta * theta;
  double polynomial = k_.back();
  for (int term = kDistortionTerms - 2; term >= 0; --term) {
    polynomial = k_[term] + theta_squared * polynomial;
  }
  return theta * (1.0 + theta_squared * polynomial);
}

std::optional<Vec2> KannalaBrandtCamera::Project(
    const Vec3& point_camera) const {
  const double off_axis = std::sqrt(point_camera.x() * point_camera.x() +
                                    point_camera.y() * point_camera.y());
  const double theta = std::atan2(off_axis, point_camera.z());
  if (theta > max_theta_) return std::nullopt;
  if (off_axis < kNearZero) {  // On the optical axis.
    if (point_camera.z() <= 0.0) return std::nullopt;
    return Vec2(cx_, cy_);
  }
  const double distorted_theta = Distort(theta);
  return Vec2(fx_ * distorted_theta * point_camera.x() / off_axis + cx_,
              fy_ * distorted_theta * point_camera.y() / off_axis + cy_);
}

std::optional<Vec3> KannalaBrandtCamera::Unproject(const Vec2& uv) const {
  const double normalized_x = (uv.x() - cx_) / fx_;
  const double normalized_y = (uv.y() - cy_) / fy_;
  const double distorted_theta =
      std::sqrt(normalized_x * normalized_x + normalized_y * normalized_y);
  if (distorted_theta < kNearZero) return Vec3(0.0, 0.0, 1.0);
  // Newton iterations on Distort(theta) = distorted_theta.
  double theta = distorted_theta;
  for (int iteration = 0; iteration < kNewtonIterations; ++iteration) {
    const double theta_squared = theta * theta;
    const double error = Distort(theta) - distorted_theta;
    // d/dtheta Distort = 1 + 3 k1 theta^2 + 5 k2 theta^4 + 7 k3 theta^6
    // + 9 k4 theta^8.
    const double slope =
        1.0 +
        theta_squared *
            (3.0 * k_[0] +
             theta_squared *
                 (5.0 * k_[1] +
                  theta_squared * (7.0 * k_[2] + theta_squared * 9.0 * k_[3])));
    if (std::abs(slope) < kNearZero) break;
    theta -= error / slope;
  }
  if (theta < 0.0 || theta > max_theta_) return std::nullopt;
  const double scale = std::sin(theta) / distorted_theta;
  return Vec3(normalized_x * scale, normalized_y * scale, std::cos(theta));
}

// ---------------------------------------------------------------------------
// Double Sphere
// ---------------------------------------------------------------------------
DoubleSphereCamera::DoubleSphereCamera(int width, int height,
                                       const std::vector<double>& params)
    : CameraModel(width, height) {
  GR_CHECK(params.size() >= kMinParams);
  fx_ = params[kFxParam];
  fy_ = params[kFyParam];
  cx_ = params[kCxParam];
  cy_ = params[kCyParam];
  xi_ = params[kDoubleSphereXiParam];
  alpha_ = params[kDoubleSphereAlphaParam];
}

std::optional<Vec2> DoubleSphereCamera::Project(
    const Vec3& point_camera) const {
  const double x = point_camera.x();
  const double y = point_camera.y();
  const double z = point_camera.z();
  const double first_sphere_distance = std::sqrt(x * x + y * y + z * z);
  // The field of view: z > -fov_bound * first_sphere_distance (w1 and w2
  // of the paper).
  const double alpha_ratio =
      alpha_ <= 0.5 ? alpha_ / (1.0 - alpha_) : (1.0 - alpha_) / alpha_;
  const double fov_bound = (alpha_ratio + xi_) /
                           std::sqrt(2.0 * alpha_ratio * xi_ + xi_ * xi_ + 1.0);
  if (z <= -fov_bound * first_sphere_distance) return std::nullopt;
  const double shifted_z = xi_ * first_sphere_distance + z;
  const double second_sphere_distance =
      std::sqrt(x * x + y * y + shifted_z * shifted_z);
  const double denominator =
      alpha_ * second_sphere_distance + (1.0 - alpha_) * shifted_z;
  if (denominator < kNearZero) return std::nullopt;
  return Vec2(fx_ * x / denominator + cx_, fy_ * y / denominator + cy_);
}

std::optional<Vec3> DoubleSphereCamera::Unproject(const Vec2& uv) const {
  const double normalized_x = (uv.x() - cx_) / fx_;
  const double normalized_y = (uv.y() - cy_) / fy_;
  const double squared_radius =
      normalized_x * normalized_x + normalized_y * normalized_y;
  if (alpha_ > 0.5 && squared_radius > 1.0 / (2.0 * alpha_ - 1.0)) {
    return std::nullopt;
  }
  const double normalized_z =
      (1.0 - alpha_ * alpha_ * squared_radius) /
      (alpha_ * std::sqrt(1.0 - (2.0 * alpha_ - 1.0) * squared_radius) + 1.0 -
       alpha_);
  const double scale =
      (normalized_z * xi_ + std::sqrt(normalized_z * normalized_z +
                                      (1.0 - xi_ * xi_) * squared_radius)) /
      (normalized_z * normalized_z + squared_radius);
  return Vec3(scale * normalized_x, scale * normalized_y,
              scale * normalized_z - xi_)
      .normalized();
}

std::shared_ptr<CameraModel> MakeCamera(const std::string& type, int width,
                                        int height,
                                        const std::vector<double>& params) {
  const int param_count = static_cast<int>(params.size());
  if (type == "pinhole") {
    if (param_count < PinholeCamera::kMinParams) return nullptr;
    return std::make_shared<PinholeCamera>(width, height, params);
  }
  if (type == "kannala_brandt" || type == "kb" || type == "fisheye" ||
      type == "opencv_fisheye") {
    if (param_count < KannalaBrandtCamera::kMinParams) return nullptr;
    return std::make_shared<KannalaBrandtCamera>(width, height, params);
  }
  if (type == "double_sphere" || type == "ds") {
    if (param_count < DoubleSphereCamera::kMinParams) return nullptr;
    return std::make_shared<DoubleSphereCamera>(width, height, params);
  }
  return nullptr;
}

}  // namespace gr
