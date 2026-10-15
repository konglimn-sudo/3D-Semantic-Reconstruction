#include "gr/camera.h"

#include <cmath>

#include "gr/check.h"

namespace gr {
namespace {

double ParamOr(const std::vector<double>& params, size_t i, double fallback) {
  return i < params.size() ? params[i] : fallback;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pinhole + radial-tangential distortion
// ---------------------------------------------------------------------------
PinholeCamera::PinholeCamera(int width, int height,
                             const std::vector<double>& params)
    : CameraModel(width, height) {
  GR_CHECK(params.size() >= kMinParams);
  fx_ = params[0];
  fy_ = params[1];
  cx_ = params[2];
  cy_ = params[3];
  k1_ = ParamOr(params, 4, 0.0);
  k2_ = ParamOr(params, 5, 0.0);
  p1_ = ParamOr(params, 6, 0.0);
  p2_ = ParamOr(params, 7, 0.0);
}

std::optional<Vec2> PinholeCamera::Project(const Vec3& point_camera) const {
  if (point_camera.z() <= 1e-6) return std::nullopt;
  const double x = point_camera.x() / point_camera.z();
  const double y = point_camera.y() / point_camera.z();
  const double r2 = x * x + y * y;
  const double radial = 1.0 + k1_ * r2 + k2_ * r2 * r2;
  const double xd = x * radial + 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
  const double yd = y * radial + p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;
  return Vec2(fx_ * xd + cx_, fy_ * yd + cy_);
}

std::optional<Vec3> PinholeCamera::Unproject(const Vec2& uv) const {
  const double xd = (uv.x() - cx_) / fx_;
  const double yd = (uv.y() - cy_) / fy_;
  double x = xd;
  double y = yd;
  for (int iteration = 0; iteration < 20; ++iteration) {
    const double r2 = x * x + y * y;
    const double radial = 1.0 + k1_ * r2 + k2_ * r2 * r2;
    const double dx = 2.0 * p1_ * x * y + p2_ * (r2 + 2.0 * x * x);
    const double dy = p1_ * (r2 + 2.0 * y * y) + 2.0 * p2_ * x * y;
    x = (xd - dx) / radial;
    y = (yd - dy) / radial;
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
  fx_ = params[0];
  fy_ = params[1];
  cx_ = params[2];
  cy_ = params[3];
  for (int i = 0; i < 4; ++i) {
    k_[i] = ParamOr(params, 4 + static_cast<size_t>(i), 0.0);
  }
  max_theta_ = ParamOr(params, 8, 1.75);
}

double KannalaBrandtCamera::Distort(double theta) const {
  const double t2 = theta * theta;
  return theta *
         (1.0 + t2 * (k_[0] + t2 * (k_[1] + t2 * (k_[2] + t2 * k_[3]))));
}

std::optional<Vec2> KannalaBrandtCamera::Project(
    const Vec3& point_camera) const {
  const double r = std::sqrt(point_camera.x() * point_camera.x() +
                             point_camera.y() * point_camera.y());
  const double theta = std::atan2(r, point_camera.z());
  if (theta > max_theta_) return std::nullopt;
  if (r < 1e-12) {
    if (point_camera.z() <= 0.0) return std::nullopt;
    return Vec2(cx_, cy_);
  }
  const double theta_d = Distort(theta);
  return Vec2(fx_ * theta_d * point_camera.x() / r + cx_,
              fy_ * theta_d * point_camera.y() / r + cy_);
}

std::optional<Vec3> KannalaBrandtCamera::Unproject(const Vec2& uv) const {
  const double mx = (uv.x() - cx_) / fx_;
  const double my = (uv.y() - cy_) / fy_;
  const double theta_d = std::sqrt(mx * mx + my * my);
  if (theta_d < 1e-12) return Vec3(0.0, 0.0, 1.0);
  // Newton iterations on Distort(theta) = theta_d.
  double theta = theta_d;
  for (int iteration = 0; iteration < 15; ++iteration) {
    const double t2 = theta * theta;
    const double f = Distort(theta) - theta_d;
    const double df =
        1.0 + t2 * (3.0 * k_[0] +
                    t2 * (5.0 * k_[1] + t2 * (7.0 * k_[2] + t2 * 9.0 * k_[3])));
    if (std::abs(df) < 1e-12) break;
    theta -= f / df;
  }
  if (theta < 0.0 || theta > max_theta_) return std::nullopt;
  const double s = std::sin(theta) / theta_d;
  return Vec3(mx * s, my * s, std::cos(theta));
}

// ---------------------------------------------------------------------------
// Double Sphere
// ---------------------------------------------------------------------------
DoubleSphereCamera::DoubleSphereCamera(int width, int height,
                                       const std::vector<double>& params)
    : CameraModel(width, height) {
  GR_CHECK(params.size() >= kMinParams);
  fx_ = params[0];
  fy_ = params[1];
  cx_ = params[2];
  cy_ = params[3];
  xi_ = params[4];
  alpha_ = params[5];
}

std::optional<Vec2> DoubleSphereCamera::Project(
    const Vec3& point_camera) const {
  const double x = point_camera.x();
  const double y = point_camera.y();
  const double z = point_camera.z();
  const double d1 = std::sqrt(x * x + y * y + z * z);
  const double w1 =
      alpha_ <= 0.5 ? alpha_ / (1.0 - alpha_) : (1.0 - alpha_) / alpha_;
  const double w2 = (w1 + xi_) / std::sqrt(2.0 * w1 * xi_ + xi_ * xi_ + 1.0);
  if (z <= -w2 * d1) return std::nullopt;
  const double zz = xi_ * d1 + z;
  const double d2 = std::sqrt(x * x + y * y + zz * zz);
  const double denominator = alpha_ * d2 + (1.0 - alpha_) * zz;
  if (denominator < 1e-12) return std::nullopt;
  return Vec2(fx_ * x / denominator + cx_, fy_ * y / denominator + cy_);
}

std::optional<Vec3> DoubleSphereCamera::Unproject(const Vec2& uv) const {
  const double mx = (uv.x() - cx_) / fx_;
  const double my = (uv.y() - cy_) / fy_;
  const double r2 = mx * mx + my * my;
  if (alpha_ > 0.5 && r2 > 1.0 / (2.0 * alpha_ - 1.0)) return std::nullopt;
  const double mz =
      (1.0 - alpha_ * alpha_ * r2) /
      (alpha_ * std::sqrt(1.0 - (2.0 * alpha_ - 1.0) * r2) + 1.0 - alpha_);
  const double k =
      (mz * xi_ + std::sqrt(mz * mz + (1.0 - xi_ * xi_) * r2)) / (mz * mz + r2);
  return Vec3(k * mx, k * my, k * mz - xi_).normalized();
}

std::shared_ptr<CameraModel> MakeCamera(const std::string& type, int width,
                                        int height,
                                        const std::vector<double>& params) {
  const int n = static_cast<int>(params.size());
  if (type == "pinhole") {
    if (n < PinholeCamera::kMinParams) return nullptr;
    return std::make_shared<PinholeCamera>(width, height, params);
  }
  if (type == "kannala_brandt" || type == "kb" || type == "fisheye" ||
      type == "opencv_fisheye") {
    if (n < KannalaBrandtCamera::kMinParams) return nullptr;
    return std::make_shared<KannalaBrandtCamera>(width, height, params);
  }
  if (type == "double_sphere" || type == "ds") {
    if (n < DoubleSphereCamera::kMinParams) return nullptr;
    return std::make_shared<DoubleSphereCamera>(width, height, params);
  }
  return nullptr;
}

}  // namespace gr
