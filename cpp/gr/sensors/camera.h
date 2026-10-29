// Camera models: pinhole (radial-tangential), Kannala-Brandt fisheye (OpenCV
// fisheye / equidistant polynomial) and Double Sphere. The camera frame
// follows the OpenCV convention: x right, y down, z along the optical axis.

#ifndef GARAGE_RECON_GR_SENSORS_CAMERA_H_
#define GARAGE_RECON_GR_SENSORS_CAMERA_H_

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gr/core/types.h"

namespace gr {

class CameraModel {
 public:
  CameraModel(int width, int height) : width_(width), height_(height) {}
  virtual ~CameraModel() = default;

  // Held through shared pointers; not copyable.
  CameraModel(const CameraModel&) = delete;
  CameraModel& operator=(const CameraModel&) = delete;

  // Pixel of a point given in the camera frame, or nullopt outside the
  // model's valid field of view (image bounds are not checked).
  virtual std::optional<Vec2> Project(const Vec3& point_camera) const = 0;
  // Unit ray in the camera frame through a pixel, or nullopt if invalid.
  virtual std::optional<Vec3> Unproject(const Vec2& uv) const = 0;
  // The model name MakeCamera takes.
  virtual std::string type() const = 0;
  // The parameters in MakeCamera's order.
  virtual std::vector<double> params() const = 0;
  // Approximate focal length in pixels (used for adaptive splat radii).
  virtual double focal() const = 0;

  int width() const { return width_; }
  int height() const { return height_; }
  // Whether a pixel lies inside the image.
  bool InImage(const Vec2& uv) const {
    return uv.x() >= 0.0 && uv.y() >= 0.0 && uv.x() < width_ &&
           uv.y() < height_;
  }

 private:
  int width_;
  int height_;
};

// params: fx, fy, cx, cy[, k1, k2, p1, p2]
class PinholeCamera : public CameraModel {
 public:
  static constexpr int kMinParams = 4;

  PinholeCamera(int width, int height, const std::vector<double>& params);

  std::optional<Vec2> Project(const Vec3& point_camera) const override;
  std::optional<Vec3> Unproject(const Vec2& uv) const override;
  std::string type() const override { return "pinhole"; }
  std::vector<double> params() const override {
    return {fx_, fy_, cx_, cy_, k1_, k2_, p1_, p2_};
  }
  double focal() const override { return 0.5 * (fx_ + fy_); }

 private:
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  // Radial and tangential distortion.
  double k1_ = 0.0;
  double k2_ = 0.0;
  double p1_ = 0.0;
  double p2_ = 0.0;
};

// params: fx, fy, cx, cy, k1, k2, k3, k4[, max_theta_rad]
class KannalaBrandtCamera : public CameraModel {
 public:
  static constexpr int kMinParams = 4;
  // The polynomial coefficients k1..k4.
  static constexpr int kDistortionTerms = 4;
  // Rays further off-axis than this (rad, ~100 deg) do not project unless
  // the parameters say otherwise.
  static constexpr double kDefaultMaxTheta = 1.75;

  KannalaBrandtCamera(int width, int height, const std::vector<double>& params);

  std::optional<Vec2> Project(const Vec3& point_camera) const override;
  std::optional<Vec3> Unproject(const Vec2& uv) const override;
  std::string type() const override { return "kannala_brandt"; }
  std::vector<double> params() const override;
  double focal() const override { return 0.5 * (fx_ + fy_); }

 private:
  // theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
  double Distort(double theta) const;

  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  std::array<double, kDistortionTerms> k_{};
  double max_theta_ = kDefaultMaxTheta;  // rad off-axis.
};

// params: fx, fy, cx, cy, xi, alpha (Usenko et al., 3DV 2018)
class DoubleSphereCamera : public CameraModel {
 public:
  static constexpr int kMinParams = 6;

  DoubleSphereCamera(int width, int height, const std::vector<double>& params);

  std::optional<Vec2> Project(const Vec3& point_camera) const override;
  std::optional<Vec3> Unproject(const Vec2& uv) const override;
  std::string type() const override { return "double_sphere"; }
  std::vector<double> params() const override {
    return {fx_, fy_, cx_, cy_, xi_, alpha_};
  }
  double focal() const override { return 0.5 * (fx_ + fy_); }

 private:
  double fx_ = 0.0;
  double fy_ = 0.0;
  double cx_ = 0.0;
  double cy_ = 0.0;
  double xi_ = 0.0;     // Offset between the two unit spheres.
  double alpha_ = 0.0;  // Weight of the second sphere against the plane.
};

// Camera of the given model ("pinhole", "kannala_brandt" / "fisheye",
// "double_sphere"); nullptr for an unknown model or too few parameters.
std::shared_ptr<CameraModel> MakeCamera(const std::string& type, int width,
                                        int height,
                                        const std::vector<double>& params);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SENSORS_CAMERA_H_
