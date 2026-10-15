#include "gr/camera_calibration.h"

#include <cmath>
#include <cstdio>
#include <optional>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// S-shaped drive at 2 m/s, knots every 50 ms.
ContinuousTrajectory SDrive() {
  std::vector<double> stamps;
  std::vector<Pose> poses;
  Vec3 position(2.0, 0.0, 0.0);
  const double dt = 0.001;
  for (int i = 0; i <= 12000; ++i) {
    const double t = i * dt;
    const double yaw = 0.5 * std::sin(0.6 * t);
    if (i % 50 == 0) {
      stamps.push_back(t);
      poses.emplace_back(So3Exp(Vec3(0.0, 0.0, yaw)), position);
    }
    position += 2.0 * dt * Vec3(std::cos(yaw), std::sin(yaw), 0.0);
  }
  return ContinuousTrajectory(stamps, poses);
}

// LiDAR surface: walls at x = 34 and y = +-9, floor z = 0, ceiling z = 3.5.
void SurfaceGrid(Points& points, Points& normals) {
  for (double a = -5.0; a <= 34.0; a += 0.1) {
    for (double b = -9.0; b <= 9.0; b += 0.1) {
      points.emplace_back(a, b, 0.0);
      normals.push_back(Vec3::UnitZ());
      points.emplace_back(a, b, 3.5);
      normals.push_back(-Vec3::UnitZ());
    }
    for (double z = 0.0; z <= 3.5; z += 0.1) {
      points.emplace_back(a, 9.0, z);
      normals.push_back(-Vec3::UnitY());
      points.emplace_back(a, -9.0, z);
      normals.push_back(Vec3::UnitY());
    }
  }
  for (double b = -9.0; b <= 9.0; b += 0.1) {
    for (double z = 0.0; z <= 3.5; z += 0.1) {
      points.emplace_back(34.0, b, z);
      normals.push_back(-Vec3::UnitX());
    }
  }
}

GR_TEST(CameraCalibrationTest, RecoversTimeOffsetAndExtrinsic) {
  const ContinuousTrajectory trajectory = SDrive();
  const auto camera = MakeCamera("pinhole", 640, 480, {380, 380, 320, 240});
  GR_ASSERT_TRUE(camera != nullptr);
  // Camera looking forward-left, slightly down (OpenCV axes).
  Mat3 vehicle_from_optical;
  vehicle_from_optical << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  CameraCalibration truth;
  truth.vehicle_from_camera = Pose(
      So3Exp(Vec3(0.0, 0.1, 0.5)) * vehicle_from_optical, Vec3(1.5, 0.3, 1.3));
  truth.time_offset = 0.015;
  truth.line_time = 30e-6;
  CameraCalibration nominal = truth;
  nominal.time_offset = 0.0;
  nominal.vehicle_from_camera =
      truth.vehicle_from_camera *
      Pose(So3Exp(Vec3(DegToRad(0.3), -DegToRad(0.25), DegToRad(0.2))),
           Vec3(0.02, -0.015, 0.01));

  std::mt19937 rng(5);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> pixel_noise(0.0, 0.3);
  // Landmarks on the walls, floor and ceiling.
  Points landmarks;
  for (int k = 0; k < 600; ++k) {
    const double u = uniform(rng);
    const double a = -5.0 + 39.0 * uniform(rng);
    const double b = -9.0 + 18.0 * uniform(rng);
    const double z = 3.5 * uniform(rng);
    if (u < 0.3) {
      landmarks.emplace_back(a, 9.0, z);
    } else if (u < 0.6) {
      landmarks.emplace_back(a, -9.0, z);
    } else if (u < 0.75) {
      landmarks.emplace_back(34.0, b, z);
    } else if (u < 0.9) {
      landmarks.emplace_back(a, b, 0.0);
    } else {
      landmarks.emplace_back(a, b, 3.5);
    }
  }
  std::vector<CalibrationImage> images;
  for (double stamp = 0.5; stamp < 11.5; stamp += 0.25) {
    images.push_back({&trajectory, stamp});
  }
  std::vector<Track> tracks(landmarks.size());
  for (size_t n = 0; n < images.size(); ++n) {
    for (size_t k = 0; k < landmarks.size(); ++k) {
      const std::optional<Vec2> uv =
          ProjectWithCalibration(*camera, truth, images[n], landmarks[k]);
      if (!uv || uv->x() < 2 || uv->y() < 2 || uv->x() > 637 || uv->y() > 477) {
        continue;
      }
      tracks[k].push_back({static_cast<int>(n),
                           *uv + Vec2(pixel_noise(rng), pixel_noise(rng))});
    }
  }
  Points surface_points;
  Points normals;
  SurfaceGrid(surface_points, normals);
  const PointCloudIndex surface(surface_points);

  const CameraCalibrationResult result =
      CalibrateCamera(*camera, nominal, images, tracks, &surface, &normals,
                      CameraCalibrationConfig());
  const Pose error = truth.vehicle_from_camera.Inverse() *
                     result.calibration.vehicle_from_camera;
  std::printf(
      "       calibration: %d landmarks, rms %.2f -> %.2f px, time offset "
      "error %.2f ms (sigma %.2f), rotation error %.4f deg, translation "
      "error %.1f mm\n",
      result.landmarks, result.rms_before, result.rms_after,
      (result.calibration.time_offset - truth.time_offset) * 1e3,
      result.time_offset_sigma * 1e3, RadToDeg(RotationAngle(error.rotation)),
      error.translation.norm() * 1e3);
  GR_EXPECT_GT(result.landmarks, 200);
  GR_EXPECT_GT(result.rms_before, 1.0);
  GR_EXPECT_LT(result.rms_after, 0.5);
  GR_EXPECT_NEAR(result.calibration.time_offset, truth.time_offset, 1.5e-3);
  GR_EXPECT_LT(RadToDeg(RotationAngle(error.rotation)), 0.05);
  GR_EXPECT_LT(error.translation.norm(), 0.01);
  GR_EXPECT_GT(result.time_offset_sigma, 0.0);
}

}  // namespace
}  // namespace gr
