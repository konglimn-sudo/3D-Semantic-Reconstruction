#include "gr/calibration/camera_calibration.h"

#include <cmath>
#include <cstdio>
#include <optional>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// The hall the camera drives through (m): x from -5 to the end wall at 34,
// side walls at y = +-9, floor at z = 0, ceiling at z = 3.5.
constexpr double kHallStartX = -5.0;
constexpr double kEndWallX = 34.0;
constexpr double kSideWallY = 9.0;
constexpr double kCeilingZ = 3.5;

// S-shaped drive at 2 m/s for 12 s, knots every 50 ms.
ContinuousTrajectory SDrive() {
  constexpr int kSteps = 12000;
  constexpr double kStep = 0.001;  // s
  constexpr int kStepsPerKnot = 50;
  constexpr double kSpeed = 2.0;  // m/s
  std::vector<double> stamps;
  std::vector<Pose> poses;
  Vec3 position(2.0, 0.0, 0.0);
  for (int i = 0; i <= kSteps; ++i) {
    const double time = i * kStep;
    const double yaw = 0.5 * std::sin(0.6 * time);
    if (i % kStepsPerKnot == 0) {
      stamps.push_back(time);
      poses.emplace_back(So3Exp(Vec3(0.0, 0.0, yaw)), position);
    }
    position += kSpeed * kStep * Vec3(std::cos(yaw), std::sin(yaw), 0.0);
  }
  return ContinuousTrajectory(stamps, poses);
}

// LiDAR surface of the hall, every 10 cm, with its inward normals.
void SurfaceGrid(Points* points, Points* normals) {
  constexpr double kSpacing = 0.1;  // m
  for (double x = kHallStartX; x <= kEndWallX; x += kSpacing) {
    for (double y = -kSideWallY; y <= kSideWallY; y += kSpacing) {
      points->emplace_back(x, y, 0.0);
      normals->push_back(Vec3::UnitZ());
      points->emplace_back(x, y, kCeilingZ);
      normals->push_back(-Vec3::UnitZ());
    }
    for (double z = 0.0; z <= kCeilingZ; z += kSpacing) {
      points->emplace_back(x, kSideWallY, z);
      normals->push_back(-Vec3::UnitY());
      points->emplace_back(x, -kSideWallY, z);
      normals->push_back(Vec3::UnitY());
    }
  }
  for (double y = -kSideWallY; y <= kSideWallY; y += kSpacing) {
    for (double z = 0.0; z <= kCeilingZ; z += kSpacing) {
      points->emplace_back(kEndWallX, y, z);
      normals->push_back(-Vec3::UnitX());
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

  constexpr int kSeed = 5;
  constexpr double kPixelNoise = 0.3;  // px
  constexpr int kLandmarks = 600;
  // Shares of the landmarks on the left wall, the right wall, the end wall,
  // the floor and (the rest) the ceiling, as cumulative bounds.
  constexpr double kLeftWallShare = 0.3;
  constexpr double kRightWallShare = 0.6;
  constexpr double kEndWallShare = 0.75;
  constexpr double kFloorShare = 0.9;
  // An image every 0.25 s from 0.5 s to 11.5 s.
  constexpr double kFirstImage = 0.5;  // s
  constexpr double kLastImage = 11.5;
  constexpr double kImagePeriod = 0.25;
  // Observations keep 2 px from the image border.
  constexpr double kBorder = 2.0;  // px
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::normal_distribution<double> pixel_noise(0.0, kPixelNoise);
  Points landmarks;
  for (int k = 0; k < kLandmarks; ++k) {
    const double choice = uniform(rng);
    const double x = kHallStartX + (kEndWallX - kHallStartX) * uniform(rng);
    const double y = -kSideWallY + 2.0 * kSideWallY * uniform(rng);
    const double z = kCeilingZ * uniform(rng);
    if (choice < kLeftWallShare) {
      landmarks.emplace_back(x, kSideWallY, z);
    } else if (choice < kRightWallShare) {
      landmarks.emplace_back(x, -kSideWallY, z);
    } else if (choice < kEndWallShare) {
      landmarks.emplace_back(kEndWallX, y, z);
    } else if (choice < kFloorShare) {
      landmarks.emplace_back(x, y, 0.0);
    } else {
      landmarks.emplace_back(x, y, kCeilingZ);
    }
  }
  std::vector<CalibrationImage> images;
  for (double stamp = kFirstImage; stamp < kLastImage; stamp += kImagePeriod) {
    images.push_back({&trajectory, stamp});
  }
  std::vector<Track> tracks(landmarks.size());
  const double max_u = camera->width() - 1 - kBorder;
  const double max_v = camera->height() - 1 - kBorder;
  for (size_t image = 0; image < images.size(); ++image) {
    for (size_t k = 0; k < landmarks.size(); ++k) {
      const std::optional<Vec2> uv =
          ProjectWithCalibration(*camera, truth, images[image], landmarks[k]);
      if (!uv || uv->x() < kBorder || uv->y() < kBorder || uv->x() > max_u ||
          uv->y() > max_v) {
        continue;
      }
      tracks[k].push_back({static_cast<int>(image),
                           *uv + Vec2(pixel_noise(rng), pixel_noise(rng))});
    }
  }
  Points surface_points;
  Points normals;
  SurfaceGrid(&surface_points, &normals);
  const PointCloudIndex surface(surface_points);

  const CameraCalibrationResult result =
      CalibrateCamera(*camera, nominal, images, tracks, &surface, &normals,
                      CameraCalibrationConfig());
  const Pose error = truth.vehicle_from_camera.Inverse() *
                     result.calibration.vehicle_from_camera;
  constexpr double kMilli = 1e3;  // ms per s, mm per m.
  constexpr int kMinLandmarks = 200;
  constexpr double kMinRmsBefore = 1.0;            // px
  constexpr double kMaxRmsAfter = 0.5;             // px
  constexpr double kTimeOffsetTolerance = 1.5e-3;  // s
  constexpr double kMaxRotationError = 0.05;       // deg
  constexpr double kMaxTranslationError = 0.01;    // m
  std::printf(
      "       calibration: %d landmarks, rms %.2f -> %.2f px, time offset "
      "error %.2f ms (sigma %.2f), rotation error %.4f deg, translation "
      "error %.1f mm\n",
      result.landmarks, result.rms_before, result.rms_after,
      (result.calibration.time_offset - truth.time_offset) * kMilli,
      result.time_offset_sigma * kMilli,
      RadToDeg(RotationAngle(error.rotation)),
      error.translation.norm() * kMilli);
  GR_EXPECT_GT(result.landmarks, kMinLandmarks);
  GR_EXPECT_GT(result.rms_before, kMinRmsBefore);
  GR_EXPECT_LT(result.rms_after, kMaxRmsAfter);
  GR_EXPECT_NEAR(result.calibration.time_offset, truth.time_offset,
                 kTimeOffsetTolerance);
  GR_EXPECT_LT(RadToDeg(RotationAngle(error.rotation)), kMaxRotationError);
  GR_EXPECT_LT(error.translation.norm(), kMaxTranslationError);
  GR_EXPECT_GT(result.time_offset_sigma, 0.0);
}

}  // namespace
}  // namespace gr
