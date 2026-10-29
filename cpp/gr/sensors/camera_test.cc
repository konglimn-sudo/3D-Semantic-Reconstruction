#include "gr/sensors/camera.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Projects random points up to `max_theta` (rad) off the optical axis and
// checks that unprojecting the pixel gives the ray back.
void ExpectRoundTrip(const CameraModel& camera, double max_theta) {
  constexpr int kSeed = 5;
  constexpr int kMaxDraws = 5000;
  constexpr int kWantedPoints = 500;
  constexpr int kMinTestedPoints = 100;
  constexpr double kPointDistance = 3.0;    // m
  constexpr double kAngleTolerance = 1e-6;  // rad
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  int tested = 0;
  for (int draw = 0; draw < kMaxDraws && tested < kWantedPoints; ++draw) {
    const Vec3 point =
        Vec3(uniform(rng), uniform(rng), uniform(rng)).normalized() *
        kPointDistance;
    if (std::acos(point.z() / point.norm()) > max_theta) continue;
    const std::optional<Vec2> uv = camera.Project(point);
    if (!uv) continue;
    const std::optional<Vec3> ray = camera.Unproject(*uv);
    GR_ASSERT_TRUE(ray.has_value());
    GR_EXPECT_NEAR(
        std::acos(std::clamp(ray->dot(point.normalized()), -1.0, 1.0)), 0.0,
        kAngleTolerance);
    ++tested;
  }
  GR_EXPECT_GE(tested, kMinTestedPoints);
}

GR_TEST(CameraTest, PinholeRoundTripWithDistortion) {
  constexpr double kMaxTheta = 0.6;  // rad
  const auto camera = MakeCamera(
      "pinhole", 640, 480, {400, 410, 320, 240, -0.1, 0.01, 0.001, -0.001});
  GR_ASSERT_TRUE(camera != nullptr);
  ExpectRoundTrip(*camera, kMaxTheta);
}

GR_TEST(CameraTest, KannalaBrandtRoundTripBeyondNinetyDegrees) {
  constexpr double kMaxTheta = 1.65;  // rad, ~95 deg.
  const auto camera =
      MakeCamera("kannala_brandt", 800, 800,
                 {250, 250, 400, 400, 0.05, -0.01, 0.002, -0.0005, 1.7});
  GR_ASSERT_TRUE(camera != nullptr);
  ExpectRoundTrip(*camera, kMaxTheta);
}

GR_TEST(CameraTest, DoubleSphereRoundTrip) {
  constexpr double kMaxTheta = 1.5;  // rad
  const auto camera =
      MakeCamera("double_sphere", 800, 800, {300, 300, 400, 400, -0.2, 0.6});
  GR_ASSERT_TRUE(camera != nullptr);
  ExpectRoundTrip(*camera, kMaxTheta);
}

GR_TEST(CameraTest, MakeCameraRejectsUnknownModelsAndMissingParameters) {
  // A 10 x 10 image with fx = fy = 1 and the principal point in the middle.
  constexpr int kSize = 10;
  GR_EXPECT_TRUE(MakeCamera("orthographic", kSize, kSize, {1, 1, 5, 5}) ==
                 nullptr);
  GR_EXPECT_TRUE(MakeCamera("pinhole", kSize, kSize, {1, 1, 5}) == nullptr);
  GR_EXPECT_TRUE(MakeCamera("double_sphere", kSize, kSize, {1, 1, 5, 5}) ==
                 nullptr);
  GR_EXPECT_TRUE(MakeCamera("fisheye", kSize, kSize, {1, 1, 5, 5}) != nullptr);
}

}  // namespace
}  // namespace gr
