#include "gr/sensors/camera.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <random>

#include "gr/testing/test.h"

namespace gr {
namespace {

void ExpectRoundTrip(const CameraModel& camera, double max_theta) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  int tested = 0;
  for (int i = 0; i < 5000 && tested < 500; ++i) {
    const Vec3 p =
        Vec3(uniform(rng), uniform(rng), uniform(rng)).normalized() * 3.0;
    if (std::acos(p.z() / p.norm()) > max_theta) continue;
    const std::optional<Vec2> uv = camera.Project(p);
    if (!uv) continue;
    const std::optional<Vec3> ray = camera.Unproject(*uv);
    GR_ASSERT_TRUE(ray.has_value());
    GR_EXPECT_NEAR(std::acos(std::clamp(ray->dot(p.normalized()), -1.0, 1.0)),
                   0.0, 1e-6);
    ++tested;
  }
  GR_EXPECT_GE(tested, 100);
}

GR_TEST(CameraTest, PinholeRoundTripWithDistortion) {
  const auto camera = MakeCamera(
      "pinhole", 640, 480, {400, 410, 320, 240, -0.1, 0.01, 0.001, -0.001});
  GR_ASSERT_TRUE(camera != nullptr);
  ExpectRoundTrip(*camera, 0.6);
}

GR_TEST(CameraTest, KannalaBrandtRoundTripBeyondNinetyDegrees) {
  const auto camera =
      MakeCamera("kannala_brandt", 800, 800,
                 {250, 250, 400, 400, 0.05, -0.01, 0.002, -0.0005, 1.7});
  GR_ASSERT_TRUE(camera != nullptr);
  ExpectRoundTrip(*camera, 1.65);
}

GR_TEST(CameraTest, DoubleSphereRoundTrip) {
  const auto camera =
      MakeCamera("double_sphere", 800, 800, {300, 300, 400, 400, -0.2, 0.6});
  GR_ASSERT_TRUE(camera != nullptr);
  ExpectRoundTrip(*camera, 1.5);
}

GR_TEST(CameraTest, MakeCameraRejectsUnknownModelsAndMissingParameters) {
  GR_EXPECT_TRUE(MakeCamera("orthographic", 10, 10, {1, 1, 5, 5}) == nullptr);
  GR_EXPECT_TRUE(MakeCamera("pinhole", 10, 10, {1, 1, 5}) == nullptr);
  GR_EXPECT_TRUE(MakeCamera("double_sphere", 10, 10, {1, 1, 5, 5}) == nullptr);
  GR_EXPECT_TRUE(MakeCamera("fisheye", 10, 10, {1, 1, 5, 5}) != nullptr);
}

}  // namespace
}  // namespace gr
