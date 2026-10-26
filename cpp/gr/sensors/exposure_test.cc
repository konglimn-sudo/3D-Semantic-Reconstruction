#include "gr/sensors/exposure.h"

#include <cmath>
#include <optional>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(CameraExposureTest, GlobalShutterUsesOnePose) {
  const auto camera = MakeCamera("pinhole", 100, 80, {50, 50, 50, 40});
  GR_ASSERT_TRUE(camera != nullptr);
  const Pose world_from_camera(So3Exp(Vec3(0.1, -0.2, 0.3)), Vec3(1, 2, 3));
  const CameraExposure exposure(world_from_camera);
  GR_EXPECT_FALSE(exposure.rolling());
  const Vec3 point = world_from_camera * Vec3(0.3, -0.2, 4.0);
  const std::optional<CameraExposure::Projection> projection =
      exposure.Project(*camera, point);
  GR_ASSERT_TRUE(projection.has_value());
  const std::optional<Vec2> expected =
      camera->Project(world_from_camera.Inverse() * point);
  GR_ASSERT_TRUE(expected.has_value());
  GR_EXPECT_NEAR((projection->uv - *expected).norm(), 0.0, 1e-9);
  // Identical first / last row poses collapse to a global shutter.
  GR_EXPECT_FALSE(
      CameraExposure(world_from_camera, world_from_camera, 80).rolling());
}

GR_TEST(CameraExposureTest, RollingShutterFindsTheSelfConsistentRow) {
  const int height = 200;
  const auto camera = MakeCamera("pinhole", 200, height, {100, 100, 100, 100});
  GR_ASSERT_TRUE(camera != nullptr);
  // The camera moves 0.3 m along x and turns 2 degrees during the readout.
  const Pose first(Mat3::Identity(), Vec3::Zero());
  const Pose last(So3Exp(Vec3(0.0, DegToRad(2.0), 0.0)), Vec3(0.3, 0.0, 0.0));
  const CameraExposure exposure(first, last, height);
  GR_EXPECT_TRUE(exposure.rolling());
  GR_EXPECT_EQ(exposure.rows(), height);
  for (const Vec3& point : {Vec3(0.5, -1.0, 5.0), Vec3(-1.0, 1.5, 4.0)}) {
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(*camera, point);
    GR_ASSERT_TRUE(projection.has_value());
    const int row = static_cast<int>(std::floor(projection->uv.y()));
    // The pixel is where the point projects with the pose of that row.
    const std::optional<Vec2> check =
        camera->Project(exposure.CameraFromWorld(row) * point);
    GR_ASSERT_TRUE(check.has_value());
    GR_EXPECT_NEAR((projection->uv - *check).norm(), 0.0, 1e-9);
    // And it differs from a global-shutter projection with the middle row.
    const std::optional<Vec2> first_row = camera->Project(first * point);
    GR_ASSERT_TRUE(first_row.has_value());
    GR_EXPECT_GT((projection->uv - *first_row).norm(), 0.5);
  }
  GR_EXPECT_NEAR((exposure.CameraFromWorld(-5).translation -
                  exposure.CameraFromWorld(0).translation)
                     .norm(),
                 0.0, 1e-12);
  GR_EXPECT_NEAR(exposure.center().x(), 0.15, 1e-12);
}

}  // namespace
}  // namespace gr
