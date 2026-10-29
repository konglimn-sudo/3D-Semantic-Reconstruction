#include "gr/sensors/exposure.h"

#include <cmath>
#include <optional>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kPixelTolerance = 1e-9;

GR_TEST(CameraExposureTest, GlobalShutterUsesOnePose) {
  constexpr int kHeight = 80;
  const auto camera = MakeCamera("pinhole", 100, kHeight, {50, 50, 50, 40});
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
  GR_EXPECT_NEAR((projection->uv - *expected).norm(), 0.0, kPixelTolerance);
  // Identical first / last row poses collapse to a global shutter.
  GR_EXPECT_FALSE(
      CameraExposure(world_from_camera, world_from_camera, kHeight).rolling());
}

GR_TEST(CameraExposureTest, RollingShutterFindsTheSelfConsistentRow) {
  constexpr int kHeight = 200;
  // The camera moves 0.3 m along x and turns 2 degrees during the readout.
  constexpr double kTravel = 0.3;  // m
  constexpr double kTurnDegrees = 2.0;
  // The rolling-shutter pixel is this far (px) from a global-shutter one.
  constexpr double kMinRollingShift = 0.5;
  constexpr double kPoseTolerance = 1e-12;
  const auto camera = MakeCamera("pinhole", 200, kHeight, {100, 100, 100, 100});
  GR_ASSERT_TRUE(camera != nullptr);
  const Pose first(Mat3::Identity(), Vec3::Zero());
  const Pose last(So3Exp(Vec3(0.0, DegToRad(kTurnDegrees), 0.0)),
                  Vec3(kTravel, 0.0, 0.0));
  const CameraExposure exposure(first, last, kHeight);
  GR_EXPECT_TRUE(exposure.rolling());
  GR_EXPECT_EQ(exposure.rows(), kHeight);
  for (const Vec3& point : {Vec3(0.5, -1.0, 5.0), Vec3(-1.0, 1.5, 4.0)}) {
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(*camera, point);
    GR_ASSERT_TRUE(projection.has_value());
    const int row = static_cast<int>(std::floor(projection->uv.y()));
    // The pixel is where the point projects with the pose of that row.
    const std::optional<Vec2> check =
        camera->Project(exposure.CameraFromWorld(row) * point);
    GR_ASSERT_TRUE(check.has_value());
    GR_EXPECT_NEAR((projection->uv - *check).norm(), 0.0, kPixelTolerance);
    // And it differs from a global-shutter projection with the first row.
    const std::optional<Vec2> first_row = camera->Project(first * point);
    GR_ASSERT_TRUE(first_row.has_value());
    GR_EXPECT_GT((projection->uv - *first_row).norm(), kMinRollingShift);
  }
  // Rows before the first one clamp to it.
  constexpr int kRowBeforeTheImage = -5;
  GR_EXPECT_NEAR((exposure.CameraFromWorld(kRowBeforeTheImage).translation -
                  exposure.CameraFromWorld(0).translation)
                     .norm(),
                 0.0, kPoseTolerance);
  // The centre of the exposure is halfway along the travel.
  GR_EXPECT_NEAR(exposure.center().x(), 0.5 * kTravel, kPoseTolerance);
}

}  // namespace
}  // namespace gr
