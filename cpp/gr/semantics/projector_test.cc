#include "gr/semantics/projector.h"

#include <vector>

#include "gr/sensors/exposure.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(ProjectorTest, RenderAndVisibilityHandleOcclusion) {
  constexpr int kImageSize = 200;      // Pixels, square.
  constexpr double kFocal = 100.0;     // Pixels; principal point in the middle.
  constexpr int kHalfWallPoints = 40;  // Points to each side of the axis.
  constexpr double kWallSpacing = 0.025;  // m
  constexpr double kWallDistance = 2.0;   // m in front of the camera.
  constexpr int kMinSplatRadius = 1;      // Pixels.
  constexpr int kMaxSplatRadius = 3;
  constexpr double kMinRange = 0.1;   // m
  constexpr double kMaxRange = 100.0;  // m
  constexpr double kDepthAbsTol = 0.05;  // m
  constexpr double kDepthRelTol = 0.02;
  const auto camera = MakeCamera("pinhole", kImageSize, kImageSize,
                                 {kFocal, kFocal, kFocal, kFocal});
  GR_ASSERT_TRUE(camera != nullptr);
  Points wall;
  for (int i = -kHalfWallPoints; i <= kHalfWallPoints; ++i) {
    for (int j = -kHalfWallPoints; j <= kHalfWallPoints; ++j) {
      wall.emplace_back(kWallSpacing * i, kWallSpacing * j, kWallDistance);
    }
  }
  const CameraExposure exposure{Pose()};
  const RenderResult render =
      RenderPoints(*camera, exposure, wall, {}, kWallSpacing, kMinSplatRadius,
                   kMaxSplatRadius, kMinRange, kMaxRange);
  // On the wall, behind it, outside the field of view.
  const Points query = {Vec3(0, 0, kWallDistance), Vec3(0.1, 0.1, 4.0),
                        Vec3(3, 0, 1)};
  const VisibilityResult visibility = ProjectVisibility(
      *camera, exposure, query, render.depth, kDepthAbsTol, kDepthRelTol);
  GR_EXPECT_EQ(visibility.visible[0], 1);
  GR_EXPECT_EQ(visibility.visible[1], 0);  // Behind the wall.
  GR_EXPECT_EQ(visibility.in_image[2], 0);
}

GR_TEST(PointCloudIndexTest, RadiusSearchesReturnSortedIndices) {
  constexpr double kRadius = 1.0;  // m
  // Point 2 is within the radius in xy only; point 1 in neither.
  const PointCloudIndex index(
      {Vec3(0, 0, 0), Vec3(10, 0, 0), Vec3(0.5, 0, 3.0), Vec3(-0.5, 0, 0)});
  GR_EXPECT_TRUE(
      (index.RadiusSearch(Vec3::Zero(), kRadius) == std::vector<int>{0, 3}));
  GR_EXPECT_TRUE((index.RadiusSearchXy(Vec2::Zero(), kRadius) ==
                  std::vector<int>{0, 2, 3}));
}

}  // namespace
}  // namespace gr
