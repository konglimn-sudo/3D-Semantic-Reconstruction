#include "gr/semantics/projector.h"

#include <vector>

#include "gr/sensors/exposure.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(ProjectorTest, RenderAndVisibilityHandleOcclusion) {
  const auto camera = MakeCamera("pinhole", 200, 200, {100, 100, 100, 100});
  GR_ASSERT_TRUE(camera != nullptr);
  Points wall;
  for (int i = -40; i <= 40; ++i) {
    for (int j = -40; j <= 40; ++j)
      wall.emplace_back(0.025 * i, 0.025 * j, 2.0);
  }
  const CameraExposure exposure{Pose()};
  const RenderResult render =
      RenderPoints(*camera, exposure, wall, {}, 0.025, 1, 3, 0.1, 100.0);
  const Points query = {Vec3(0, 0, 2.0), Vec3(0.1, 0.1, 4.0), Vec3(3, 0, 1)};
  const VisibilityResult visibility =
      ProjectVisibility(*camera, exposure, query, render.depth, 0.05, 0.02);
  GR_EXPECT_EQ(visibility.visible[0], 1);
  GR_EXPECT_EQ(visibility.visible[1], 0);  // Behind the wall.
  GR_EXPECT_EQ(visibility.in_image[2], 0);
}

GR_TEST(PointCloudIndexTest, RadiusSearchesReturnSortedIndices) {
  const PointCloudIndex index(
      {Vec3(0, 0, 0), Vec3(10, 0, 0), Vec3(0.5, 0, 3.0), Vec3(-0.5, 0, 0)});
  GR_EXPECT_TRUE(
      (index.RadiusSearch(Vec3::Zero(), 1.0) == std::vector<int>{0, 3}));
  GR_EXPECT_TRUE(
      (index.RadiusSearchXy(Vec2::Zero(), 1.0) == std::vector<int>{0, 2, 3}));
}

}  // namespace
}  // namespace gr
