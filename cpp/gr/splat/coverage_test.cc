#include "gr/splat/coverage.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Points of the plane z = `z`, x and y in [-half, half], 2 cm apart.
Points Plane(double z, double half, double x0 = 0.0, double y0 = 0.0) {
  Points points;
  const int n = static_cast<int>(std::lround(half / 0.02));
  for (int i = -n; i <= n; ++i) {
    for (int j = -n; j <= n; ++j) {
      points.emplace_back(x0 + 0.02 * i, y0 + 0.02 * j, z);
    }
  }
  return points;
}

GR_TEST(CameraCoverageTest, CountsUnoccludedWellExposedViews) {
  const auto camera = MakeCamera("pinhole", 200, 200, {100, 100, 100, 100});
  GR_ASSERT_TRUE(camera != nullptr);
  Points points = Plane(4.0, 5.0);
  const size_t wall = points.size();
  for (const Vec3& p : Plane(2.0, 0.3, -0.8, 0.0)) points.push_back(p);
  const Points normals(points.size(), Vec3(0, 0, -1));
  CameraCoverage coverage(points, normals, CoverageConfig());
  // Left half (u < 100) mid-grey, right half dark; the top rows are masked.
  std::vector<uint8_t> rgb(200 * 200 * 3, 0);
  std::vector<uint8_t> mask(200 * 200, 0);
  for (int v = 0; v < 200; ++v) {
    for (int u = 0; u < 200; ++u) {
      const size_t p = static_cast<size_t>(v * 200 + u);
      const uint8_t value = u < 100 ? 128 : 5;
      for (int c = 0; c < 3; ++c) rgb[3 * p + static_cast<size_t>(c)] = value;
      if (v < 20) mask[p] = 1;
    }
  }
  const ImageView image{rgb.data(), 200, 200, 3};
  coverage.AddFrame(*camera, CameraExposure(Pose()), image, mask.data());
  const CoverageResult result = coverage.Result();
  auto index_of = [&](const Vec3& target) {
    size_t best = 0;
    for (size_t i = 0; i < wall; ++i) {
      if ((points[i] - target).norm() < (points[best] - target).norm()) {
        best = i;
      }
    }
    return best;
  };
  const size_t lit = index_of(Vec3(-0.4, 1.0, 4.0));
  const size_t dark = index_of(Vec3(1.0, 1.0, 4.0));
  const size_t hidden = index_of(Vec3(-1.6, 0.0, 4.0));  // Behind the patch.
  const size_t masked = index_of(Vec3(-1.0, -3.5, 4.0));
  const size_t outside = index_of(Vec3(4.8, 4.8, 4.0));  // Out of the image.
  GR_EXPECT_EQ(result.views[lit], 1);
  GR_EXPECT_EQ(result.good_views[lit], 1);
  GR_EXPECT_NEAR(result.color[lit].x(), 128.0 / 255.0, 1e-6);
  GR_EXPECT_NEAR(result.best_gsd[lit], 4.0 * std::sqrt(1 + 0.01 + 0.0625) /
                                           100.0, 1e-3);
  GR_EXPECT_EQ(result.views[dark], 1);
  GR_EXPECT_EQ(result.good_views[dark], 0);
  GR_EXPECT_EQ(result.views[hidden], 0);
  GR_EXPECT_EQ(result.views[masked], 1);
  GR_EXPECT_EQ(result.good_views[masked], 0);
  GR_EXPECT_EQ(result.views[outside], 0);
  GR_EXPECT_EQ(result.good_views[wall], 1);  // The occluding patch itself.
  // Without an image every unoccluded view counts.
  CameraCoverage planned(points, normals, CoverageConfig());
  planned.AddFrame(*camera, CameraExposure(Pose()), ImageView(), nullptr);
  GR_EXPECT_EQ(planned.Result().good_views[dark], 1);
  GR_EXPECT_EQ(planned.Result().good_views[hidden], 0);
}

GR_TEST(TrainingMaskTest, MovingChangedAndBoxBits) {
  const auto camera = MakeCamera("pinhole", 200, 200, {100, 100, 100, 100});
  GR_ASSERT_TRUE(camera != nullptr);
  const PointCloudIndex statics(Plane(6.0, 4.0));
  const Points moving = Plane(3.0, 0.2, -1.0, 0.0);   // Pixel u = 67.
  const Points hidden = Plane(8.0, 0.4, 0.0, -1.5);   // Behind the wall.
  const Points changed = Plane(3.0, 0.2, 1.0, 0.0);   // Pixel u = 133.
  Points all_moving = moving;
  for (const Vec3& p : hidden) all_moving.push_back(p);
  ObjectBox box;
  box.center = Vec3(0.0, 1.5, 4.0);  // Pixel v = 137.
  box.size = Vec3(0.6, 0.6, 0.6);
  TrainingMaskConfig config;
  config.dilate = 2;
  const std::vector<uint8_t> mask =
      RenderTrainingMask(*camera, CameraExposure(Pose()), statics, all_moving,
                         changed, {box}, config);
  GR_ASSERT_TRUE(mask.size() == 200u * 200u);
  auto at = [&](int u, int v) {
    return mask[static_cast<size_t>(v * 200 + u)];
  };
  GR_EXPECT_EQ(at(67, 100), kMaskMoving);
  GR_EXPECT_EQ(at(133, 100), kMaskChanged);
  GR_EXPECT_EQ(at(100, 137), kMaskMoving);
  GR_EXPECT_EQ(at(100, 100), 0);
  GR_EXPECT_EQ(at(100, 81), 0);  // The hidden moving points.
  // Dilation: 0.2 m at 3 m is +-6.7 px, plus 2.
  GR_EXPECT_EQ(at(67 + 8, 100), kMaskMoving);
  GR_EXPECT_EQ(at(67 + 11, 100), 0);
}

}  // namespace
}  // namespace gr
