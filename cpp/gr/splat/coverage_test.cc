#include "gr/splat/coverage.h"

#include <cmath>
#include <memory>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// A 200 x 200 pinhole camera with a 100 px focal length, centred.
constexpr int kImageSize = 200;
constexpr double kFocal = 100.0;
constexpr int kImageCentre = kImageSize / 2;
// Plane points are this far (m) apart.
constexpr double kPlaneStep = 0.02;

std::shared_ptr<CameraModel> TestCamera() {
  return MakeCamera("pinhole", kImageSize, kImageSize,
                    {kFocal, kFocal, kFocal, kFocal});
}

// Points of the plane z = `z`, x and y within `half` of (center_x,
// center_y), kPlaneStep apart.
Points Plane(double z, double half, double center_x = 0.0,
             double center_y = 0.0) {
  Points points;
  const int steps = static_cast<int>(std::lround(half / kPlaneStep));
  for (int i = -steps; i <= steps; ++i) {
    for (int j = -steps; j <= steps; ++j) {
      points.emplace_back(center_x + kPlaneStep * i, center_y + kPlaneStep * j,
                          z);
    }
  }
  return points;
}

GR_TEST(CameraCoverageTest, CountsUnoccludedWellExposedViews) {
  // A wall at z = 4 m and a patch in front of it at z = 2 m, x = -0.8 m.
  constexpr double kWallZ = 4.0;
  constexpr double kWallHalf = 5.0;
  constexpr double kPatchZ = 2.0;
  constexpr double kPatchHalf = 0.3;
  constexpr double kPatchX = -0.8;
  // Left half (u < 100) mid-grey, right half dark; the top rows are masked.
  constexpr uint8_t kGrey = 128;
  constexpr uint8_t kDark = 5;
  constexpr int kMaskedRows = 20;
  constexpr double kMaxByte = 255.0;
  constexpr double kColourTolerance = 1e-6;
  constexpr double kGsdTolerance = 1e-3;  // m
  const auto camera = TestCamera();
  GR_ASSERT_TRUE(camera != nullptr);
  Points points = Plane(kWallZ, kWallHalf);
  const size_t wall = points.size();
  for (const Vec3& point : Plane(kPatchZ, kPatchHalf, kPatchX, 0.0)) {
    points.push_back(point);
  }
  const Points normals(points.size(), Vec3(0, 0, -1));
  CameraCoverage coverage(points, normals, CoverageConfig());
  std::vector<uint8_t> rgb(kImageSize * kImageSize * kRgbChannels, 0);
  std::vector<uint8_t> mask(kImageSize * kImageSize, 0);
  for (int v = 0; v < kImageSize; ++v) {
    for (int u = 0; u < kImageSize; ++u) {
      const size_t pixel = static_cast<size_t>(v * kImageSize + u);
      const uint8_t value = u < kImageCentre ? kGrey : kDark;
      for (int channel = 0; channel < kRgbChannels; ++channel) {
        rgb[kRgbChannels * pixel + static_cast<size_t>(channel)] = value;
      }
      if (v < kMaskedRows) mask[pixel] = 1;
    }
  }
  const ImageView image{rgb.data(), kImageSize, kImageSize, kRgbChannels};
  coverage.AddFrame(*camera, CameraExposure(Pose()), image, mask.data());
  const CoverageResult result = coverage.Result();
  const auto index_of = [&](const Vec3& target) {
    size_t best = 0;
    for (size_t point = 0; point < wall; ++point) {
      if ((points[point] - target).norm() < (points[best] - target).norm()) {
        best = point;
      }
    }
    return best;
  };
  const Vec3 lit_point(-0.4, 1.0, kWallZ);
  const size_t lit = index_of(lit_point);
  const size_t dark = index_of(Vec3(1.0, 1.0, kWallZ));
  const size_t hidden = index_of(Vec3(-1.6, 0.0, kWallZ));  // Behind the patch.
  const size_t masked = index_of(Vec3(-1.0, -3.5, kWallZ));
  const size_t outside = index_of(Vec3(4.8, 4.8, kWallZ));  // Out of image.
  GR_EXPECT_EQ(result.views[lit], 1);
  GR_EXPECT_EQ(result.good_views[lit], 1);
  GR_EXPECT_NEAR(result.color[lit].x(), kGrey / kMaxByte, kColourTolerance);
  // Range over focal length.
  GR_EXPECT_NEAR(result.best_gsd[lit], lit_point.norm() / kFocal,
                 kGsdTolerance);
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
  // Static wall at 6 m; moving points at 3 m, 1 m left (pixel u = 67) and
  // more behind the wall; changed points at 3 m, 1 m right (u = 133); a box
  // at 4 m, 1.5 m down (v = 137).
  constexpr int kMovingU = 67;
  constexpr int kChangedU = 133;
  constexpr int kBoxV = 137;
  constexpr int kHiddenV = 81;
  constexpr int kDilate = 2;  // Pixels.
  // Dilation: 0.2 m at 3 m is +-6.7 px, plus kDilate.
  constexpr int kInsideDilation = 8;
  constexpr int kOutsideDilation = 11;
  const auto camera = TestCamera();
  GR_ASSERT_TRUE(camera != nullptr);
  const PointCloudIndex statics(Plane(6.0, 4.0));
  const Points moving = Plane(3.0, 0.2, -1.0, 0.0);
  const Points hidden = Plane(8.0, 0.4, 0.0, -1.5);  // Behind the wall.
  const Points changed = Plane(3.0, 0.2, 1.0, 0.0);
  Points all_moving = moving;
  for (const Vec3& point : hidden) all_moving.push_back(point);
  ObjectBox box;
  box.center = Vec3(0.0, 1.5, 4.0);
  box.size = Vec3(0.6, 0.6, 0.6);
  TrainingMaskConfig config;
  config.dilate = kDilate;
  const std::vector<uint8_t> mask =
      RenderTrainingMask(*camera, CameraExposure(Pose()), statics, all_moving,
                         changed, {box}, config);
  GR_ASSERT_TRUE(mask.size() == static_cast<size_t>(kImageSize) * kImageSize);
  const auto at = [&](int u, int v) {
    return mask[static_cast<size_t>(v * kImageSize + u)];
  };
  GR_EXPECT_EQ(at(kMovingU, kImageCentre), kMaskMoving);
  GR_EXPECT_EQ(at(kChangedU, kImageCentre), kMaskChanged);
  GR_EXPECT_EQ(at(kImageCentre, kBoxV), kMaskMoving);
  GR_EXPECT_EQ(at(kImageCentre, kImageCentre), 0);
  GR_EXPECT_EQ(at(kImageCentre, kHiddenV), 0);  // The hidden moving points.
  GR_EXPECT_EQ(at(kMovingU + kInsideDilation, kImageCentre), kMaskMoving);
  GR_EXPECT_EQ(at(kMovingU + kOutsideDilation, kImageCentre), 0);
}

}  // namespace
}  // namespace gr
