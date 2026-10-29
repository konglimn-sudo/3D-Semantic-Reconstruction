#include "gr/mapping/raster.h"

#include <vector>

#include "gr/sensors/exposure.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(RasterTest, Reducers) {
  constexpr double kTolerance = 1e-12;
  constexpr double kEmpty = -1.0;  // Fill of empty cells.
  constexpr int kGridSide = 2;     // Cells of 1 m.
  RowMatX2d xy(4, kPlaneDim);
  xy << 0.1, 0.1, 0.2, 0.2, 1.1, 0.1, 5.0, 5.0;  // The last one is outside.
  Eigen::VectorXd values(4);
  values << 1.0, 3.0, 7.0, 9.0;
  Grid2D grid;
  grid.res = 1.0;
  grid.width = kGridSide;
  grid.height = kGridSide;
  const Eigen::MatrixXd mean =
      Rasterize(xy, values, grid, Reducer::kMean, kEmpty);
  GR_EXPECT_NEAR(mean(0, 0), 2.0, kTolerance);  // Mean of 1 and 3.
  GR_EXPECT_NEAR(mean(0, 1), 7.0, kTolerance);
  GR_EXPECT_NEAR(mean(1, 1), kEmpty, kTolerance);
  GR_EXPECT_NEAR(Rasterize(xy, values, grid, Reducer::kMax, 0.0)(0, 0), 3.0,
                 kTolerance);
  GR_EXPECT_NEAR(Rasterize(xy, values, grid, Reducer::kCount, 0.0)(0, 0), 2.0,
                 kTolerance);
}

GR_TEST(FloorTextureFuserTest, TakesTheMedianColourOfTheBestViews) {
  constexpr int kGridSide = 20;       // Cells of kCell.
  constexpr double kCell = 0.1;       // m
  constexpr int kTopK = 3;
  constexpr int kImageSide = 64;      // px
  constexpr double kFocal = 40.0;     // px
  constexpr double kHeight = 2.0;     // m above the floor.
  constexpr double kFarHeight = 2.5;  // m
  constexpr double kDepthTolerance = 0.1;  // m
  constexpr double kMaxRange = 10.0;       // m
  constexpr double kMinCos = 0.1;
  constexpr uint8_t kGood[kRgbChannels] = {100, 150, 200};
  constexpr uint8_t kBad[kRgbChannels] = {255, 0, 0};
  Grid2D grid;
  grid.x0 = -kGridSide * kCell / 2;
  grid.y0 = -kGridSide * kCell / 2;
  grid.res = kCell;
  grid.width = kGridSide;
  grid.height = kGridSide;
  FloorTextureFuser fuser(grid, Eigen::MatrixXd::Zero(kGridSide, kGridSide),
                          kTopK);
  const auto camera = MakeCamera(
      "pinhole", kImageSide, kImageSide,
      {kFocal, kFocal, kImageSide / 2.0, kImageSide / 2.0});
  GR_ASSERT_TRUE(camera != nullptr);
  // The camera looks straight down at the floor.
  Mat3 down;
  down << 1, 0, 0, 0, -1, 0, 0, 0, -1;  // Camera z -> world -z.
  const Pose world_from_camera(down, Vec3(0, 0, kHeight));
  const size_t bytes =
      static_cast<size_t>(kImageSide) * kImageSide * kRgbChannels;
  std::vector<uint8_t> good(bytes);
  std::vector<uint8_t> bad(bytes);
  for (size_t pixel = 0; pixel < bytes; pixel += kRgbChannels) {
    for (int channel = 0; channel < kRgbChannels; ++channel) {
      good[pixel + static_cast<size_t>(channel)] = kGood[channel];
      bad[pixel + static_cast<size_t>(channel)] = kBad[channel];
    }
  }
  const ImageView good_view{good.data(), kImageSide, kImageSide,
                            kRgbChannels};
  const ImageView bad_view{bad.data(), kImageSide, kImageSide, kRgbChannels};
  const CameraExposure exposure(world_from_camera);
  fuser.AddFrame(*camera, exposure, good_view, nullptr, kDepthTolerance,
                 kMaxRange, kMinCos);
  fuser.AddFrame(*camera, exposure, good_view, nullptr, kDepthTolerance,
                 kMaxRange, kMinCos);
  // A worse (farther) view with a wrong colour.
  const CameraExposure far(Pose(down, Vec3(0, 0, kFarHeight)));
  fuser.AddFrame(*camera, far, bad_view, nullptr, kDepthTolerance, kMaxRange,
                 kMinCos);
  const FloorTextureFuser::Texture texture = fuser.ComputeTexture();
  const size_t center_cell = (kGridSide / 2) * kGridSide + kGridSide / 2;
  for (int channel = 0; channel < kRgbChannels; ++channel) {
    GR_EXPECT_EQ(texture.rgb[center_cell * kRgbChannels +
                             static_cast<size_t>(channel)],
                 kGood[channel]);
  }
  GR_EXPECT_EQ(texture.observations[center_cell], 3);
}

}  // namespace
}  // namespace gr
