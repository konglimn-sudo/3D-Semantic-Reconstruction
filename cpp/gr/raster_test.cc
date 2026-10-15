#include "gr/raster.h"

#include <vector>

#include "gr/exposure.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(RasterTest, Reducers) {
  RowMatX2d xy(4, 2);
  xy << 0.1, 0.1, 0.2, 0.2, 1.1, 0.1, 5.0, 5.0;
  Eigen::VectorXd values(4);
  values << 1.0, 3.0, 7.0, 9.0;
  Grid2D grid;
  grid.res = 1.0;
  grid.width = 2;
  grid.height = 2;
  const Eigen::MatrixXd mean =
      Rasterize(xy, values, grid, Reducer::kMean, -1.0);
  GR_EXPECT_NEAR(mean(0, 0), 2.0, 1e-12);
  GR_EXPECT_NEAR(mean(0, 1), 7.0, 1e-12);
  GR_EXPECT_NEAR(mean(1, 1), -1.0, 1e-12);
  GR_EXPECT_NEAR(Rasterize(xy, values, grid, Reducer::kMax, 0.0)(0, 0), 3.0,
                 1e-12);
  GR_EXPECT_NEAR(Rasterize(xy, values, grid, Reducer::kCount, 0.0)(0, 0), 2.0,
                 1e-12);
}

GR_TEST(FloorTextureFuserTest, TakesTheMedianColourOfTheBestViews) {
  Grid2D grid;
  grid.x0 = -1;
  grid.y0 = -1;
  grid.res = 0.1;
  grid.width = 20;
  grid.height = 20;
  FloorTextureFuser fuser(grid, Eigen::MatrixXd::Zero(20, 20), 3);
  const auto camera = MakeCamera("pinhole", 64, 64, {40, 40, 32, 32});
  GR_ASSERT_TRUE(camera != nullptr);
  // Camera 2 m above the floor looking straight down.
  Mat3 down;
  down << 1, 0, 0, 0, -1, 0, 0, 0, -1;  // Camera z -> world -z.
  const Pose world_from_camera(down, Vec3(0, 0, 2));
  std::vector<uint8_t> good(64 * 64 * 3);
  std::vector<uint8_t> bad(64 * 64 * 3);
  for (size_t i = 0; i < good.size(); i += 3) {
    good[i] = 100;
    good[i + 1] = 150;
    good[i + 2] = 200;
    bad[i] = 255;
    bad[i + 1] = 0;
    bad[i + 2] = 0;
  }
  const ImageView good_view{good.data(), 64, 64, 3};
  const ImageView bad_view{bad.data(), 64, 64, 3};
  const CameraExposure exposure(world_from_camera);
  fuser.AddFrame(*camera, exposure, good_view, nullptr, 0.1, 10.0, 0.1);
  fuser.AddFrame(*camera, exposure, good_view, nullptr, 0.1, 10.0, 0.1);
  // A worse (farther) view with a wrong colour.
  const CameraExposure far(Pose(down, Vec3(0, 0, 2.5)));
  fuser.AddFrame(*camera, far, bad_view, nullptr, 0.1, 10.0, 0.1);
  const FloorTextureFuser::Texture texture = fuser.ComputeTexture();
  const size_t center = (10 * 20 + 10) * 3;
  GR_EXPECT_EQ(texture.rgb[center], 100);
  GR_EXPECT_EQ(texture.rgb[center + 1], 150);
  GR_EXPECT_EQ(texture.rgb[center + 2], 200);
  GR_EXPECT_EQ(texture.observations[10 * 20 + 10], 3);
}

}  // namespace
}  // namespace gr
