#include "gr/core/voxel.h"

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(VoxelTest, DownsampleKeepsOnePointPerVoxel) {
  // A 10 x 10 grid of points 1 cm apart: a 10 cm x 10 cm patch.
  constexpr int kGridSide = 10;
  constexpr double kSpacing = 0.01;
  // 5 cm voxels split the patch into 2 x 2; a 1 m voxel holds all of it.
  constexpr double kSmallVoxel = 0.05;
  constexpr size_t kSmallVoxelsOccupied = 4;
  constexpr double kLargeVoxel = 1.0;
  Points points;
  for (int i = 0; i < kGridSide; ++i) {
    for (int j = 0; j < kGridSide; ++j) {
      points.emplace_back(kSpacing * i, kSpacing * j, 0.0);
    }
  }
  GR_EXPECT_EQ(VoxelDownsample(points, kSmallVoxel).size(),
               kSmallVoxelsOccupied);
  GR_EXPECT_EQ(VoxelDownsample(points, kLargeVoxel).size(), 1u);
  // A voxel size of zero keeps every point.
  GR_EXPECT_EQ(VoxelDownsample(points, 0.0).size(), points.size());
  // The first point of every voxel is kept, in input order.
  GR_EXPECT_EQ(VoxelDownsampleIndices(points, kLargeVoxel).front(), 0);
}

}  // namespace
}  // namespace gr
