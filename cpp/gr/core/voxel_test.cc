#include "gr/core/voxel.h"

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(VoxelTest, DownsampleKeepsOnePointPerVoxel) {
  Points points;
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      points.emplace_back(0.01 * i, 0.01 * j, 0.0);  // 10 cm x 10 cm patch.
    }
  }
  GR_EXPECT_EQ(VoxelDownsample(points, 0.05).size(), 4u);
  GR_EXPECT_EQ(VoxelDownsample(points, 1.0).size(), 1u);
  GR_EXPECT_EQ(VoxelDownsample(points, 0.0).size(), points.size());
  // The first point of every voxel is kept, in input order.
  GR_EXPECT_EQ(VoxelDownsampleIndices(points, 1.0).front(), 0);
}

}  // namespace
}  // namespace gr
