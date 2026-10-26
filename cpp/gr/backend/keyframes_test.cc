#include "gr/backend/keyframes.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

Pose PoseXy(double x, double y) {
  return Pose(Mat3::Identity(), Vec3(x, y, 0));
}

GR_TEST(KeyframeSetTest, SubmapAndPathDistance) {
  KeyframeSet keyframes({0.1, 0.5});
  const Points sweep{Vec3(1, 0, 0), Vec3(1.02, 0, 0), Vec3(0, 3, 0)};
  for (int k = 0; k < 5; ++k) {
    Keyframe keyframe;
    keyframe.trip = k < 3 ? 0 : 1;
    keyframe.frame = k;
    keyframe.stamp = k;
    keyframe.odom = PoseXy(2.0 * k, 0);
    keyframes.Add(keyframe, sweep);
  }
  GR_EXPECT_EQ(keyframes.num_trips(), 2);
  GR_EXPECT_TRUE((keyframes.TripIndices(1) == std::vector<int>{3, 4}));
  GR_EXPECT_TRUE(keyframes.TripIndices(7).empty());
  // The two points 2 cm apart share a 10 cm voxel.
  GR_EXPECT_EQ(keyframes.Cloud(0, 0.1).size(), 2u);
  GR_EXPECT_TRUE(keyframes.HasVoxel(0.5));
  GR_EXPECT_FALSE(keyframes.HasVoxel(0.3));
  const std::vector<double> distance =
      keyframes.PathDistance(keyframes.OdomPoses());
  GR_EXPECT_NEAR(distance[2], 4.0, 1e-12);
  GR_EXPECT_NEAR(distance[3], 0.0, 1e-12);  // Restarts per trip.
  // Neighbours of keyframe 1 on its trip, in its frame: x = 1 - 2, 1, 1 + 2.
  const Points submap = keyframes.Submap(1, 1, 0.1, keyframes.OdomPoses());
  int on_axis = 0;
  for (const Vec3& p : submap) {
    if (std::abs(p.y()) < 1e-9 &&
        (std::abs(p.x() + 1) < 0.05 || std::abs(p.x() - 3) < 0.05)) {
      ++on_axis;
    }
  }
  GR_EXPECT_EQ(submap.size(), 6u);
  GR_EXPECT_EQ(on_axis, 2);
}

}  // namespace
}  // namespace gr
