#include "gr/backend/keyframes.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kExactTolerance = 1e-12;

Pose PoseXy(double x, double y) {
  return Pose(Mat3::Identity(), Vec3(x, y, 0));
}

GR_TEST(KeyframeSetTest, SubmapAndPathDistance) {
  // Five keyframes 2 m apart along x, the first three on trip 0, each with
  // the same three-point sweep, kept at a fine and a coarse voxel.
  constexpr double kFineVoxel = 0.1;    // m
  constexpr double kCoarseVoxel = 0.5;  // m
  constexpr double kOtherVoxel = 0.3;   // m, not kept
  constexpr int kKeyframes = 5;
  constexpr int kFirstTrip1 = 3;
  constexpr double kSpacing = 2.0;  // m
  constexpr int kHalfWindow = 1;    // Submap neighbours on each side.
  constexpr double kOnAxisTolerance = 1e-9;
  constexpr double kMatchRadius = 0.05;  // m
  KeyframeSet keyframes({kFineVoxel, kCoarseVoxel});
  const Points sweep{Vec3(1, 0, 0), Vec3(1.02, 0, 0), Vec3(0, 3, 0)};
  for (int index = 0; index < kKeyframes; ++index) {
    Keyframe keyframe;
    keyframe.trip = index < kFirstTrip1 ? 0 : 1;
    keyframe.frame = index;
    keyframe.stamp = index;
    keyframe.odom = PoseXy(kSpacing * index, 0);
    keyframes.Add(keyframe, sweep);
  }
  GR_EXPECT_EQ(keyframes.num_trips(), 2);
  GR_EXPECT_TRUE((keyframes.TripIndices(1) == std::vector<int>{3, 4}));
  GR_EXPECT_TRUE(keyframes.TripIndices(7).empty());
  // The two points 2 cm apart share a 10 cm voxel.
  GR_EXPECT_EQ(keyframes.Cloud(0, kFineVoxel).size(), 2u);
  GR_EXPECT_TRUE(keyframes.HasVoxel(kCoarseVoxel));
  GR_EXPECT_FALSE(keyframes.HasVoxel(kOtherVoxel));
  const std::vector<double> distance =
      keyframes.PathDistance(keyframes.OdomPoses());
  GR_EXPECT_NEAR(distance[2], 4.0, kExactTolerance);
  GR_EXPECT_NEAR(distance[3], 0.0, kExactTolerance);  // Restarts per trip.
  // Neighbours of keyframe 1 on its trip, in its frame: x = 1 - 2, 1, 1 + 2.
  const Points submap =
      keyframes.Submap(1, kHalfWindow, kFineVoxel, keyframes.OdomPoses());
  int on_axis = 0;
  for (const Vec3& point : submap) {
    if (std::abs(point.y()) < kOnAxisTolerance &&
        (std::abs(point.x() + 1) < kMatchRadius ||
         std::abs(point.x() - 3) < kMatchRadius)) {
      ++on_axis;
    }
  }
  GR_EXPECT_EQ(submap.size(), 6u);
  GR_EXPECT_EQ(on_axis, 2);
}

}  // namespace
}  // namespace gr
