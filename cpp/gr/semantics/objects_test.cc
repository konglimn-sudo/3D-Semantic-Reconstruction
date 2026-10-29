#include "gr/semantics/objects.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "gr/core/pose.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr int kCarLabel = 12;
constexpr double kCarCenterZ = 0.75;  // m: half the car height.

ObjectBox MakeCarBox(int track_id, const Vec3& center, double yaw) {
  ObjectBox box;
  box.track_id = track_id;
  box.label = kCarLabel;
  box.center = center;
  box.size = Vec3(4.6, 1.9, 2.0 * kCarCenterZ);
  box.yaw = yaw;
  return box;
}

GR_TEST(BoxTimelineTest, InterpolatesTracksAlongTheShortArc) {
  constexpr int kParkedTrack = 1;
  constexpr int kMovingTrack = 5;
  constexpr int kOtherTripTrack = 9;
  // The moving car: x 0 at t 1 s to x 2 at t 2 s, yaw 3 to -3 rad.
  constexpr double kStartTime = 1.0;
  constexpr double kEndTime = 2.0;
  constexpr double kEndX = 2.0;
  constexpr double kStartYaw = 3.0;
  constexpr double kMiddleTime = 1.5;
  constexpr double kTolerance = 1e-12;
  constexpr double kYawTolerance = 1e-9;
  BoxTimeline timeline;
  timeline.AddStatic(0, MakeCarBox(kParkedTrack, Vec3(10, 0, kCarCenterZ),
                                   0.0));
  // Added out of order on purpose.
  timeline.AddMoving(0, kEndTime,
                     MakeCarBox(kMovingTrack, Vec3(kEndX, 0, kCarCenterZ),
                                -kStartYaw));
  timeline.AddMoving(0, kStartTime,
                     MakeCarBox(kMovingTrack, Vec3(0, 0, kCarCenterZ),
                                kStartYaw));
  timeline.AddMoving(1, kStartTime,
                     MakeCarBox(kOtherTripTrack, Vec3(0, 0, kCarCenterZ),
                                0.0));
  const std::vector<ObjectBox> at = timeline.At(0, kMiddleTime);
  GR_ASSERT_TRUE(at.size() == 2);
  GR_EXPECT_EQ(at[0].track_id, kParkedTrack);
  GR_EXPECT_EQ(at[1].track_id, kMovingTrack);
  GR_EXPECT_NEAR(at[1].center.x(), 0.5 * kEndX, kTolerance);
  // 3.0 -> -3.0 passes through pi, not through 0.
  GR_EXPECT_NEAR(WrapAngle(at[1].yaw - kPi), 0.0, kYawTolerance);
  GR_EXPECT_FALSE(at[1].is_static);
  // Within and beyond the slack after the track's end.
  constexpr double kSlackShare = 0.8;
  constexpr double kBeyondShare = 1.2;
  GR_EXPECT_EQ(
      timeline.At(0, kEndTime + kSlackShare * BoxTimeline::kDefaultSlack)
          .size(),
      2u);
  GR_EXPECT_EQ(
      timeline.At(0, kEndTime + kBeyondShare * BoxTimeline::kDefaultSlack)
          .size(),
      1u);
  constexpr double kOtherTripTime = 1.2;
  const std::vector<ObjectBox> other_trip = timeline.At(1, kOtherTripTime);
  GR_ASSERT_TRUE(other_trip.size() == 1);
  GR_EXPECT_EQ(other_trip[0].track_id, kOtherTripTrack);
}

GR_TEST(ObjectBoxTest, SurfaceSamplesLieOnTheBox) {
  constexpr double kStep = 0.1;          // m
  constexpr size_t kMinSamples = 1000;
  constexpr double kOnSurface = 1e-9;    // m: margin of the containment test.
  constexpr double kInsideMargin = -0.02;  // m: strictly inside.
  constexpr double kTolerance = 1e-9;
  const ObjectBox box = MakeCarBox(3, Vec3(1, 2, kCarCenterZ), 0.4);
  const Points surface = box.SampleSurface(kStep);
  GR_EXPECT_GT(surface.size(), kMinSamples);
  double z_min = std::numeric_limits<double>::infinity();
  int outside = 0;
  int inside = 0;
  for (const Vec3& point : surface) {
    outside += !box.Contains(point, kOnSurface);
    inside += box.Contains(point, kInsideMargin);  // On the boundary only.
    z_min = std::min(z_min, point.z());
  }
  GR_EXPECT_EQ(outside, 0);
  GR_EXPECT_EQ(inside, 0);
  // The top is sampled, the bottom (at z 0) only by the sides.
  GR_EXPECT_NEAR(z_min, 0.0, kTolerance);
}

}  // namespace
}  // namespace gr
