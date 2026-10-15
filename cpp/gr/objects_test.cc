#include "gr/objects.h"

#include <algorithm>
#include <vector>

#include "gr/pose.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

ObjectBox MakeCarBox(int track_id, const Vec3& center, double yaw) {
  ObjectBox box;
  box.track_id = track_id;
  box.label = 12;
  box.center = center;
  box.size = Vec3(4.6, 1.9, 1.5);
  box.yaw = yaw;
  return box;
}

GR_TEST(BoxTimelineTest, InterpolatesTracksAlongTheShortArc) {
  BoxTimeline timeline;
  timeline.AddStatic(0, MakeCarBox(1, Vec3(10, 0, 0.75), 0.0));
  // Added out of order on purpose.
  timeline.AddMoving(0, 2.0, MakeCarBox(5, Vec3(2, 0, 0.75), -3.0));
  timeline.AddMoving(0, 1.0, MakeCarBox(5, Vec3(0, 0, 0.75), 3.0));
  timeline.AddMoving(1, 1.0, MakeCarBox(9, Vec3(0, 0, 0.75), 0.0));
  const std::vector<ObjectBox> at = timeline.At(0, 1.5);
  GR_ASSERT_TRUE(at.size() == 2);
  GR_EXPECT_EQ(at[0].track_id, 1);
  GR_EXPECT_EQ(at[1].track_id, 5);
  GR_EXPECT_NEAR(at[1].center.x(), 1.0, 1e-12);
  // 3.0 -> -3.0 passes through pi, not through 0.
  GR_EXPECT_NEAR(WrapAngle(at[1].yaw - kPi), 0.0, 1e-9);
  GR_EXPECT_FALSE(at[1].is_static);
  GR_EXPECT_EQ(timeline.At(0, 2.4).size(), 2u);  // Within the slack.
  GR_EXPECT_EQ(timeline.At(0, 2.6).size(), 1u);
  const std::vector<ObjectBox> other_trip = timeline.At(1, 1.2);
  GR_ASSERT_TRUE(other_trip.size() == 1);
  GR_EXPECT_EQ(other_trip[0].track_id, 9);
}

GR_TEST(ObjectBoxTest, SurfaceSamplesLieOnTheBox) {
  const ObjectBox box = MakeCarBox(3, Vec3(1, 2, 0.75), 0.4);
  const Points surface = box.SampleSurface(0.1);
  GR_EXPECT_GT(surface.size(), 1000u);
  double z_min = 1e9;
  int outside = 0;
  int inside = 0;
  for (const Vec3& p : surface) {
    outside += !box.Contains(p, 1e-9);
    inside += box.Contains(p, -0.02);  // On the boundary, not inside.
    z_min = std::min(z_min, p.z());
  }
  GR_EXPECT_EQ(outside, 0);
  GR_EXPECT_EQ(inside, 0);
  GR_EXPECT_NEAR(z_min, 0.0, 1e-9);
}

}  // namespace
}  // namespace gr
