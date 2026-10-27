#include "gr/nav/features.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(FeaturesTest, FootprintsLandOnTheNearestEdge) {
  std::vector<Vec2> xy;
  // A barrier arm cabinet next to the first aisle (x = 5, y = 1.2 .. 1.6),
  // a speed bump across the second (x = 20 .. 20.4, y = 8 .. 12), and two
  // stray points.
  for (int i = 0; i < 20; ++i) xy.emplace_back(5.0 + 0.01 * i, 1.2 + 0.02 * i);
  for (int i = 0; i < 40; ++i) {
    for (int k = 0; k < 4; ++k) xy.emplace_back(20.0 + 0.1 * k, 8.0 + 0.1 * i);
  }
  xy.emplace_back(40.0, 40.0);
  xy.emplace_back(-30.0, 0.0);
  const std::vector<Footprint> footprints = ObjectFootprints(xy, 0.2, 5);
  GR_ASSERT_TRUE(footprints.size() == 2);
  GR_EXPECT_EQ(footprints[0].points, 160);  // Largest first: the bump.
  GR_EXPECT_NEAR(footprints[1].min.y(), 1.2, 1e-12);
  const std::vector<std::vector<Vec2>> edges = {
      {Vec2(0, 0), Vec2(30, 0)}, {Vec2(10, 10), Vec2(30, 10)}};
  const std::vector<FootprintEdge> hits =
      EdgesAtFootprints(footprints, edges, 1.5);
  GR_ASSERT_TRUE(hits.size() == 2);
  GR_EXPECT_EQ(hits[0].edge, 1);
  GR_EXPECT_NEAR(hits[0].distance, 0.0, 1e-9);
  GR_EXPECT_NEAR(hits[0].along, 10.0, 0.06);
  GR_EXPECT_EQ(hits[1].edge, 0);
  GR_EXPECT_NEAR(hits[1].distance, 1.2, 1e-9);
  GR_EXPECT_NEAR(hits[1].along, 5.0, 0.06);
  GR_EXPECT_EQ(static_cast<int>(EdgesAtFootprints(footprints, edges, 1.0)
                                    .size()),
               1);
}

}  // namespace
}  // namespace gr
