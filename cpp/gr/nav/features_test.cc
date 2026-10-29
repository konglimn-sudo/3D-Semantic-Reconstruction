#include "gr/nav/features.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(FeaturesTest, FootprintsLandOnTheNearestEdge) {
  constexpr double kCell = 0.2;            // m
  constexpr int kMinPoints = 5;
  constexpr int kBumpPoints = 160;         // 40 x 4 below.
  constexpr double kExact = 1e-12;
  constexpr double kTolerance = 1e-9;
  constexpr double kAlongTolerance = 0.06;  // m, the edge sampling.
  constexpr double kMargin = 1.5;          // m
  constexpr double kNarrowMargin = 1.0;    // m, below the cabinet's 1.2 m.
  // A barrier arm cabinet next to the first aisle (x = 5, y = 1.2 .. 1.6),
  // a speed bump across the second (x = 20 .. 20.4, y = 8 .. 12), and two
  // stray points.
  std::vector<Vec2> xy;
  for (int i = 0; i < 20; ++i) xy.emplace_back(5.0 + 0.01 * i, 1.2 + 0.02 * i);
  for (int row = 0; row < 40; ++row) {
    for (int column = 0; column < 4; ++column) {
      xy.emplace_back(20.0 + 0.1 * column, 8.0 + 0.1 * row);
    }
  }
  xy.emplace_back(40.0, 40.0);
  xy.emplace_back(-30.0, 0.0);
  const std::vector<Footprint> footprints =
      ObjectFootprints(xy, kCell, kMinPoints);
  GR_ASSERT_TRUE(footprints.size() == 2);
  GR_EXPECT_EQ(footprints[0].points, kBumpPoints);  // Largest first.
  GR_EXPECT_NEAR(footprints[1].min.y(), 1.2, kExact);
  const std::vector<std::vector<Vec2>> edges = {
      {Vec2(0, 0), Vec2(30, 0)}, {Vec2(10, 10), Vec2(30, 10)}};
  const std::vector<FootprintEdge> hits =
      EdgesAtFootprints(footprints, edges, kMargin);
  GR_ASSERT_TRUE(hits.size() == 2);
  GR_EXPECT_EQ(hits[0].edge, 1);  // The bump straddles the second aisle.
  GR_EXPECT_NEAR(hits[0].distance, 0.0, kTolerance);
  GR_EXPECT_NEAR(hits[0].along, 10.0, kAlongTolerance);
  GR_EXPECT_EQ(hits[1].edge, 0);  // The cabinet is 1.2 m off the first.
  GR_EXPECT_NEAR(hits[1].distance, 1.2, kTolerance);
  GR_EXPECT_NEAR(hits[1].along, 5.0, kAlongTolerance);
  GR_EXPECT_EQ(
      static_cast<int>(EdgesAtFootprints(footprints, edges, kNarrowMargin)
                           .size()),
      1);
}

}  // namespace
}  // namespace gr
