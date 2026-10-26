#include "gr/nav/traffic.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// A 40 x 20 m loop: corners 0 (0, 0), 1 (40, 0), 2 (40, 20), 3 (0, 20); the
// bottom aisle split at x = 20 (node 4); a slot connector hangs off node 4.
std::vector<TrafficEdge> Loop() {
  auto edge = [](int a, int b, Vec2 p, Vec2 q, bool drivable = true) {
    return TrafficEdge{a, b, 0, drivable, {p, q}};
  };
  return {edge(0, 4, Vec2(0, 0), Vec2(20, 0)),
          edge(4, 1, Vec2(20, 0), Vec2(40, 0)),
          edge(1, 2, Vec2(40, 0), Vec2(40, 20)),
          edge(2, 3, Vec2(40, 20), Vec2(0, 20)),
          edge(3, 0, Vec2(0, 20), Vec2(0, 0)),
          edge(4, 5, Vec2(20, 0), Vec2(20, -3), false)};
}

TrafficArrow Arrow(double x, double y, double dx) {
  return {0, Vec2(x, y), Vec2(dx, 0.0)};
}

GR_TEST(TrafficTest, CorridorsJoinEdgesBetweenJunctions) {
  // Only corners with a turn above the limit end a corridor: the bottom
  // aisle (two edges joined through the connector's node) is one corridor.
  const auto corridors = Corridors(Loop(), 35.0);
  GR_ASSERT_TRUE(corridors.size() == 4);
  GR_EXPECT_EQ(static_cast<int>(corridors[0].size()), 2);
}

GR_TEST(TrafficTest, CentredArrowsMakeAOneWayCorridor) {
  const std::vector<Corridor> out =
      DecideTraffic(Loop(), {Arrow(10, 0.2, -1), Arrow(30, -0.3, -1)});
  GR_ASSERT_TRUE(out.size() == 1);
  GR_EXPECT_TRUE(out[0].verdict == TrafficVerdict::kOneWay);
  GR_EXPECT_EQ(out[0].arrows, 2);
  // Travel against the corridor's step direction (it runs 0 -> 4 -> 1).
  GR_EXPECT_EQ(out[0].travel * (out[0].steps[0].forward ? 1 : -1), -1);
}

GR_TEST(TrafficTest, OtherPatternsStayTwoWay) {
  const auto verdict = [](const std::vector<TrafficArrow>& arrows) {
    const std::vector<Corridor> out = DecideTraffic(Loop(), arrows);
    return out.empty() ? TrafficVerdict::kTwoWay : out[0].verdict;
  };
  GR_EXPECT_TRUE(verdict({Arrow(10, -1.5, 1), Arrow(30, 1.5, -1)}) ==
                 TrafficVerdict::kTwoWay);
  GR_EXPECT_TRUE(verdict({Arrow(10, -1.5, 1), Arrow(30, -1.5, 1)}) ==
                 TrafficVerdict::kOneLaneSeen);
  GR_EXPECT_TRUE(verdict({Arrow(10, 0.0, 1)}) == TrafficVerdict::kOneArrow);
  GR_EXPECT_TRUE(verdict({Arrow(10, 0.0, 1), Arrow(30, 0.0, 1),
                          Arrow(31, 1.6, -1)}) == TrafficVerdict::kTwoWay);
}

}  // namespace
}  // namespace gr
