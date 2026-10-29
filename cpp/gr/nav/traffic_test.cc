#include "gr/nav/traffic.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// A 40 x 20 m loop: corners 0 (0, 0), 1 (40, 0), 2 (40, 20), 3 (0, 20); the
// bottom aisle split at x = 20 (node 4); a slot connector hangs off node 4.
std::vector<TrafficEdge> Loop() {
  const auto edge = [](int start_node, int end_node, Vec2 start, Vec2 end,
                       bool drivable = true) {
    return TrafficEdge{start_node, end_node, 0, drivable, {start, end}};
  };
  return {edge(0, 4, Vec2(0, 0), Vec2(20, 0)),
          edge(4, 1, Vec2(20, 0), Vec2(40, 0)),
          edge(1, 2, Vec2(40, 0), Vec2(40, 20)),
          edge(2, 3, Vec2(40, 20), Vec2(0, 20)),
          edge(3, 0, Vec2(0, 20), Vec2(0, 0)),
          edge(4, 5, Vec2(20, 0), Vec2(20, -3), false)};
}

// An arrow on level 0 at (x, y) pointing along +x (`direction_x` = 1) or -x
// (-1).
TrafficArrow Arrow(double x, double y, double direction_x) {
  return {0, Vec2(x, y), Vec2(direction_x, 0.0)};
}

GR_TEST(TrafficTest, CorridorsJoinEdgesBetweenJunctions) {
  // Only corners with a turn above the limit end a corridor: the bottom
  // aisle (two edges joined through the connector's node) is one corridor.
  constexpr double kMaxTurn = 35.0;  // deg
  const auto corridors = Corridors(Loop(), kMaxTurn);
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

// A drive along the bottom aisle from x0 to x1 in its right lane (1 m off
// the centre line).
DrivenPath Drive(double x0, double x1) {
  constexpr int kSamples = 60;
  constexpr double kLaneOffset = 1.0;  // m
  DrivenPath path;
  for (int k = 0; k <= kSamples; ++k) {
    path.points.emplace_back(x0 + (x1 - x0) * k / kSamples,
                             x1 > x0 ? -kLaneOffset : kLaneOffset);
  }
  return path;
}

GR_TEST(TrafficTest, PassesCountDrivesAlongEdges) {
  const std::vector<EdgePasses> passes =
      CountPasses(Loop(), {Drive(1, 39), Drive(39, 1), Drive(2, 13)},
                  TrafficConfig());
  GR_EXPECT_EQ(passes[0].forward, 2);  // 0 -> 4: two drives (one short).
  GR_EXPECT_EQ(passes[0].backward, 1);
  GR_EXPECT_EQ(passes[1].forward, 1);
  GR_EXPECT_EQ(passes[5].forward + passes[5].backward, 0);  // Connector.
}

GR_TEST(TrafficTest, OneArrowNeedsDrivesOrASign) {
  const auto decide = [](const std::vector<DrivenPath>& paths,
                         const std::vector<TrafficSign>& signs) {
    const std::vector<Corridor> out =
        DecideTraffic(Loop(), {Arrow(10, 0.1, -1)}, TrafficConfig(), paths,
                      signs);
    return out.empty() ? TrafficVerdict::kTwoWay : out[0].verdict;
  };
  GR_EXPECT_TRUE(decide({}, {}) == TrafficVerdict::kOneArrow);
  // Two drives with the arrow (-x), none against: one-way.
  GR_EXPECT_TRUE(decide({Drive(39, 1), Drive(38, 2)}, {}) ==
                 TrafficVerdict::kOneWayDriven);
  // One drive against it: two-way.
  GR_EXPECT_TRUE(decide({Drive(39, 1), Drive(38, 2), Drive(1, 39)}, {}) ==
                 TrafficVerdict::kOneArrow);
  TrafficSign one_way;
  one_way.xy = Vec2(20.0, 2.0);
  GR_EXPECT_TRUE(decide({}, {one_way}) == TrafficVerdict::kOneWaySign);
  GR_EXPECT_TRUE(decide({Drive(1, 39)}, {one_way}) ==
                 TrafficVerdict::kConflict);
}

GR_TEST(TrafficTest, NoEntrySignsDecideAlone) {
  // A no-entry sign near the corridor's start (x = 3): traffic leaves
  // there, i.e. travels -x, against the steps 0 -> 4 -> 1.
  TrafficSign sign;
  sign.kind = TrafficSign::Kind::kNoEntry;
  sign.xy = Vec2(3.0, -2.5);
  std::vector<Corridor> out = DecideTraffic(Loop(), {}, TrafficConfig(), {},
                                            {sign});
  GR_ASSERT_TRUE(out.size() == 1);
  GR_EXPECT_TRUE(out[0].verdict == TrafficVerdict::kOneWaySign);
  GR_EXPECT_EQ(out[0].travel * (out[0].steps[0].forward ? 1 : -1), -1);
  // At the corner (0, 0) two corridors end; its facing (+y, towards the
  // left aisle's readers) picks the left aisle (3 -> 0), travelling +y out
  // of the corner... i.e. towards the readers.
  sign.xy = Vec2(1.0, 1.0);
  sign.facing = Vec2(0.0, 1.0);
  out = DecideTraffic(Loop(), {}, TrafficConfig(), {}, {sign});
  GR_ASSERT_TRUE(out.size() == 1);
  GR_EXPECT_EQ(static_cast<int>(out[0].steps.size()), 1);
  GR_EXPECT_EQ(out[0].steps[0].edge, 4);
  // Edge 4 runs 3 (0, 20) -> 0 (0, 0): travel towards +y is against it.
  GR_EXPECT_EQ(out[0].travel * (out[0].steps[0].forward ? 1 : -1), -1);
  // A drive against the sign's direction: conflicting evidence.
  sign.facing = Vec2::Zero();
  sign.xy = Vec2(3.0, -2.5);
  out = DecideTraffic(Loop(), {}, TrafficConfig(), {Drive(1, 39)}, {sign});
  GR_EXPECT_TRUE(out[0].verdict == TrafficVerdict::kConflict);
}

}  // namespace
}  // namespace gr
