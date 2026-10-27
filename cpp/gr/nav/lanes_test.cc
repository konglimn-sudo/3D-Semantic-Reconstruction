#include "gr/nav/lanes.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// A 2.5 x 5 m slot with its entrance on y = y_entrance, the slot above it
// (into = +1) or below it (into = -1).
LaneSlot Slot(const std::string& row, double x0, double y_entrance,
              double into) {
  const double y_back = y_entrance + 5.0 * into;
  LaneSlot s;
  s.row = row;
  s.width = 2.5;
  s.corners = {Vec2(x0, y_entrance), Vec2(x0 + 2.5, y_entrance),
               Vec2(x0 + 2.5, y_back), Vec2(x0, y_back)};
  if (into < 0) {
    s.corners = {Vec2(x0 + 2.5, y_entrance), Vec2(x0, y_entrance),
                 Vec2(x0, y_back), Vec2(x0 + 2.5, y_back)};
  }
  return s;
}

GR_TEST(LanesTest, FacingRowsShareAMidwayLaneAndSingleRowsGetTheirOwn) {
  std::vector<LaneSlot> slots;
  for (int k = 0; k < 8; ++k) slots.push_back(Slot("A", 2.5 * k, 0.0, -1));
  for (int k = 0; k < 8; ++k) slots.push_back(Slot("B", 2.5 * k, 7.0, 1));
  for (int k = 0; k < 4; ++k) slots.push_back(Slot("C", 2.5 * k, 30.0, 1));
  const RasterFrame frame{-10.0, -10.0, 0.1, 400, 500};
  const std::vector<uint8_t> navigable(400 * 500, 1);
  const LaneConfig config;
  const LanePlan plan = PlanLanes(slots, navigable, frame, {}, config);
  GR_EXPECT_EQ(plan.rows, 3);
  GR_ASSERT_TRUE(plan.lanes.size() == 2);
  GR_EXPECT_NEAR(plan.lanes[0][0].y(), 3.5, 1e-9);
  GR_EXPECT_NEAR(plan.lanes[1][0].y(), 30.0 - config.lane_offset, 1e-9);
  // A skeleton already along the paired lane: only the single row's lane is
  // needed.
  const LanePlan followed = PlanLanes(
      slots, navigable, frame, {{Vec2(-1.0, 3.5), Vec2(21.0, 3.5)}}, config);
  GR_EXPECT_EQ(followed.followed, 1);
  GR_EXPECT_EQ(static_cast<int>(followed.lanes.size()), 1);
  // A skeleton edge 1.5 m off the lane, along it, gives way; a crossing one
  // does not.
  GR_EXPECT_TRUE(CoveredByLanes({Vec2(2, 5), Vec2(15, 5)}, plan.lanes,
                                config));
  GR_EXPECT_FALSE(CoveredByLanes({Vec2(5, -2), Vec2(5, 9)}, plan.lanes,
                                 config));
}

GR_TEST(LanesTest, LongWallsProposeLanesOnlyInOpenAreas) {
  // A 60 x 40 m open apron with a wall along y = 0 (all navigable), and a
  // 6 m wide aisle along a wall at y = 100 (rows of slots beyond it).
  const RasterFrame frame{-10.0, -10.0, 0.1, 800, 1300};
  std::vector<uint8_t> navigable(800 * 1300, 0);
  for (int r = 0; r < 1300; ++r) {
    for (int c = 0; c < 800; ++c) {
      const double y = -10.0 + (r + 0.5) * 0.1;
      navigable[static_cast<size_t>(r) * 800 + c] =
          (y > 0.5 && y < 40.0) || (y > 94.0 && y < 99.5);
    }
  }
  const LaneConfig config;
  const std::vector<LaneWall> walls = {
      {Vec2(0.0, 0.0), Vec2(60.0, 0.0), 0.3},
      {Vec2(0.0, 100.0), Vec2(60.0, 100.0), 0.3}};
  const LanePlan plan = PlanLanes({}, navigable, frame, {}, config, walls);
  GR_EXPECT_EQ(plan.walls, 1);
  GR_ASSERT_TRUE(plan.lanes.size() == 1);
  GR_EXPECT_NEAR(plan.lanes[0][0].y(), 0.15 + config.wall_offset, 1e-9);
  GR_EXPECT_EQ(static_cast<int>(plan.source[0]), 1);
  // A drive along the apron, 20 m out, proposes its straight stretch; a
  // second pass along the same line adds nothing.
  std::vector<Vec2> drive;
  for (int k = 0; k <= 50; ++k) drive.emplace_back(5.0 + k, 20.0 + 0.05 * (k % 2));
  const LanePlan driven = PlanLanes({}, navigable, frame, {}, config, walls,
                                    {drive, drive});
  GR_EXPECT_EQ(driven.drives, 1);
  GR_ASSERT_TRUE(driven.lanes.size() == 2);
  GR_EXPECT_EQ(static_cast<int>(driven.source[1]), 2);
  GR_EXPECT_NEAR(driven.lanes[1][0].y(), 20.0, 0.06);
}

}  // namespace
}  // namespace gr
