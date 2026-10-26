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

}  // namespace
}  // namespace gr
