#include "gr/nav/lanes.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kSlotWidth = 2.5;  // m
constexpr double kSlotDepth = 5.0;  // m
constexpr double kTolerance = 1e-9;

// A slot with its entrance on y = y_entrance, the slot above it (into = +1)
// or below it (into = -1).
LaneSlot Slot(const std::string& row, double x0, double y_entrance,
              double into) {
  const double y_back = y_entrance + kSlotDepth * into;
  LaneSlot slot;
  slot.row = row;
  slot.width = kSlotWidth;
  slot.corners = {Vec2(x0, y_entrance), Vec2(x0 + kSlotWidth, y_entrance),
                  Vec2(x0 + kSlotWidth, y_back), Vec2(x0, y_back)};
  if (into < 0) {
    slot.corners = {Vec2(x0 + kSlotWidth, y_entrance), Vec2(x0, y_entrance),
                    Vec2(x0, y_back), Vec2(x0 + kSlotWidth, y_back)};
  }
  return slot;
}

GR_TEST(LanesTest, FacingRowsShareAMidwayLaneAndSingleRowsGetTheirOwn) {
  // Rows A (entrance y = 0, facing up) and B (y = 7, facing down) face each
  // other; row C (y = 30) is alone.
  constexpr double kRowAY = 0.0;
  constexpr double kRowBY = 7.0;
  constexpr double kRowCY = 30.0;
  constexpr int kPairedSlots = 8;
  constexpr int kSingleSlots = 4;
  constexpr int kWidth = 400;   // Cells of 0.1 m from (-10, -10).
  constexpr int kHeight = 500;
  constexpr double kMidway = (kRowAY + kRowBY) / 2;
  std::vector<LaneSlot> slots;
  for (int k = 0; k < kPairedSlots; ++k) {
    slots.push_back(Slot("A", kSlotWidth * k, kRowAY, -1));
  }
  for (int k = 0; k < kPairedSlots; ++k) {
    slots.push_back(Slot("B", kSlotWidth * k, kRowBY, 1));
  }
  for (int k = 0; k < kSingleSlots; ++k) {
    slots.push_back(Slot("C", kSlotWidth * k, kRowCY, 1));
  }
  const RasterFrame frame{-10.0, -10.0, 0.1, kWidth, kHeight};
  const std::vector<uint8_t> navigable(kWidth * kHeight, 1);
  const LaneConfig config;
  const LanePlan plan = PlanLanes(slots, navigable, frame, {}, config);
  GR_EXPECT_EQ(plan.rows, 3);
  GR_ASSERT_TRUE(plan.lanes.size() == 2);
  GR_EXPECT_NEAR(plan.lanes[0][0].y(), kMidway, kTolerance);
  GR_EXPECT_NEAR(plan.lanes[1][0].y(), kRowCY - config.lane_offset,
                 kTolerance);
  // A skeleton already along the paired lane: only the single row's lane is
  // needed.
  const LanePlan followed =
      PlanLanes(slots, navigable, frame,
                {{Vec2(-1.0, kMidway), Vec2(21.0, kMidway)}}, config);
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
  constexpr int kWidth = 800;    // Cells of kCell from (-10, -10).
  constexpr int kHeight = 1300;
  constexpr double kCell = 0.1;  // m
  constexpr double kOrigin = -10.0;
  constexpr double kWallThickness = 0.3;  // m
  constexpr double kDriveY = 20.0;        // m
  constexpr double kDriveJitter = 0.05;   // m
  constexpr int kDriveSamples = 50;       // 1 m apart.
  constexpr double kDriveTolerance = 0.06;  // m
  const RasterFrame frame{kOrigin, kOrigin, kCell, kWidth, kHeight};
  std::vector<uint8_t> navigable(kWidth * kHeight, 0);
  for (int row = 0; row < kHeight; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      const double y = kOrigin + (row + 0.5) * kCell;
      navigable[static_cast<size_t>(row) * kWidth + column] =
          (y > 0.5 && y < 40.0) || (y > 94.0 && y < 99.5);
    }
  }
  const LaneConfig config;
  const std::vector<LaneWall> walls = {
      {Vec2(0.0, 0.0), Vec2(60.0, 0.0), kWallThickness},
      {Vec2(0.0, 100.0), Vec2(60.0, 100.0), kWallThickness}};
  const LanePlan plan = PlanLanes({}, navigable, frame, {}, config, walls);
  GR_EXPECT_EQ(plan.walls, 1);
  GR_ASSERT_TRUE(plan.lanes.size() == 1);
  GR_EXPECT_NEAR(plan.lanes[0][0].y(),
                 kWallThickness / 2 + config.wall_offset, kTolerance);
  GR_EXPECT_EQ(static_cast<int>(plan.source[0]), kWallLane);
  // A drive along the apron proposes its straight stretch; a second pass
  // along the same line adds nothing.
  std::vector<Vec2> drive;
  for (int k = 0; k <= kDriveSamples; ++k) {
    drive.emplace_back(5.0 + k, kDriveY + kDriveJitter * (k % 2));
  }
  const LanePlan driven = PlanLanes({}, navigable, frame, {}, config, walls,
                                    {drive, drive});
  GR_EXPECT_EQ(driven.drives, 1);
  GR_ASSERT_TRUE(driven.lanes.size() == 2);
  GR_EXPECT_EQ(static_cast<int>(driven.source[1]), kDriveLane);
  GR_EXPECT_NEAR(driven.lanes[1][0].y(), kDriveY, kDriveTolerance);
}

}  // namespace
}  // namespace gr
