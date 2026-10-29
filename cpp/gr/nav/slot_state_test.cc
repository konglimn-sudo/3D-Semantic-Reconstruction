#include "gr/nav/slot_state.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kSlotWidth = 2.5;  // m
constexpr double kSlotDepth = 5.0;  // m

SlotGeometry Slot(double x0) {
  SlotGeometry slot;
  slot.corners = {Vec2(x0, 0.0), Vec2(x0 + kSlotWidth, 0.0),
                  Vec2(x0 + kSlotWidth, kSlotDepth), Vec2(x0, kSlotDepth)};
  return slot;
}

// Floor returns every kStep over [x0, x1] x [0, kSlotDepth], and a car body
// (a box surface kCarHeight high) over the second slot.
Points Sweep(double x0, double x1, bool car) {
  constexpr double kStep = 0.1;          // m
  constexpr double kFloorHeight = 0.02;  // m
  constexpr double kCarHeight = 1.4;     // m
  Points points;
  for (double x = x0; x < x1; x += kStep) {
    for (double y = kStep / 2; y < kSlotDepth; y += kStep) {
      const bool under_car = car && x > 3.0 && x < 4.5 && y > 0.5 && y < 4.5;
      points.emplace_back(x, y, under_car ? kCarHeight : kFloorHeight);
    }
  }
  return points;
}

GR_TEST(SlotStateTest, CarsOccupyAndSeenFloorFrees) {
  constexpr double kFarSlotX = 50.0;  // m, out of range.
  constexpr double kMinFloorShare = 0.9;
  constexpr double kPartialSweepEnd = 1.2;  // m: half of the first slot.
  constexpr int kCarPoints = 100;
  constexpr double kFreeShare = 0.8;
  const std::vector<SlotGeometry> slots = {Slot(0.0), Slot(kSlotWidth),
                                           Slot(kFarSlotX)};
  const SlotStateConfig config;
  const Vec3 sensor(2.5, -4.0, 1.8);
  std::vector<SlotObservation> seen = ObserveSlots(
      Sweep(0.0, 2 * kSlotWidth, true), sensor, slots, config);
  GR_ASSERT_TRUE(seen.size() == 2);  // The third slot is out of range.
  GR_EXPECT_TRUE(seen[0].state == SlotState::kFree);
  GR_EXPECT_GT(seen[0].floor_share, kMinFloorShare);
  GR_EXPECT_TRUE(seen[1].state == SlotState::kOccupied);
  // Only half of the first slot's floor seen... still free; none: unknown.
  seen = ObserveSlots(Sweep(0.0, kPartialSweepEnd, false), sensor, slots,
                      config);
  GR_EXPECT_TRUE(seen[0].state == SlotState::kFree);
  GR_EXPECT_TRUE(seen[1].state == SlotState::kUnknown);

  // The tracker flips a known state after `confirm` (2) observations.
  SlotStateTracker tracker(slots.size(), config);
  tracker.Update(1.0, {{1, SlotState::kOccupied, kCarPoints, 0.0}});
  tracker.Update(2.0, {{1, SlotState::kFree, 0, kFreeShare}});
  GR_EXPECT_TRUE(tracker.entries()[1].state == SlotState::kOccupied);
  tracker.Update(3.0, {{1, SlotState::kFree, 0, kFreeShare}});
  GR_EXPECT_TRUE(tracker.entries()[1].state == SlotState::kFree);
  GR_EXPECT_NEAR(tracker.entries()[1].stamp, 3.0, 0.0);
  GR_EXPECT_EQ(tracker.entries()[1].observations, 3);
  GR_EXPECT_TRUE(tracker.entries()[0].state == SlotState::kUnknown);
}

}  // namespace
}  // namespace gr
