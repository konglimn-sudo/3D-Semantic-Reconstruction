#include "gr/nav/slot_state.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

SlotGeometry Slot(double x0) {
  SlotGeometry slot;
  slot.corners = {Vec2(x0, 0.0), Vec2(x0 + 2.5, 0.0), Vec2(x0 + 2.5, 5.0),
                  Vec2(x0, 5.0)};
  return slot;
}

// Floor returns every 0.1 m over [x0, x1] x [0, 5], and a car body (a 1.4 m
// high box surface) over the second slot.
Points Sweep(double x0, double x1, bool car) {
  Points points;
  for (double x = x0; x < x1; x += 0.1) {
    for (double y = 0.05; y < 5.0; y += 0.1) {
      const bool under_car = car && x > 3.0 && x < 4.5 && y > 0.5 && y < 4.5;
      points.emplace_back(x, y, under_car ? 1.4 : 0.02);
    }
  }
  return points;
}

GR_TEST(SlotStateTest, CarsOccupyAndSeenFloorFrees) {
  const std::vector<SlotGeometry> slots = {Slot(0.0), Slot(2.5), Slot(50.0)};
  const SlotStateConfig config;
  const Vec3 sensor(2.5, -4.0, 1.8);
  std::vector<SlotObservation> seen =
      ObserveSlots(Sweep(0.0, 5.0, true), sensor, slots, config);
  GR_ASSERT_TRUE(seen.size() == 2);  // The third slot is out of range.
  GR_EXPECT_TRUE(seen[0].state == SlotState::kFree);
  GR_EXPECT_GT(seen[0].floor_share, 0.9);
  GR_EXPECT_TRUE(seen[1].state == SlotState::kOccupied);
  // Only half of the first slot's floor seen... still free; none: unknown.
  seen = ObserveSlots(Sweep(0.0, 1.2, false), sensor, slots, config);
  GR_EXPECT_TRUE(seen[0].state == SlotState::kFree);
  GR_EXPECT_TRUE(seen[1].state == SlotState::kUnknown);

  // The tracker flips a known state after `confirm` observations.
  SlotStateTracker tracker(slots.size(), config);
  tracker.Update(1.0, {{1, SlotState::kOccupied, 100, 0.0}});
  tracker.Update(2.0, {{1, SlotState::kFree, 0, 0.8}});
  GR_EXPECT_TRUE(tracker.entries()[1].state == SlotState::kOccupied);
  tracker.Update(3.0, {{1, SlotState::kFree, 0, 0.8}});
  GR_EXPECT_TRUE(tracker.entries()[1].state == SlotState::kFree);
  GR_EXPECT_NEAR(tracker.entries()[1].stamp, 3.0, 0.0);
  GR_EXPECT_EQ(tracker.entries()[1].observations, 3);
  GR_EXPECT_TRUE(tracker.entries()[0].state == SlotState::kUnknown);
}

}  // namespace
}  // namespace gr
