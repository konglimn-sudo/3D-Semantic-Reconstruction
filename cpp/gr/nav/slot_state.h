// Live occupancy of the parking slots from a localized vehicle's own sweeps
// (the map only knows the capture time).
//
// Per sweep (map frame), each slot within `max_range` of the sensor is
// looked at inside its outline shrunk by `inset`:
//
//   occupied   at least `min_body_points` returns `min_height` ..
//              `max_height` above the slot floor (a car body);
//   free       fewer than `max_free_points` such returns, and floor returns
//              in at least `min_floor_share` of its `cell`-sized cells (the
//              floor is seen, not hidden behind a neighbour);
//   unknown    otherwise (too far, occluded, or in between).
//
// A tracker keeps the latest state per slot and flips a known state only
// after `confirm` consecutive observations of the other one (a passing car
// or a pedestrian does not free an occupied slot for one sweep).

#ifndef GARAGE_RECON_GR_NAV_SLOT_STATE_H_
#define GARAGE_RECON_GR_NAV_SLOT_STATE_H_

#include <array>
#include <string>
#include <vector>

#include "gr/core/types.h"

namespace gr {

struct SlotStateConfig {
  double max_range = 15.0;  // m from the sensor (horizontal).
  double inset = 0.3;       // m inside the outline.
  double min_height = 0.3;  // m above the slot floor ...
  double max_height = 2.2;  // ... for car-body returns.
  double floor_band = 0.1;  // |height| of floor returns (m).
  int min_body_points = 25;
  int max_free_points = 3;
  double cell = 0.25;  // m, floor coverage cells.
  double min_floor_share = 0.25;
  int confirm = 2;
};

// A parking slot is a quadrilateral.
inline constexpr int kSlotCorners = 4;

struct SlotGeometry {
  std::string slot_id;
  int level = 0;
  std::array<Vec2, kSlotCorners> corners;
  double z = 0.0;  // Floor height.
};

enum class SlotState { kUnknown = 0, kFree = 1, kOccupied = 2 };
// The number of SlotState values.
inline constexpr int kSlotStateCount = 3;

struct SlotObservation {
  int slot = -1;
  SlotState state = SlotState::kUnknown;
  int body_points = 0;
  double floor_share = 0.0;
};

// The slots within range of `sensor` and what one sweep (map frame) says
// about each.
std::vector<SlotObservation> ObserveSlots(
    const Points& map_points, const Vec3& sensor,
    const std::vector<SlotGeometry>& slots, const SlotStateConfig& config);

class SlotStateTracker {
 public:
  struct Entry {
    SlotState state = SlotState::kUnknown;
    double stamp = 0.0;  // Of the last observation that was not unknown.
    int observations = 0;
  };

  SlotStateTracker(size_t num_slots, const SlotStateConfig& config)
      : config_(config), entries_(num_slots), pending_(num_slots) {}

  void Update(double stamp, const std::vector<SlotObservation>& observed);
  const std::vector<Entry>& entries() const { return entries_; }

 private:
  struct Pending {
    SlotState state = SlotState::kUnknown;
    int count = 0;
  };
  SlotStateConfig config_;
  std::vector<Entry> entries_;
  std::vector<Pending> pending_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_SLOT_STATE_H_
