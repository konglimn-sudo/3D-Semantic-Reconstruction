// Where a localized vehicle is, in navigation terms: its level (the floor
// under it nearest to its origin), the nearest drivable edge of the route
// graph on that level (distance, position along it, lateral offset, heading
// relative to the edge direction) and the slot whose footprint contains it.

#ifndef GARAGE_RECON_GR_NAV_POSITION_H_
#define GARAGE_RECON_GR_NAV_POSITION_H_

#include <optional>
#include <string>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"

namespace gr {

struct NavEdge {
  int level = 0;
  std::string kind;  // aisle, lane, bridge, ramp, connector, ...
  std::vector<Vec2> points;
};

struct NavSlot {
  int level = 0;
  std::string slot_id;
  std::vector<Vec2> corners;
};

// Floor heights of a level (nav layer floor_z), row-major from (x0, y0);
// NaN where unknown.
struct FloorHeights {
  int level = 0;
  double x0 = 0.0;
  double y0 = 0.0;
  double res = 0.1;
  int width = 0;
  int height = 0;
  std::vector<double> z;

  // NaN outside the grid.
  double At(const Vec2& xy) const;
};

struct EdgePosition {
  static constexpr int kNoEdge = -1;
  int edge = kNoEdge;
  std::string kind;
  double distance = 0.0;  // m from the edge.
  double along = 0.0;     // m from its start.
  double lateral = 0.0;   // m, left of the edge direction.
  double heading_rel_deg = 0.0;  // Vehicle yaw minus the edge direction.
};

struct NavPosition {
  Vec3 xyz = Vec3::Zero();
  double yaw_deg = 0.0;
  std::optional<int> level;
  std::optional<EdgePosition> edge;
  std::optional<std::string> slot;
};

// Levels are tried in the order of `floors` (ties keep the first).
NavPosition Where(const Pose& map_from_vehicle,
                  const std::vector<NavEdge>& edges,
                  const std::vector<NavSlot>& slots,
                  const std::vector<FloorHeights>& floors);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_POSITION_H_
