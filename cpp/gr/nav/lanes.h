// Lanes along the rows of slots (geometry of the Python stage nav/lanes.py,
// which edits the route graph).
//
// The aisle skeleton follows the middle of the free space. Between two rows
// that is the lane, but in an open lot (one row facing open ground, a wide
// apron) the skeleton wanders through the open space and forms a web, away
// from where vehicles drive. Every row of slots therefore proposes a lane
// parallel to its entrance line:
//
//   facing rows   rows whose entrances face each other `min_aisle` ..
//                 `max_aisle` apart: the lane runs midway;
//   single row    `lane_offset` in front of the entrance line.
//
// Lane pieces must stay on navigable cells (the caller's mask: drivable,
// outside slots and ramps, with the vehicle's clearance). Pieces the
// skeleton already runs along (`skip_followed` of them within
// `follow_distance`) are not needed; skeleton edges that run along a lane
// (within `replace_distance` and `replace_angle`, over `replace_share` of
// their length) give way to it.

#ifndef GARAGE_RECON_GR_NAV_LANES_H_
#define GARAGE_RECON_GR_NAV_LANES_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "gr/core/types.h"
#include "gr/semantics/markings.h"

namespace gr {

struct LaneConfig {
  double max_row_angle = 10.0;  // Facing rows: entrance normals opposite ...
  double min_aisle = 4.0;       // ... and their entrance lines this far
  double max_aisle = 10.0;      // apart (m) ...
  double min_overlap = 3.0;     // ... overlapping along the rows (m).
  double lane_offset = 3.0;     // Single row: lane in front (m).
  double min_length = 4.0;      // Shorter pieces are dropped (m).
  double dedupe_distance = 1.5;  // Pieces near a taken lane are dropped (m).
  double follow_distance = 0.5;
  double skip_followed = 0.8;
  double replace_distance = 3.0;
  double replace_angle = 30.0;  // deg
  double replace_share = 0.5;
};

// A slot of the level (corners: entrance-left, entrance-right, back-right,
// back-left).
struct LaneSlot {
  std::string row;
  std::array<Vec2, 4> corners;
  double z = 0.0;
  double width = 0.0;
};

struct LaneRow {
  Vec2 u = Vec2::UnitX();  // Along the row.
  Vec2 n = Vec2::UnitY();  // From the slots towards the aisle.
  double offset = 0.0;     // Entrance line at n . x = offset ...
  double lo = 0.0;         // ... spanning u . x in [lo, hi].
  double hi = 0.0;
  double z = 0.0;
  int slots = 0;
};

// Rows of the slots by their row id, in order of first appearance.
std::vector<LaneRow> SlotRows(const std::vector<LaneSlot>& slots);

// Proposed lanes, facing pairs first: end points.
std::vector<std::array<Vec2, 2>> LaneLines(const std::vector<LaneRow>& rows,
                                           const LaneConfig& config);

struct LanePlan {
  int rows = 0;
  int followed = 0;  // Pieces the skeleton already runs along.
  std::vector<std::array<Vec2, 2>> lanes;
};

// The lane pieces of one level: proposed lanes cut to the navigable cells
// (`navigable`, row-major in `frame`), away from lanes already taken, and
// not where the skeleton already runs along them.
LanePlan PlanLanes(const std::vector<LaneSlot>& slots,
                   const std::vector<uint8_t>& navigable,
                   const RasterFrame& frame,
                   const std::vector<std::vector<Vec2>>& skeleton,
                   const LaneConfig& config = {});

// Whether a skeleton edge runs along one of the lanes.
bool CoveredByLanes(const std::vector<Vec2>& edge,
                    const std::vector<std::array<Vec2, 2>>& lanes,
                    const LaneConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_LANES_H_
