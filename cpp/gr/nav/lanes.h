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
// In open areas without slot rows (an apron, a lot along a wall) long walls
// propose lanes too: `wall_offset` from the wall face, on a side that stays
// navigable for `wall_open_width` beyond the lane (so a wall along a normal
// aisle, whose skeleton already runs midway, proposes nothing). Last, the
// survey drives propose their long straight stretches (Douglas-Peucker with
// `drive_tolerance`, at least `drive_min_length`): in an open lot without
// walls, where the vehicle drove is the best evidence of a lane. Earlier
// proposals win: later ones near a lane already taken are dropped.
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
  double wall_offset = 3.5;       // Wall lane from the wall face (m) ...
  double min_wall_length = 12.0;  // ... of walls at least this long (m) ...
  double wall_open_width = 8.0;   // ... with this much free beyond it (m).
  double drive_tolerance = 0.5;   // Drives simplified (m) ...
  double drive_min_length = 15.0;  // ... into straight stretches this long.
};

// A wall of the level (structure stage): centre line and thickness.
struct LaneWall {
  Vec2 p0 = Vec2::Zero();
  Vec2 p1 = Vec2::Zero();
  double thickness = 0.0;
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
  int walls = 0;     // Walls that proposed a lane.
  int drives = 0;    // Drive stretches that proposed a lane.
  int followed = 0;  // Pieces the skeleton already runs along.
  std::vector<std::array<Vec2, 2>> lanes;
  // Per lane: 0 slot row, 1 wall, 2 drive.
  std::vector<uint8_t> source;
};

// The lane pieces of one level: proposed lanes (slot rows, walls in open
// areas, then straight stretches of the drives) cut to the navigable cells
// (`navigable`, row-major in `frame`), away from lanes already taken, and
// not where the skeleton already runs along them.
LanePlan PlanLanes(const std::vector<LaneSlot>& slots,
                   const std::vector<uint8_t>& navigable,
                   const RasterFrame& frame,
                   const std::vector<std::vector<Vec2>>& skeleton,
                   const LaneConfig& config = {},
                   const std::vector<LaneWall>& walls = {},
                   const std::vector<std::vector<Vec2>>& drives = {});

// Whether a skeleton edge runs along one of the lanes.
bool CoveredByLanes(const std::vector<Vec2>& edge,
                    const std::vector<std::array<Vec2, 2>>& lanes,
                    const LaneConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_LANES_H_
