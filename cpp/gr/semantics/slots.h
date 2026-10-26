// Parking slots from vectorised markings (section 5, module 3).
//
// Separators are grouped into rows, consecutive separators with a plausible
// spacing form slots, gaps of k typical widths are split into k inferred
// slots (separators hidden by cars), and the entrance side is the side facing
// the driven aisle.
//
// Rows are often a little rotated against the structure grid (or the grid
// yaw is off outdoors, without a column grid): each row's direction is
// refined from its separator ends (Theil-Sen, kept only if significant) and
// the row is assembled in its own frame. Back-to-back rows are told from
// single rows by the 10 / 90 % quantiles of the separator ends, so a few long
// fragments do not split a single row.
//
// An inferred slot needs a reason for the missing line: a hypothetical
// separator on observed, empty floor without a paint ridge in the LiDAR
// intensity would have been seen, so such gaps are not split (a ridge means
// paint that vectorisation missed; hidden floor means a parked car). Inferred
// slots the vehicle drove through, that overlap a slot with paint, or that
// lie mostly on unobserved ground (behind walls) are dropped as well.
//
// Corner order: entrance-left, entrance-right, back-right, back-left
// (looking into the slot).

#ifndef GARAGE_RECON_GR_SEMANTICS_SLOTS_H_
#define GARAGE_RECON_GR_SEMANTICS_SLOTS_H_

#include <array>
#include <string>
#include <vector>

#include "gr/core/logging.h"
#include "gr/core/types.h"
#include "gr/semantics/markings.h"

namespace gr {

struct SlotConfig {
  double width_min = 2.2;  // Slot width range (m).
  double width_max = 3.4;
  double depth_min = 4.4;  // Slot depth range (m).
  double depth_max = 6.2;
  double min_separator_length = 1.0;  // Partly visible separators count.
  double typical_depth = 5.3;  // Completes rows whose back ends are hidden.
  double angle_tol = 10.0;     // deg off the row direction.
  double row_overlap = 0.6;
  double infer_tolerance = 0.25;
  double occupied_height = 0.5;
  double occupied_fraction = 0.25;
  // A hidden separator this much on seen, empty floor and with less paint
  // contrast than this share of the row's: no inferred slots there.
  double visible_separator_share = 0.7;
  double paint_ridge_share = 0.4;
  double inferred_max_overlap = 0.1;   // Area share with kept slots.
  double inferred_min_evidence = 0.5;  // Observed floor or a parked car.
  double column_overlap_area = 0.1;    // m^2 of a column inside a slot.
};

// The rasters of a level the slots need (row-major, `frame`); NaN where not
// measured. floor_count and intensity may be empty.
struct SlotRasters {
  RasterFrame frame;
  std::vector<float> height_max;
  std::vector<float> floor_z;
  std::vector<float> floor_count;
  std::vector<float> intensity;
};

struct SlotColumn {
  Vec2 center = Vec2::Zero();
  Vec2 size = Vec2::Zero();
};

struct ParkingSlot {
  std::string slot_id;  // "L<level>-R<row>-<index>".
  std::string row;      // "R<n>".
  int level = 0;
  std::array<Vec2, 4> corners;
  double z = 0.0;
  double width = 0.0;
  double depth = 0.0;
  int occupied = -1;  // At capture time: -1 not measured, 0 / 1.
  bool inferred = false;
  double confidence = 1.0;
};

std::vector<ParkingSlot> AssembleSlots(
    const std::vector<MarkingSegment>& segments, int level, double yaw,
    const std::vector<Vec2>& path_xy, const std::vector<SlotColumn>& columns,
    const SlotRasters& rasters, const SlotConfig& config = {},
    const Logger& log = nullptr);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_SLOTS_H_
