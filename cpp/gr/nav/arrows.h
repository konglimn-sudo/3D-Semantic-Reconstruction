// Floor arrows in a plan-view paint mask (one level).
//
// Each connected component of painted cells is described by its principal
// axis and its width profile along that axis (painted area per bin / bin
// length, robust to the gaps between LiDAR rings). An arrow is a narrow
// shaft with a head at one end: the widest part of the profile, at least
// `min_head_ratio` times the shaft (the median width), tapering to a tip
// beyond it. Parking lines (constant width) and line junctions (the widest
// part at the very end, no tip) fail these tests. The direction points from
// the shaft to the head; curved or forked arrows (turns) give their main
// axis.

#ifndef GARAGE_RECON_GR_NAV_ARROWS_H_
#define GARAGE_RECON_GR_NAV_ARROWS_H_

#include <cstdint>
#include <vector>

#include "gr/core/types.h"

namespace gr {

struct ArrowConfig {
  double min_length = 1.8;  // Extent along the principal axis (m).
  double max_length = 7.0;
  double min_area = 0.2;  // Painted area (m^2).
  double max_area = 5.0;
  double max_shaft_width = 0.45;  // m
  double min_head_width = 0.4;    // m
  double min_head_ratio = 2.0;    // Head width / shaft width.
  // The head must lie in the outer part of the length: its widest bin at
  // least this fraction of the length away from the middle ...
  double min_head_offset = 0.15;
  // ... and taper over at least this length beyond it (m).
  double min_tip_length = 0.25;
  double bin = 0.1;  // Width profile bin (m).
};

struct FloorArrow {
  Vec2 center = Vec2::Zero();     // Painted centroid (m).
  Vec2 direction = Vec2::UnitX(); // Unit, from the shaft to the head.
  Vec2 tail = Vec2::Zero();       // Ends along the axis (m).
  Vec2 tip = Vec2::Zero();
  double length = 0.0;
  double shaft_width = 0.0;
  double head_width = 0.0;
  double area = 0.0;
  std::vector<int> cells;  // Mask indices (row * width + col) of its paint.
};

// `mask` is row-major (height x width); cell (row, col) has its centre at
// (x0 + (col + 0.5) res, y0 + (row + 0.5) res).
std::vector<FloorArrow> DetectArrows(const std::vector<uint8_t>& mask,
                                     int width, int height, double res,
                                     double x0, double y0,
                                     const ArrowConfig& config = {});

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_ARROWS_H_
