// Road features of the route graph from labelled map points: barrier gates
// (a vehicle stops there; also a point of interest) and speed bumps (slow
// down). The points of one class on a level are clustered into footprints
// (8-connected cells of `cell` metres); each footprint is assigned to the
// nearest drivable edge that passes within `margin` of it, with the arc
// length where it does.

#ifndef GARAGE_RECON_GR_NAV_FEATURES_H_
#define GARAGE_RECON_GR_NAV_FEATURES_H_

#include <vector>

#include "gr/core/types.h"

namespace gr {

struct Footprint {
  Vec2 min = Vec2::Zero();
  Vec2 max = Vec2::Zero();
  Vec2 center = Vec2::Zero();  // Mean of its points.
  int points = 0;
};

// Footprints of 8-connected occupied cells with at least `min_points`
// points, largest first.
std::vector<Footprint> ObjectFootprints(const std::vector<Vec2>& xy,
                                        double cell, int min_points);

struct FootprintEdge {
  // A footprint or edge index that is not set.
  static constexpr int kNone = -1;

  int footprint = kNone;
  int edge = kNone;
  double along = 0.0;     // Arc length of the closest point (m).
  double distance = 0.0;  // From the footprint's box (0 inside).
};

// The nearest edge (polylines in xy) within `margin` of each footprint's
// box; footprints without one are left out.
std::vector<FootprintEdge> EdgesAtFootprints(
    const std::vector<Footprint>& footprints,
    const std::vector<std::vector<Vec2>>& edges, double margin);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_FEATURES_H_
