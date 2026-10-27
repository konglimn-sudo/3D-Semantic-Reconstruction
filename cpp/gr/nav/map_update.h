// What the structural changes a trip found (change_detection.h) mean for the
// navigation map: the level of each change (the floor below it) and the
// drivable edges whose vehicle corridor (`half_width` to either side) it
// reaches: an added obstacle there narrows or blocks the route, a removed
// one (a duct) changes the headroom. Vehicle-like clusters and clusters
// standing in a slot are left out: parked cars (also when seen only in
// part) are occupancy (slot_state.h), not map changes.

#ifndef GARAGE_RECON_GR_NAV_MAP_UPDATE_H_
#define GARAGE_RECON_GR_NAV_MAP_UPDATE_H_

#include <optional>
#include <vector>

#include "gr/mapping/change_detection.h"
#include "gr/nav/position.h"

namespace gr {

struct MapChange {
  ChangeCluster cluster;
  std::optional<int> level;
  std::vector<int> edges;  // Drivable edges whose corridor it reaches.
};

std::vector<MapChange> AssessChanges(
    const std::vector<ChangeCluster>& clusters,
    const std::vector<NavEdge>& edges, const std::vector<NavSlot>& slots,
    const std::vector<FloorHeights>& floors, double half_width);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_MAP_UPDATE_H_
