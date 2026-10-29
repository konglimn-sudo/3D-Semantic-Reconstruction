#include "gr/nav/map_update.h"

#include <cmath>
#include <limits>
#include <string>

#include "gr/core/geometry.h"
#include "gr/nav/features.h"

namespace gr {
namespace {

// A change belongs to a floor up to this far (m) above its bottom: the
// floor under a new object's lowest returns.
constexpr double kFloorAboveBottom = 0.5;
// Slots are polygons of at least this many corners.
constexpr size_t kMinPolygonCorners = 3;

}  // namespace

std::vector<MapChange> AssessChanges(
    const std::vector<ChangeCluster>& clusters,
    const std::vector<NavEdge>& edges, const std::vector<NavSlot>& slots,
    const std::vector<FloorHeights>& floors, double half_width) {
  std::vector<MapChange> out;
  for (const ChangeCluster& cluster : clusters) {
    if (cluster.vehicle_like) continue;
    MapChange change;
    change.cluster = cluster;
    // The highest floor not above the cluster's bottom.
    double best = std::numeric_limits<double>::infinity();
    for (const FloorHeights& floor : floors) {
      const double z = floor.At(cluster.center.head<2>());
      if (!std::isfinite(z) || z > cluster.min.z() + kFloorAboveBottom) {
        continue;
      }
      const double gap = cluster.min.z() - z;
      if (gap < best) {
        best = gap;
        change.level = floor.level;
      }
    }
    bool in_slot = false;
    for (const NavSlot& slot : slots) {
      in_slot |= change.level && slot.level == *change.level &&
                 slot.corners.size() >= kMinPolygonCorners &&
                 PointInPolygon(slot.corners, cluster.center.head<2>());
    }
    if (in_slot) continue;
    if (change.level) {
      Footprint footprint;
      footprint.min = cluster.min.head<2>();
      footprint.max = cluster.max.head<2>();
      for (size_t index = 0; index < edges.size(); ++index) {
        const NavEdge& edge = edges[index];
        if (edge.level != *change.level || edge.kind == "connector" ||
            edge.points.size() < 2) {
          continue;
        }
        if (!EdgesAtFootprints({footprint}, {edge.points}, half_width)
                 .empty()) {
          change.edges.push_back(static_cast<int>(index));
        }
      }
    }
    out.push_back(change);
  }
  return out;
}

}  // namespace gr
