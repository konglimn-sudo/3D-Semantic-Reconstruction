// Traffic direction of the aisles from the floor arrows (arrows.h).
//
// An arrow belongs to the nearest drivable edge (aisle or lane) of its level
// that runs along it. Edges that continue each other without a junction
// form a corridor (the stretch between two intersections; slot connectors do
// not count as junctions). Per corridor:
//
//   arrows in both directions, or an opposite arrow    two-way;
//   within `opposite_radius` of its arrows
//   arrows off the centre line (one lane's arrows)      two-way (one lane);
//   fewer than `min_arrows`                             unknown;
//   otherwise                                           one-way.
//
// The skeleton wiggles at junctions, where a lane's arrow can fall on it; the
// count and the opposite arrows keep such stretches two-way.
//
// A single centred arrow (worn or hidden arrows) or none is completed by
// other evidence, unless any of it disagrees:
//
//   no-entry sign (禁止驶入) at a corridor end    one-way, out through that
//                                                 end (its facing, when
//                                                 known, picks the corridor
//                                                 at a junction);
//   one arrow and a one-way sign (单行), or one     one-way along the arrow;
//   arrow and `min_passes` passes of the survey
//   drives in its direction and none against
//   a one-way sign and such passes                one-way along the passes.
//
// Passes alone never make a corridor one-way: a two-way aisle may simply
// have been driven one way.

#ifndef GARAGE_RECON_GR_NAV_TRAFFIC_H_
#define GARAGE_RECON_GR_NAV_TRAFFIC_H_

#include <vector>

#include "gr/core/types.h"

namespace gr {

struct TrafficConfig {
  double arrow_max_distance = 3.5;  // Arrow to its edge (m) ...
  double arrow_max_angle = 30.0;    // ... running along it (deg).
  double center_tolerance = 0.8;    // One-way arrows sit this near the axis.
  int min_arrows = 2;
  double opposite_radius = 8.0;  // m
  double max_turn = 35.0;        // deg between edges of one corridor.
  // Driven passes: path steps within `pass_max_distance` of an edge and
  // within `pass_max_angle` of its direction; a pass covers
  // min(`pass_min_cover` of the edge, `pass_min_length`).
  double pass_max_distance = 2.5;  // m
  double pass_max_angle = 30.0;    // deg
  double pass_min_cover = 0.6;
  double pass_min_length = 10.0;   // m
  int min_passes = 2;
  double sign_distance = 8.0;      // m: a sign's corridor end ...
  double sign_lateral = 6.0;       // ... and offset from the corridor.
};

// A graph edge as the traffic rules see it (all edges of the graph, so that
// junction degrees are right; only drivable ones carry arrows).
struct TrafficEdge {
  int start_node = -1;
  int end_node = -1;
  int level = 0;
  bool drivable = false;     // Aisle or lane.
  std::vector<Vec2> points;  // From start_node to end_node.
};

struct TrafficArrow {
  int level = 0;
  Vec2 center = Vec2::Zero();
  Vec2 direction = Vec2::UnitX();  // Unit.
};

// One drive along a level, positions in time order.
struct DrivenPath {
  int level = 0;
  std::vector<Vec2> points;
};

struct TrafficSign {
  enum class Kind { kNoEntry, kOneWay };
  int level = 0;
  Vec2 xy = Vec2::Zero();
  Kind kind = Kind::kOneWay;
  Vec2 facing = Vec2::Zero();  // Towards its readers (unit); zero: unknown.
};

struct EdgePasses {
  int forward = 0;   // start_node -> end_node
  int backward = 0;
};

// Passes of the driven paths along every edge (drivable ones only).
std::vector<EdgePasses> CountPasses(const std::vector<TrafficEdge>& edges,
                                    const std::vector<DrivenPath>& paths,
                                    const TrafficConfig& config);

struct ArrowAssignment {
  static constexpr int kNoEdge = -1;
  int edge = kNoEdge;
  int sign = 0;          // +1 along start_node -> end_node, else -1.
  double lateral = 0.0;  // Left of start_node -> end_node (m).
};

std::vector<ArrowAssignment> AssignArrows(
    const std::vector<TrafficEdge>& edges,
    const std::vector<TrafficArrow>& arrows, const TrafficConfig& config);

struct CorridorStep {
  int edge = ArrowAssignment::kNoEdge;
  bool forward = true;  // Travelled start_node -> end_node.
};

// Chains of drivable edges between junctions.
std::vector<std::vector<CorridorStep>> Corridors(
    const std::vector<TrafficEdge>& edges, double max_turn_deg);

enum class TrafficVerdict {
  kTwoWay,
  kOneLaneSeen,
  kOneArrow,
  kOneWay,         // Arrows.
  kOneWayDriven,   // One arrow (or a one-way sign) and the driven passes.
  kOneWaySign,     // A no-entry sign, or one arrow and a one-way sign.
  kConflict,       // Evidence in both directions: two-way.
};

struct Corridor {
  std::vector<CorridorStep> steps;
  int level = 0;
  double length = 0.0;
  int arrows = 0;
  int passes_along = 0;    // Driven passes along the steps ...
  int passes_against = 0;  // ... and against them.
  int no_entry_signs = 0;
  int one_way_signs = 0;
  TrafficVerdict verdict = TrafficVerdict::kTwoWay;
  int travel = 0;  // One-way: +1 along the corridor's steps, -1 against.
};

// The corridors that carry arrows or signs, with their verdicts.
std::vector<Corridor> DecideTraffic(
    const std::vector<TrafficEdge>& edges,
    const std::vector<TrafficArrow>& arrows, const TrafficConfig& config = {},
    const std::vector<DrivenPath>& paths = {},
    const std::vector<TrafficSign>& signs = {});

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_TRAFFIC_H_
