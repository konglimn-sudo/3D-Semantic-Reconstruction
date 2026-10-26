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
};

// A graph edge as the traffic rules see it (all edges of the graph, so that
// junction degrees are right; only drivable ones carry arrows).
struct TrafficEdge {
  int a = -1;
  int b = -1;
  int level = 0;
  bool drivable = false;     // Aisle or lane.
  std::vector<Vec2> points;  // From node a to node b.
};

struct TrafficArrow {
  int level = 0;
  Vec2 center = Vec2::Zero();
  Vec2 direction = Vec2::UnitX();  // Unit.
};

struct ArrowAssignment {
  int edge = -1;         // -1: no edge.
  int sign = 0;          // +1 along a -> b.
  double lateral = 0.0;  // Left of a -> b (m).
};

std::vector<ArrowAssignment> AssignArrows(
    const std::vector<TrafficEdge>& edges,
    const std::vector<TrafficArrow>& arrows, const TrafficConfig& config);

struct CorridorStep {
  int edge = -1;
  bool forward = true;  // Travelled a -> b along the corridor.
};

// Chains of drivable edges between junctions.
std::vector<std::vector<CorridorStep>> Corridors(
    const std::vector<TrafficEdge>& edges, double max_turn_deg);

enum class TrafficVerdict { kTwoWay, kOneLaneSeen, kOneArrow, kOneWay };

struct Corridor {
  std::vector<CorridorStep> steps;
  int level = 0;
  double length = 0.0;
  int arrows = 0;
  TrafficVerdict verdict = TrafficVerdict::kTwoWay;
  int travel = 0;  // One-way: +1 along the corridor's steps, -1 against.
};

// The corridors that carry arrows, with their verdicts.
std::vector<Corridor> DecideTraffic(const std::vector<TrafficEdge>& edges,
                                    const std::vector<TrafficArrow>& arrows,
                                    const TrafficConfig& config = {});

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_TRAFFIC_H_
