#include "gr/nav/traffic.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <set>

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// Lengths below this (m) have no direction.
constexpr double kMinLength = 1e-12;
// Path steps shorter than this (m) are standing still.
constexpr double kMinStep = 1e-6;
// Edges continue a corridor through nodes with exactly this many edges.
constexpr size_t kCorridorNodeDegree = 2;
// Arrows facing each other: their directions' cosine below this.
constexpr double kOppositeMaxCos = -0.7;
// A sign with a known facing scores this many metres per unit of
// misalignment (1 - |cos|) with a corridor.
constexpr double kFacingWeight = 10.0;
// No corridor / edge chosen yet.
constexpr int kNone = -1;

// Distance of `xy` to a polyline, the unit tangent (start to end) of the
// nearest segment and the signed lateral offset (left of it).
struct Projection {
  double distance = kInf;
  Vec2 tangent = Vec2::UnitX();
  double lateral = 0.0;
  double along = 0.0;  // Arc length of the nearest point.
};

Projection Project(const std::vector<Vec2>& points, const Vec2& xy) {
  Projection best;
  double start = 0.0;  // Arc length at the segment start.
  for (size_t k = 0; k + 1 < points.size(); ++k) {
    const Vec2 segment = points[k + 1] - points[k];
    const double length = std::max(segment.norm(), kMinLength);
    const double fraction = std::clamp(
        (xy - points[k]).dot(segment) / (length * length), 0.0, 1.0);
    const Vec2 foot = points[k] + fraction * segment;
    const double distance = (xy - foot).norm();
    if (distance < best.distance) {
      best.distance = distance;
      best.along = start + fraction * segment.norm();
      best.tangent = segment / length;
      const Vec2 offset = xy - foot;
      best.lateral = best.tangent.x() * offset.y() -
                     best.tangent.y() * offset.x();
    }
    start += segment.norm();
  }
  return best;
}

Vec2 Unit(const Vec2& vector) {
  return vector / std::max(vector.norm(), kMinLength);
}

double Length(const std::vector<Vec2>& points) {
  double total = 0.0;
  for (size_t k = 0; k + 1 < points.size(); ++k) {
    total += (points[k + 1] - points[k]).norm();
  }
  return total;
}

}  // namespace

std::vector<ArrowAssignment> AssignArrows(
    const std::vector<TrafficEdge>& edges,
    const std::vector<TrafficArrow>& arrows, const TrafficConfig& config) {
  const double cos_max = std::cos(DegToRad(config.arrow_max_angle));
  std::vector<ArrowAssignment> out(arrows.size());
  for (size_t arrow = 0; arrow < arrows.size(); ++arrow) {
    double best = kInf;
    for (size_t index = 0; index < edges.size(); ++index) {
      const TrafficEdge& edge = edges[index];
      if (edge.level != arrows[arrow].level || !edge.drivable ||
          edge.points.size() < 2) {
        continue;
      }
      const Projection projection = Project(edge.points, arrows[arrow].center);
      const double alignment =
          arrows[arrow].direction.dot(projection.tangent);
      if (projection.distance > config.arrow_max_distance ||
          std::abs(alignment) < cos_max || projection.distance >= best) {
        continue;
      }
      best = projection.distance;
      out[arrow] = {static_cast<int>(index), alignment > 0.0 ? 1 : -1,
                    projection.lateral};
    }
  }
  return out;
}

std::vector<std::vector<CorridorStep>> Corridors(
    const std::vector<TrafficEdge>& edges, double max_turn_deg) {
  std::map<int, std::vector<int>> incident;
  std::vector<int> drivable;
  for (size_t index = 0; index < edges.size(); ++index) {
    if (!edges[index].drivable || edges[index].points.size() < 2) continue;
    drivable.push_back(static_cast<int>(index));
    incident[edges[index].start_node].push_back(static_cast<int>(index));
    incident[edges[index].end_node].push_back(static_cast<int>(index));
  }
  const double cos_max = std::cos(DegToRad(max_turn_deg));
  // Unit direction of travel leaving an edge at its start or end node.
  const auto leaving = [&](const TrafficEdge& edge, bool at_end) {
    const std::vector<Vec2>& points = edge.points;
    return Unit(at_end ? points[points.size() - 1] - points[points.size() - 2]
                       : points[0] - points[1]);
  };
  std::set<int> visited;
  std::vector<std::vector<CorridorStep>> chains;
  for (int start : drivable) {
    if (visited.count(start)) continue;
    visited.insert(start);
    std::deque<CorridorStep> chain = {{start, true}};
    for (bool ahead : {true, false}) {
      int index = start;
      bool forward = true;
      while (true) {
        const TrafficEdge& edge = edges[static_cast<size_t>(index)];
        const bool leaving_end = forward == ahead;
        const int node = leaving_end ? edge.end_node : edge.start_node;
        const std::vector<int>& at_node = incident[node];
        if (at_node.size() != kCorridorNodeDegree) break;
        const int next = at_node[0] == index ? at_node[1] : at_node[0];
        if (next == index || visited.count(next)) break;
        const TrafficEdge& other = edges[static_cast<size_t>(next)];
        const Vec2 out = leaving(edge, leaving_end);
        const std::vector<Vec2>& other_points = other.points;
        const Vec2 into = Unit(other.start_node == node
                                   ? other_points[1] - other_points[0]
                                   : other_points[other_points.size() - 2] -
                                         other_points.back());
        if (out.dot(into) < cos_max) break;
        visited.insert(next);
        const bool next_forward = (other.start_node == node) == ahead;
        if (ahead) {
          chain.push_back({next, next_forward});
        } else {
          chain.push_front({next, next_forward});
        }
        index = next;
        forward = next_forward;
      }
    }
    chains.emplace_back(chain.begin(), chain.end());
  }
  return chains;
}

std::vector<EdgePasses> CountPasses(const std::vector<TrafficEdge>& edges,
                                    const std::vector<DrivenPath>& paths,
                                    const TrafficConfig& config) {
  std::vector<EdgePasses> out(edges.size());
  const double cos_max = std::cos(DegToRad(config.pass_max_angle));
  for (const DrivenPath& path : paths) {
    // Runs of consecutive steps on one edge in one direction: their extent
    // along the edge.
    int run_edge = kNone;
    int run_sign = 0;
    double lo = 0.0;
    double hi = 0.0;
    const auto close_run = [&] {
      if (run_edge == kNone) return;
      const double length =
          Length(edges[static_cast<size_t>(run_edge)].points);
      if (hi - lo >= std::min(config.pass_min_cover * length,
                              config.pass_min_length)) {
        EdgePasses& passes = out[static_cast<size_t>(run_edge)];
        (run_sign > 0 ? passes.forward : passes.backward) += 1;
      }
      run_edge = kNone;
    };
    for (size_t k = 0; k + 1 < path.points.size(); ++k) {
      const Vec2 step = path.points[k + 1] - path.points[k];
      if (step.norm() < kMinStep) continue;
      const Vec2 mid = 0.5 * (path.points[k] + path.points[k + 1]);
      int best = kNone;
      int sign = 0;
      double best_distance = config.pass_max_distance;
      double along = 0.0;
      for (size_t index = 0; index < edges.size(); ++index) {
        const TrafficEdge& edge = edges[index];
        if (!edge.drivable || edge.level != path.level ||
            edge.points.size() < 2) {
          continue;
        }
        const Projection projection = Project(edge.points, mid);
        const double alignment = Unit(step).dot(projection.tangent);
        if (projection.distance > best_distance ||
            std::abs(alignment) < cos_max) {
          continue;
        }
        best = static_cast<int>(index);
        best_distance = projection.distance;
        sign = alignment > 0.0 ? 1 : -1;
        along = projection.along;
      }
      if (best != run_edge || sign != run_sign) {
        close_run();
        if (best == kNone) continue;
        run_edge = best;
        run_sign = sign;
        lo = hi = along;
      } else {
        lo = std::min(lo, along);
        hi = std::max(hi, along);
      }
    }
    close_run();
  }
  return out;
}

std::vector<Corridor> DecideTraffic(const std::vector<TrafficEdge>& edges,
                                    const std::vector<TrafficArrow>& arrows,
                                    const TrafficConfig& config,
                                    const std::vector<DrivenPath>& paths,
                                    const std::vector<TrafficSign>& signs) {
  const std::vector<ArrowAssignment> assigned =
      AssignArrows(edges, arrows, config);
  std::map<int, std::vector<int>> arrows_by_edge;
  for (size_t arrow = 0; arrow < assigned.size(); ++arrow) {
    if (assigned[arrow].edge != ArrowAssignment::kNoEdge) {
      arrows_by_edge[assigned[arrow].edge].push_back(static_cast<int>(arrow));
    }
  }
  const std::vector<EdgePasses> passes = CountPasses(edges, paths, config);
  const std::vector<std::vector<CorridorStep>> chains =
      Corridors(edges, config.max_turn);
  // Signs to corridors: the corridor whose polyline the sign is nearest to
  // (within `sign_lateral`; a no-entry sign also within `sign_distance` of
  // one of its ends); with a known facing, the corridor most aligned with
  // it. Each sign votes for travel +1 / -1 along the chosen corridor's steps
  // (no-entry) or just for one-way.
  struct SignVote {
    int travel;  // 0 for a one-way sign.
  };
  std::vector<std::vector<SignVote>> sign_votes(chains.size());
  for (const TrafficSign& sign : signs) {
    int best = kNone;
    int best_travel = 0;
    double best_score = kInf;
    for (size_t chain = 0; chain < chains.size(); ++chain) {
      // The corridor as one polyline in step order.
      std::vector<Vec2> line;
      int level = 0;
      for (const CorridorStep& step : chains[chain]) {
        const TrafficEdge& edge = edges[static_cast<size_t>(step.edge)];
        level = edge.level;
        std::vector<Vec2> points = edge.points;
        if (!step.forward) std::reverse(points.begin(), points.end());
        line.insert(line.end(), points.begin() + (line.empty() ? 0 : 1),
                    points.end());
      }
      if (level != sign.level || line.size() < 2) continue;
      const Projection projection = Project(line, sign.xy);
      const double length = Length(line);
      const bool near_start = projection.along <= config.sign_distance;
      const bool near_end = length - projection.along <= config.sign_distance;
      // A no-entry sign stands at an end; a one-way sign anywhere along.
      if (projection.distance > config.sign_lateral ||
          (sign.kind == TrafficSign::Kind::kNoEntry && !near_start &&
           !near_end)) {
        continue;
      }
      double score = projection.distance;
      if (sign.facing.squaredNorm() > 0.0) {
        score += kFacingWeight *
                 (1.0 - std::abs(sign.facing.dot(projection.tangent)));
      }
      if (score >= best_score) continue;
      best_score = score;
      best = static_cast<int>(chain);
      if (sign.kind == TrafficSign::Kind::kOneWay) {
        best_travel = 0;
      } else if (sign.facing.squaredNorm() > 0.0) {
        // It faces the traffic it stops: travel goes towards its readers.
        best_travel = sign.facing.dot(projection.tangent) > 0.0 ? 1 : -1;
      } else {
        // Out through the end it stands at.
        best_travel = near_end && !near_start ? 1
                      : near_start && !near_end
                          ? -1
                          : (projection.along > 0.5 * length ? 1 : -1);
      }
    }
    if (best != kNone) {
      sign_votes[static_cast<size_t>(best)].push_back(
          {sign.kind == TrafficSign::Kind::kOneWay ? 0 : best_travel});
    }
  }
  std::vector<Corridor> out;
  for (size_t index = 0; index < chains.size(); ++index) {
    const std::vector<CorridorStep>& chain = chains[index];
    struct ArrowVote {
      int sign;
      double lateral;
      int arrow;
    };
    std::vector<ArrowVote> votes;
    Corridor corridor;
    for (const CorridorStep& step : chain) {
      const EdgePasses& edge_passes = passes[static_cast<size_t>(step.edge)];
      corridor.passes_along +=
          step.forward ? edge_passes.forward : edge_passes.backward;
      corridor.passes_against +=
          step.forward ? edge_passes.backward : edge_passes.forward;
      const auto found = arrows_by_edge.find(step.edge);
      if (found == arrows_by_edge.end()) continue;
      const int factor = step.forward ? 1 : -1;
      for (int arrow : found->second) {
        const ArrowAssignment& assignment =
            assigned[static_cast<size_t>(arrow)];
        votes.push_back(
            {assignment.sign * factor, assignment.lateral * factor, arrow});
      }
    }
    int no_entry_travel = 0;
    bool no_entry_conflict = false;
    for (const SignVote& vote : sign_votes[index]) {
      if (vote.travel == 0) {
        ++corridor.one_way_signs;
        continue;
      }
      ++corridor.no_entry_signs;
      if (no_entry_travel != 0 && vote.travel != no_entry_travel) {
        no_entry_conflict = true;
      }
      no_entry_travel = vote.travel;
    }
    if (votes.empty() && sign_votes[index].empty()) continue;
    bool plus = false;
    bool minus = false;
    bool centred = true;
    bool opposite = false;
    for (const ArrowVote& vote : votes) {
      (vote.sign > 0 ? plus : minus) = true;
      centred &= std::abs(vote.lateral) <= config.center_tolerance;
      const TrafficArrow& mine = arrows[static_cast<size_t>(vote.arrow)];
      for (const TrafficArrow& other : arrows) {
        if (other.level == mine.level &&
            (other.center - mine.center).norm() <= config.opposite_radius &&
            other.direction.dot(mine.direction) < kOppositeMaxCos) {
          opposite = true;
        }
      }
    }
    corridor.steps = chain;
    corridor.level = edges[static_cast<size_t>(chain[0].edge)].level;
    for (const CorridorStep& step : chain) {
      corridor.length += Length(edges[static_cast<size_t>(step.edge)].points);
    }
    corridor.arrows = static_cast<int>(votes.size());
    // Travel direction the passes support (0: none or both).
    int driven = 0;
    if (corridor.passes_along >= config.min_passes &&
        corridor.passes_against == 0) {
      driven = 1;
    } else if (corridor.passes_against >= config.min_passes &&
               corridor.passes_along == 0) {
      driven = -1;
    }
    const bool any_against = [&](int travel) {
      return travel > 0 ? corridor.passes_against > 0
                        : corridor.passes_along > 0;
    }(no_entry_travel);
    const int arrow = votes.empty() ? 0 : votes[0].sign;
    if ((plus && minus) || opposite) {
      corridor.verdict = TrafficVerdict::kTwoWay;
    } else if (!centred) {
      corridor.verdict = TrafficVerdict::kOneLaneSeen;
    } else if (static_cast<int>(votes.size()) >= config.min_arrows) {
      corridor.verdict = no_entry_travel != 0 && no_entry_travel != arrow
                             ? TrafficVerdict::kConflict
                             : TrafficVerdict::kOneWay;
      corridor.travel = arrow;
    } else if (no_entry_travel != 0) {
      if (no_entry_conflict || any_against ||
          (arrow != 0 && arrow != no_entry_travel)) {
        corridor.verdict = TrafficVerdict::kConflict;
      } else {
        corridor.verdict = TrafficVerdict::kOneWaySign;
        corridor.travel = no_entry_travel;
      }
    } else if (arrow != 0 && corridor.one_way_signs > 0) {
      const bool against = arrow > 0 ? corridor.passes_against > 0
                                     : corridor.passes_along > 0;
      corridor.verdict =
          against ? TrafficVerdict::kConflict : TrafficVerdict::kOneWaySign;
      corridor.travel = against ? 0 : arrow;
    } else if (arrow != 0 && driven == arrow) {
      corridor.verdict = TrafficVerdict::kOneWayDriven;
      corridor.travel = arrow;
    } else if (arrow == 0 && corridor.one_way_signs > 0 && driven != 0) {
      corridor.verdict = TrafficVerdict::kOneWayDriven;
      corridor.travel = driven;
    } else if (arrow != 0) {
      corridor.verdict = TrafficVerdict::kOneArrow;
    } else {
      corridor.verdict = TrafficVerdict::kTwoWay;
    }
    out.push_back(std::move(corridor));
  }
  return out;
}

}  // namespace gr
