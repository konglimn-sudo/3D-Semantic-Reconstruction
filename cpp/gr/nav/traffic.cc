#include "gr/nav/traffic.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <set>

namespace gr {
namespace {

// Distance of `xy` to a polyline, the unit tangent (a -> b) of the nearest
// segment and the signed lateral offset (left of a -> b).
struct Projection {
  double distance = std::numeric_limits<double>::infinity();
  Vec2 tangent = Vec2::UnitX();
  double lateral = 0.0;
  double along = 0.0;  // Arc length of the nearest point.
};

Projection Project(const std::vector<Vec2>& points, const Vec2& xy) {
  Projection best;
  double start = 0.0;
  for (size_t k = 0; k + 1 < points.size(); ++k) {
    const Vec2 seg = points[k + 1] - points[k];
    const double length = std::max(seg.norm(), 1e-12);
    const double t = std::clamp((xy - points[k]).dot(seg) / (length * length),
                                0.0, 1.0);
    const Vec2 foot = points[k] + t * seg;
    const double distance = (xy - foot).norm();
    if (distance < best.distance) {
      best.distance = distance;
      best.along = start + t * seg.norm();
      best.tangent = seg / length;
      const Vec2 offset = xy - foot;
      best.lateral = best.tangent.x() * offset.y() -
                     best.tangent.y() * offset.x();
    }
    start += seg.norm();
  }
  return best;
}

Vec2 Unit(const Vec2& v) { return v / std::max(v.norm(), 1e-12); }

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
  for (size_t i = 0; i < arrows.size(); ++i) {
    double best = std::numeric_limits<double>::infinity();
    for (size_t k = 0; k < edges.size(); ++k) {
      const TrafficEdge& edge = edges[k];
      if (edge.level != arrows[i].level || !edge.drivable ||
          edge.points.size() < 2) {
        continue;
      }
      const Projection p = Project(edge.points, arrows[i].center);
      const double alignment = arrows[i].direction.dot(p.tangent);
      if (p.distance > config.arrow_max_distance ||
          std::abs(alignment) < cos_max || p.distance >= best) {
        continue;
      }
      best = p.distance;
      out[i] = {static_cast<int>(k), alignment > 0.0 ? 1 : -1, p.lateral};
    }
  }
  return out;
}

std::vector<std::vector<CorridorStep>> Corridors(
    const std::vector<TrafficEdge>& edges, double max_turn_deg) {
  std::map<int, std::vector<int>> incident;
  std::vector<int> drivable;
  for (size_t k = 0; k < edges.size(); ++k) {
    if (!edges[k].drivable || edges[k].points.size() < 2) continue;
    drivable.push_back(static_cast<int>(k));
    incident[edges[k].a].push_back(static_cast<int>(k));
    incident[edges[k].b].push_back(static_cast<int>(k));
  }
  const double cos_max = std::cos(DegToRad(max_turn_deg));
  // Unit direction of travel leaving an edge at its a or b end.
  auto leaving = [&](const TrafficEdge& e, bool at_b) {
    const std::vector<Vec2>& p = e.points;
    return Unit(at_b ? p[p.size() - 1] - p[p.size() - 2] : p[0] - p[1]);
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
        const bool leaving_b = forward == ahead;
        const int node = leaving_b ? edge.b : edge.a;
        const std::vector<int>& at = incident[node];
        if (at.size() != 2) break;
        const int next = at[0] == index ? at[1] : at[0];
        if (next == index || visited.count(next)) break;
        const TrafficEdge& other = edges[static_cast<size_t>(next)];
        const Vec2 out = leaving(edge, leaving_b);
        const std::vector<Vec2>& q = other.points;
        const Vec2 into = Unit(other.a == node ? q[1] - q[0]
                                               : q[q.size() - 2] - q.back());
        if (out.dot(into) < cos_max) break;
        visited.insert(next);
        const bool next_forward = (other.a == node) == ahead;
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
    int run_edge = -1;
    int run_sign = 0;
    double lo = 0.0;
    double hi = 0.0;
    const auto close_run = [&] {
      if (run_edge < 0) return;
      const double length = Length(edges[static_cast<size_t>(run_edge)].points);
      if (hi - lo >= std::min(config.pass_min_cover * length,
                              config.pass_min_length)) {
        EdgePasses& passes = out[static_cast<size_t>(run_edge)];
        (run_sign > 0 ? passes.forward : passes.backward) += 1;
      }
      run_edge = -1;
    };
    for (size_t k = 0; k + 1 < path.points.size(); ++k) {
      const Vec2 step = path.points[k + 1] - path.points[k];
      if (step.norm() < 1e-6) continue;
      const Vec2 mid = 0.5 * (path.points[k] + path.points[k + 1]);
      int best = -1;
      int sign = 0;
      double best_distance = config.pass_max_distance;
      double along = 0.0;
      for (size_t e = 0; e < edges.size(); ++e) {
        const TrafficEdge& edge = edges[e];
        if (!edge.drivable || edge.level != path.level ||
            edge.points.size() < 2) {
          continue;
        }
        const Projection p = Project(edge.points, mid);
        const double alignment = Unit(step).dot(p.tangent);
        if (p.distance > best_distance || std::abs(alignment) < cos_max) {
          continue;
        }
        best = static_cast<int>(e);
        best_distance = p.distance;
        sign = alignment > 0.0 ? 1 : -1;
        along = p.along;
      }
      if (best != run_edge || sign != run_sign) {
        close_run();
        if (best < 0) continue;
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
  std::map<int, std::vector<int>> by_edge;  // Edge -> arrows.
  for (size_t i = 0; i < assigned.size(); ++i) {
    if (assigned[i].edge >= 0) {
      by_edge[assigned[i].edge].push_back(static_cast<int>(i));
    }
  }
  const std::vector<EdgePasses> passes = CountPasses(edges, paths, config);
  const std::vector<std::vector<CorridorStep>> chains =
      Corridors(edges, config.max_turn);
  // Signs to corridors: the corridor whose polyline the sign is nearest to
  // (within `sign_lateral`; a no-entry sign also within `sign_distance` of
  // one of its ends); with a known facing, the corridor most aligned with
  // it. Each sign
  // votes for travel +1 / -1 along the chosen corridor's steps (no-entry)
  // or just for one-way.
  struct SignVote {
    int travel;  // 0 for a one-way sign.
  };
  std::vector<std::vector<SignVote>> sign_votes(chains.size());
  for (const TrafficSign& sign : signs) {
    int best = -1;
    int best_travel = 0;
    double best_score = std::numeric_limits<double>::infinity();
    for (size_t c = 0; c < chains.size(); ++c) {
      // The corridor as one polyline in step order.
      std::vector<Vec2> line;
      int level = 0;
      for (const CorridorStep& step : chains[c]) {
        const TrafficEdge& edge = edges[static_cast<size_t>(step.edge)];
        level = edge.level;
        std::vector<Vec2> points = edge.points;
        if (!step.forward) std::reverse(points.begin(), points.end());
        line.insert(line.end(), points.begin() + (line.empty() ? 0 : 1),
                    points.end());
      }
      if (level != sign.level || line.size() < 2) continue;
      const Projection p = Project(line, sign.xy);
      const double length = Length(line);
      const bool near_start = p.along <= config.sign_distance;
      const bool near_end = length - p.along <= config.sign_distance;
      // A no-entry sign stands at an end; a one-way sign anywhere along.
      if (p.distance > config.sign_lateral ||
          (sign.kind == TrafficSign::Kind::kNoEntry && !near_start &&
           !near_end)) {
        continue;
      }
      double score = p.distance;
      if (sign.facing.squaredNorm() > 0.0) {
        score += 10.0 * (1.0 - std::abs(sign.facing.dot(p.tangent)));
      }
      if (score >= best_score) continue;
      best_score = score;
      best = static_cast<int>(c);
      if (sign.kind == TrafficSign::Kind::kOneWay) {
        best_travel = 0;
      } else if (sign.facing.squaredNorm() > 0.0) {
        // It faces the traffic it stops: travel goes towards its readers.
        best_travel = sign.facing.dot(p.tangent) > 0.0 ? 1 : -1;
      } else {
        // Out through the end it stands at.
        best_travel = near_end && !near_start ? 1
                      : near_start && !near_end
                          ? -1
                          : (p.along > 0.5 * length ? 1 : -1);
      }
    }
    if (best >= 0) {
      sign_votes[static_cast<size_t>(best)].push_back(
          {sign.kind == TrafficSign::Kind::kOneWay ? 0 : best_travel});
    }
  }
  std::vector<Corridor> out;
  for (size_t c = 0; c < chains.size(); ++c) {
    const std::vector<CorridorStep>& chain = chains[c];
    struct Vote {
      int sign;
      double lateral;
      int arrow;
    };
    std::vector<Vote> votes;
    Corridor corridor;
    for (const CorridorStep& step : chain) {
      const EdgePasses& p = passes[static_cast<size_t>(step.edge)];
      corridor.passes_along += step.forward ? p.forward : p.backward;
      corridor.passes_against += step.forward ? p.backward : p.forward;
      const auto found = by_edge.find(step.edge);
      if (found == by_edge.end()) continue;
      const int factor = step.forward ? 1 : -1;
      for (int i : found->second) {
        votes.push_back({assigned[static_cast<size_t>(i)].sign * factor,
                         assigned[static_cast<size_t>(i)].lateral * factor,
                         i});
      }
    }
    int no_entry_travel = 0;
    bool no_entry_conflict = false;
    for (const SignVote& v : sign_votes[c]) {
      if (v.travel == 0) {
        ++corridor.one_way_signs;
        continue;
      }
      ++corridor.no_entry_signs;
      if (no_entry_travel != 0 && v.travel != no_entry_travel) {
        no_entry_conflict = true;
      }
      no_entry_travel = v.travel;
    }
    if (votes.empty() && sign_votes[c].empty()) continue;
    bool plus = false;
    bool minus = false;
    bool centred = true;
    bool opposite = false;
    for (const Vote& v : votes) {
      (v.sign > 0 ? plus : minus) = true;
      centred &= std::abs(v.lateral) <= config.center_tolerance;
      const TrafficArrow& mine = arrows[static_cast<size_t>(v.arrow)];
      for (const TrafficArrow& other : arrows) {
        if (other.level == mine.level &&
            (other.center - mine.center).norm() <= config.opposite_radius &&
            other.direction.dot(mine.direction) < -0.7) {
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
