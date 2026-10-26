#include "gr/nav/traffic.h"

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
};

Projection Project(const std::vector<Vec2>& points, const Vec2& xy) {
  Projection best;
  for (size_t k = 0; k + 1 < points.size(); ++k) {
    const Vec2 seg = points[k + 1] - points[k];
    const double length = std::max(seg.norm(), 1e-12);
    const double t = std::clamp((xy - points[k]).dot(seg) / (length * length),
                                0.0, 1.0);
    const Vec2 foot = points[k] + t * seg;
    const double distance = (xy - foot).norm();
    if (distance < best.distance) {
      best.distance = distance;
      best.tangent = seg / length;
      const Vec2 offset = xy - foot;
      best.lateral = best.tangent.x() * offset.y() -
                     best.tangent.y() * offset.x();
    }
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

std::vector<Corridor> DecideTraffic(const std::vector<TrafficEdge>& edges,
                                    const std::vector<TrafficArrow>& arrows,
                                    const TrafficConfig& config) {
  const std::vector<ArrowAssignment> assigned =
      AssignArrows(edges, arrows, config);
  std::map<int, std::vector<int>> by_edge;  // Edge -> arrows.
  for (size_t i = 0; i < assigned.size(); ++i) {
    if (assigned[i].edge >= 0) {
      by_edge[assigned[i].edge].push_back(static_cast<int>(i));
    }
  }
  std::vector<Corridor> out;
  for (const std::vector<CorridorStep>& chain :
       Corridors(edges, config.max_turn)) {
    struct Vote {
      int sign;
      double lateral;
      int arrow;
    };
    std::vector<Vote> votes;
    for (const CorridorStep& step : chain) {
      const auto found = by_edge.find(step.edge);
      if (found == by_edge.end()) continue;
      const int factor = step.forward ? 1 : -1;
      for (int i : found->second) {
        votes.push_back({assigned[static_cast<size_t>(i)].sign * factor,
                         assigned[static_cast<size_t>(i)].lateral * factor,
                         i});
      }
    }
    if (votes.empty()) continue;
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
    Corridor corridor;
    corridor.steps = chain;
    corridor.level = edges[static_cast<size_t>(chain[0].edge)].level;
    for (const CorridorStep& step : chain) {
      corridor.length += Length(edges[static_cast<size_t>(step.edge)].points);
    }
    corridor.arrows = static_cast<int>(votes.size());
    if ((plus && minus) || opposite) {
      corridor.verdict = TrafficVerdict::kTwoWay;
    } else if (!centred) {
      corridor.verdict = TrafficVerdict::kOneLaneSeen;
    } else if (static_cast<int>(votes.size()) < config.min_arrows) {
      corridor.verdict = TrafficVerdict::kOneArrow;
    } else {
      corridor.verdict = TrafficVerdict::kOneWay;
      corridor.travel = votes[0].sign;
    }
    out.push_back(std::move(corridor));
  }
  return out;
}

}  // namespace gr
