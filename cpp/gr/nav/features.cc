#include "gr/nav/features.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// Footprints are 8-connected: the cells this many steps around.
constexpr int64_t kNeighbourhoodRadius = 1;
// Edge segments are sampled this finely (m) for their closest approach to a
// footprint box of a few decimetres.
constexpr double kEdgeSampleStep = 0.05;

using Cell = std::pair<int64_t, int64_t>;

// Distance from a point to an axis-aligned box (0 inside).
double BoxDistance(const Vec2& point, const Vec2& lo, const Vec2& hi) {
  const Vec2 outside =
      (lo - point).cwiseMax(point - hi).cwiseMax(Vec2::Zero());
  return outside.norm();
}

}  // namespace

std::vector<Footprint> ObjectFootprints(const std::vector<Vec2>& xy,
                                        double cell, int min_points) {
  // Points per occupied cell, then 8-connected components of the cells.
  std::map<Cell, std::vector<int>> points_by_cell;
  for (size_t k = 0; k < xy.size(); ++k) {
    points_by_cell[{static_cast<int64_t>(std::floor(xy[k].x() / cell)),
                    static_cast<int64_t>(std::floor(xy[k].y() / cell))}]
        .push_back(static_cast<int>(k));
  }
  std::map<Cell, bool> visited;
  std::vector<Footprint> out;
  for (const auto& [start, unused] : points_by_cell) {
    if (visited[start]) continue;
    visited[start] = true;
    std::vector<Cell> stack = {start};
    Footprint footprint;
    footprint.min = Vec2::Constant(kInf);
    footprint.max = -footprint.min;
    while (!stack.empty()) {
      const Cell current = stack.back();
      stack.pop_back();
      for (int point : points_by_cell.at(current)) {
        const Vec2& position = xy[static_cast<size_t>(point)];
        footprint.min = footprint.min.cwiseMin(position);
        footprint.max = footprint.max.cwiseMax(position);
        footprint.center += position;
        ++footprint.points;
      }
      for (int64_t dx = -kNeighbourhoodRadius; dx <= kNeighbourhoodRadius;
           ++dx) {
        for (int64_t dy = -kNeighbourhoodRadius; dy <= kNeighbourhoodRadius;
             ++dy) {
          const Cell neighbour = {current.first + dx, current.second + dy};
          if (points_by_cell.count(neighbour) && !visited[neighbour]) {
            visited[neighbour] = true;
            stack.push_back(neighbour);
          }
        }
      }
    }
    if (footprint.points < min_points) continue;
    footprint.center /= footprint.points;
    out.push_back(footprint);
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const Footprint& left, const Footprint& right) {
                     return left.points > right.points;
                   });
  return out;
}

std::vector<FootprintEdge> EdgesAtFootprints(
    const std::vector<Footprint>& footprints,
    const std::vector<std::vector<Vec2>>& edges, double margin) {
  std::vector<FootprintEdge> out;
  for (size_t index = 0; index < footprints.size(); ++index) {
    const Footprint& footprint = footprints[index];
    FootprintEdge best;
    best.distance = kInf;
    for (size_t edge = 0; edge < edges.size(); ++edge) {
      const std::vector<Vec2>& line = edges[edge];
      double start = 0.0;  // Arc length at the segment start.
      for (size_t k = 0; k + 1 < line.size(); ++k) {
        const Vec2 segment = line[k + 1] - line[k];
        const double length = segment.norm();
        const int steps =
            std::max(1, static_cast<int>(std::ceil(length / kEdgeSampleStep)));
        for (int step = 0; step <= steps; ++step) {
          const double fraction = static_cast<double>(step) / steps;
          const double distance = BoxDistance(line[k] + fraction * segment,
                                              footprint.min, footprint.max);
          if (distance < best.distance) {
            best = {static_cast<int>(index), static_cast<int>(edge),
                    start + fraction * length, distance};
          }
        }
        start += length;
      }
    }
    if (best.edge != FootprintEdge::kNone && best.distance <= margin) {
      out.push_back(best);
    }
  }
  return out;
}

}  // namespace gr
