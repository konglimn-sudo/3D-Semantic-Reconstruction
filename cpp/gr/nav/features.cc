#include "gr/nav/features.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

namespace gr {
namespace {

// Distance from a point to an axis-aligned box (0 inside).
double BoxDistance(const Vec2& p, const Vec2& lo, const Vec2& hi) {
  const Vec2 d = (lo - p).cwiseMax(p - hi).cwiseMax(Vec2::Zero());
  return d.norm();
}

}  // namespace

std::vector<Footprint> ObjectFootprints(const std::vector<Vec2>& xy,
                                        double cell, int min_points) {
  // Points per occupied cell, then 8-connected components of the cells.
  std::map<std::pair<int64_t, int64_t>, std::vector<int>> cells;
  for (size_t k = 0; k < xy.size(); ++k) {
    cells[{static_cast<int64_t>(std::floor(xy[k].x() / cell)),
           static_cast<int64_t>(std::floor(xy[k].y() / cell))}]
        .push_back(static_cast<int>(k));
  }
  std::map<std::pair<int64_t, int64_t>, bool> visited;
  std::vector<Footprint> out;
  for (const auto& [start, unused] : cells) {
    if (visited[start]) continue;
    visited[start] = true;
    std::vector<std::pair<int64_t, int64_t>> stack = {start};
    Footprint f;
    f.min = Vec2::Constant(std::numeric_limits<double>::infinity());
    f.max = -f.min;
    while (!stack.empty()) {
      const auto c = stack.back();
      stack.pop_back();
      for (int k : cells.at(c)) {
        f.min = f.min.cwiseMin(xy[static_cast<size_t>(k)]);
        f.max = f.max.cwiseMax(xy[static_cast<size_t>(k)]);
        f.center += xy[static_cast<size_t>(k)];
        ++f.points;
      }
      for (int64_t dx = -1; dx <= 1; ++dx) {
        for (int64_t dy = -1; dy <= 1; ++dy) {
          const std::pair<int64_t, int64_t> n = {c.first + dx, c.second + dy};
          if (cells.count(n) && !visited[n]) {
            visited[n] = true;
            stack.push_back(n);
          }
        }
      }
    }
    if (f.points < min_points) continue;
    f.center /= f.points;
    out.push_back(f);
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const Footprint& a, const Footprint& b) {
                     return a.points > b.points;
                   });
  return out;
}

std::vector<FootprintEdge> EdgesAtFootprints(
    const std::vector<Footprint>& footprints,
    const std::vector<std::vector<Vec2>>& edges, double margin) {
  std::vector<FootprintEdge> out;
  for (size_t f = 0; f < footprints.size(); ++f) {
    const Footprint& fp = footprints[f];
    FootprintEdge best;
    best.distance = std::numeric_limits<double>::infinity();
    for (size_t e = 0; e < edges.size(); ++e) {
      const std::vector<Vec2>& line = edges[e];
      double start = 0.0;
      for (size_t k = 0; k + 1 < line.size(); ++k) {
        const Vec2 d = line[k + 1] - line[k];
        const double length = d.norm();
        // Closest approach of the segment to the box: sampled finely
        // enough for boxes of a few decimetres.
        const int steps = std::max(1, static_cast<int>(std::ceil(length /
                                                                 0.05)));
        for (int i = 0; i <= steps; ++i) {
          const double t = static_cast<double>(i) / steps;
          const double distance =
              BoxDistance(line[k] + t * d, fp.min, fp.max);
          if (distance < best.distance) {
            best = {static_cast<int>(f), static_cast<int>(e),
                    start + t * length, distance};
          }
        }
        start += length;
      }
    }
    if (best.edge >= 0 && best.distance <= margin) out.push_back(best);
  }
  return out;
}

}  // namespace gr
