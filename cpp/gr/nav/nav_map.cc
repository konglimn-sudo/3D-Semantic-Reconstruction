#include "gr/nav/nav_map.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <utility>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// The diagonal step of the 8-neighbourhood (std::sqrt(2.0)).
constexpr double kSqrt2 = 1.41421356237309504880;
// 8-neighbourhood from east, alternating straight (even) and diagonal (odd)
// steps.
constexpr int kNeighbourCount = 8;
constexpr int kStepX[kNeighbourCount] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kStepY[kNeighbourCount] = {0, 1, 1, 1, 0, -1, -1, -1};
// Skeleton cells with this many skeleton neighbours are chain cells; the
// others belong to nodes. kOffSkeleton marks the cells off the skeleton.
constexpr int8_t kChainNeighbours = 2;
constexpr int8_t kOffSkeleton = -1;
// Node degrees: an end point and the least of a junction.
constexpr int kEndPointDegree = 1;
constexpr int kMinJunctionDegree = 3;
// A cell, node or parent that is not set.
constexpr int kNone = -1;
// Chords shorter than this (cells, m) measure distances to their start.
constexpr double kMinChordLength = 1e-12;

size_t Index(int x, int y, int width) {
  return static_cast<size_t>(y) * static_cast<size_t>(width) +
         static_cast<size_t>(x);
}

}  // namespace

SkeletonGraph ExtractSkeletonGraph(const std::vector<uint8_t>& skeleton,
                                   int width, int height, double min_spur) {
  GR_CHECK(skeleton.size() == static_cast<size_t>(width) * height);
  const auto on = [&](int x, int y) {
    return x >= 0 && y >= 0 && x < width && y < height &&
           skeleton[Index(x, y, width)] != 0;
  };
  const size_t num_cells = skeleton.size();
  // Skeleton neighbours per cell.
  std::vector<int8_t> neighbours(num_cells, kOffSkeleton);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (!on(x, y)) continue;
      int8_t count = 0;
      for (int step = 0; step < kNeighbourCount; ++step) {
        count += on(x + kStepX[step], y + kStepY[step]) ? 1 : 0;
      }
      neighbours[Index(x, y, width)] = count;
    }
  }
  // Node clusters.
  std::vector<int> cluster(num_cells, kNone);
  std::vector<std::vector<Eigen::Vector2i>> members;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t cell = Index(x, y, width);
      if (neighbours[cell] == kOffSkeleton ||
          neighbours[cell] == kChainNeighbours || cluster[cell] != kNone) {
        continue;
      }
      const int id = static_cast<int>(members.size());
      members.emplace_back();
      std::vector<Eigen::Vector2i> stack = {{x, y}};
      cluster[cell] = id;
      while (!stack.empty()) {
        const Eigen::Vector2i current = stack.back();
        stack.pop_back();
        members[static_cast<size_t>(id)].push_back(current);
        for (int step = 0; step < kNeighbourCount; ++step) {
          const int next_x = current.x() + kStepX[step];
          const int next_y = current.y() + kStepY[step];
          if (!on(next_x, next_y)) continue;
          const size_t next = Index(next_x, next_y, width);
          if (neighbours[next] == kChainNeighbours || cluster[next] != kNone) {
            continue;
          }
          cluster[next] = id;
          stack.emplace_back(next_x, next_y);
        }
      }
    }
  }
  struct RawEdge {
    int start_node;
    int end_node;
    std::vector<Vec2> cells;
    double length;
    bool alive;
  };
  std::vector<Vec2> node_position;
  for (const auto& cells : members) {
    Vec2 mean = Vec2::Zero();
    for (const Eigen::Vector2i& cell : cells) mean += cell.cast<double>();
    node_position.push_back(mean / static_cast<double>(cells.size()));
  }
  std::vector<RawEdge> edges;
  std::vector<uint8_t> visited(num_cells, 0);
  // Walks a chain from `start` (a node cell or the loop seed) into `first`.
  const auto walk = [&](const Eigen::Vector2i& start,
                        const Eigen::Vector2i& first, int from_node) {
    RawEdge edge{from_node, from_node, {start.cast<double>()}, 0.0, true};
    Eigen::Vector2i previous = start;
    Eigen::Vector2i current = first;
    while (true) {
      edge.length += (current - previous).cast<double>().norm();
      edge.cells.push_back(current.cast<double>());
      const size_t current_cell = Index(current.x(), current.y(), width);
      if (cluster[current_cell] != kNone) {
        edge.end_node = cluster[current_cell];
        break;
      }
      if (current == start) break;  // Closed loop back at the seed.
      visited[current_cell] = 1;
      bool advanced = false;
      for (int step = 0; step < kNeighbourCount; ++step) {
        const Eigen::Vector2i next(current.x() + kStepX[step],
                                   current.y() + kStepY[step]);
        if (next == previous || !on(next.x(), next.y())) continue;
        const size_t next_cell = Index(next.x(), next.y(), width);
        if (visited[next_cell] && next != start) continue;
        previous = current;
        current = next;
        advanced = true;
        break;
      }
      if (!advanced) break;
    }
    edges.push_back(std::move(edge));
  };
  for (size_t id = 0; id < members.size(); ++id) {
    for (const Eigen::Vector2i& member : members[id]) {
      for (int step = 0; step < kNeighbourCount; ++step) {
        const Eigen::Vector2i next(member.x() + kStepX[step],
                                   member.y() + kStepY[step]);
        if (!on(next.x(), next.y())) continue;
        const size_t next_cell = Index(next.x(), next.y(), width);
        if (cluster[next_cell] != kNone || visited[next_cell]) continue;
        walk(member, next, static_cast<int>(id));
      }
    }
  }
  // Closed loops without any node.
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t cell = Index(x, y, width);
      if (neighbours[cell] != kChainNeighbours || visited[cell]) continue;
      const int id = static_cast<int>(node_position.size());
      node_position.emplace_back(x, y);
      cluster[cell] = id;  // The seed acts as the node.
      for (int step = 0; step < kNeighbourCount; ++step) {
        const Eigen::Vector2i next(x + kStepX[step], y + kStepY[step]);
        if (!on(next.x(), next.y()) ||
            visited[Index(next.x(), next.y(), width)]) {
          continue;
        }
        walk({x, y}, next, id);
        break;
      }
      visited[cell] = 1;
    }
  }

  const size_t num_nodes = node_position.size();
  const auto degrees = [&]() {
    std::vector<int> degree(num_nodes, 0);
    for (const RawEdge& edge : edges) {
      if (!edge.alive) continue;
      ++degree[static_cast<size_t>(edge.start_node)];
      ++degree[static_cast<size_t>(edge.end_node)];
    }
    return degree;
  };
  // Spur pruning.
  bool changed = true;
  while (changed) {
    changed = false;
    const std::vector<int> degree = degrees();
    for (RawEdge& edge : edges) {
      if (!edge.alive || edge.length >= min_spur) continue;
      if (edge.start_node == edge.end_node) {
        // Tiny loop around a staircase of node cells.
        edge.alive = false;
        changed = true;
        break;
      }
      const int start_degree = degree[static_cast<size_t>(edge.start_node)];
      const int end_degree = degree[static_cast<size_t>(edge.end_node)];
      if ((start_degree == kEndPointDegree &&
           end_degree >= kMinJunctionDegree) ||
          (end_degree == kEndPointDegree &&
           start_degree >= kMinJunctionDegree) ||
          (start_degree == kEndPointDegree &&
           end_degree == kEndPointDegree)) {
        edge.alive = false;
        changed = true;
        break;  // Degrees changed.
      }
    }
  }
  // Merge nodes with exactly two distinct edges.
  changed = true;
  while (changed) {
    changed = false;
    std::vector<std::vector<int>> incident(num_nodes);
    for (size_t index = 0; index < edges.size(); ++index) {
      const RawEdge& edge = edges[index];
      if (!edge.alive) continue;
      incident[static_cast<size_t>(edge.start_node)].push_back(
          static_cast<int>(index));
      if (edge.end_node != edge.start_node) {
        incident[static_cast<size_t>(edge.end_node)].push_back(
            static_cast<int>(index));
      }
    }
    for (size_t node = 0; node < num_nodes; ++node) {
      if (incident[node].size() != kChainNeighbours) continue;
      RawEdge& incoming = edges[static_cast<size_t>(incident[node][0])];
      RawEdge& outgoing = edges[static_cast<size_t>(incident[node][1])];
      if (incoming.start_node == incoming.end_node ||
          outgoing.start_node == outgoing.end_node) {
        continue;
      }
      // Orient `incoming` to end at `node` and `outgoing` to start there.
      if (incoming.end_node != static_cast<int>(node)) {
        std::reverse(incoming.cells.begin(), incoming.cells.end());
        std::swap(incoming.start_node, incoming.end_node);
      }
      if (outgoing.start_node != static_cast<int>(node)) {
        std::reverse(outgoing.cells.begin(), outgoing.cells.end());
        std::swap(outgoing.start_node, outgoing.end_node);
      }
      incoming.cells.insert(incoming.cells.end(), outgoing.cells.begin() + 1,
                            outgoing.cells.end());
      incoming.length += outgoing.length;
      incoming.end_node = outgoing.end_node;
      outgoing.alive = false;
      changed = true;
      break;
    }
  }
  // Drop isolated short components and compact.
  SkeletonGraph graph;
  std::vector<int> remap(num_nodes, kNone);
  for (RawEdge& edge : edges) {
    if (!edge.alive) continue;
    if (edge.start_node == edge.end_node && edge.length < min_spur) continue;
    for (int end : {edge.start_node, edge.end_node}) {
      int& id = remap[static_cast<size_t>(end)];
      if (id == kNone) {
        id = static_cast<int>(graph.nodes.size());
        graph.nodes.push_back(node_position[static_cast<size_t>(end)]);
      }
    }
    SkeletonEdge out;
    out.start_node = remap[static_cast<size_t>(edge.start_node)];
    out.end_node = remap[static_cast<size_t>(edge.end_node)];
    out.cells = std::move(edge.cells);
    out.length = edge.length;
    graph.edges.push_back(std::move(out));
  }
  return graph;
}

std::vector<Vec2> SimplifyPolyline(const std::vector<Vec2>& points,
                                   double tolerance) {
  const size_t count = points.size();
  if (count <= 2) return points;
  std::vector<uint8_t> keep(count, 0);
  keep[0] = keep[count - 1] = 1;
  std::vector<std::pair<size_t, size_t>> stack = {{0, count - 1}};
  while (!stack.empty()) {
    const auto [first, last] = stack.back();
    stack.pop_back();
    const Vec2 start = points[first];
    const Vec2 chord = points[last] - start;
    const double length = chord.norm();
    double worst = -1.0;  // Below any distance.
    size_t worst_index = first;
    for (size_t i = first + 1; i < last; ++i) {
      const Vec2 offset = points[i] - start;
      const double distance =
          length > kMinChordLength
              ? std::abs(chord.x() * offset.y() - chord.y() * offset.x()) /
                    length
              : offset.norm();
      if (distance > worst) {
        worst = distance;
        worst_index = i;
      }
    }
    if (worst > tolerance) {
      keep[worst_index] = 1;
      stack.emplace_back(first, worst_index);
      stack.emplace_back(worst_index, last);
    }
  }
  std::vector<Vec2> out;
  for (size_t i = 0; i < count; ++i) {
    if (keep[i]) out.push_back(points[i]);
  }
  return out;
}

RouteGraph::RouteGraph(int num_nodes)
    : num_nodes_(num_nodes), arcs_(static_cast<size_t>(num_nodes)) {
  GR_CHECK(num_nodes >= 0);
}

int RouteGraph::AddEdge(int from, int to, double cost, bool one_way) {
  GR_CHECK(from >= 0 && from < num_nodes_ && to >= 0 && to < num_nodes_);
  GR_CHECK(cost >= 0.0);
  const int edge = static_cast<int>(costs_.size());
  costs_.push_back(cost);
  arcs_[static_cast<size_t>(from)].push_back({to, edge});
  if (!one_way) arcs_[static_cast<size_t>(to)].push_back({from, edge});
  return edge;
}

std::vector<double> RouteGraph::Run(int source, int target,
                                    std::vector<Arc>* parent) const {
  GR_CHECK(source >= 0 && source < num_nodes_);
  std::vector<double> distance(static_cast<size_t>(num_nodes_), kInf);
  if (parent != nullptr) {
    parent->assign(static_cast<size_t>(num_nodes_), Arc{kNone, kNone});
  }
  using Item = std::pair<double, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
  distance[static_cast<size_t>(source)] = 0.0;
  queue.emplace(0.0, source);
  while (!queue.empty()) {
    const auto [node_distance, node] = queue.top();
    queue.pop();
    if (node_distance > distance[static_cast<size_t>(node)]) continue;
    if (node == target) break;
    for (const Arc& arc : arcs_[static_cast<size_t>(node)]) {
      const double candidate =
          node_distance + costs_[static_cast<size_t>(arc.edge)];
      if (candidate < distance[static_cast<size_t>(arc.to)]) {
        distance[static_cast<size_t>(arc.to)] = candidate;
        if (parent != nullptr) {
          (*parent)[static_cast<size_t>(arc.to)] = {node, arc.edge};
        }
        queue.emplace(candidate, arc.to);
      }
    }
  }
  return distance;
}

std::vector<double> RouteGraph::Distances(int source) const {
  return Run(source, kNone, nullptr);
}

GraphPath RouteGraph::ShortestPath(int source, int target) const {
  GR_CHECK(target >= 0 && target < num_nodes_);
  std::vector<Arc> parent;
  const std::vector<double> distance = Run(source, target, &parent);
  GraphPath path;
  if (!std::isfinite(distance[static_cast<size_t>(target)])) return path;
  path.found = true;
  path.cost = distance[static_cast<size_t>(target)];
  for (int node = target; node != source;) {
    const Arc& arc = parent[static_cast<size_t>(node)];
    path.nodes.push_back(node);
    path.edges.push_back(arc.edge);
    node = arc.to;
  }
  path.nodes.push_back(source);
  std::reverse(path.nodes.begin(), path.nodes.end());
  std::reverse(path.edges.begin(), path.edges.end());
  return path;
}

std::vector<Eigen::Vector2i> GridAStar(const std::vector<float>& cost,
                                       int width, int height,
                                       const Eigen::Vector2i& start,
                                       const Eigen::Vector2i& goal) {
  GR_CHECK(cost.size() == static_cast<size_t>(width) * height);
  const auto inside = [&](const Eigen::Vector2i& cell) {
    return cell.x() >= 0 && cell.y() >= 0 && cell.x() < width &&
           cell.y() < height;
  };
  const auto blocked = [&](size_t cell) {
    return !std::isfinite(cost[cell]) || cost[cell] < 0.0f;
  };
  if (!inside(start) || !inside(goal)) return {};
  const size_t start_cell = Index(start.x(), start.y(), width);
  const size_t goal_cell = Index(goal.x(), goal.y(), width);
  if (blocked(start_cell) || blocked(goal_cell)) return {};
  // Octile distance to the goal.
  const auto heuristic = [&](int x, int y) {
    const double dx = std::abs(x - goal.x());
    const double dy = std::abs(y - goal.y());
    return std::max(dx, dy) + (kSqrt2 - 1.0) * std::min(dx, dy);
  };
  std::vector<double> cost_so_far(cost.size(), kInf);
  std::vector<int64_t> parent(cost.size(), kNone);
  std::vector<uint8_t> closed(cost.size(), 0);
  using Item = std::pair<double, int64_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  cost_so_far[start_cell] = 0.0;
  open.emplace(heuristic(start.x(), start.y()),
               static_cast<int64_t>(start_cell));
  while (!open.empty()) {
    const size_t cell = static_cast<size_t>(open.top().second);
    open.pop();
    if (closed[cell]) continue;
    closed[cell] = 1;
    if (cell == goal_cell) break;
    const int x = static_cast<int>(cell % static_cast<size_t>(width));
    const int y = static_cast<int>(cell / static_cast<size_t>(width));
    for (int step = 0; step < kNeighbourCount; ++step) {
      const Eigen::Vector2i next(x + kStepX[step], y + kStepY[step]);
      if (!inside(next)) continue;
      const size_t next_cell = Index(next.x(), next.y(), width);
      if (closed[next_cell] || blocked(next_cell)) continue;
      const double length = step % 2 == 0 ? 1.0 : kSqrt2;
      const double candidate =
          cost_so_far[cell] + length * (1.0 + cost[next_cell]);
      if (candidate < cost_so_far[next_cell]) {
        cost_so_far[next_cell] = candidate;
        parent[next_cell] = static_cast<int64_t>(cell);
        open.emplace(candidate + heuristic(next.x(), next.y()),
                     static_cast<int64_t>(next_cell));
      }
    }
  }
  if (!std::isfinite(cost_so_far[goal_cell])) return {};
  std::vector<Eigen::Vector2i> path;
  for (int64_t cell = static_cast<int64_t>(goal_cell); cell >= 0;
       cell = parent[static_cast<size_t>(cell)]) {
    path.emplace_back(static_cast<int>(cell % width),
                      static_cast<int>(cell / width));
    if (static_cast<size_t>(cell) == start_cell) break;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

}  // namespace gr
