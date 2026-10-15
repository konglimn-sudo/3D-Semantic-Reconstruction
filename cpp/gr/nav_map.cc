#include "gr/nav_map.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <utility>

#include "gr/check.h"

namespace gr {
namespace {

constexpr double kBig = 1e20;
constexpr double kInf = std::numeric_limits<double>::infinity();
// 8-neighbourhood from east, alternating straight and diagonal steps.
constexpr int kDu[8] = {1, 1, 0, -1, -1, -1, 0, 1};
constexpr int kDv[8] = {0, 1, 1, 1, 0, -1, -1, -1};

// Squared distance transform of one line (Felzenszwalb & Huttenlocher).
void SquaredDistance1d(const std::vector<double>& f, int n,
                       std::vector<double>& d, std::vector<int>& v,
                       std::vector<double>& z) {
  int k = 0;
  v[0] = 0;
  z[0] = -kInf;
  z[1] = kInf;
  for (int q = 1; q < n; ++q) {
    auto intersection = [&](int p) {
      return ((f[q] + static_cast<double>(q) * q) -
              (f[p] + static_cast<double>(p) * p)) /
             (2.0 * q - 2.0 * p);
    };
    double s = intersection(v[k]);
    while (s <= z[k]) {
      --k;
      s = intersection(v[k]);
    }
    ++k;
    v[k] = q;
    z[k] = s;
    z[k + 1] = kInf;
  }
  k = 0;
  for (int q = 0; q < n; ++q) {
    while (z[k + 1] < q) ++k;
    d[q] = static_cast<double>(q - v[k]) * (q - v[k]) + f[v[k]];
  }
}

size_t Index(int u, int v, int width) {
  return static_cast<size_t>(v) * static_cast<size_t>(width) +
         static_cast<size_t>(u);
}

}  // namespace

std::vector<float> DistanceTransform(const std::vector<uint8_t>& sources,
                                     int width, int height) {
  GR_CHECK(width >= 0 && height >= 0);
  GR_CHECK(sources.size() == static_cast<size_t>(width) * height);
  std::vector<double> grid(sources.size());
  for (size_t i = 0; i < sources.size(); ++i) {
    grid[i] = sources[i] != 0 ? 0.0 : kBig;
  }
  const int n = std::max(width, height);
  std::vector<double> f(static_cast<size_t>(n)), d(static_cast<size_t>(n));
  std::vector<double> z(static_cast<size_t>(n) + 1);
  std::vector<int> v(static_cast<size_t>(n));
  for (int u = 0; u < width; ++u) {
    for (int r = 0; r < height; ++r) f[r] = grid[Index(u, r, width)];
    SquaredDistance1d(f, height, d, v, z);
    for (int r = 0; r < height; ++r) grid[Index(u, r, width)] = d[r];
  }
  for (int r = 0; r < height; ++r) {
    for (int u = 0; u < width; ++u) f[u] = grid[Index(u, r, width)];
    SquaredDistance1d(f, width, d, v, z);
    for (int u = 0; u < width; ++u) grid[Index(u, r, width)] = d[u];
  }
  std::vector<float> out(grid.size());
  for (size_t i = 0; i < grid.size(); ++i) {
    out[i] = grid[i] >= kBig * 0.5
                 ? std::numeric_limits<float>::infinity()
                 : static_cast<float>(std::sqrt(grid[i]));
  }
  return out;
}

std::vector<uint8_t> ThinMask(const std::vector<uint8_t>& mask, int width,
                              int height) {
  GR_CHECK(mask.size() == static_cast<size_t>(width) * height);
  std::vector<uint8_t> image(mask.size());
  for (size_t i = 0; i < mask.size(); ++i) image[i] = mask[i] != 0 ? 1 : 0;
  auto at = [&](int u, int v) -> int {
    if (u < 0 || v < 0 || u >= width || v >= height) return 0;
    return image[Index(u, v, width)];
  };
  std::vector<size_t> remove;
  bool changed = true;
  while (changed) {
    changed = false;
    for (int pass = 0; pass < 2; ++pass) {
      remove.clear();
      for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
          if (image[Index(u, v, width)] == 0) continue;
          // P2..P9 clockwise from north.
          const int p[8] = {at(u, v - 1),     at(u + 1, v - 1),
                            at(u + 1, v),     at(u + 1, v + 1),
                            at(u, v + 1),     at(u - 1, v + 1),
                            at(u - 1, v),     at(u - 1, v - 1)};
          int b = 0;
          int a = 0;
          for (int k = 0; k < 8; ++k) {
            b += p[k];
            if (p[k] == 0 && p[(k + 1) % 8] == 1) ++a;
          }
          if (b < 2 || b > 6 || a != 1) continue;
          const bool keep =
              pass == 0 ? (p[0] * p[2] * p[4] != 0 || p[2] * p[4] * p[6] != 0)
                        : (p[0] * p[2] * p[6] != 0 || p[0] * p[4] * p[6] != 0);
          if (!keep) remove.push_back(Index(u, v, width));
        }
      }
      for (size_t i : remove) image[i] = 0;
      changed |= !remove.empty();
    }
  }
  return image;
}

SkeletonGraph ExtractSkeletonGraph(const std::vector<uint8_t>& skeleton,
                                   int width, int height, double min_spur) {
  GR_CHECK(skeleton.size() == static_cast<size_t>(width) * height);
  auto on = [&](int u, int v) {
    return u >= 0 && v >= 0 && u < width && v < height &&
           skeleton[Index(u, v, width)] != 0;
  };
  const size_t num_cells = skeleton.size();
  std::vector<int8_t> count(num_cells, -1);
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      if (!on(u, v)) continue;
      int8_t c = 0;
      for (int k = 0; k < 8; ++k) c += on(u + kDu[k], v + kDv[k]) ? 1 : 0;
      count[Index(u, v, width)] = c;
    }
  }
  // Node clusters.
  std::vector<int> cluster(num_cells, -1);
  std::vector<std::vector<Eigen::Vector2i>> members;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const size_t i = Index(u, v, width);
      if (count[i] < 0 || count[i] == 2 || cluster[i] >= 0) continue;
      const int id = static_cast<int>(members.size());
      members.emplace_back();
      std::vector<Eigen::Vector2i> stack = {{u, v}};
      cluster[i] = id;
      while (!stack.empty()) {
        const Eigen::Vector2i c = stack.back();
        stack.pop_back();
        members[static_cast<size_t>(id)].push_back(c);
        for (int k = 0; k < 8; ++k) {
          const int uu = c.x() + kDu[k];
          const int vv = c.y() + kDv[k];
          if (!on(uu, vv)) continue;
          const size_t j = Index(uu, vv, width);
          if (count[j] == 2 || cluster[j] >= 0) continue;
          cluster[j] = id;
          stack.emplace_back(uu, vv);
        }
      }
    }
  }
  struct RawEdge {
    int a;
    int b;
    std::vector<Vec2> cells;
    double length;
    bool alive;
  };
  std::vector<Vec2> node_pos;
  for (const auto& cells : members) {
    Vec2 mean = Vec2::Zero();
    for (const Eigen::Vector2i& c : cells) mean += c.cast<double>();
    node_pos.push_back(mean / static_cast<double>(cells.size()));
  }
  std::vector<RawEdge> edges;
  std::vector<uint8_t> visited(num_cells, 0);
  // Walks a chain from `start` (a node cell or the loop seed) into `first`.
  auto walk = [&](const Eigen::Vector2i& start, const Eigen::Vector2i& first,
                  int from_node) {
    RawEdge edge{from_node, from_node, {start.cast<double>()}, 0.0, true};
    Eigen::Vector2i prev = start;
    Eigen::Vector2i cur = first;
    while (true) {
      edge.length += (cur - prev).cast<double>().norm();
      edge.cells.push_back(cur.cast<double>());
      const size_t ci = Index(cur.x(), cur.y(), width);
      if (cluster[ci] >= 0) {
        edge.b = cluster[ci];
        break;
      }
      if (cur == start) break;  // Closed loop back at the seed.
      visited[ci] = 1;
      bool advanced = false;
      for (int k = 0; k < 8; ++k) {
        const Eigen::Vector2i next(cur.x() + kDu[k], cur.y() + kDv[k]);
        if (next == prev || !on(next.x(), next.y())) continue;
        const size_t ni = Index(next.x(), next.y(), width);
        if (visited[ni] && next != start) continue;
        prev = cur;
        cur = next;
        advanced = true;
        break;
      }
      if (!advanced) break;
    }
    edges.push_back(std::move(edge));
  };
  for (size_t id = 0; id < members.size(); ++id) {
    for (const Eigen::Vector2i& c : members[id]) {
      for (int k = 0; k < 8; ++k) {
        const Eigen::Vector2i n(c.x() + kDu[k], c.y() + kDv[k]);
        if (!on(n.x(), n.y())) continue;
        const size_t ni = Index(n.x(), n.y(), width);
        if (cluster[ni] >= 0 || visited[ni]) continue;
        walk(c, n, static_cast<int>(id));
      }
    }
  }
  // Closed loops without any node.
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      const size_t i = Index(u, v, width);
      if (count[i] != 2 || visited[i]) continue;
      const int id = static_cast<int>(node_pos.size());
      node_pos.emplace_back(u, v);
      cluster[i] = id;  // The seed acts as the node.
      for (int k = 0; k < 8; ++k) {
        const Eigen::Vector2i n(u + kDu[k], v + kDv[k]);
        if (!on(n.x(), n.y()) || visited[Index(n.x(), n.y(), width)]) continue;
        walk({u, v}, n, id);
        break;
      }
      visited[i] = 1;
    }
  }

  const size_t num_nodes = node_pos.size();
  auto degrees = [&]() {
    std::vector<int> degree(num_nodes, 0);
    for (const RawEdge& e : edges) {
      if (!e.alive) continue;
      ++degree[static_cast<size_t>(e.a)];
      ++degree[static_cast<size_t>(e.b)];
    }
    return degree;
  };
  // Spur pruning.
  bool changed = true;
  while (changed) {
    changed = false;
    const std::vector<int> degree = degrees();
    for (RawEdge& e : edges) {
      if (!e.alive || e.length >= min_spur) continue;
      if (e.a == e.b) {  // Tiny loop around a staircase of node cells.
        e.alive = false;
        changed = true;
        break;
      }
      const int da = degree[static_cast<size_t>(e.a)];
      const int db = degree[static_cast<size_t>(e.b)];
      if ((da == 1 && db >= 3) || (db == 1 && da >= 3) ||
          (da == 1 && db == 1)) {
        e.alive = false;
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
    for (size_t k = 0; k < edges.size(); ++k) {
      const RawEdge& e = edges[k];
      if (!e.alive) continue;
      incident[static_cast<size_t>(e.a)].push_back(static_cast<int>(k));
      if (e.b != e.a) {
        incident[static_cast<size_t>(e.b)].push_back(static_cast<int>(k));
      }
    }
    for (size_t node = 0; node < num_nodes; ++node) {
      if (incident[node].size() != 2) continue;
      RawEdge& e1 = edges[static_cast<size_t>(incident[node][0])];
      RawEdge& e2 = edges[static_cast<size_t>(incident[node][1])];
      if (e1.a == e1.b || e2.a == e2.b) continue;
      // Orient e1 to end at `node` and e2 to start there.
      if (e1.b != static_cast<int>(node)) {
        std::reverse(e1.cells.begin(), e1.cells.end());
        std::swap(e1.a, e1.b);
      }
      if (e2.a != static_cast<int>(node)) {
        std::reverse(e2.cells.begin(), e2.cells.end());
        std::swap(e2.a, e2.b);
      }
      e1.cells.insert(e1.cells.end(), e2.cells.begin() + 1, e2.cells.end());
      e1.length += e2.length;
      e1.b = e2.b;
      e2.alive = false;
      changed = true;
      break;
    }
  }
  // Drop isolated short components and compact.
  SkeletonGraph graph;
  std::vector<int> remap(num_nodes, -1);
  for (RawEdge& e : edges) {
    if (!e.alive) continue;
    if (e.a == e.b && e.length < min_spur) continue;
    for (int end : {e.a, e.b}) {
      int& id = remap[static_cast<size_t>(end)];
      if (id < 0) {
        id = static_cast<int>(graph.nodes.size());
        graph.nodes.push_back(node_pos[static_cast<size_t>(end)]);
      }
    }
    SkeletonEdge out;
    out.a = remap[static_cast<size_t>(e.a)];
    out.b = remap[static_cast<size_t>(e.b)];
    out.cells = std::move(e.cells);
    out.length = e.length;
    graph.edges.push_back(std::move(out));
  }
  return graph;
}

std::vector<Vec2> SimplifyPolyline(const std::vector<Vec2>& points,
                                   double tolerance) {
  const size_t n = points.size();
  if (n <= 2) return points;
  std::vector<uint8_t> keep(n, 0);
  keep[0] = keep[n - 1] = 1;
  std::vector<std::pair<size_t, size_t>> stack = {{0, n - 1}};
  while (!stack.empty()) {
    const auto [first, last] = stack.back();
    stack.pop_back();
    const Vec2 a = points[first];
    const Vec2 ab = points[last] - a;
    const double length = ab.norm();
    double worst = -1.0;
    size_t worst_index = first;
    for (size_t i = first + 1; i < last; ++i) {
      const Vec2 ap = points[i] - a;
      const double d =
          length > 1e-12 ? std::abs(ab.x() * ap.y() - ab.y() * ap.x()) / length
                         : ap.norm();
      if (d > worst) {
        worst = d;
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
  for (size_t i = 0; i < n; ++i) {
    if (keep[i]) out.push_back(points[i]);
  }
  return out;
}

RouteGraph::RouteGraph(int num_nodes)
    : num_nodes_(num_nodes), arcs_(static_cast<size_t>(num_nodes)) {
  GR_CHECK(num_nodes >= 0);
}

int RouteGraph::AddEdge(int a, int b, double cost, bool one_way) {
  GR_CHECK(a >= 0 && a < num_nodes_ && b >= 0 && b < num_nodes_);
  GR_CHECK(cost >= 0.0);
  const int edge = static_cast<int>(costs_.size());
  costs_.push_back(cost);
  arcs_[static_cast<size_t>(a)].push_back({b, edge});
  if (!one_way) arcs_[static_cast<size_t>(b)].push_back({a, edge});
  return edge;
}

std::vector<double> RouteGraph::Run(int source, int target,
                                    std::vector<Arc>* parent) const {
  GR_CHECK(source >= 0 && source < num_nodes_);
  std::vector<double> distance(static_cast<size_t>(num_nodes_), kInf);
  if (parent != nullptr) {
    parent->assign(static_cast<size_t>(num_nodes_), Arc{-1, -1});
  }
  using Item = std::pair<double, int>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> queue;
  distance[static_cast<size_t>(source)] = 0.0;
  queue.emplace(0.0, source);
  while (!queue.empty()) {
    const auto [d, node] = queue.top();
    queue.pop();
    if (d > distance[static_cast<size_t>(node)]) continue;
    if (node == target) break;
    for (const Arc& arc : arcs_[static_cast<size_t>(node)]) {
      const double candidate = d + costs_[static_cast<size_t>(arc.edge)];
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
  return Run(source, -1, nullptr);
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
  auto inside = [&](const Eigen::Vector2i& c) {
    return c.x() >= 0 && c.y() >= 0 && c.x() < width && c.y() < height;
  };
  auto blocked = [&](size_t i) {
    return !std::isfinite(cost[i]) || cost[i] < 0.0f;
  };
  if (!inside(start) || !inside(goal)) return {};
  const size_t s = Index(start.x(), start.y(), width);
  const size_t g = Index(goal.x(), goal.y(), width);
  if (blocked(s) || blocked(g)) return {};
  auto heuristic = [&](int u, int v) {
    const double dx = std::abs(u - goal.x());
    const double dy = std::abs(v - goal.y());
    return std::max(dx, dy) + (std::sqrt(2.0) - 1.0) * std::min(dx, dy);
  };
  std::vector<double> g_cost(cost.size(), kInf);
  std::vector<int64_t> parent(cost.size(), -1);
  std::vector<uint8_t> closed(cost.size(), 0);
  using Item = std::pair<double, int64_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  g_cost[s] = 0.0;
  open.emplace(heuristic(start.x(), start.y()), static_cast<int64_t>(s));
  while (!open.empty()) {
    const size_t i = static_cast<size_t>(open.top().second);
    open.pop();
    if (closed[i]) continue;
    closed[i] = 1;
    if (i == g) break;
    const int u = static_cast<int>(i % static_cast<size_t>(width));
    const int v = static_cast<int>(i / static_cast<size_t>(width));
    for (int k = 0; k < 8; ++k) {
      const Eigen::Vector2i n(u + kDu[k], v + kDv[k]);
      if (!inside(n)) continue;
      const size_t j = Index(n.x(), n.y(), width);
      if (closed[j] || blocked(j)) continue;
      const double step = (k % 2 == 0) ? 1.0 : std::sqrt(2.0);
      const double candidate = g_cost[i] + step * (1.0 + cost[j]);
      if (candidate < g_cost[j]) {
        g_cost[j] = candidate;
        parent[j] = static_cast<int64_t>(i);
        open.emplace(candidate + heuristic(n.x(), n.y()),
                     static_cast<int64_t>(j));
      }
    }
  }
  if (!std::isfinite(g_cost[g])) return {};
  std::vector<Eigen::Vector2i> path;
  for (int64_t i = static_cast<int64_t>(g); i >= 0;
       i = parent[static_cast<size_t>(i)]) {
    path.emplace_back(static_cast<int>(i % width), static_cast<int>(i / width));
    if (static_cast<size_t>(i) == s) break;
  }
  std::reverse(path.begin(), path.end());
  return path;
}

}  // namespace gr
