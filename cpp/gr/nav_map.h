// Navigation map support: the geometry behind the semantic navigation map
// (drivable layers, aisle route graph, slot goals).
//
// Rasters are row-major (height x width) like Grid2D in raster.h. The route
// graph of a level is the skeleton of its drivable aisles: the aisle mask is
// first shrunk by the vehicle's half width, so every point of the skeleton
// keeps that clearance to walls, columns and other obstacles, and a route
// along the graph cannot pass through them.

#ifndef GARAGE_RECON_GR_NAV_MAP_H_
#define GARAGE_RECON_GR_NAV_MAP_H_

#include <cstdint>
#include <limits>
#include <vector>

#include "gr/types.h"

namespace gr {

// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher): distance
// in cells from every cell to the nearest nonzero cell of `sources`; +inf
// without any source.
std::vector<float> DistanceTransform(const std::vector<uint8_t>& sources,
                                     int width, int height);

// Zhang-Suen thinning of a binary mask to a one-cell-wide 8-connected
// skeleton.
std::vector<uint8_t> ThinMask(const std::vector<uint8_t>& mask, int width,
                              int height);

struct SkeletonEdge {
  int a = -1;               // Node indices (a == b for a closed loop).
  int b = -1;
  std::vector<Vec2> cells;  // (col, row) from node a to node b.
  double length = 0.0;      // Cells.
};

struct SkeletonGraph {
  std::vector<Vec2> nodes;  // (col, row) of junctions and end points.
  std::vector<SkeletonEdge> edges;
};

// Graph of a skeleton: nodes at junctions and end points (8-connected
// clusters of cells with other than two skeleton neighbours), edges along the
// chains between them. End-point spurs and loops shorter than `min_spur`
// cells are removed repeatedly and nodes left with two edges are merged;
// components shorter than `min_spur` in total are dropped.
SkeletonGraph ExtractSkeletonGraph(const std::vector<uint8_t>& skeleton,
                                   int width, int height, double min_spur);

// Douglas-Peucker simplification (keeps the end points).
std::vector<Vec2> SimplifyPolyline(const std::vector<Vec2>& points,
                                   double tolerance);

struct GraphPath {
  bool found = false;
  double cost = std::numeric_limits<double>::infinity();
  std::vector<int> nodes;  // From source to target.
  std::vector<int> edges;
};

// Directed / undirected weighted graph with Dijkstra queries.
class RouteGraph {
 public:
  explicit RouteGraph(int num_nodes);

  // Returns the edge index. A one-way edge is travelled from `a` to `b` only.
  // `cost` must be non-negative.
  int AddEdge(int a, int b, double cost, bool one_way);

  // Cost from `source` to every node (+inf where unreachable).
  std::vector<double> Distances(int source) const;
  GraphPath ShortestPath(int source, int target) const;

  int num_nodes() const { return num_nodes_; }
  int num_edges() const { return static_cast<int>(costs_.size()); }

 private:
  struct Arc {
    int to;
    int edge;
  };
  // Dijkstra from `source`; fills the incoming arc of every settled node.
  std::vector<double> Run(int source, int target,
                          std::vector<Arc>* parent) const;

  int num_nodes_;
  std::vector<std::vector<Arc>> arcs_;
  std::vector<double> costs_;
};

// 8-connected A* on a cost grid: entering a cell costs the step length times
// (1 + cost); cells with negative or non-finite cost are blocked. Returns the
// cells (col, row) from start to goal, or nothing if the goal is unreachable.
std::vector<Eigen::Vector2i> GridAStar(const std::vector<float>& cost,
                                       int width, int height,
                                       const Eigen::Vector2i& start,
                                       const Eigen::Vector2i& goal);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_MAP_H_
