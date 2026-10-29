// Static k-d tree over points of the xy plane: nearest-neighbour queries with
// an upper distance bound (the scipy cKDTree.query(k=1,
// distance_upper_bound) of the Python tooling). Built once, queried from many
// threads.

#ifndef GARAGE_RECON_GR_CORE_KDTREE_H_
#define GARAGE_RECON_GR_CORE_KDTREE_H_

#include <limits>
#include <vector>

#include "gr/core/types.h"

namespace gr {

class KdTree2 {
 public:
  // A query result.
  struct Hit {
    int index = kNoPoint;  // Into points(); kNoPoint: none within the bound.
    double distance = std::numeric_limits<double>::infinity();
  };

  // Hit::index when no point lies within the bound.
  static constexpr int kNoPoint = -1;

  explicit KdTree2(std::vector<Vec2> points);
  KdTree2(const KdTree2&) = default;
  KdTree2& operator=(const KdTree2&) = default;
  KdTree2(KdTree2&&) = default;
  KdTree2& operator=(KdTree2&&) = default;

  // The nearest point strictly closer than `max_distance` (ties: the lowest
  // index).
  Hit Nearest(
      const Vec2& query,
      double max_distance = std::numeric_limits<double>::infinity()) const;

  size_t size() const { return points_.size(); }
  const std::vector<Vec2>& points() const { return points_; }

 private:
  struct Node {
    int begin = 0;  // Range of order_.
    int end = 0;
    int axis = kLeafAxis;  // Split axis; kLeafAxis for a leaf.
    double split = 0.0;    // Points below go left, the others right.
    int left = kNoNode;
    int right = kNoNode;
  };

  // Node::axis of a leaf.
  static constexpr int kLeafAxis = -1;
  // Node::left and Node::right of a leaf.
  static constexpr int kNoNode = -1;

  // Adds the subtree over order_[begin, end) and returns its root.
  int Build(int begin, int end);
  // Lowers *best (and *best_squared_distance, its squared distance) to the
  // nearest point of the subtree at `node_index` that beats it.
  void Search(int node_index, const Vec2& query, Hit* best,
              double* best_squared_distance) const;
  // The point at `position` of order_.
  const Vec2& OrderedPoint(int position) const {
    return points_[static_cast<size_t>(order_[static_cast<size_t>(position)])];
  }

  std::vector<Vec2> points_;
  std::vector<int> order_;  // Indices into points_, grouped by node.
  std::vector<Node> nodes_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_KDTREE_H_
