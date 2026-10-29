#include "gr/core/kdtree.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace gr {
namespace {

// Nodes with at most this many points are leaves.
constexpr int kLeafSize = 16;
// Split axes.
constexpr int kXAxis = 0;
constexpr int kYAxis = 1;

}  // namespace

KdTree2::KdTree2(std::vector<Vec2> points) : points_(std::move(points)) {
  order_.resize(points_.size());
  for (size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<int>(i);
  if (!points_.empty()) Build(0, static_cast<int>(points_.size()));
}

int KdTree2::Build(int begin, int end) {
  const int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(Node{begin, end});
  if (end - begin <= kLeafSize) return node_index;
  // Split along the wider extent.
  Vec2 box_min = OrderedPoint(begin);
  Vec2 box_max = box_min;
  for (int position = begin; position < end; ++position) {
    box_min = box_min.cwiseMin(OrderedPoint(position));
    box_max = box_max.cwiseMax(OrderedPoint(position));
  }
  const Vec2 extent = box_max - box_min;
  const int axis = extent.x() >= extent.y() ? kXAxis : kYAxis;
  const int middle = begin + (end - begin) / 2;
  std::nth_element(order_.begin() + begin, order_.begin() + middle,
                   order_.begin() + end, [&](int a, int b) {
                     return points_[static_cast<size_t>(a)](axis) <
                            points_[static_cast<size_t>(b)](axis);
                   });
  const double split = OrderedPoint(middle)(axis);
  const int left = Build(begin, middle);
  const int right = Build(middle, end);
  Node& node = nodes_[static_cast<size_t>(node_index)];
  node.axis = axis;
  node.split = split;
  node.left = left;
  node.right = right;
  return node_index;
}

void KdTree2::Search(int node_index, const Vec2& query, Hit* best,
                     double* best_squared_distance) const {
  const Node& node = nodes_[static_cast<size_t>(node_index)];
  if (node.axis == kLeafAxis) {
    for (int position = node.begin; position < node.end; ++position) {
      const int index = order_[static_cast<size_t>(position)];
      const double squared_distance =
          (points_[static_cast<size_t>(index)] - query).squaredNorm();
      if (squared_distance < *best_squared_distance ||
          (squared_distance == *best_squared_distance &&
           best->index != kNoPoint && index < best->index)) {
        *best_squared_distance = squared_distance;
        best->index = index;
      }
    }
    return;
  }
  const double offset_from_split = query(node.axis) - node.split;
  const int near_child = offset_from_split < 0.0 ? node.left : node.right;
  const int far_child = offset_from_split < 0.0 ? node.right : node.left;
  Search(near_child, query, best, best_squared_distance);
  // The far side can only hold a better point within the best distance of
  // the split.
  if (offset_from_split * offset_from_split <= *best_squared_distance) {
    Search(far_child, query, best, best_squared_distance);
  }
}

KdTree2::Hit KdTree2::Nearest(const Vec2& query, double max_distance) const {
  Hit best;
  if (points_.empty()) return best;
  double best_squared_distance = max_distance * max_distance;
  Search(/*node_index=*/0, query, &best, &best_squared_distance);
  if (best.index != kNoPoint) best.distance = std::sqrt(best_squared_distance);
  return best;
}

}  // namespace gr
