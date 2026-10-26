#include "gr/core/kdtree.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace gr {
namespace {

constexpr int kLeafSize = 16;

}  // namespace

KdTree2::KdTree2(std::vector<Vec2> points) : points_(std::move(points)) {
  order_.resize(points_.size());
  for (size_t i = 0; i < order_.size(); ++i) order_[i] = static_cast<int>(i);
  if (!points_.empty()) Build(0, static_cast<int>(points_.size()), 0);
}

int KdTree2::Build(int begin, int end, int depth) {
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back(Node{begin, end, -1, 0.0, -1, -1});
  if (end - begin <= kLeafSize) return id;
  // Split along the wider extent.
  Vec2 lo = points_[static_cast<size_t>(order_[static_cast<size_t>(begin)])];
  Vec2 hi = lo;
  for (int k = begin; k < end; ++k) {
    const Vec2& p =
        points_[static_cast<size_t>(order_[static_cast<size_t>(k)])];
    lo = lo.cwiseMin(p);
    hi = hi.cwiseMax(p);
  }
  const int axis = (hi - lo).x() >= (hi - lo).y() ? 0 : 1;
  const int middle = begin + (end - begin) / 2;
  std::nth_element(order_.begin() + begin, order_.begin() + middle,
                   order_.begin() + end, [&](int a, int b) {
                     return points_[static_cast<size_t>(a)](axis) <
                            points_[static_cast<size_t>(b)](axis);
                   });
  const double split =
      points_[static_cast<size_t>(order_[static_cast<size_t>(middle)])](axis);
  const int left = Build(begin, middle, depth + 1);
  const int right = Build(middle, end, depth + 1);
  Node& node = nodes_[static_cast<size_t>(id)];
  node.axis = axis;
  node.split = split;
  node.left = left;
  node.right = right;
  return id;
}

void KdTree2::Search(int id, const Vec2& query, Hit& best,
                     double& best_squared) const {
  const Node& node = nodes_[static_cast<size_t>(id)];
  if (node.axis < 0) {
    for (int k = node.begin; k < node.end; ++k) {
      const int index = order_[static_cast<size_t>(k)];
      const double d2 = (points_[static_cast<size_t>(index)] - query)
                            .squaredNorm();
      if (d2 < best_squared || (d2 == best_squared && best.index >= 0 &&
                                index < best.index)) {
        best_squared = d2;
        best.index = index;
      }
    }
    return;
  }
  const double delta = query(node.axis) - node.split;
  const int near = delta < 0.0 ? node.left : node.right;
  const int far = delta < 0.0 ? node.right : node.left;
  Search(near, query, best, best_squared);
  if (delta * delta <= best_squared) Search(far, query, best, best_squared);
}

KdTree2::Hit KdTree2::Nearest(const Vec2& query, double max_distance) const {
  Hit best;
  if (points_.empty()) return best;
  double best_squared = max_distance * max_distance;
  Search(0, query, best, best_squared);
  if (best.index >= 0) best.distance = std::sqrt(best_squared);
  return best;
}

}  // namespace gr
