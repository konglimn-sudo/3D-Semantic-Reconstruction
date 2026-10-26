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
  explicit KdTree2(std::vector<Vec2> points);
  KdTree2(const KdTree2&) = default;
  KdTree2& operator=(const KdTree2&) = default;
  KdTree2(KdTree2&&) = default;
  KdTree2& operator=(KdTree2&&) = default;

  struct Hit {
    int index = -1;  // -1: none within the bound.
    double distance = std::numeric_limits<double>::infinity();
  };

  // The nearest point strictly closer than `max_distance` (ties: the lowest
  // index).
  Hit Nearest(const Vec2& query, double max_distance =
                                     std::numeric_limits<double>::infinity())
      const;

  size_t size() const { return points_.size(); }
  const std::vector<Vec2>& points() const { return points_; }

 private:
  struct Node {
    int begin = 0;  // Range of order_.
    int end = 0;
    int axis = -1;  // -1: leaf.
    double split = 0.0;
    int left = -1;
    int right = -1;
  };

  int Build(int begin, int end, int depth);
  void Search(int node, const Vec2& query, Hit& best,
              double& best_squared) const;

  std::vector<Vec2> points_;
  std::vector<int> order_;
  std::vector<Node> nodes_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_KDTREE_H_
