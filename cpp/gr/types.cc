#include "gr/types.h"

namespace gr {

Points ToPoints(const Eigen::Ref<const RowMatX3d>& matrix) {
  Points points(static_cast<size_t>(matrix.rows()));
  for (Eigen::Index i = 0; i < matrix.rows(); ++i) {
    points[static_cast<size_t>(i)] = matrix.row(i).transpose();
  }
  return points;
}

RowMatX3d ToMatrix(const Points& points) {
  RowMatX3d matrix(static_cast<Eigen::Index>(points.size()), 3);
  for (size_t i = 0; i < points.size(); ++i) {
    matrix.row(static_cast<Eigen::Index>(i)) = points[i].transpose();
  }
  return matrix;
}

}  // namespace gr
