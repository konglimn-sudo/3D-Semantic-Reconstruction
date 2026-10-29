#include "gr/core/types.h"

namespace gr {

Points ToPoints(const Eigen::Ref<const RowMatX3d>& matrix) {
  Points points(static_cast<size_t>(matrix.rows()));
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    points[static_cast<size_t>(row)] = matrix.row(row).transpose();
  }
  return points;
}

RowMatX3d ToMatrix(const Points& points) {
  RowMatX3d matrix(static_cast<Eigen::Index>(points.size()), kSpaceDim);
  for (size_t i = 0; i < points.size(); ++i) {
    matrix.row(static_cast<Eigen::Index>(i)) = points[i].transpose();
  }
  return matrix;
}

}  // namespace gr
