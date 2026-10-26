// Plane factor of the LiDAR bundle adjustment (BALM-style): the points of
// several scans on one planar surface patch.
//
// Each scan contributes the weighted first and second moments of its points
// in its own frame (a PointCluster), so the cost and its derivatives never
// touch individual points. The plane is not a variable: at every evaluation
// the best-fit plane of all points at the current poses is used, and the
// cost is the weighted sum of squared point-to-plane distances, W * l_min,
// where l_min is the smallest eigenvalue of the merged covariance. The
// Gauss-Newton linearisation eliminates the plane parameters with a Schur
// complement. Pose perturbation per scan: [t (world), phi (right)], the
// convention of the pose graph.

#ifndef GARAGE_RECON_GR_BACKEND_PLANE_FACTOR_H_
#define GARAGE_RECON_GR_BACKEND_PLANE_FACTOR_H_

#include <optional>
#include <vector>

#include "Eigen/Core"

#include "gr/core/pose.h"
#include "gr/core/types.h"

namespace gr {

// Weighted point statistics: sum of weights, sum of w * p, sum of w * p p^T.
class PointCluster {
 public:
  PointCluster() = default;

  void Add(const Vec3& point, double weight = 1.0);
  void Add(const PointCluster& other);
  // Multiplies every weight by `factor` (> 0).
  void Scale(double factor);
  // Statistics of the points mapped by `pose`.
  PointCluster Transformed(const Pose& pose) const;

  double weight() const { return weight_; }
  const Vec3& sum() const { return sum_; }
  const Mat3& outer() const { return outer_; }
  // Require weight() > 0.
  Vec3 Mean() const { return sum_ / weight_; }
  Mat3 Covariance() const;

 private:
  double weight_ = 0.0;
  Vec3 sum_ = Vec3::Zero();
  Mat3 outer_ = Mat3::Zero();
};

// Principal axes of a cluster. `normal` (the axis of the smallest
// eigenvalue) and `offset` define the plane normal . x + offset = 0.
struct ClusterPlane {
  Vec3 centroid = Vec3::Zero();
  Vec3 normal = Vec3::UnitZ();
  double offset = 0.0;
  Vec3 eigenvalues = Vec3::Zero();  // Ascending (m^2).
  Mat3 axes = Mat3::Identity();     // Columns: the matching eigenvectors.
};

// The cluster must have a positive weight.
ClusterPlane FitClusterPlane(const PointCluster& cluster);

// The points of one scan (node) on the plane, in the node's frame.
struct PlaneObservation {
  int node = 0;
  PointCluster points;
};

// Merged world-frame statistics; `poses[a]` is world_from_node of
// observations[a].
PointCluster MergeObservations(
    const std::vector<PlaneObservation>& observations,
    const std::vector<Pose>& poses);

// Weighted sum of squared distances to the best-fit plane, W * l_min.
double PlaneCost(const std::vector<PlaneObservation>& observations,
                 const std::vector<Pose>& poses);

struct PlaneLinearization {
  double cost = 0.0;    // W * l_min at the linearisation point.
  double weight = 0.0;  // W
  ClusterPlane plane;
  // Gauss-Newton normal equations of the residuals r = n . x + d with the
  // plane eliminated: 6m x 6m and 6m for m observations, blocks in the order
  // of `observations`. The gradient is J^T r (half the cost gradient).
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;
};

// nullopt if the points do not span a plane (no in-plane extent) or the
// weight is zero.
std::optional<PlaneLinearization> LinearizePlane(
    const std::vector<PlaneObservation>& observations,
    const std::vector<Pose>& poses);

}  // namespace gr

#endif  // GARAGE_RECON_GR_BACKEND_PLANE_FACTOR_H_
