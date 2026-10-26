#include "gr/backend/plane_factor.h"

#include <algorithm>

#include "Eigen/Eigenvalues"

#include "gr/core/check.h"

namespace gr {

void PointCluster::Add(const Vec3& point, double weight) {
  weight_ += weight;
  sum_ += weight * point;
  outer_ += weight * point * point.transpose();
}

void PointCluster::Add(const PointCluster& other) {
  weight_ += other.weight_;
  sum_ += other.sum_;
  outer_ += other.outer_;
}

void PointCluster::Scale(double factor) {
  GR_CHECK(factor > 0.0);
  weight_ *= factor;
  sum_ *= factor;
  outer_ *= factor;
}

PointCluster PointCluster::Transformed(const Pose& pose) const {
  const Mat3& r = pose.rotation;
  const Vec3& t = pose.translation;
  const Vec3 rotated_sum = r * sum_;
  PointCluster out;
  out.weight_ = weight_;
  out.sum_ = rotated_sum + weight_ * t;
  out.outer_ = r * outer_ * r.transpose() + rotated_sum * t.transpose() +
               t * rotated_sum.transpose() + weight_ * t * t.transpose();
  return out;
}

Mat3 PointCluster::Covariance() const {
  GR_CHECK(weight_ > 0.0);
  const Vec3 mean = Mean();
  return outer_ / weight_ - mean * mean.transpose();
}

ClusterPlane FitClusterPlane(const PointCluster& cluster) {
  GR_CHECK(cluster.weight() > 0.0);
  const Eigen::SelfAdjointEigenSolver<Mat3> solver(cluster.Covariance());
  ClusterPlane plane;
  plane.centroid = cluster.Mean();
  plane.eigenvalues = solver.eigenvalues().cwiseMax(0.0);
  plane.axes = solver.eigenvectors();
  plane.normal = plane.axes.col(0);
  plane.offset = -plane.normal.dot(plane.centroid);
  return plane;
}

PointCluster MergeObservations(
    const std::vector<PlaneObservation>& observations,
    const std::vector<Pose>& poses) {
  GR_CHECK(observations.size() == poses.size());
  PointCluster merged;
  for (size_t a = 0; a < observations.size(); ++a) {
    merged.Add(observations[a].points.Transformed(poses[a]));
  }
  return merged;
}

double PlaneCost(const std::vector<PlaneObservation>& observations,
                 const std::vector<Pose>& poses) {
  const PointCluster merged = MergeObservations(observations, poses);
  if (merged.weight() <= 0.0) return 0.0;
  return merged.weight() * FitClusterPlane(merged).eigenvalues[0];
}

std::optional<PlaneLinearization> LinearizePlane(
    const std::vector<PlaneObservation>& observations,
    const std::vector<Pose>& poses) {
  const PointCluster merged = MergeObservations(observations, poses);
  const double total_weight = merged.weight();
  if (total_weight <= 0.0) return std::nullopt;
  PlaneLinearization out;
  out.plane = FitClusterPlane(merged);
  out.weight = total_weight;
  out.cost = total_weight * out.plane.eigenvalues[0];
  // The in-plane spread must be positive for the plane to be determined.
  const double scale = std::max(out.plane.eigenvalues[2], 1e-300);
  if (out.plane.eigenvalues[1] <= 1e-12 * scale) return std::nullopt;

  const Vec3& n = out.plane.normal;
  const double d = out.plane.offset;
  const Vec3& mu = out.plane.centroid;
  Eigen::Matrix<double, 3, 2> tangent;
  tangent << out.plane.axes.col(1), out.plane.axes.col(2);
  // Plane parameters: normal tilt (2, about the centroid) and offset (1).
  // Their normal-equation block is diagonal for this choice of axes.
  const Vec3 plane_diagonal(total_weight * out.plane.eigenvalues[1],
                            total_weight * out.plane.eigenvalues[2],
                            total_weight);

  const int m = static_cast<int>(observations.size());
  out.hessian = Eigen::MatrixXd::Zero(6 * m, 6 * m);
  out.gradient = Eigen::VectorXd::Zero(6 * m);
  Eigen::MatrixXd pose_by_plane = Eigen::MatrixXd::Zero(6 * m, 3);
  for (int a = 0; a < m; ++a) {
    const PointCluster& cluster = observations[static_cast<size_t>(a)].points;
    const Pose& pose = poses[static_cast<size_t>(a)];
    const double w = cluster.weight();
    const Vec3& s = cluster.sum();
    const Mat3& p = cluster.outer();
    const Vec3 m_local = pose.rotation.transpose() * n;  // Normal, node frame.
    const double c = n.dot(pose.translation) + d;
    const Mat3 m_hat = Hat(m_local);
    // Residual of a point: m_local . p + c. Jacobian row: [n^T, (p x m)^T].
    const Vec3 s_cross_m = s.cross(m_local);
    const int row = 6 * a;
    out.hessian.block<3, 3>(row, row) = w * n * n.transpose();
    out.hessian.block<3, 3>(row, row + 3) = n * s_cross_m.transpose();
    out.hessian.block<3, 3>(row + 3, row) = s_cross_m * n.transpose();
    out.hessian.block<3, 3>(row + 3, row + 3) = m_hat * p * m_hat.transpose();
    out.gradient.segment<3>(row) = n * (m_local.dot(s) + w * c);
    out.gradient.segment<3>(row + 3) = -m_hat * (p * m_local + c * s);
    // Coupling with the plane parameters; plane Jacobian row:
    // [(x - mu)^T B, 1].
    const Vec3 world_sum = pose.rotation * s + w * pose.translation;
    pose_by_plane.block<3, 2>(row, 0) =
        n * ((world_sum - w * mu).transpose() * tangent);
    pose_by_plane.block<3, 1>(row, 2) = n * w;
    pose_by_plane.block<3, 2>(row + 3, 0) =
        -m_hat *
        (p * pose.rotation.transpose() +
         s * (pose.translation - mu).transpose()) *
        tangent;
    pose_by_plane.block<3, 1>(row + 3, 2) = -m_hat * s;
  }
  // Schur complement of the plane block (diagonal). The plane gradient is
  // zero at the best fit, so the gradient needs no correction.
  for (int k = 0; k < 3; ++k) {
    if (plane_diagonal[k] <= 0.0) continue;
    const Eigen::VectorXd column = pose_by_plane.col(k);
    out.hessian.noalias() -= column * column.transpose() / plane_diagonal[k];
  }
  return out;
}

}  // namespace gr
