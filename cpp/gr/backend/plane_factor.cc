#include "gr/backend/plane_factor.h"

#include <algorithm>

#include "Eigen/Eigenvalues"

#include "gr/core/check.h"

namespace gr {
namespace {

// Positions of SelfAdjointEigenSolver's eigenvalues and eigenvectors
// (ascending): the plane normal, then the two in-plane axes.
constexpr int kNormalAxis = 0;
constexpr int kFirstTangent = 1;
constexpr int kSecondTangent = 2;
// Planes are degenerate when the smaller in-plane spread is below this
// fraction of the larger one (which is floored at kMinScale).
constexpr double kMinInPlaneRatio = 1e-12;
constexpr double kMinScale = 1e-300;
// The plane parameters eliminated by the Schur complement: the normal's
// tilts about the two in-plane axes, then the offset.
constexpr int kPlaneTilts = 2;
constexpr int kPlaneOffsetParam = kPlaneTilts;
constexpr int kPlaneParams = kPlaneTilts + 1;

}  // namespace

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
  const Mat3& rotation = pose.rotation;
  const Vec3& translation = pose.translation;
  const Vec3 rotated_sum = rotation * sum_;
  PointCluster out;
  out.weight_ = weight_;
  out.sum_ = rotated_sum + weight_ * translation;
  out.outer_ = rotation * outer_ * rotation.transpose() +
               rotated_sum * translation.transpose() +
               translation * rotated_sum.transpose() +
               weight_ * translation * translation.transpose();
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
  plane.normal = plane.axes.col(kNormalAxis);
  plane.offset = -plane.normal.dot(plane.centroid);
  return plane;
}

PointCluster MergeObservations(
    const std::vector<PlaneObservation>& observations,
    const std::vector<Pose>& poses) {
  GR_CHECK(observations.size() == poses.size());
  PointCluster merged;
  for (size_t index = 0; index < observations.size(); ++index) {
    merged.Add(observations[index].points.Transformed(poses[index]));
  }
  return merged;
}

double PlaneCost(const std::vector<PlaneObservation>& observations,
                 const std::vector<Pose>& poses) {
  const PointCluster merged = MergeObservations(observations, poses);
  if (merged.weight() <= 0.0) return 0.0;
  return merged.weight() * FitClusterPlane(merged).eigenvalues[kNormalAxis];
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
  out.cost = total_weight * out.plane.eigenvalues[kNormalAxis];
  // The in-plane spread must be positive for the plane to be determined.
  const double scale =
      std::max(out.plane.eigenvalues[kSecondTangent], kMinScale);
  if (out.plane.eigenvalues[kFirstTangent] <= kMinInPlaneRatio * scale) {
    return std::nullopt;
  }

  const Vec3& normal = out.plane.normal;
  const double offset = out.plane.offset;
  const Vec3& centroid = out.plane.centroid;
  Eigen::Matrix<double, kSpaceDim, kPlaneTilts> tangent;
  tangent << out.plane.axes.col(kFirstTangent),
      out.plane.axes.col(kSecondTangent);
  // Plane parameters: normal tilt (about the centroid) and offset. Their
  // normal-equation block is diagonal for this choice of axes.
  const Vec3 plane_diagonal(
      total_weight * out.plane.eigenvalues[kFirstTangent],
      total_weight * out.plane.eigenvalues[kSecondTangent], total_weight);

  const int count = static_cast<int>(observations.size());
  out.hessian = Eigen::MatrixXd::Zero(kPoseDim * count, kPoseDim * count);
  out.gradient = Eigen::VectorXd::Zero(kPoseDim * count);
  Eigen::MatrixXd pose_by_plane =
      Eigen::MatrixXd::Zero(kPoseDim * count, kPlaneParams);
  for (int index = 0; index < count; ++index) {
    const PointCluster& cluster =
        observations[static_cast<size_t>(index)].points;
    const Pose& pose = poses[static_cast<size_t>(index)];
    const double weight = cluster.weight();
    const Vec3& sum = cluster.sum();
    const Mat3& outer = cluster.outer();
    // The normal in the node frame, and the plane offset seen from there.
    const Vec3 local_normal = pose.rotation.transpose() * normal;
    const double local_offset = normal.dot(pose.translation) + offset;
    const Mat3 local_normal_hat = Hat(local_normal);
    // Residual of a point p: local_normal . p + local_offset. Jacobian row:
    // [n^T, (p x local_normal)^T].
    const Vec3 sum_cross_normal = sum.cross(local_normal);
    // The node's translation and rotation rows.
    const int translation_row = kPoseDim * index + kTranslationOffset;
    const int rotation_row = kPoseDim * index + kRotationOffset;
    out.hessian.block<kSpaceDim, kSpaceDim>(translation_row, translation_row) =
        weight * normal * normal.transpose();
    out.hessian.block<kSpaceDim, kSpaceDim>(translation_row, rotation_row) =
        normal * sum_cross_normal.transpose();
    out.hessian.block<kSpaceDim, kSpaceDim>(rotation_row, translation_row) =
        sum_cross_normal * normal.transpose();
    out.hessian.block<kSpaceDim, kSpaceDim>(rotation_row, rotation_row) =
        local_normal_hat * outer * local_normal_hat.transpose();
    out.gradient.segment<kSpaceDim>(translation_row) =
        normal * (local_normal.dot(sum) + weight * local_offset);
    out.gradient.segment<kSpaceDim>(rotation_row) =
        -local_normal_hat * (outer * local_normal + local_offset * sum);
    // Coupling with the plane parameters; plane Jacobian row:
    // [(x - mu)^T B, 1].
    const Vec3 world_sum = pose.rotation * sum + weight * pose.translation;
    pose_by_plane.block<kSpaceDim, kPlaneTilts>(translation_row, 0) =
        normal * ((world_sum - weight * centroid).transpose() * tangent);
    pose_by_plane.block<kSpaceDim, 1>(translation_row, kPlaneOffsetParam) =
        normal * weight;
    pose_by_plane.block<kSpaceDim, kPlaneTilts>(rotation_row, 0) =
        -local_normal_hat *
        (outer * pose.rotation.transpose() +
         sum * (pose.translation - centroid).transpose()) *
        tangent;
    pose_by_plane.block<kSpaceDim, 1>(rotation_row, kPlaneOffsetParam) =
        -local_normal_hat * sum;
  }
  // Schur complement of the plane block (diagonal). The plane gradient is
  // zero at the best fit, so the gradient needs no correction.
  for (int k = 0; k < kPlaneParams; ++k) {
    if (plane_diagonal[k] <= 0.0) continue;
    const Eigen::VectorXd column = pose_by_plane.col(k);
    out.hessian.noalias() -= column * column.transpose() / plane_diagonal[k];
  }
  return out;
}

}  // namespace gr
