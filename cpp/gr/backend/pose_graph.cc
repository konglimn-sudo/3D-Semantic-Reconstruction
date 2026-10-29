#include "gr/backend/pose_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "Eigen/Cholesky"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseCore"

#include "gr/core/check.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gr {
namespace {

using Triplets = std::vector<Eigen::Triplet<double>>;
using SpaceByPose = Eigen::Matrix<double, kSpaceDim, kPoseDim>;
using ImuByPose = Eigen::Matrix<double, kImuResidualDim, kPoseDim>;

// Normal-equation offset of a fixed variable.
constexpr int kFixedVariable = -1;
// An inertial state is [velocity, gyro bias, accel bias]; a pose is [t, phi]
// (kTranslationOffset, kRotationOffset).
constexpr int kVelocityOffset = 0;
constexpr int kBiasOffset = kVelocityOffset + kSpaceDim;
constexpr int kInertialDim = kBiasOffset + kBiasDim;
// Local Jacobian columns of the factors on inertial states, by variable.
// IMU: [pose_i, pose_j, inertial_i, inertial_j].
constexpr int kImuPoseICol = 0;
constexpr int kImuPoseJCol = kImuPoseICol + kPoseDim;
constexpr int kImuInertialICol = kImuPoseJCol + kPoseDim;
constexpr int kImuInertialJCol = kImuInertialICol + kInertialDim;
constexpr int kImuFactorCols = kImuInertialJCol + kInertialDim;
// Body velocity: [pose, inertial].
constexpr int kBodyVelocityPoseCol = 0;
constexpr int kBodyVelocityInertialCol = kBodyVelocityPoseCol + kPoseDim;
constexpr int kBodyVelocityCols = kBodyVelocityInertialCol + kInertialDim;
// Bias: [inertial_i, inertial_j]; a prior has inertial_i only.
constexpr int kBiasInertialICol = 0;
constexpr int kBiasInertialJCol = kBiasInertialICol + kInertialDim;
constexpr int kBiasFactorCols = kBiasInertialJCol + kInertialDim;
// A between factor's Jacobian columns: [pose_i, pose_j].
constexpr int kBetweenCols = 2 * kPoseDim;
// A plane factor relates two nodes at least.
constexpr size_t kMinPlaneObservations = 2;
// Plane factors are linearized in batches of this many.
constexpr int64_t kPlanesPerBatch = 256;
// Levenberg-Marquardt damping: lambda grows by kDampingFactor after a
// rejected step and shrinks by it after an accepted one (never below
// kMinLambda), at most kMaxDampingAttempts times per iteration. It scales the
// Hessian diagonal, floored at kMinDampingDiagonal.
constexpr int kMaxDampingAttempts = 12;
constexpr double kDampingFactor = 10.0;
constexpr double kMinLambda = 1e-12;
constexpr double kMinDampingDiagonal = 1e-9;
// Guards the relative cost decrease against a zero cost.
constexpr double kMinCost = 1e-300;

// Cauchy loss of `chi2` with `scale` on sqrt(chi2); chi2 itself if `scale`
// <= 0.
double RobustRho(double chi2, double scale) {
  if (scale <= 0.0) return chi2;
  const double squared_scale = scale * scale;
  return squared_scale * std::log1p(chi2 / squared_scale);
}

// Derivative of RobustRho with respect to chi2 (the IRLS weight).
double RobustWeight(double chi2, double scale) {
  if (scale <= 0.0) return 1.0;
  return 1.0 / (1.0 + chi2 / (scale * scale));
}

Vec6 PriorResidual(const Pose& pose, const Pose& measurement, Mat6* jacobian) {
  const Vec3 rotation_error =
      So3Log(measurement.rotation.transpose() * pose.rotation);
  Vec6 residual;
  residual.segment<kSpaceDim>(kTranslationOffset) =
      pose.translation - measurement.translation;
  residual.segment<kSpaceDim>(kRotationOffset) = rotation_error;
  if (jacobian != nullptr) {
    jacobian->setZero();
    jacobian->block<kSpaceDim, kSpaceDim>(
        kTranslationOffset, kTranslationOffset) = Mat3::Identity();
    jacobian->block<kSpaceDim, kSpaceDim>(kRotationOffset, kRotationOffset) =
        So3RightJacobianInverse(rotation_error);
  }
  return residual;
}

Vec3 PositionResidual(const Pose& pose, const PositionFactor& factor,
                      SpaceByPose* jacobian) {
  if (jacobian != nullptr) {
    jacobian->block<kSpaceDim, kSpaceDim>(0, kTranslationOffset) =
        Mat3::Identity();
    jacobian->block<kSpaceDim, kSpaceDim>(0, kRotationOffset) =
        -pose.rotation * Hat(factor.lever);
  }
  return pose.translation + pose.rotation * factor.lever - factor.measurement;
}

// e = R^T z_world - up_body; d e / d phi (right perturbation) = Hat(R^T z).
Vec3 GravityResidual(const Pose& pose, const GravityFactor& factor,
                     Mat3* jacobian) {
  const Vec3 up = pose.rotation.transpose() * Vec3::UnitZ();
  if (jacobian != nullptr) *jacobian = Hat(up);
  return up - factor.up_body;
}

// Adds the nonzero entries of a pose-pose block at (first_row, first_col).
void AddBlock(int first_row, int first_col, const Mat6& block,
              Triplets* triplets) {
  for (int row = 0; row < kPoseDim; ++row) {
    for (int col = 0; col < kPoseDim; ++col) {
      if (block(row, col) != 0.0) {
        triplets->emplace_back(first_row + row, first_col + col,
                               block(row, col));
      }
    }
  }
}

// Columns [first, first + size) of a factor Jacobian belong to the variable
// at normal-equation offset `offset` (kFixedVariable: fixed variable).
struct Segment {
  int first = 0;
  int size = 0;
  int offset = kFixedVariable;
};

// Adds J^T W J to `triplets` and J^T W e to `gradient` for a factor with a
// dense local Jacobian.
void AddDenseFactor(const Eigen::MatrixXd& jacobian,
                    const Eigen::MatrixXd& information,
                    const Eigen::VectorXd& residual,
                    const std::vector<Segment>& segments, Triplets* triplets,
                    Eigen::VectorXd* gradient) {
  const Eigen::MatrixXd weighted_transpose = jacobian.transpose() * information;
  const Eigen::MatrixXd factor_hessian = weighted_transpose * jacobian;
  const Eigen::VectorXd factor_gradient = weighted_transpose * residual;
  for (const Segment& row_block : segments) {
    if (row_block.offset < 0) continue;
    gradient->segment(row_block.offset, row_block.size) +=
        factor_gradient.segment(row_block.first, row_block.size);
    for (const Segment& col_block : segments) {
      if (col_block.offset < 0) continue;
      for (int row = 0; row < row_block.size; ++row) {
        for (int col = 0; col < col_block.size; ++col) {
          const double value =
              factor_hessian(row_block.first + row, col_block.first + col);
          if (value != 0.0) {
            triplets->emplace_back(row_block.offset + row,
                                   col_block.offset + col, value);
          }
        }
      }
    }
  }
}

// The IMU body pose of a node and the Jacobian of the body's [position,
// rotation] perturbation with respect to the node's [t, phi] perturbation.
struct BodyPose {
  Pose world_from_body;
  Mat6 body_by_node = Mat6::Zero();
};

BodyPose ToBody(const Pose& world_from_node, const Pose& node_from_body) {
  BodyPose body;
  body.world_from_body = world_from_node * node_from_body;
  body.body_by_node.block<kSpaceDim, kSpaceDim>(
      kTranslationOffset, kTranslationOffset) = Mat3::Identity();
  body.body_by_node.block<kSpaceDim, kSpaceDim>(kTranslationOffset,
                                                kRotationOffset) =
      -world_from_node.rotation * Hat(node_from_body.translation);
  body.body_by_node.block<kSpaceDim, kSpaceDim>(
      kRotationOffset, kRotationOffset) = node_from_body.rotation.transpose();
  return body;
}

Vec9 ImuResidual(const ImuFactor& factor, const Pose& world_from_node_i,
                 const Pose& world_from_node_j, const Vec3& velocity_i,
                 const Vec3& velocity_j, const Vec6& bias_i) {
  return factor.preintegration.Residual(
      world_from_node_i * factor.node_from_body, velocity_i,
      world_from_node_j * factor.node_from_body, velocity_j, bias_i);
}

Vec3 BodyVelocityResidual(const BodyVelocityFactor& factor,
                          const Pose& world_from_node, const Vec3& velocity) {
  const Mat3 body_rotation =
      world_from_node.rotation * factor.node_from_body.rotation;
  return body_rotation.transpose() * velocity - factor.velocity_body;
}

int ThreadIndex() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

int ThreadCount() {
#ifdef _OPENMP
  return omp_get_num_threads();
#else
  return 1;
#endif
}

std::vector<Pose> ObservationPoses(const PlaneFactor& factor,
                                   const std::vector<Pose>& poses) {
  std::vector<Pose> observation_poses;
  observation_poses.reserve(factor.observations.size());
  for (const PlaneObservation& observation : factor.observations) {
    observation_poses.push_back(poses[static_cast<size_t>(observation.node)]);
  }
  return observation_poses;
}

// Weighted Gauss-Newton terms of one plane factor; `pose_variables` holds the
// pose variable index of every observation (kFixedVariable: fixed node).
struct PlaneTerm {
  bool valid = false;
  std::vector<int> pose_variables;
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;
};

}  // namespace

int PoseGraph::AddNode(const Pose& initial, bool fixed) {
  poses_.push_back(initial);
  fixed_.push_back(fixed);
  inertial_index_.push_back(kNoInertialState);
  return static_cast<int>(poses_.size()) - 1;
}

void PoseGraph::SetFixed(int node, bool fixed) {
  GR_CHECK(HasNode(node));
  fixed_[static_cast<size_t>(node)] = fixed;
}

void PoseGraph::SetPose(int node, const Pose& pose) {
  GR_CHECK(HasNode(node));
  poses_[static_cast<size_t>(node)] = pose;
}

void PoseGraph::AddInertialState(int node, const Vec3& velocity,
                                 const Vec6& bias) {
  GR_CHECK(HasNode(node) && !HasInertialState(node));
  inertial_index_[static_cast<size_t>(node)] =
      static_cast<int>(velocities_.size());
  velocities_.push_back(velocity);
  biases_.push_back(bias);
}

bool PoseGraph::HasInertialState(int node) const {
  return HasNode(node) && InertialIndex(node) != kNoInertialState;
}

const Vec3& PoseGraph::Velocity(int node) const {
  GR_CHECK(HasInertialState(node));
  return velocities_[static_cast<size_t>(InertialIndex(node))];
}

const Vec6& PoseGraph::Bias(int node) const {
  GR_CHECK(HasInertialState(node));
  return biases_[static_cast<size_t>(InertialIndex(node))];
}

void PoseGraph::AddImu(int i, int j, const ImuPreintegration& preintegration,
                       const Pose& node_from_body) {
  GR_CHECK(HasInertialState(i) && HasInertialState(j));
  ImuFactor factor;
  factor.i = i;
  factor.j = j;
  factor.preintegration = preintegration;
  factor.node_from_body = node_from_body;
  factor.information = preintegration.covariance().inverse();
  imu_.push_back(factor);
}

void PoseGraph::AddBodyVelocity(int node, const Vec3& velocity_body,
                                const Vec3& sigma, const Pose& node_from_body) {
  GR_CHECK(HasInertialState(node));
  GR_CHECK((sigma.array() > 0.0).all());
  BodyVelocityFactor factor;
  factor.i = node;
  factor.velocity_body = velocity_body;
  factor.node_from_body = node_from_body;
  factor.information = sigma.cwiseProduct(sigma).cwiseInverse().asDiagonal();
  body_velocities_.push_back(factor);
}

void PoseGraph::AddBiasBetween(int i, int j, const Vec6& sigma) {
  GR_CHECK(HasInertialState(i) && HasInertialState(j));
  GR_CHECK((sigma.array() > 0.0).all());
  BiasFactor factor;
  factor.i = i;
  factor.j = j;
  factor.information = sigma.cwiseProduct(sigma).cwiseInverse().asDiagonal();
  bias_factors_.push_back(factor);
}

void PoseGraph::AddBiasPrior(int node, const Vec6& bias, const Vec6& sigma) {
  GR_CHECK(HasInertialState(node));
  GR_CHECK((sigma.array() > 0.0).all());
  BiasFactor factor;
  factor.i = node;
  factor.bias = bias;
  factor.information = sigma.cwiseProduct(sigma).cwiseInverse().asDiagonal();
  bias_factors_.push_back(factor);
}

void PoseGraph::AddBetween(int i, int j, const Pose& measurement,
                           const Mat6& information, double robust_scale) {
  GR_CHECK(HasNode(i) && HasNode(j));
  between_.push_back({i, j, measurement, information, robust_scale});
}

void PoseGraph::AddPrior(int node, const Pose& measurement,
                         const Mat6& information) {
  GR_CHECK(HasNode(node));
  priors_.push_back({node, measurement, information});
}

void PoseGraph::AddPosition(int node, const Vec3& measurement,
                            const Mat3& information, const Vec3& lever) {
  GR_CHECK(HasNode(node));
  PositionFactor factor;
  factor.i = node;
  factor.measurement = measurement;
  factor.lever = lever;
  factor.information = information;
  positions_.push_back(factor);
}

void PoseGraph::AddGravity(int node, const Vec3& up_body, double sigma_rad) {
  GR_CHECK(HasNode(node));
  GravityFactor factor;
  factor.i = node;
  factor.up_body = up_body.normalized();
  factor.information = 1.0 / (sigma_rad * sigma_rad);
  gravity_.push_back(factor);
}

void PoseGraph::AddPlane(std::vector<PlaneObservation> observations,
                         double sigma, double robust_scale) {
  GR_CHECK(sigma > 0.0 && observations.size() >= kMinPlaneObservations);
  PlaneFactor factor;
  for (size_t index = 0; index < observations.size(); ++index) {
    GR_CHECK(HasNode(observations[index].node));
    GR_CHECK(observations[index].points.weight() > 0.0);
    for (size_t earlier = 0; earlier < index; ++earlier) {
      GR_CHECK(observations[index].node != observations[earlier].node);
    }
    factor.weight += observations[index].points.weight();
  }
  factor.observations = std::move(observations);
  factor.information = 1.0 / (sigma * sigma);
  factor.robust_scale = robust_scale;
  planes_.push_back(std::move(factor));
}

Vec6 PoseGraph::BetweenResidual(const Pose& pose_i, const Pose& pose_j,
                                const Pose& measurement, Mat6* jacobian_i,
                                Mat6* jacobian_j) {
  const Mat3 rotation_i_inverse = pose_i.rotation.transpose();
  const Vec3 displacement = pose_j.translation - pose_i.translation;
  const Vec3 rotation_error = So3Log(measurement.rotation.transpose() *
                                     rotation_i_inverse * pose_j.rotation);
  Vec6 residual;
  residual.segment<kSpaceDim>(kTranslationOffset) =
      rotation_i_inverse * displacement - measurement.translation;
  residual.segment<kSpaceDim>(kRotationOffset) = rotation_error;
  if (jacobian_i == nullptr && jacobian_j == nullptr) return residual;
  const Mat3 right_jacobian_inverse = So3RightJacobianInverse(rotation_error);
  if (jacobian_i != nullptr) {
    jacobian_i->setZero();
    jacobian_i->block<kSpaceDim, kSpaceDim>(
        kTranslationOffset, kTranslationOffset) = -rotation_i_inverse;
    jacobian_i->block<kSpaceDim, kSpaceDim>(kTranslationOffset,
                                            kRotationOffset) =
        Hat(rotation_i_inverse * displacement);
    jacobian_i->block<kSpaceDim, kSpaceDim>(kRotationOffset, kRotationOffset) =
        -right_jacobian_inverse * pose_j.rotation.transpose() * pose_i.rotation;
  }
  if (jacobian_j != nullptr) {
    jacobian_j->setZero();
    jacobian_j->block<kSpaceDim, kSpaceDim>(
        kTranslationOffset, kTranslationOffset) = rotation_i_inverse;
    jacobian_j->block<kSpaceDim, kSpaceDim>(kRotationOffset, kRotationOffset) =
        right_jacobian_inverse;
  }
  return residual;
}

PoseGraph::State PoseGraph::CurrentState() const {
  return State{poses_, velocities_, biases_};
}

double PoseGraph::PlaneChi2(const PlaneFactor& factor,
                            const std::vector<Pose>& poses) const {
  const double cost =
      PlaneCost(factor.observations, ObservationPoses(factor, poses));
  const double mean_chi2 = cost / factor.weight * factor.information;
  return factor.weight * RobustRho(mean_chi2, factor.robust_scale);
}

std::vector<double> PoseGraph::PlaneRms() const {
  std::vector<double> rms(planes_.size());
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
#ifdef _OPENMP
  constexpr int kPlanesPerChunk = 64;
#pragma omp parallel for schedule(dynamic, kPlanesPerChunk)
#endif
  for (int64_t plane = 0; plane < num_planes; ++plane) {
    const PlaneFactor& factor = planes_[static_cast<size_t>(plane)];
    const double cost =
        PlaneCost(factor.observations, ObservationPoses(factor, poses_));
    rms[static_cast<size_t>(plane)] = std::sqrt(cost / factor.weight);
  }
  return rms;
}

double PoseGraph::TotalCost(const State& state) const {
  const std::vector<Pose>& poses = state.poses;
  const auto pose_at = [&](int node) -> const Pose& {
    return poses[static_cast<size_t>(node)];
  };
  const auto inertial = [&](int node) {
    return static_cast<size_t>(InertialIndex(node));
  };
  double cost = 0.0;
  for (const BetweenFactor& factor : between_) {
    const Vec6 residual = BetweenResidual(pose_at(factor.i), pose_at(factor.j),
                                          factor.measurement);
    cost += RobustRho(residual.dot(factor.information * residual),
                      factor.robust_scale);
  }
  for (const PriorFactor& factor : priors_) {
    const Vec6 residual =
        PriorResidual(pose_at(factor.i), factor.measurement, nullptr);
    cost += residual.dot(factor.information * residual);
  }
  for (const PositionFactor& factor : positions_) {
    const Vec3 residual = PositionResidual(pose_at(factor.i), factor, nullptr);
    cost += residual.dot(factor.information * residual);
  }
  for (const GravityFactor& factor : gravity_) {
    cost += factor.information *
            GravityResidual(pose_at(factor.i), factor, nullptr).squaredNorm();
  }
  for (const ImuFactor& factor : imu_) {
    const Vec9 residual = ImuResidual(
        factor, pose_at(factor.i), pose_at(factor.j),
        state.velocities[inertial(factor.i)],
        state.velocities[inertial(factor.j)], state.biases[inertial(factor.i)]);
    cost += residual.dot(factor.information * residual);
  }
  for (const BodyVelocityFactor& factor : body_velocities_) {
    const Vec3 residual = BodyVelocityResidual(
        factor, pose_at(factor.i), state.velocities[inertial(factor.i)]);
    cost += residual.dot(factor.information * residual);
  }
  for (const BiasFactor& factor : bias_factors_) {
    const Vec6 residual =
        factor.j < 0 ? Vec6(state.biases[inertial(factor.i)] - factor.bias)
                     : Vec6(state.biases[inertial(factor.j)] -
                            state.biases[inertial(factor.i)]);
    cost += residual.dot(factor.information * residual);
  }
  // Summed in plane order, so that the cost does not depend on the number of
  // threads.
  std::vector<double> plane_chi2(planes_.size());
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
#ifdef _OPENMP
  constexpr int kPlanesPerChunk = 64;
#pragma omp parallel for schedule(dynamic, kPlanesPerChunk)
#endif
  for (int64_t plane = 0; plane < num_planes; ++plane) {
    plane_chi2[static_cast<size_t>(plane)] =
        PlaneChi2(planes_[static_cast<size_t>(plane)], poses);
  }
  double plane_cost = 0.0;
  for (const double chi2 : plane_chi2) plane_cost += chi2;
  return 0.5 * (cost + plane_cost);
}

std::vector<double> PoseGraph::BetweenChi2() const {
  std::vector<double> chi2;
  chi2.reserve(between_.size());
  for (const BetweenFactor& factor : between_) {
    const Vec6 residual = BetweenResidual(poses_[static_cast<size_t>(factor.i)],
                                          poses_[static_cast<size_t>(factor.j)],
                                          factor.measurement);
    chi2.push_back(residual.dot(factor.information * residual));
  }
  return chi2;
}

std::vector<double> PoseGraph::ImuChi2() const {
  std::vector<double> chi2;
  chi2.reserve(imu_.size());
  for (const ImuFactor& factor : imu_) {
    const size_t state_i = static_cast<size_t>(InertialIndex(factor.i));
    const size_t state_j = static_cast<size_t>(InertialIndex(factor.j));
    const Vec9 residual =
        ImuResidual(factor, poses_[static_cast<size_t>(factor.i)],
                    poses_[static_cast<size_t>(factor.j)], velocities_[state_i],
                    velocities_[state_j], biases_[state_i]);
    chi2.push_back(residual.dot(factor.information * residual));
  }
  return chi2;
}

struct PoseGraph::LinearSystem {
  // All factors, except the pose-pose blocks of the plane factors when the
  // pose block is dense.
  Eigen::SparseMatrix<double> hessian;
  Eigen::VectorXd gradient;
  Eigen::MatrixXd pose_hessian;  // Dense layout: plane pose-pose blocks.
};

void PoseGraph::LinearizePlanes(const State& state, const Layout& layout,
                                LinearSystem* system,
                                Triplets* triplets) const {
  const int num_variables = layout.pose_dim / kPoseDim;
  if (layout.dense) {
    system->pose_hessian =
        Eigen::MatrixXd::Zero(layout.pose_dim, layout.pose_dim);
  }
  // Sparse layout: blocks per row variable, keyed by the column variable.
  std::vector<std::unordered_map<int, Mat6>> rows;
  if (!layout.dense) rows.resize(static_cast<size_t>(num_variables));
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
  std::vector<PlaneTerm> terms(static_cast<size_t>(kPlanesPerBatch));
  for (int64_t start = 0; start < num_planes; start += kPlanesPerBatch) {
    const int64_t count = std::min(kPlanesPerBatch, num_planes - start);
#ifdef _OPENMP
    constexpr int kPlanesPerChunk = 4;
#pragma omp parallel for schedule(dynamic, kPlanesPerChunk)
#endif
    for (int64_t slot = 0; slot < count; ++slot) {
      const PlaneFactor& factor = planes_[static_cast<size_t>(start + slot)];
      PlaneTerm& term = terms[static_cast<size_t>(slot)];
      term.valid = false;
      const std::optional<PlaneLinearization> linearization = LinearizePlane(
          factor.observations, ObservationPoses(factor, state.poses));
      if (!linearization) continue;
      const double mean_chi2 =
          linearization->cost / factor.weight * factor.information;
      const double scale =
          factor.information * RobustWeight(mean_chi2, factor.robust_scale);
      term.pose_variables.clear();
      for (const PlaneObservation& observation : factor.observations) {
        const int offset = layout.pose[static_cast<size_t>(observation.node)];
        term.pose_variables.push_back(offset < 0 ? kFixedVariable
                                                 : offset / kPoseDim);
      }
      term.hessian = scale * linearization->hessian;
      term.gradient = scale * linearization->gradient;
      term.valid = true;
    }
    for (int64_t slot = 0; slot < count; ++slot) {
      const PlaneTerm& term = terms[static_cast<size_t>(slot)];
      if (!term.valid) continue;
      for (size_t observation = 0; observation < term.pose_variables.size();
           ++observation) {
        const int variable = term.pose_variables[observation];
        if (variable < 0) continue;
        system->gradient.segment<kPoseDim>(kPoseDim * variable) +=
            term.gradient.segment<kPoseDim>(
                static_cast<Eigen::Index>(kPoseDim * observation));
      }
    }
    // Each thread adds the blocks of the row variables it owns.
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      const int thread = ThreadIndex();
      const int num_threads = ThreadCount();
      for (int64_t slot = 0; slot < count; ++slot) {
        const PlaneTerm& term = terms[static_cast<size_t>(slot)];
        if (!term.valid) continue;
        const size_t num_observations = term.pose_variables.size();
        for (size_t row = 0; row < num_observations; ++row) {
          const int row_variable = term.pose_variables[row];
          if (row_variable < 0 || row_variable % num_threads != thread) {
            continue;
          }
          for (size_t col = 0; col < num_observations; ++col) {
            const int col_variable = term.pose_variables[col];
            if (col_variable < 0) continue;
            const Mat6 block = term.hessian.block<kPoseDim, kPoseDim>(
                static_cast<Eigen::Index>(kPoseDim * row),
                static_cast<Eigen::Index>(kPoseDim * col));
            if (layout.dense) {
              system->pose_hessian.block<kPoseDim, kPoseDim>(
                  kPoseDim * row_variable, kPoseDim * col_variable) += block;
            } else {
              auto [entry, inserted] =
                  rows[static_cast<size_t>(row_variable)].try_emplace(
                      col_variable, block);
              if (!inserted) entry->second += block;
            }
          }
        }
      }
    }
  }
  for (size_t row_variable = 0; row_variable < rows.size(); ++row_variable) {
    for (const auto& [col_variable, block] : rows[row_variable]) {
      AddBlock(kPoseDim * static_cast<int>(row_variable),
               kPoseDim * col_variable, block, triplets);
    }
  }
}

PoseGraph::LinearSystem PoseGraph::Linearize(const State& state,
                                             const Layout& layout) const {
  const int dim = layout.dim;
  // Pose blocks come first, so kPoseDim * pose_variable(node) is the node's
  // offset.
  const auto pose_variable = [&](int node) {
    const int offset = layout.pose[static_cast<size_t>(node)];
    return offset < 0 ? kFixedVariable : offset / kPoseDim;
  };
  const auto pose_at = [&](int node) -> const Pose& {
    return state.poses[static_cast<size_t>(node)];
  };
  const auto pose_offset = [&](int node) {
    return layout.pose[static_cast<size_t>(node)];
  };
  const auto inertial_offset = [&](int node) {
    return layout.inertial[static_cast<size_t>(InertialIndex(node))];
  };
  // Hessian entries of each kind of factor (at most).
  constexpr size_t kBetweenEntries = kBetweenCols * kBetweenCols;
  constexpr size_t kPoseEntries = kPoseDim * kPoseDim;
  constexpr size_t kImuEntries = kImuFactorCols * kImuFactorCols;
  constexpr size_t kBodyVelocityEntries = kBodyVelocityCols * kBodyVelocityCols;
  constexpr size_t kBiasEntries = kBiasFactorCols * kBiasFactorCols;
  Triplets triplets;
  triplets.reserve(between_.size() * kBetweenEntries +
                   (priors_.size() + positions_.size() + gravity_.size()) *
                       kPoseEntries +
                   imu_.size() * kImuEntries +
                   body_velocities_.size() * kBodyVelocityEntries +
                   bias_factors_.size() * kBiasEntries);
  LinearSystem system;
  system.gradient = Eigen::VectorXd::Zero(dim);
  for (const BetweenFactor& factor : between_) {
    Mat6 jacobian_i;
    Mat6 jacobian_j;
    const Vec6 residual =
        BetweenResidual(pose_at(factor.i), pose_at(factor.j),
                        factor.measurement, &jacobian_i, &jacobian_j);
    const Mat6 weight =
        RobustWeight(residual.dot(factor.information * residual),
                     factor.robust_scale) *
        factor.information;
    const int variable_i = pose_variable(factor.i);
    const int variable_j = pose_variable(factor.j);
    if (variable_i >= 0) {
      AddBlock(kPoseDim * variable_i, kPoseDim * variable_i,
               jacobian_i.transpose() * weight * jacobian_i, &triplets);
      system.gradient.segment<kPoseDim>(kPoseDim * variable_i) +=
          jacobian_i.transpose() * weight * residual;
    }
    if (variable_j >= 0) {
      AddBlock(kPoseDim * variable_j, kPoseDim * variable_j,
               jacobian_j.transpose() * weight * jacobian_j, &triplets);
      system.gradient.segment<kPoseDim>(kPoseDim * variable_j) +=
          jacobian_j.transpose() * weight * residual;
    }
    if (variable_i >= 0 && variable_j >= 0) {
      const Mat6 coupling = jacobian_i.transpose() * weight * jacobian_j;
      AddBlock(kPoseDim * variable_i, kPoseDim * variable_j, coupling,
               &triplets);
      AddBlock(kPoseDim * variable_j, kPoseDim * variable_i,
               coupling.transpose(), &triplets);
    }
  }
  for (const PriorFactor& factor : priors_) {
    const int variable = pose_variable(factor.i);
    if (variable < 0) continue;
    Mat6 jacobian;
    const Vec6 residual =
        PriorResidual(pose_at(factor.i), factor.measurement, &jacobian);
    AddBlock(kPoseDim * variable, kPoseDim * variable,
             jacobian.transpose() * factor.information * jacobian, &triplets);
    system.gradient.segment<kPoseDim>(kPoseDim * variable) +=
        jacobian.transpose() * factor.information * residual;
  }
  for (const PositionFactor& factor : positions_) {
    const int variable = pose_variable(factor.i);
    if (variable < 0) continue;
    SpaceByPose jacobian;
    const Vec3 residual =
        PositionResidual(pose_at(factor.i), factor, &jacobian);
    AddBlock(kPoseDim * variable, kPoseDim * variable,
             jacobian.transpose() * factor.information * jacobian, &triplets);
    system.gradient.segment<kPoseDim>(kPoseDim * variable) +=
        jacobian.transpose() * factor.information * residual;
  }
  for (const GravityFactor& factor : gravity_) {
    const int variable = pose_variable(factor.i);
    if (variable < 0) continue;
    Mat3 jacobian;
    const Vec3 residual = GravityResidual(pose_at(factor.i), factor, &jacobian);
    Mat6 block = Mat6::Zero();
    block.block<kSpaceDim, kSpaceDim>(kRotationOffset, kRotationOffset) =
        factor.information * jacobian.transpose() * jacobian;
    AddBlock(kPoseDim * variable, kPoseDim * variable, block, &triplets);
    system.gradient.segment<kSpaceDim>(kPoseDim * variable + kRotationOffset) +=
        factor.information * jacobian.transpose() * residual;
  }
  for (const ImuFactor& factor : imu_) {
    const size_t state_i = static_cast<size_t>(InertialIndex(factor.i));
    const size_t state_j = static_cast<size_t>(InertialIndex(factor.j));
    const BodyPose body_i = ToBody(pose_at(factor.i), factor.node_from_body);
    const BodyPose body_j = ToBody(pose_at(factor.j), factor.node_from_body);
    ImuPreintegration::Jacobian by_preintegration;
    const Vec9 residual = factor.preintegration.Residual(
        body_i.world_from_body, state.velocities[state_i],
        body_j.world_from_body, state.velocities[state_j],
        state.biases[state_i], &by_preintegration);
    const auto columns = [&by_preintegration](int first) {
      return by_preintegration.block<kImuResidualDim, kSpaceDim>(0, first);
    };
    // The body pose blocks in the [position, rotation] order of a pose.
    ImuByPose by_body_i;
    by_body_i << columns(ImuPreintegration::kPositionICol),
        columns(ImuPreintegration::kRotationICol);
    ImuByPose by_body_j;
    by_body_j << columns(ImuPreintegration::kPositionJCol),
        columns(ImuPreintegration::kRotationJCol);
    Eigen::MatrixXd jacobian =
        Eigen::MatrixXd::Zero(kImuResidualDim, kImuFactorCols);
    jacobian.block<kImuResidualDim, kPoseDim>(0, kImuPoseICol) =
        by_body_i * body_i.body_by_node;
    jacobian.block<kImuResidualDim, kPoseDim>(0, kImuPoseJCol) =
        by_body_j * body_j.body_by_node;
    jacobian.block<kImuResidualDim, kSpaceDim>(
        0, kImuInertialICol + kVelocityOffset) =
        columns(ImuPreintegration::kVelocityICol);
    jacobian.block<kImuResidualDim, kBiasDim>(0,
                                              kImuInertialICol + kBiasOffset) =
        by_preintegration.block<kImuResidualDim, kBiasDim>(
            0, ImuPreintegration::kGyroBiasCol);
    jacobian.block<kImuResidualDim, kSpaceDim>(
        0, kImuInertialJCol + kVelocityOffset) =
        columns(ImuPreintegration::kVelocityJCol);
    AddDenseFactor(
        jacobian, factor.information, residual,
        {{kImuPoseICol, kPoseDim, pose_offset(factor.i)},
         {kImuPoseJCol, kPoseDim, pose_offset(factor.j)},
         {kImuInertialICol, kInertialDim, inertial_offset(factor.i)},
         {kImuInertialJCol, kInertialDim, inertial_offset(factor.j)}},
        &triplets, &system.gradient);
  }
  for (const BodyVelocityFactor& factor : body_velocities_) {
    const Vec3& velocity =
        state.velocities[static_cast<size_t>(InertialIndex(factor.i))];
    const BodyPose body = ToBody(pose_at(factor.i), factor.node_from_body);
    const Mat3 body_from_world = body.world_from_body.rotation.transpose();
    const Vec3 residual =
        BodyVelocityResidual(factor, pose_at(factor.i), velocity);
    // By the body [position, rotation].
    SpaceByPose by_body;
    by_body << Mat3::Zero(), Hat(body_from_world * velocity);
    Eigen::MatrixXd jacobian =
        Eigen::MatrixXd::Zero(kSpaceDim, kBodyVelocityCols);
    jacobian.block<kSpaceDim, kPoseDim>(0, kBodyVelocityPoseCol) =
        by_body * body.body_by_node;
    jacobian.block<kSpaceDim, kSpaceDim>(
        0, kBodyVelocityInertialCol + kVelocityOffset) = body_from_world;
    AddDenseFactor(
        jacobian, factor.information, residual,
        {{kBodyVelocityPoseCol, kPoseDim, pose_offset(factor.i)},
         {kBodyVelocityInertialCol, kInertialDim, inertial_offset(factor.i)}},
        &triplets, &system.gradient);
  }
  for (const BiasFactor& factor : bias_factors_) {
    const Vec6& bias_i =
        state.biases[static_cast<size_t>(InertialIndex(factor.i))];
    if (factor.j < 0) {
      Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(kBiasDim, kInertialDim);
      jacobian.block<kBiasDim, kBiasDim>(0, kBiasInertialICol + kBiasOffset) =
          Mat6::Identity();
      AddDenseFactor(
          jacobian, factor.information, bias_i - factor.bias,
          {{kBiasInertialICol, kInertialDim, inertial_offset(factor.i)}},
          &triplets, &system.gradient);
      continue;
    }
    const Vec6& bias_j =
        state.biases[static_cast<size_t>(InertialIndex(factor.j))];
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(kBiasDim, kBiasFactorCols);
    jacobian.block<kBiasDim, kBiasDim>(0, kBiasInertialICol + kBiasOffset) =
        -Mat6::Identity();
    jacobian.block<kBiasDim, kBiasDim>(0, kBiasInertialJCol + kBiasOffset) =
        Mat6::Identity();
    AddDenseFactor(
        jacobian, factor.information, bias_j - bias_i,
        {{kBiasInertialICol, kInertialDim, inertial_offset(factor.i)},
         {kBiasInertialJCol, kInertialDim, inertial_offset(factor.j)}},
        &triplets, &system.gradient);
  }
  if (!planes_.empty()) LinearizePlanes(state, layout, &system, &triplets);
  system.hessian.resize(dim, dim);
  system.hessian.setFromTriplets(triplets.begin(), triplets.end());
  return system;
}

bool PoseGraph::Solve(const LinearSystem& system, const Layout& layout,
                      double lambda, Eigen::VectorXd* step) const {
  const int dim = layout.dim;
  Eigen::VectorXd diagonal = system.hessian.diagonal();
  if (layout.dense) {
    diagonal.head(layout.pose_dim) += system.pose_hessian.diagonal();
  }
  const Eigen::VectorXd damping =
      lambda *
      diagonal.cwiseMax(Eigen::VectorXd::Constant(dim, kMinDampingDiagonal));
  if (!layout.dense) {
    Eigen::SparseMatrix<double> damped = system.hessian;
    for (int index = 0; index < dim; ++index) {
      damped.coeffRef(index, index) += damping[index];
    }
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(damped);
    if (solver.info() != Eigen::Success) return false;
    *step = solver.solve(-system.gradient);
    return step->allFinite();
  }
  // Dense pose block: eliminate the inertial states (sparse, chain
  // structured), then factor the reduced pose system.
  const int pose_dim = layout.pose_dim;
  const int inertial_dim = dim - pose_dim;
  Eigen::MatrixXd reduced =
      system.pose_hessian +
      Eigen::MatrixXd(system.hessian.topLeftCorner(pose_dim, pose_dim));
  reduced.diagonal() += damping.head(pose_dim);
  Eigen::VectorXd rhs = -system.gradient.head(pose_dim);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> inertial_solver;
  Eigen::SparseMatrix<double> coupling;  // pose_dim x inertial_dim
  if (inertial_dim > 0) {
    Eigen::SparseMatrix<double> damped_inertial =
        system.hessian.bottomRightCorner(inertial_dim, inertial_dim);
    for (int index = 0; index < inertial_dim; ++index) {
      damped_inertial.coeffRef(index, index) += damping[pose_dim + index];
    }
    inertial_solver.compute(damped_inertial);
    if (inertial_solver.info() != Eigen::Success) return false;
    coupling = system.hessian.topRightCorner(pose_dim, inertial_dim);
    const Eigen::MatrixXd coupling_transpose =
        Eigen::MatrixXd(coupling.transpose());
    // damped_inertial^-1 coupling^T
    Eigen::MatrixXd solved(inertial_dim, pose_dim);
    constexpr int kColumnsPerChunk = 64;
    const int64_t num_chunks =
        (pose_dim + kColumnsPerChunk - 1) / kColumnsPerChunk;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      const int first = static_cast<int>(chunk) * kColumnsPerChunk;
      const int width = std::min(kColumnsPerChunk, pose_dim - first);
      solved.middleCols(first, width) =
          inertial_solver.solve(coupling_transpose.middleCols(first, width));
    }
    reduced.noalias() -= coupling * solved;
    rhs += coupling * inertial_solver.solve(system.gradient.tail(inertial_dim));
  }
  const Eigen::LLT<Eigen::MatrixXd> llt(reduced);
  if (llt.info() != Eigen::Success) return false;
  step->resize(dim);
  step->head(pose_dim) = llt.solve(rhs);
  if (inertial_dim > 0) {
    step->tail(inertial_dim) =
        inertial_solver.solve(-system.gradient.tail(inertial_dim) -
                              coupling.transpose() * step->head(pose_dim));
  }
  return step->allFinite();
}

bool PoseGraph::DampedStep(const LinearSystem& system, const Layout& layout,
                           double* lambda, State* state, double* cost) const {
  for (int attempt = 0; attempt < kMaxDampingAttempts; ++attempt) {
    Eigen::VectorXd step;
    if (!Solve(system, layout, *lambda, &step)) {
      *lambda *= kDampingFactor;
      continue;
    }
    State candidate = *state;
    for (size_t node = 0; node < candidate.poses.size(); ++node) {
      const int offset = layout.pose[node];
      if (offset < 0) continue;
      Pose& pose = candidate.poses[node];
      pose.translation += step.segment<kSpaceDim>(offset + kTranslationOffset);
      pose.rotation = pose.rotation *
                      So3Exp(step.segment<kSpaceDim>(offset + kRotationOffset));
      pose.Normalize();
    }
    for (size_t inertial = 0; inertial < candidate.velocities.size();
         ++inertial) {
      const int offset = layout.inertial[inertial];
      candidate.velocities[inertial] +=
          step.segment<kSpaceDim>(offset + kVelocityOffset);
      candidate.biases[inertial] +=
          step.segment<kBiasDim>(offset + kBiasOffset);
    }
    const double new_cost = TotalCost(candidate);
    if (new_cost < *cost) {
      std::swap(*state, candidate);
      *cost = new_cost;
      *lambda = std::max(*lambda / kDampingFactor, kMinLambda);
      return true;
    }
    *lambda *= kDampingFactor;
  }
  return false;
}

PoseGraphSummary PoseGraph::Optimize(const PoseGraphOptions& options) {
  PoseGraphSummary summary;
  if (poses_.empty()) return summary;

  // Gauge: fix the first node if nothing anchors the graph. Gravity alone
  // leaves x, y, z and yaw free, so it does not anchor it.
  std::vector<bool> fixed = fixed_;
  const bool anchored = !priors_.empty() || !positions_.empty() ||
                        std::any_of(fixed.begin(), fixed.end(),
                                    [](bool node_fixed) { return node_fixed; });
  if (!anchored) fixed[0] = true;
  Layout layout;
  layout.pose.assign(poses_.size(), kFixedVariable);
  for (size_t node = 0; node < layout.pose.size(); ++node) {
    if (!fixed[node]) {
      layout.pose[node] = layout.dim;
      layout.dim += kPoseDim;
    }
  }
  layout.pose_dim = layout.dim;
  layout.dense = !planes_.empty() &&
                 layout.pose_dim / kPoseDim <= options.dense_max_pose_nodes;
  layout.inertial.resize(velocities_.size());
  for (int& offset : layout.inertial) {
    offset = layout.dim;
    layout.dim += kInertialDim;
  }
  if (layout.dim == 0) return summary;

  State state = CurrentState();
  double cost = TotalCost(state);
  summary.initial_cost = cost;
  double lambda = options.lambda_init;
  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    const LinearSystem system = Linearize(state, layout);
    const double previous_cost = cost;
    const bool accepted = DampedStep(system, layout, &lambda, &state, &cost);
    summary.iterations = iteration + 1;
    if (!accepted) {
      summary.converged = true;  // No further decrease possible.
      break;
    }
    const double relative_decrease =
        (previous_cost - cost) / std::max(previous_cost, kMinCost);
    if (relative_decrease < options.rel_tol) {
      summary.converged = true;
      break;
    }
  }
  poses_ = std::move(state.poses);
  velocities_ = std::move(state.velocities);
  biases_ = std::move(state.biases);
  summary.final_cost = cost;
  return summary;
}

}  // namespace gr
