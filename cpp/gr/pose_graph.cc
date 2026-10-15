#include "gr/pose_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>

#include "Eigen/Cholesky"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseCore"

#include "gr/check.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gr {
namespace {

using Triplets = std::vector<Eigen::Triplet<double>>;

double RobustRho(double chi2, double scale) {
  if (scale <= 0.0) return chi2;
  const double s2 = scale * scale;
  return s2 * std::log1p(chi2 / s2);
}

double RobustWeight(double chi2, double scale) {
  if (scale <= 0.0) return 1.0;
  return 1.0 / (1.0 + chi2 / (scale * scale));
}

Vec6 PriorResidual(const Pose& pose, const Pose& measurement, Mat6* jacobian) {
  const Vec3 rotation_error =
      So3Log(measurement.rotation.transpose() * pose.rotation);
  Vec6 e;
  e.head<3>() = pose.translation - measurement.translation;
  e.tail<3>() = rotation_error;
  if (jacobian != nullptr) {
    jacobian->setZero();
    jacobian->block<3, 3>(0, 0) = Mat3::Identity();
    jacobian->block<3, 3>(3, 3) = So3RightJacobianInverse(rotation_error);
  }
  return e;
}

Vec3 PositionResidual(const Pose& pose, const PositionFactor& factor,
                      Eigen::Matrix<double, 3, 6>* jacobian) {
  if (jacobian != nullptr) {
    jacobian->block<3, 3>(0, 0) = Mat3::Identity();
    jacobian->block<3, 3>(0, 3) = -pose.rotation * Hat(factor.lever);
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

void AddBlock(Triplets& triplets, int row0, int col0, const Mat6& block) {
  for (int r = 0; r < 6; ++r) {
    for (int c = 0; c < 6; ++c) {
      if (block(r, c) != 0.0) {
        triplets.emplace_back(row0 + r, col0 + c, block(r, c));
      }
    }
  }
}

// Columns [first, first + size) of a factor Jacobian belong to the variable
// at normal-equation offset `offset` (< 0: fixed variable).
struct Segment {
  int first = 0;
  int size = 0;
  int offset = -1;
};

// Adds J^T W J and J^T W e of a factor with a dense local Jacobian.
void AddDenseFactor(const Eigen::MatrixXd& jacobian,
                    const Eigen::MatrixXd& information,
                    const Eigen::VectorXd& residual,
                    const std::vector<Segment>& segments, Triplets& triplets,
                    Eigen::VectorXd& b) {
  const Eigen::MatrixXd jt_w = jacobian.transpose() * information;
  const Eigen::MatrixXd h = jt_w * jacobian;
  const Eigen::VectorXd g = jt_w * residual;
  for (const Segment& row : segments) {
    if (row.offset < 0) continue;
    b.segment(row.offset, row.size) += g.segment(row.first, row.size);
    for (const Segment& col : segments) {
      if (col.offset < 0) continue;
      for (int r = 0; r < row.size; ++r) {
        for (int c = 0; c < col.size; ++c) {
          const double value = h(row.first + r, col.first + c);
          if (value != 0.0) {
            triplets.emplace_back(row.offset + r, col.offset + c, value);
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
  body.body_by_node.block<3, 3>(0, 0) = Mat3::Identity();
  body.body_by_node.block<3, 3>(0, 3) =
      -world_from_node.rotation * Hat(node_from_body.translation);
  body.body_by_node.block<3, 3>(3, 3) = node_from_body.rotation.transpose();
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
  std::vector<Pose> out;
  out.reserve(factor.observations.size());
  for (const PlaneObservation& observation : factor.observations) {
    out.push_back(poses[static_cast<size_t>(observation.node)]);
  }
  return out;
}

// Weighted Gauss-Newton terms of one plane factor; `vars` holds the pose
// variable index of every observation (-1: fixed node).
struct PlaneTerm {
  bool valid = false;
  std::vector<int> vars;
  Eigen::MatrixXd hessian;
  Eigen::VectorXd gradient;
};

}  // namespace

int PoseGraph::AddNode(const Pose& initial, bool fixed) {
  poses_.push_back(initial);
  fixed_.push_back(fixed);
  inertial_index_.push_back(-1);
  return static_cast<int>(poses_.size()) - 1;
}

void PoseGraph::SetFixed(int i, bool fixed) {
  GR_CHECK(HasNode(i));
  fixed_[static_cast<size_t>(i)] = fixed;
}

void PoseGraph::SetPose(int i, const Pose& pose) {
  GR_CHECK(HasNode(i));
  poses_[static_cast<size_t>(i)] = pose;
}

void PoseGraph::AddInertialState(int i, const Vec3& velocity,
                                 const Vec6& bias) {
  GR_CHECK(HasNode(i) && !HasInertialState(i));
  inertial_index_[static_cast<size_t>(i)] =
      static_cast<int>(velocities_.size());
  velocities_.push_back(velocity);
  biases_.push_back(bias);
}

bool PoseGraph::HasInertialState(int i) const {
  return HasNode(i) && InertialIndex(i) >= 0;
}

const Vec3& PoseGraph::Velocity(int i) const {
  GR_CHECK(HasInertialState(i));
  return velocities_[static_cast<size_t>(InertialIndex(i))];
}

const Vec6& PoseGraph::Bias(int i) const {
  GR_CHECK(HasInertialState(i));
  return biases_[static_cast<size_t>(InertialIndex(i))];
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

void PoseGraph::AddBodyVelocity(int i, const Vec3& velocity_body,
                                const Vec3& sigma, const Pose& node_from_body) {
  GR_CHECK(HasInertialState(i));
  GR_CHECK((sigma.array() > 0.0).all());
  BodyVelocityFactor factor;
  factor.i = i;
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

void PoseGraph::AddBiasPrior(int i, const Vec6& bias, const Vec6& sigma) {
  GR_CHECK(HasInertialState(i));
  GR_CHECK((sigma.array() > 0.0).all());
  BiasFactor factor;
  factor.i = i;
  factor.bias = bias;
  factor.information = sigma.cwiseProduct(sigma).cwiseInverse().asDiagonal();
  bias_factors_.push_back(factor);
}

void PoseGraph::AddBetween(int i, int j, const Pose& measurement,
                           const Mat6& information, double robust_scale) {
  GR_CHECK(HasNode(i) && HasNode(j));
  between_.push_back({i, j, measurement, information, robust_scale});
}

void PoseGraph::AddPrior(int i, const Pose& measurement,
                         const Mat6& information) {
  GR_CHECK(HasNode(i));
  priors_.push_back({i, measurement, information});
}

void PoseGraph::AddPosition(int i, const Vec3& measurement,
                            const Mat3& information, const Vec3& lever) {
  GR_CHECK(HasNode(i));
  PositionFactor factor;
  factor.i = i;
  factor.measurement = measurement;
  factor.lever = lever;
  factor.information = information;
  positions_.push_back(factor);
}

void PoseGraph::AddGravity(int i, const Vec3& up_body, double sigma_rad) {
  GR_CHECK(HasNode(i));
  GravityFactor factor;
  factor.i = i;
  factor.up_body = up_body.normalized();
  factor.information = 1.0 / (sigma_rad * sigma_rad);
  gravity_.push_back(factor);
}

void PoseGraph::AddPlane(std::vector<PlaneObservation> observations,
                         double sigma, double robust_scale) {
  GR_CHECK(sigma > 0.0 && observations.size() >= 2);
  PlaneFactor factor;
  for (size_t a = 0; a < observations.size(); ++a) {
    GR_CHECK(HasNode(observations[a].node));
    GR_CHECK(observations[a].points.weight() > 0.0);
    for (size_t b = 0; b < a; ++b) {
      GR_CHECK(observations[a].node != observations[b].node);
    }
    factor.weight += observations[a].points.weight();
  }
  factor.observations = std::move(observations);
  factor.information = 1.0 / (sigma * sigma);
  factor.robust_scale = robust_scale;
  planes_.push_back(std::move(factor));
}

Vec6 PoseGraph::BetweenResidual(const Pose& pose_i, const Pose& pose_j,
                                const Pose& measurement, Mat6* jacobian_i,
                                Mat6* jacobian_j) {
  const Mat3 ri_t = pose_i.rotation.transpose();
  const Vec3 d = pose_j.translation - pose_i.translation;
  const Vec3 rotation_error =
      So3Log(measurement.rotation.transpose() * ri_t * pose_j.rotation);
  Vec6 e;
  e.head<3>() = ri_t * d - measurement.translation;
  e.tail<3>() = rotation_error;
  if (jacobian_i == nullptr && jacobian_j == nullptr) return e;
  const Mat3 jr_inv = So3RightJacobianInverse(rotation_error);
  if (jacobian_i != nullptr) {
    jacobian_i->setZero();
    jacobian_i->block<3, 3>(0, 0) = -ri_t;
    jacobian_i->block<3, 3>(0, 3) = Hat(ri_t * d);
    jacobian_i->block<3, 3>(3, 3) =
        -jr_inv * pose_j.rotation.transpose() * pose_i.rotation;
  }
  if (jacobian_j != nullptr) {
    jacobian_j->setZero();
    jacobian_j->block<3, 3>(0, 0) = ri_t;
    jacobian_j->block<3, 3>(3, 3) = jr_inv;
  }
  return e;
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
  std::vector<double> out(planes_.size());
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
  for (int64_t p = 0; p < num_planes; ++p) {
    const PlaneFactor& factor = planes_[static_cast<size_t>(p)];
    const double cost =
        PlaneCost(factor.observations, ObservationPoses(factor, poses_));
    out[static_cast<size_t>(p)] = std::sqrt(cost / factor.weight);
  }
  return out;
}

double PoseGraph::TotalCost(const State& state) const {
  const std::vector<Pose>& poses = state.poses;
  const auto at = [&](int i) -> const Pose& {
    return poses[static_cast<size_t>(i)];
  };
  const auto inertial = [&](int i) {
    return static_cast<size_t>(InertialIndex(i));
  };
  double cost = 0.0;
  for (const BetweenFactor& f : between_) {
    const Vec6 e = BetweenResidual(at(f.i), at(f.j), f.measurement);
    cost += RobustRho(e.dot(f.information * e), f.robust_scale);
  }
  for (const PriorFactor& f : priors_) {
    const Vec6 e = PriorResidual(at(f.i), f.measurement, nullptr);
    cost += e.dot(f.information * e);
  }
  for (const PositionFactor& f : positions_) {
    const Vec3 e = PositionResidual(at(f.i), f, nullptr);
    cost += e.dot(f.information * e);
  }
  for (const GravityFactor& f : gravity_) {
    cost += f.information * GravityResidual(at(f.i), f, nullptr).squaredNorm();
  }
  for (const ImuFactor& f : imu_) {
    const Vec9 e = ImuResidual(
        f, at(f.i), at(f.j), state.velocities[inertial(f.i)],
        state.velocities[inertial(f.j)], state.biases[inertial(f.i)]);
    cost += e.dot(f.information * e);
  }
  for (const BodyVelocityFactor& f : body_velocities_) {
    const Vec3 e =
        BodyVelocityResidual(f, at(f.i), state.velocities[inertial(f.i)]);
    cost += e.dot(f.information * e);
  }
  for (const BiasFactor& f : bias_factors_) {
    const Vec6 e =
        f.j < 0
            ? Vec6(state.biases[inertial(f.i)] - f.bias)
            : Vec6(state.biases[inertial(f.j)] - state.biases[inertial(f.i)]);
    cost += e.dot(f.information * e);
  }
  double plane_cost = 0.0;
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : plane_cost) schedule(dynamic, 64)
#endif
  for (int64_t p = 0; p < num_planes; ++p) {
    plane_cost += PlaneChi2(planes_[static_cast<size_t>(p)], poses);
  }
  return 0.5 * (cost + plane_cost);
}

std::vector<double> PoseGraph::BetweenChi2() const {
  std::vector<double> out;
  out.reserve(between_.size());
  for (const BetweenFactor& f : between_) {
    const Vec6 e =
        BetweenResidual(poses_[static_cast<size_t>(f.i)],
                        poses_[static_cast<size_t>(f.j)], f.measurement);
    out.push_back(e.dot(f.information * e));
  }
  return out;
}

std::vector<double> PoseGraph::ImuChi2() const {
  std::vector<double> out;
  out.reserve(imu_.size());
  for (const ImuFactor& f : imu_) {
    const size_t si = static_cast<size_t>(InertialIndex(f.i));
    const size_t sj = static_cast<size_t>(InertialIndex(f.j));
    const Vec9 e = ImuResidual(f, poses_[static_cast<size_t>(f.i)],
                               poses_[static_cast<size_t>(f.j)],
                               velocities_[si], velocities_[sj], biases_[si]);
    out.push_back(e.dot(f.information * e));
  }
  return out;
}

struct PoseGraph::LinearSystem {
  // All factors, except the pose-pose blocks of the plane factors when the
  // pose block is dense.
  Eigen::SparseMatrix<double> h;
  Eigen::VectorXd b;
  Eigen::MatrixXd pose_hessian;  // Dense layout: plane pose-pose blocks.
};

void PoseGraph::LinearizePlanes(const State& state, const Layout& layout,
                                LinearSystem& system,
                                Triplets& triplets) const {
  const int num_vars = layout.pose_dim / 6;
  if (layout.dense) {
    system.pose_hessian =
        Eigen::MatrixXd::Zero(layout.pose_dim, layout.pose_dim);
  }
  // Sparse layout: blocks per row variable, keyed by the column variable.
  std::vector<std::unordered_map<int, Mat6>> rows;
  if (!layout.dense) rows.resize(static_cast<size_t>(num_vars));
  const int64_t num_planes = static_cast<int64_t>(planes_.size());
  constexpr int64_t kChunk = 256;
  std::vector<PlaneTerm> terms(static_cast<size_t>(kChunk));
  for (int64_t start = 0; start < num_planes; start += kChunk) {
    const int64_t count = std::min(kChunk, num_planes - start);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int64_t q = 0; q < count; ++q) {
      const PlaneFactor& f = planes_[static_cast<size_t>(start + q)];
      PlaneTerm& term = terms[static_cast<size_t>(q)];
      term.valid = false;
      const std::optional<PlaneLinearization> linearization =
          LinearizePlane(f.observations, ObservationPoses(f, state.poses));
      if (!linearization) continue;
      const double mean_chi2 = linearization->cost / f.weight * f.information;
      const double scale =
          f.information * RobustWeight(mean_chi2, f.robust_scale);
      term.vars.clear();
      for (const PlaneObservation& observation : f.observations) {
        const int offset = layout.pose[static_cast<size_t>(observation.node)];
        term.vars.push_back(offset < 0 ? -1 : offset / 6);
      }
      term.hessian = scale * linearization->hessian;
      term.gradient = scale * linearization->gradient;
      term.valid = true;
    }
    for (int64_t q = 0; q < count; ++q) {
      const PlaneTerm& term = terms[static_cast<size_t>(q)];
      if (!term.valid) continue;
      for (size_t a = 0; a < term.vars.size(); ++a) {
        if (term.vars[a] < 0) continue;
        system.b.segment<6>(6 * term.vars[a]) +=
            term.gradient.segment<6>(static_cast<Eigen::Index>(6 * a));
      }
    }
    // Each thread adds the blocks of the row variables it owns.
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      const int thread = ThreadIndex();
      const int threads = ThreadCount();
      for (int64_t q = 0; q < count; ++q) {
        const PlaneTerm& term = terms[static_cast<size_t>(q)];
        if (!term.valid) continue;
        const size_t m = term.vars.size();
        for (size_t a = 0; a < m; ++a) {
          const int va = term.vars[a];
          if (va < 0 || va % threads != thread) continue;
          for (size_t c = 0; c < m; ++c) {
            const int vc = term.vars[c];
            if (vc < 0) continue;
            const Mat6 block =
                term.hessian.block<6, 6>(static_cast<Eigen::Index>(6 * a),
                                         static_cast<Eigen::Index>(6 * c));
            if (layout.dense) {
              system.pose_hessian.block<6, 6>(6 * va, 6 * vc) += block;
            } else {
              auto [it, inserted] =
                  rows[static_cast<size_t>(va)].try_emplace(vc, block);
              if (!inserted) it->second += block;
            }
          }
        }
      }
    }
  }
  for (size_t va = 0; va < rows.size(); ++va) {
    for (const auto& [vc, block] : rows[va]) {
      AddBlock(triplets, 6 * static_cast<int>(va), 6 * vc, block);
    }
  }
}

PoseGraph::LinearSystem PoseGraph::Linearize(const State& state,
                                             const Layout& layout) const {
  const int dim = layout.dim;
  // Pose blocks come first, so 6 * var(i) is node i's offset.
  const auto var = [&](int i) {
    const int offset = layout.pose[static_cast<size_t>(i)];
    return offset < 0 ? -1 : offset / 6;
  };
  const auto pose_at = [&](int i) -> const Pose& {
    return state.poses[static_cast<size_t>(i)];
  };
  const auto pose_offset = [&](int i) {
    return layout.pose[static_cast<size_t>(i)];
  };
  const auto inertial_offset = [&](int i) {
    return layout.inertial[static_cast<size_t>(InertialIndex(i))];
  };
  Triplets triplets;
  triplets.reserve(between_.size() * 144 +
                   (priors_.size() + positions_.size() + gravity_.size()) * 36 +
                   imu_.size() * 900 + body_velocities_.size() * 225 +
                   bias_factors_.size() * 324);
  LinearSystem system;
  system.b = Eigen::VectorXd::Zero(dim);
  for (const BetweenFactor& f : between_) {
    Mat6 ji, jj;
    const Vec6 e =
        BetweenResidual(pose_at(f.i), pose_at(f.j), f.measurement, &ji, &jj);
    const Mat6 w =
        RobustWeight(e.dot(f.information * e), f.robust_scale) * f.information;
    const int vi = var(f.i);
    const int vj = var(f.j);
    if (vi >= 0) {
      AddBlock(triplets, 6 * vi, 6 * vi, ji.transpose() * w * ji);
      system.b.segment<6>(6 * vi) += ji.transpose() * w * e;
    }
    if (vj >= 0) {
      AddBlock(triplets, 6 * vj, 6 * vj, jj.transpose() * w * jj);
      system.b.segment<6>(6 * vj) += jj.transpose() * w * e;
    }
    if (vi >= 0 && vj >= 0) {
      const Mat6 hij = ji.transpose() * w * jj;
      AddBlock(triplets, 6 * vi, 6 * vj, hij);
      AddBlock(triplets, 6 * vj, 6 * vi, hij.transpose());
    }
  }
  for (const PriorFactor& f : priors_) {
    const int vi = var(f.i);
    if (vi < 0) continue;
    Mat6 j;
    const Vec6 e = PriorResidual(pose_at(f.i), f.measurement, &j);
    AddBlock(triplets, 6 * vi, 6 * vi, j.transpose() * f.information * j);
    system.b.segment<6>(6 * vi) += j.transpose() * f.information * e;
  }
  for (const PositionFactor& f : positions_) {
    const int vi = var(f.i);
    if (vi < 0) continue;
    Eigen::Matrix<double, 3, 6> j;
    const Vec3 e = PositionResidual(pose_at(f.i), f, &j);
    AddBlock(triplets, 6 * vi, 6 * vi, j.transpose() * f.information * j);
    system.b.segment<6>(6 * vi) += j.transpose() * f.information * e;
  }
  for (const GravityFactor& f : gravity_) {
    const int vi = var(f.i);
    if (vi < 0) continue;
    Mat3 j;
    const Vec3 e = GravityResidual(pose_at(f.i), f, &j);
    Mat6 block = Mat6::Zero();
    block.block<3, 3>(3, 3) = f.information * j.transpose() * j;
    AddBlock(triplets, 6 * vi, 6 * vi, block);
    system.b.segment<3>(6 * vi + 3) += f.information * j.transpose() * e;
  }
  for (const ImuFactor& f : imu_) {
    const size_t si = static_cast<size_t>(InertialIndex(f.i));
    const size_t sj = static_cast<size_t>(InertialIndex(f.j));
    const BodyPose body_i = ToBody(pose_at(f.i), f.node_from_body);
    const BodyPose body_j = ToBody(pose_at(f.j), f.node_from_body);
    ImuPreintegration::Jacobian j24;
    const Vec9 e = f.preintegration.Residual(
        body_i.world_from_body, state.velocities[si], body_j.world_from_body,
        state.velocities[sj], state.biases[si], &j24);
    // Local columns: [pose_i, pose_j, inertial_i, inertial_j] (6, 6, 9, 9).
    Eigen::Matrix<double, 9, 6> by_body_i;
    by_body_i << j24.block<9, 3>(0, 3), j24.block<9, 3>(0, 0);
    Eigen::Matrix<double, 9, 6> by_body_j;
    by_body_j << j24.block<9, 3>(0, 12), j24.block<9, 3>(0, 9);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(9, 30);
    jacobian.block<9, 6>(0, 0) = by_body_i * body_i.body_by_node;
    jacobian.block<9, 6>(0, 6) = by_body_j * body_j.body_by_node;
    jacobian.block<9, 3>(0, 12) = j24.block<9, 3>(0, 6);
    jacobian.block<9, 6>(0, 15) = j24.block<9, 6>(0, 18);
    jacobian.block<9, 3>(0, 21) = j24.block<9, 3>(0, 15);
    AddDenseFactor(jacobian, f.information, e,
                   {{0, 6, pose_offset(f.i)},
                    {6, 6, pose_offset(f.j)},
                    {12, 9, inertial_offset(f.i)},
                    {21, 9, inertial_offset(f.j)}},
                   triplets, system.b);
  }
  for (const BodyVelocityFactor& f : body_velocities_) {
    const Vec3& velocity =
        state.velocities[static_cast<size_t>(InertialIndex(f.i))];
    const BodyPose body = ToBody(pose_at(f.i), f.node_from_body);
    const Mat3 body_from_world = body.world_from_body.rotation.transpose();
    const Vec3 e = BodyVelocityResidual(f, pose_at(f.i), velocity);
    Eigen::Matrix<double, 3, 6> by_body;
    by_body << Mat3::Zero(), Hat(body_from_world * velocity);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3, 15);
    jacobian.block<3, 6>(0, 0) = by_body * body.body_by_node;
    jacobian.block<3, 3>(0, 6) = body_from_world;
    AddDenseFactor(jacobian, f.information, e,
                   {{0, 6, pose_offset(f.i)}, {6, 9, inertial_offset(f.i)}},
                   triplets, system.b);
  }
  for (const BiasFactor& f : bias_factors_) {
    const Vec6& bias_i = state.biases[static_cast<size_t>(InertialIndex(f.i))];
    if (f.j < 0) {
      Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 9);
      jacobian.block<6, 6>(0, 3) = Mat6::Identity();
      AddDenseFactor(jacobian, f.information, bias_i - f.bias,
                     {{0, 9, inertial_offset(f.i)}}, triplets, system.b);
      continue;
    }
    const Vec6& bias_j = state.biases[static_cast<size_t>(InertialIndex(f.j))];
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(6, 18);
    jacobian.block<6, 6>(0, 3) = -Mat6::Identity();
    jacobian.block<6, 6>(0, 12) = Mat6::Identity();
    AddDenseFactor(jacobian, f.information, bias_j - bias_i,
                   {{0, 9, inertial_offset(f.i)}, {9, 9, inertial_offset(f.j)}},
                   triplets, system.b);
  }
  if (!planes_.empty()) LinearizePlanes(state, layout, system, triplets);
  system.h.resize(dim, dim);
  system.h.setFromTriplets(triplets.begin(), triplets.end());
  return system;
}

bool PoseGraph::Solve(const LinearSystem& system, const Layout& layout,
                      double lambda, Eigen::VectorXd& dx) const {
  const int dim = layout.dim;
  Eigen::VectorXd diagonal = system.h.diagonal();
  if (layout.dense) {
    diagonal.head(layout.pose_dim) += system.pose_hessian.diagonal();
  }
  const Eigen::VectorXd damping =
      lambda * diagonal.cwiseMax(Eigen::VectorXd::Constant(dim, 1e-9));
  if (!layout.dense) {
    Eigen::SparseMatrix<double> a = system.h;
    for (int k = 0; k < dim; ++k) a.coeffRef(k, k) += damping[k];
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(a);
    if (solver.info() != Eigen::Success) return false;
    dx = solver.solve(-system.b);
    return dx.allFinite();
  }
  // Dense pose block: eliminate the inertial states (sparse, chain
  // structured), then factor the reduced pose system.
  const int p = layout.pose_dim;
  const int q = dim - p;
  Eigen::MatrixXd reduced =
      system.pose_hessian + Eigen::MatrixXd(system.h.topLeftCorner(p, p));
  reduced.diagonal() += damping.head(p);
  Eigen::VectorXd rhs = -system.b.head(p);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> inertial_solver;
  Eigen::SparseMatrix<double> coupling;  // p x q
  if (q > 0) {
    Eigen::SparseMatrix<double> a_ii = system.h.bottomRightCorner(q, q);
    for (int k = 0; k < q; ++k) a_ii.coeffRef(k, k) += damping[p + k];
    inertial_solver.compute(a_ii);
    if (inertial_solver.info() != Eigen::Success) return false;
    coupling = system.h.topRightCorner(p, q);
    const Eigen::MatrixXd coupling_t = Eigen::MatrixXd(coupling.transpose());
    Eigen::MatrixXd solved(q, p);  // a_ii^-1 coupling^T
    constexpr int kColumns = 64;
    const int64_t num_chunks = (p + kColumns - 1) / kColumns;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
      const int first = static_cast<int>(chunk) * kColumns;
      const int width = std::min(kColumns, p - first);
      solved.middleCols(first, width) =
          inertial_solver.solve(coupling_t.middleCols(first, width));
    }
    reduced.noalias() -= coupling * solved;
    rhs += coupling * inertial_solver.solve(system.b.tail(q));
  }
  const Eigen::LLT<Eigen::MatrixXd> llt(reduced);
  if (llt.info() != Eigen::Success) return false;
  dx.resize(dim);
  dx.head(p) = llt.solve(rhs);
  if (q > 0) {
    dx.tail(q) = inertial_solver.solve(-system.b.tail(q) -
                                       coupling.transpose() * dx.head(p));
  }
  return dx.allFinite();
}

bool PoseGraph::DampedStep(const LinearSystem& system, const Layout& layout,
                           double& lambda, State& state, double& cost) const {
  for (int attempt = 0; attempt < 12; ++attempt) {
    Eigen::VectorXd dx;
    if (!Solve(system, layout, lambda, dx)) {
      lambda *= 10.0;
      continue;
    }
    State candidate = state;
    for (size_t i = 0; i < candidate.poses.size(); ++i) {
      const int offset = layout.pose[i];
      if (offset < 0) continue;
      Pose& p = candidate.poses[i];
      p.translation += dx.segment<3>(offset);
      p.rotation = p.rotation * So3Exp(dx.segment<3>(offset + 3));
      p.Normalize();
    }
    for (size_t s = 0; s < candidate.velocities.size(); ++s) {
      const int offset = layout.inertial[s];
      candidate.velocities[s] += dx.segment<3>(offset);
      candidate.biases[s] += dx.segment<6>(offset + 3);
    }
    const double new_cost = TotalCost(candidate);
    if (new_cost < cost) {
      std::swap(state, candidate);
      cost = new_cost;
      lambda = std::max(lambda / 10.0, 1e-12);
      return true;
    }
    lambda *= 10.0;
  }
  return false;
}

PoseGraphSummary PoseGraph::Optimize(const PoseGraphOptions& options) {
  PoseGraphSummary summary;
  const int n = static_cast<int>(poses_.size());
  if (n == 0) return summary;

  // Gauge: fix the first node if nothing anchors the graph. Gravity alone
  // leaves x, y, z and yaw free, so it does not anchor it.
  std::vector<bool> fixed = fixed_;
  const bool anchored =
      !priors_.empty() || !positions_.empty() ||
      std::any_of(fixed.begin(), fixed.end(), [](bool b) { return b; });
  if (!anchored) fixed[0] = true;
  Layout layout;
  layout.pose.assign(static_cast<size_t>(n), -1);
  for (size_t i = 0; i < layout.pose.size(); ++i) {
    if (!fixed[i]) {
      layout.pose[i] = layout.dim;
      layout.dim += 6;
    }
  }
  layout.pose_dim = layout.dim;
  layout.dense =
      !planes_.empty() && layout.pose_dim / 6 <= options.dense_max_pose_nodes;
  layout.inertial.resize(velocities_.size());
  for (int& offset : layout.inertial) {
    offset = layout.dim;
    layout.dim += 9;
  }
  if (layout.dim == 0) return summary;

  State x = CurrentState();
  double cost = TotalCost(x);
  summary.initial_cost = cost;
  double lambda = options.lambda_init;
  for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
    const LinearSystem system = Linearize(x, layout);
    const double previous_cost = cost;
    const bool accepted = DampedStep(system, layout, lambda, x, cost);
    summary.iterations = iteration + 1;
    if (!accepted) {
      summary.converged = true;  // No further decrease possible.
      break;
    }
    const double relative_decrease =
        (previous_cost - cost) / std::max(previous_cost, 1e-300);
    if (relative_decrease < options.rel_tol) {
      summary.converged = true;
      break;
    }
  }
  poses_ = std::move(x.poses);
  velocities_ = std::move(x.velocities);
  biases_ = std::move(x.biases);
  summary.final_cost = cost;
  return summary;
}

}  // namespace gr
