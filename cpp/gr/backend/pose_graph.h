// Sparse pose-graph optimiser (Levenberg-Marquardt on R^3 x SO(3), right
// perturbation). Supports relative-pose (odometry / loop) factors with an
// optional Cauchy kernel, absolute pose priors, position factors (GNSS /
// surveyed targets with a lever arm) and gravity (roll / pitch) factors.
// Residual and perturbation order: [t, phi].
//
// Plane factors make it a LiDAR bundle adjustment: the points of several
// nodes on one plane, with the plane eliminated (plane_factor.h). Their
// Schur complements couple many nodes, so with plane factors the pose block
// is solved densely (inertial states eliminated first) up to
// PoseGraphOptions::dense_max_pose_nodes free nodes.
//
// Nodes may also carry an inertial state: the world-frame velocity of the IMU
// body and the IMU bias [gyro, accel]. IMU preintegration factors relate the
// states of consecutive nodes, body-velocity factors (wheel speed with the
// non-holonomic constraint) observe the velocity, and random-walk factors
// connect the biases. The IMU body frame is attached to the node frame by a
// fixed `node_from_body` transform (e.g. lidar_from_vehicle).

#ifndef GARAGE_RECON_GR_BACKEND_POSE_GRAPH_H_
#define GARAGE_RECON_GR_BACKEND_POSE_GRAPH_H_

#include <vector>

#include "Eigen/SparseCore"

#include "gr/backend/plane_factor.h"
#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/sensors/imu_preintegration.h"

namespace gr {

// Factors refer to nodes by index (`i`, `j`).
struct BetweenFactor {
  int i = 0;
  int j = 0;
  Pose measurement;  // Measured T_i^-1 T_j.
  Mat6 information = Mat6::Identity();
  double robust_scale = 0.0;  // Cauchy scale on sqrt(chi2); <= 0 disables it.
};

struct PriorFactor {
  int i = 0;
  Pose measurement;
  Mat6 information = Mat6::Identity();
};

struct PositionFactor {
  int i = 0;
  Vec3 measurement = Vec3::Zero();  // World position of `lever` (node frame).
  Vec3 lever = Vec3::Zero();
  Mat3 information = Mat3::Identity();
};

// Gravity direction measured in the node frame (accelerometer, motion
// compensated): constrains roll and pitch only. Essential for multi-level
// garages whose levels are connected by ramps alone.
struct GravityFactor {
  int i = 0;
  Vec3 up_body = Vec3::UnitZ();  // World +z expressed in the node frame.
  double information = 1.0;      // 1 / sigma^2 (rad^-2).
};

// IMU preintegration between the inertial states of nodes i and j.
struct ImuFactor {
  int i = 0;
  int j = 0;
  ImuPreintegration preintegration;
  Pose node_from_body;
  Mat9 information = Mat9::Identity();
};

// Velocity of the IMU body measured in the body frame at node i, e.g. the
// wheel speed along x with zero lateral and vertical speed.
struct BodyVelocityFactor {
  int i = 0;
  Vec3 velocity_body = Vec3::Zero();
  Pose node_from_body;
  Mat3 information = Mat3::Identity();
};

// Bias random walk (b_j - b_i ~ 0) or prior (b_i - bias ~ 0).
struct BiasFactor {
  int i = 0;
  int j = -1;  // -1: prior on node i.
  Vec6 bias = Vec6::Zero();
  Mat6 information = Mat6::Identity();
};

// Points of several nodes on one plane; the plane is eliminated.
struct PlaneFactor {
  std::vector<PlaneObservation> observations;  // Distinct nodes.
  double weight = 0.0;                         // Total point weight W.
  double information = 1.0;   // 1 / sigma^2 of a unit-weight point (m^-2).
  double robust_scale = 0.0;  // Cauchy scale on the mean point chi2.
};

struct PoseGraphOptions {
  int max_iterations = 50;
  double lambda_init = 1e-4;  // Initial Levenberg-Marquardt damping.
  double rel_tol = 1e-10;     // Stop below this relative cost decrease.
  // With plane factors, up to this many free nodes the pose block is solved
  // as a dense matrix after eliminating the inertial states; beyond it the
  // whole system goes to the sparse solver.
  int dense_max_pose_nodes = 2500;
};

struct PoseGraphSummary {
  int iterations = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  bool converged = false;
};

class PoseGraph {
 public:
  PoseGraph() = default;
  PoseGraph(const PoseGraph&) = default;
  PoseGraph& operator=(const PoseGraph&) = default;
  PoseGraph(PoseGraph&&) = default;
  PoseGraph& operator=(PoseGraph&&) = default;

  // Returns the node index.
  int AddNode(const Pose& initial, bool fixed = false);
  void SetFixed(int node, bool fixed);
  void SetPose(int node, const Pose& pose);
  // Gives the node a body velocity (world frame) and an IMU bias [gyro,
  // accel] to estimate; the inertial state is free even if the pose is fixed.
  void AddInertialState(int node, const Vec3& velocity, const Vec6& bias);

  // Node indices must refer to existing nodes.
  void AddBetween(int i, int j, const Pose& measurement,
                  const Mat6& information, double robust_scale = 0.0);
  void AddPrior(int node, const Pose& measurement, const Mat6& information);
  void AddPosition(int node, const Vec3& measurement, const Mat3& information,
                   const Vec3& lever = Vec3::Zero());
  void AddGravity(int node, const Vec3& up_body, double sigma_rad);
  // Factors on inertial states; the nodes must have one.
  void AddImu(int i, int j, const ImuPreintegration& preintegration,
              const Pose& node_from_body);
  void AddBodyVelocity(int node, const Vec3& velocity_body, const Vec3& sigma,
                       const Pose& node_from_body);
  // Random walk between the biases of i and j, standard deviation `sigma`.
  void AddBiasBetween(int i, int j, const Vec6& sigma);
  void AddBiasPrior(int node, const Vec6& bias, const Vec6& sigma);
  // `observations` must refer to distinct existing nodes and carry a positive
  // weight; `sigma` (m) is the point-to-plane noise of a unit-weight point.
  void AddPlane(std::vector<PlaneObservation> observations, double sigma,
                double robust_scale = 0.0);

  PoseGraphSummary Optimize(const PoseGraphOptions& options = {});

  size_t num_nodes() const { return poses_.size(); }
  const std::vector<Pose>& poses() const { return poses_; }
  bool HasInertialState(int node) const;
  // The node must have an inertial state.
  const Vec3& Velocity(int node) const;
  const Vec6& Bias(int node) const;
  // chi2 = e^T info e of every between / IMU factor at the current estimate.
  std::vector<double> BetweenChi2() const;
  std::vector<double> ImuChi2() const;
  size_t num_planes() const { return planes_.size(); }
  // RMS point-to-plane distance (m) of every plane factor.
  std::vector<double> PlaneRms() const;

  // Residual of a between factor and, if requested, its Jacobians.
  static Vec6 BetweenResidual(const Pose& pose_i, const Pose& pose_j,
                              const Pose& measurement,
                              Mat6* jacobian_i = nullptr,
                              Mat6* jacobian_j = nullptr);

 private:
  // inertial_index_ of a node without inertial state.
  static constexpr int kNoInertialState = -1;

  struct State {
    std::vector<Pose> poses;
    std::vector<Vec3> velocities;  // Per inertial state.
    std::vector<Vec6> biases;
  };
  // Column offsets of the variables in the normal equations (-1: fixed).
  // Pose blocks come first, then the inertial states.
  struct Layout {
    std::vector<int> pose;      // Per node.
    std::vector<int> inertial;  // Per inertial state: [v, gyro b, accel b].
    int pose_dim = 0;
    int dim = 0;
    bool dense = false;  // Dense pose block (plane factors).
  };
  struct LinearSystem;  // Gauss-Newton normal equations, defined in the .cc.

  bool HasNode(int node) const {
    return node >= 0 && node < static_cast<int>(poses_.size());
  }
  int InertialIndex(int node) const {
    return inertial_index_[static_cast<size_t>(node)];
  }
  State CurrentState() const;
  double TotalCost(const State& state) const;
  // chi2 of a plane factor (robust) at the given node poses.
  double PlaneChi2(const PlaneFactor& factor,
                   const std::vector<Pose>& poses) const;
  LinearSystem Linearize(const State& state, const Layout& layout) const;
  // Adds the plane factors to `system` (dense pose block) or `triplets`.
  void LinearizePlanes(const State& state, const Layout& layout,
                       LinearSystem* system,
                       std::vector<Eigen::Triplet<double>>* triplets) const;
  // Solves the damped normal equations for `step`; false if they are not
  // positive definite.
  bool Solve(const LinearSystem& system, const Layout& layout, double lambda,
             Eigen::VectorXd* step) const;
  // Levenberg-Marquardt steps with growing damping until the cost decreases
  // below `cost`; updates `lambda`, and `state` and `cost` on success.
  // Returns false if no step decreases it.
  bool DampedStep(const LinearSystem& system, const Layout& layout,
                  double* lambda, State* state, double* cost) const;

  std::vector<Pose> poses_;
  std::vector<bool> fixed_;
  std::vector<int> inertial_index_;  // Per node, or kNoInertialState.
  std::vector<Vec3> velocities_;
  std::vector<Vec6> biases_;
  std::vector<BetweenFactor> between_;
  std::vector<PriorFactor> priors_;
  std::vector<PositionFactor> positions_;
  std::vector<GravityFactor> gravity_;
  std::vector<ImuFactor> imu_;
  std::vector<BodyVelocityFactor> body_velocities_;
  std::vector<BiasFactor> bias_factors_;
  std::vector<PlaneFactor> planes_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_BACKEND_POSE_GRAPH_H_
