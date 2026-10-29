#include "gr/backend/pose_graph.h"

#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include "Eigen/QR"

#include "gr/sensors/imu_preintegration.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

// The simulated vehicle drives a circle at kSpeed and kYawRate, its IMU at
// the vehicle origin sampled every kImuPeriod; graph nodes follow every
// kNodePeriod from kFirstNodeStamp.
constexpr double kSpeed = 3.0;           // m/s
constexpr double kYawRate = 0.3;         // rad/s
constexpr double kImuPeriod = 0.005;     // s
constexpr double kNodePeriod = 0.5;      // s
constexpr double kFirstNodeStamp = 1.0;  // s
// Shared factor noise.
constexpr double kAnchorSigmaT = 0.001;  // m
constexpr double kAnchorSigmaR = 0.01;   // deg
constexpr double kGyroBiasWalk = 1e-5;   // rad/s per node interval
constexpr double kAccelBiasWalk = 1e-4;  // m/s^2 per node interval
constexpr double kGyroBiasPrior = 0.02;  // rad/s
constexpr double kAccelBiasPrior = 0.2;  // m/s^2
constexpr double kPointSigma = 0.005;    // m, plane factors
constexpr double kRobustScale = 3.0;
constexpr double kOdometryInformation = 100.0;
// A 8 x 8 grid of points per sampled rectangle.
constexpr int kGridSize = 8;

Pose RandomPose(std::mt19937& rng, double scale) {
  constexpr double kRotationScale = 0.5;  // rad
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  return Pose(
      So3Exp(Vec3(uniform(rng), uniform(rng), uniform(rng)) * kRotationScale),
      Vec3(uniform(rng), uniform(rng), uniform(rng)) * scale);
}

Pose PerturbRight(const Pose& pose, const Vec6& delta) {
  return Pose(pose.rotation * So3Exp(delta.segment<kSpaceDim>(kRotationOffset)),
              pose.translation + delta.segment<kSpaceDim>(kTranslationOffset));
}

// A uniform perturbation of up to `max_translation` (m) and
// `max_rotation_deg` along every axis.
Vec6 RandomDelta(std::mt19937& rng, double max_translation,
                 double max_rotation_deg) {
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  Vec6 delta;
  for (int axis = 0; axis < kPoseDim; ++axis) delta[axis] = uniform(rng);
  delta.segment<kSpaceDim>(kTranslationOffset) *= max_translation;
  delta.segment<kSpaceDim>(kRotationOffset) *= DegToRad(max_rotation_deg);
  return delta;
}

// The vehicle (= IMU body) on its circle at time `t`.
Pose CirclePose(double t) {
  const double yaw = kYawRate * t;
  const double radius = kSpeed / kYawRate;
  return Pose(
      So3Exp(Vec3(0.0, 0.0, yaw)),
      Vec3(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0));
}

Vec3 CircleVelocity(double t) {
  return kSpeed * Vec3(std::cos(kYawRate * t), std::sin(kYawRate * t), 0.0);
}

// IMU samples of the circle drive over [0, duration] with constant biases.
ImuSamples SimulateCircleImu(double duration, const Vec6& bias) {
  ImuSamples imu;
  const int last_sample = static_cast<int>(std::lround(duration / kImuPeriod));
  for (int sample = 0; sample <= last_sample; ++sample) {
    imu.stamps.push_back(kImuPeriod * sample);
    imu.gyro.push_back(Vec3(0.0, 0.0, kYawRate) +
                       bias.segment<kSpaceDim>(kGyroBiasOffset));
    imu.accel.push_back(Vec3(0.0, kSpeed * kYawRate, kGravity) +
                        bias.segment<kSpaceDim>(kAccelBiasOffset));
  }
  return imu;
}

std::vector<double> NodeStamps(double last_stamp) {
  std::vector<double> stamps;
  for (double stamp = kFirstNodeStamp; stamp <= last_stamp;
       stamp += kNodePeriod) {
    stamps.push_back(stamp);
  }
  return stamps;
}

// [gyro, accel] sigmas.
Vec6 BiasSigma(double gyro, double accel) {
  Vec6 sigma;
  sigma << Vec3::Constant(gyro), Vec3::Constant(accel);
  return sigma;
}

GR_TEST(PoseGraphTest, BetweenJacobiansMatchNumericDerivatives) {
  constexpr int kSeed = 11;
  constexpr double kPoseScale = 3.0;         // m
  constexpr double kMeasurementNoise = 0.1;  // m
  constexpr double kStep = 1e-7;
  constexpr double kTolerance = 1e-5;
  std::mt19937 rng(kSeed);
  const Pose pose_i = RandomPose(rng, kPoseScale);
  const Pose pose_j = RandomPose(rng, kPoseScale);
  const Pose measurement =
      pose_i.Inverse() * pose_j * RandomPose(rng, kMeasurementNoise);
  Mat6 jacobian_i;
  Mat6 jacobian_j;
  const Vec6 residual = PoseGraph::BetweenResidual(pose_i, pose_j, measurement,
                                                   &jacobian_i, &jacobian_j);
  for (int axis = 0; axis < kPoseDim; ++axis) {
    Vec6 delta = Vec6::Zero();
    delta[axis] = kStep;
    const Vec6 residual_i = PoseGraph::BetweenResidual(
        PerturbRight(pose_i, delta), pose_j, measurement);
    const Vec6 residual_j = PoseGraph::BetweenResidual(
        pose_i, PerturbRight(pose_j, delta), measurement);
    GR_EXPECT_NEAR(
        ((residual_i - residual) / kStep - jacobian_i.col(axis)).norm(), 0.0,
        kTolerance);
    GR_EXPECT_NEAR(
        ((residual_j - residual) / kStep - jacobian_j.col(axis)).norm(), 0.0,
        kTolerance);
  }
}

GR_TEST(PoseGraphTest, LoopClosureReducesDrift) {
  // Square loop of 4 x 10 poses 2 m apart; the odometry carries a yaw bias,
  // one loop closure.
  constexpr int kSides = 4;
  constexpr int kPosesPerSide = 10;
  constexpr double kSpacing = 2.0;                 // m
  constexpr double kQuarterTurn = kPi / 2.0;       // rad
  constexpr double kYawBias = 0.01;                // rad per step
  constexpr double kLoopInformationFactor = 10.0;  // Over the odometry.
  constexpr double kMinDrift = 1.0;                // m
  constexpr double kMaxDriftRatio = 0.3;
  std::vector<Pose> truth;
  const Vec3 corners[kSides] = {Vec3(0, 0, 0), Vec3(20, 0, 0), Vec3(20, 20, 0),
                                Vec3(0, 20, 0)};
  for (int index = 0; index < kSides * kPosesPerSide; ++index) {
    const int side = index / kPosesPerSide;
    const double yaw = side * kQuarterTurn;
    const Vec3 direction(std::cos(yaw), std::sin(yaw), 0.0);
    truth.emplace_back(
        Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
        corners[side] + direction * kSpacing * (index % kPosesPerSide));
  }
  PoseGraph graph;
  Pose current = truth[0];
  graph.AddNode(current, /*fixed=*/true);
  const Mat6 information = Mat6::Identity() * kOdometryInformation;
  for (size_t index = 1; index < truth.size(); ++index) {
    Pose measurement = truth[index - 1].Inverse() * truth[index];
    measurement.rotation = measurement.rotation * So3Exp(Vec3(0, 0, kYawBias));
    current = current * measurement;
    graph.AddNode(current);
    graph.AddBetween(static_cast<int>(index) - 1, static_cast<int>(index),
                     measurement, information);
  }
  const int last = static_cast<int>(truth.size()) - 1;
  const double before =
      (graph.poses().back().translation - truth.back().translation).norm();
  graph.AddBetween(last, 0, truth.back().Inverse() * truth[0],
                   information * kLoopInformationFactor, kRobustScale);
  const PoseGraphSummary summary = graph.Optimize();
  const double after =
      (graph.poses().back().translation - truth.back().translation).norm();
  GR_EXPECT_LT(summary.final_cost, summary.initial_cost);
  GR_EXPECT_GT(before, kMinDrift);
  GR_EXPECT_LT(after, kMaxDriftRatio * before);
}

GR_TEST(PoseGraphTest, PositionFactorsAnchorTheGraph) {
  constexpr double kPositionInformation = 1e4;
  constexpr double kTolerance = 1e-3;  // m
  PoseGraph graph;
  graph.AddNode(Pose(Mat3::Identity(), Vec3(1, 1, 1)));
  graph.AddNode(Pose(Mat3::Identity(), Vec3(2, 1, 1)));
  graph.AddBetween(0, 1, Pose(Mat3::Identity(), Vec3(1, 0, 0)),
                   Mat6::Identity() * kOdometryInformation);
  graph.AddPosition(0, Vec3(0, 0, 0), Mat3::Identity() * kPositionInformation);
  graph.AddPosition(1, Vec3(1, 0, 0), Mat3::Identity() * kPositionInformation);
  graph.Optimize();
  GR_EXPECT_NEAR(graph.poses()[0].translation.norm(), 0.0, kTolerance);
  GR_EXPECT_NEAR((graph.poses()[1].translation - Vec3(1, 0, 0)).norm(), 0.0,
                 kTolerance);
}

GR_TEST(PoseGraphTest, GravityFactorLevelsATiltedChain) {
  // A chain whose relative measurements carry a pitch bias drifts in z; one
  // gravity factor per node removes the tilt while yaw stays free.
  constexpr int kNodes = 30;
  constexpr double kPitchBias = 0.003;     // rad per step
  constexpr double kStep = 1.0;            // m
  constexpr double kGravitySigma = 0.002;  // rad
  constexpr double kMinDrift = 1.0;        // m
  constexpr double kMaxDriftRatio = 0.3;
  constexpr double kMaxTilt = 0.01;  // Of the up direction.
  PoseGraph graph;
  Pose current;
  graph.AddNode(current, /*fixed=*/true);
  const Mat6 information = Mat6::Identity() * kOdometryInformation;
  for (int node = 1; node < kNodes; ++node) {
    const Pose measurement(So3Exp(Vec3(0, kPitchBias, 0)), Vec3(kStep, 0, 0));
    current = current * measurement;
    graph.AddNode(current);
    graph.AddBetween(node - 1, node, measurement, information);
  }
  const double z_before = std::abs(graph.poses().back().translation.z());
  for (int node = 0; node < kNodes; ++node) {
    graph.AddGravity(node, Vec3::UnitZ(), kGravitySigma);
  }
  graph.Optimize();
  const Pose last = graph.poses().back();
  GR_EXPECT_GT(z_before, kMinDrift);
  GR_EXPECT_NEAR(std::abs(last.translation.z()), 0.0,
                 kMaxDriftRatio * z_before);
  GR_EXPECT_NEAR(
      (last.rotation.transpose() * Vec3::UnitZ() - Vec3::UnitZ()).norm(), 0.0,
      kMaxTilt);
}

GR_TEST(PoseGraphTest, ImuAndWheelFactorsRecoverTheImuBias) {
  // Vehicle on a circle (body = vehicle frame); graph nodes are LiDAR poses
  // with a lever arm. LiDAR-like relative poses fix the trajectory, so the IMU
  // factors observe the gyro and accelerometer biases.
  constexpr double kImuDuration = 22.0;     // s
  constexpr double kLastNodeStamp = 21.0;   // s
  constexpr double kWheelSigma = 0.02;      // m/s
  constexpr double kOdometrySigmaT = 0.01;  // m
  constexpr double kOdometrySigmaR = 0.05;  // deg
  constexpr double kMaxCostRatio = 1e-3;
  // Nodes this far from either end are checked.
  constexpr int kEndMargin = 5;
  constexpr double kMaxGyroBiasError = 2e-4;   // rad/s
  constexpr double kMaxAccelBiasError = 5e-3;  // m/s^2
  constexpr double kMaxVelocityError = 0.01;   // m/s
  constexpr double kMaxPositionError = 0.01;   // m
  constexpr double kMaxImuChi2 = 1.0;
  Vec6 bias;
  bias << 0.002, -0.003, 0.004, 0.04, -0.05, 0.03;
  const ImuSamples imu = SimulateCircleImu(kImuDuration, bias);
  const Pose vehicle_from_lidar(So3Exp(Vec3(0.0, 0.0, 0.1)),
                                Vec3(1.0, 0.0, 1.8));
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();

  PoseGraph graph;
  const std::vector<double> stamps = NodeStamps(kLastNodeStamp);
  const int num_nodes = static_cast<int>(stamps.size());
  for (int node = 0; node < num_nodes; ++node) {
    const double t = stamps[static_cast<size_t>(node)];
    graph.AddNode(CirclePose(t) * vehicle_from_lidar);
    graph.AddInertialState(node, CirclePose(t).rotation * Vec3(kSpeed, 0, 0),
                           Vec6::Zero());
    graph.AddBodyVelocity(node, Vec3(kSpeed, 0.0, 0.0),
                          Vec3::Constant(kWheelSigma), lidar_from_vehicle);
  }
  graph.AddPrior(0, graph.poses()[0],
                 PoseInformation(kAnchorSigmaT, kAnchorSigmaR));
  const Vec6 walk = BiasSigma(kGyroBiasWalk, kAccelBiasWalk);
  graph.AddBiasPrior(0, Vec6::Zero(),
                     BiasSigma(kGyroBiasPrior, kAccelBiasPrior));
  for (int node = 1; node < num_nodes; ++node) {
    const Pose& previous = graph.poses()[static_cast<size_t>(node - 1)];
    const Pose& current = graph.poses()[static_cast<size_t>(node)];
    graph.AddBetween(node - 1, node, previous.Inverse() * current,
                     PoseInformation(kOdometrySigmaT, kOdometrySigmaR));
    const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
        imu, stamps[static_cast<size_t>(node - 1)],
        stamps[static_cast<size_t>(node)], Vec6::Zero(), ImuNoise());
    GR_ASSERT_TRUE(preintegration.has_value());
    graph.AddImu(node - 1, node, *preintegration, lidar_from_vehicle);
    graph.AddBiasBetween(node - 1, node, walk);
  }
  const PoseGraphSummary summary = graph.Optimize();
  GR_EXPECT_LT(summary.final_cost, kMaxCostRatio * summary.initial_cost);
  for (int node : {kEndMargin, num_nodes / 2, num_nodes - kEndMargin}) {
    const Vec6 error = graph.Bias(node) - bias;
    GR_EXPECT_LT(error.segment<kSpaceDim>(kGyroBiasOffset).norm(),
                 kMaxGyroBiasError);
    GR_EXPECT_LT(error.segment<kSpaceDim>(kAccelBiasOffset).norm(),
                 kMaxAccelBiasError);
    const double t = stamps[static_cast<size_t>(node)];
    GR_EXPECT_LT((graph.Velocity(node) - CircleVelocity(t)).norm(),
                 kMaxVelocityError);
  }
  // The trajectory itself is unchanged.
  const Pose last = graph.poses().back() * lidar_from_vehicle;
  GR_EXPECT_LT(
      (last.translation - CirclePose(stamps.back()).translation).norm(),
      kMaxPositionError);
  for (double chi2 : graph.ImuChi2()) GR_EXPECT_LT(chi2, kMaxImuChi2);
}

// Grid of points on the parallelogram origin + a * u + b * v (a in
// [0, length_u], b in [0, length_v]) as seen from `world_from_node`, in the
// node frame, with a node-specific offset of the grid and noise along the
// plane normal.
PointCluster SampleRectangle(const Vec3& origin, const Vec3& u, const Vec3& v,
                             double length_u, double length_v,
                             const Pose& world_from_node, std::mt19937& rng,
                             double noise) {
  std::uniform_real_distribution<double> jitter(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  const Vec3 normal = u.cross(v).normalized();
  const Pose node_from_world = world_from_node.Inverse();
  PointCluster cluster;
  const double shift_u = jitter(rng);
  const double shift_v = jitter(rng);
  for (int row = 0; row < kGridSize; ++row) {
    for (int col = 0; col < kGridSize; ++col) {
      const double a = length_u * (row + shift_u) / kGridSize;
      const double b = length_v * (col + shift_v) / kGridSize;
      const Vec3 world = origin + a * u + b * v + noise * gauss(rng) * normal;
      cluster.Add(node_from_world * world);
    }
  }
  return cluster;
}

GR_TEST(PoseGraphTest, PlaneFactorsAlignTheScansOfARoom) {
  // A 10 x 6 x 3 m room scanned from 8 nodes 1 m apart, turning 0.8 rad per
  // node; all but the first start perturbed.
  constexpr int kSeed = 5;
  constexpr int kNodes = 8;
  constexpr double kMaxTilt = 0.02;           // rad
  constexpr double kYawPerNode = 0.8;         // rad
  constexpr double kFirstX = 1.5;             // m
  constexpr double kCenterY = 3.0;            // m
  constexpr double kSensorHeight = 1.8;       // m
  constexpr double kStartTranslation = 0.05;  // m
  constexpr double kStartRotation = 1.0;      // deg
  constexpr double kPointNoise = 0.002;       // m
  constexpr double kMaxPositionError = 2e-3;  // m
  constexpr double kMaxRotationError = 0.02;  // deg
  constexpr double kMinRmsBefore = 0.01;      // m
  constexpr double kMaxRmsAfter = 0.003;      // m
  struct Rectangle {
    Vec3 origin, u, v;
    double length_u, length_v;
  };
  const std::vector<Rectangle> planes = {
      {Vec3(0, 0, 0), Vec3::UnitX(), Vec3::UnitY(), 10, 6},  // Floor.
      {Vec3(0, 0, 3), Vec3::UnitX(), Vec3::UnitY(), 10, 6},  // Ceiling.
      {Vec3(0, 0, 0), Vec3::UnitY(), Vec3::UnitZ(), 6, 3},   // Walls.
      {Vec3(10, 0, 0), Vec3::UnitY(), Vec3::UnitZ(), 6, 3},
      {Vec3(0, 0, 0), Vec3::UnitX(), Vec3::UnitZ(), 10, 3},
      {Vec3(0, 6, 0), Vec3::UnitX(), Vec3::UnitZ(), 10, 3},
  };
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<Pose> truth;
  for (int node = 0; node < kNodes; ++node) {
    truth.emplace_back(
        So3Exp(Vec3(kMaxTilt * uniform(rng), kMaxTilt * uniform(rng),
                    kYawPerNode * node)),
        Vec3(kFirstX + node, kCenterY + uniform(rng), kSensorHeight));
  }
  for (const int dense_limit : {PoseGraphOptions().dense_max_pose_nodes, 0}) {
    PoseGraph graph;
    for (int node = 0; node < kNodes; ++node) {
      const Vec6 delta = RandomDelta(rng, kStartTranslation, kStartRotation);
      graph.AddNode(node == 0 ? truth[0] : PerturbRight(truth[node], delta),
                    /*fixed=*/node == 0);
    }
    for (const Rectangle& plane : planes) {
      std::vector<PlaneObservation> observations;
      for (int node = 0; node < kNodes; ++node) {
        observations.push_back(
            {node,
             SampleRectangle(plane.origin, plane.u, plane.v, plane.length_u,
                             plane.length_v, truth[static_cast<size_t>(node)],
                             rng, kPointNoise)});
      }
      graph.AddPlane(observations, kPointSigma, kRobustScale);
    }
    GR_EXPECT_EQ(graph.num_planes(), planes.size());
    const std::vector<double> rms_before = graph.PlaneRms();
    PoseGraphOptions options;
    options.dense_max_pose_nodes = dense_limit;
    const PoseGraphSummary summary = graph.Optimize(options);
    GR_EXPECT_TRUE(summary.converged);
    for (size_t node = 0; node < truth.size(); ++node) {
      const Pose& pose = graph.poses()[node];
      GR_EXPECT_LT((pose.translation - truth[node].translation).norm(),
                   kMaxPositionError);
      GR_EXPECT_LT(RadToDeg(RotationAngle(pose.rotation.transpose() *
                                          truth[node].rotation)),
                   kMaxRotationError);
    }
    const std::vector<double> rms_after = graph.PlaneRms();
    for (size_t plane = 0; plane < rms_after.size(); ++plane) {
      GR_EXPECT_GT(rms_before[plane], kMinRmsBefore);
      GR_EXPECT_LT(rms_after[plane], kMaxRmsAfter);
    }
  }
}

GR_TEST(PoseGraphTest, DenseAndSparseSolversAgreeWithInertialStates) {
  // A drive on a circle over a floor tiled into 4 m plane patches: plane,
  // IMU, wheel and bias factors together. The dense solver eliminates the
  // inertial states before factoring the pose block; both solvers must reach
  // the same optimum.
  constexpr int kSeed = 9;
  constexpr double kImuDuration = 12.0;       // s
  constexpr double kLastNodeStamp = 11.0;     // s
  constexpr double kStartTranslation = 0.02;  // m
  constexpr double kStartRotation = 0.3;      // deg
  constexpr double kTileSize = 4.0;           // m
  constexpr double kTilesMinX = -12.0;        // m
  constexpr double kTilesMaxX = 12.0;         // m
  constexpr double kTilesMinY = -2.0;         // m
  constexpr double kTilesMaxY = 22.0;         // m
  constexpr double kFloorHeight = -0.3;       // m
  constexpr double kVisibleRange = 7.0;       // m, horizontal
  constexpr double kPointNoise = 0.003;       // m
  constexpr double kWheelSigma = 0.05;        // m/s
  constexpr double kWheelLateralSigma = 0.3;  // m/s
  constexpr size_t kMinTileObservations = 2;  // A plane factor's minimum.
  constexpr size_t kMinTiles = 5;
  constexpr double kSolverTolerance = 1e-5;    // m and bias units
  constexpr double kMaxDriveTilt = 0.01;       // deg
  constexpr double kMaxHeightResidual = 5e-4;  // m
  // Height error model: a tilted plane [x, y, 1].
  constexpr int kTiltParams = 3;
  Vec6 bias;
  bias << 0.001, -0.002, 0.0, 0.02, 0.0, 0.0;
  const ImuSamples imu = SimulateCircleImu(kImuDuration, bias);
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  const std::vector<double> stamps = NodeStamps(kLastNodeStamp);
  const int num_nodes = static_cast<int>(stamps.size());
  std::mt19937 rng(kSeed);
  std::vector<Pose> truth;
  std::vector<Pose> start;
  for (double t : stamps) {
    truth.push_back(CirclePose(t) * vehicle_from_lidar);
    start.push_back(PerturbRight(
        truth.back(), RandomDelta(rng, kStartTranslation, kStartRotation)));
  }
  // Floor tiles seen by the nodes within kVisibleRange.
  std::vector<std::vector<PlaneObservation>> tiles;
  for (double x = kTilesMinX; x < kTilesMaxX; x += kTileSize) {
    for (double y = kTilesMinY; y < kTilesMaxY; y += kTileSize) {
      std::vector<PlaneObservation> observations;
      for (int node = 0; node < num_nodes; ++node) {
        const Vec3 center(x + 0.5 * kTileSize, y + 0.5 * kTileSize,
                          kFloorHeight);
        const Pose& pose = truth[static_cast<size_t>(node)];
        if ((pose.translation - center).head<kPlaneDim>().norm() >
            kVisibleRange) {
          continue;
        }
        observations.push_back(
            {node, SampleRectangle(Vec3(x, y, kFloorHeight), Vec3::UnitX(),
                                   Vec3::UnitY(), kTileSize, kTileSize, pose,
                                   rng, kPointNoise)});
      }
      if (observations.size() >= kMinTileObservations) {
        tiles.push_back(observations);
      }
    }
  }
  GR_ASSERT_TRUE(tiles.size() > kMinTiles);
  std::vector<std::vector<Pose>> solutions;
  std::vector<std::vector<Vec6>> biases;
  for (const int dense_limit : {PoseGraphOptions().dense_max_pose_nodes, 0}) {
    PoseGraph graph;
    for (int node = 0; node < num_nodes; ++node) {
      graph.AddNode(start[static_cast<size_t>(node)]);
      graph.AddInertialState(
          node,
          CirclePose(stamps[static_cast<size_t>(node)]).rotation *
              Vec3(kSpeed, 0.0, 0.0),
          Vec6::Zero());
      graph.AddBodyVelocity(
          node, Vec3(kSpeed, 0.0, 0.0),
          Vec3(kWheelSigma, kWheelLateralSigma, kWheelLateralSigma),
          lidar_from_vehicle);
    }
    graph.AddPrior(0, truth[0], PoseInformation(kAnchorSigmaT, kAnchorSigmaR));
    const Vec6 walk = BiasSigma(kGyroBiasWalk, kAccelBiasWalk);
    graph.AddBiasPrior(0, Vec6::Zero(),
                       BiasSigma(kGyroBiasPrior, kAccelBiasPrior));
    for (int node = 1; node < num_nodes; ++node) {
      const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
          imu, stamps[static_cast<size_t>(node - 1)],
          stamps[static_cast<size_t>(node)], Vec6::Zero(), ImuNoise());
      GR_ASSERT_TRUE(preintegration.has_value());
      graph.AddImu(node - 1, node, *preintegration, lidar_from_vehicle);
      graph.AddBiasBetween(node - 1, node, walk);
    }
    for (const std::vector<PlaneObservation>& tile : tiles) {
      graph.AddPlane(tile, kPointSigma);
    }
    PoseGraphOptions options;
    options.dense_max_pose_nodes = dense_limit;
    const PoseGraphSummary summary = graph.Optimize(options);
    GR_EXPECT_LT(summary.final_cost, summary.initial_cost);
    solutions.push_back(graph.poses());
    biases.emplace_back();
    for (int node = 0; node < num_nodes; ++node) {
      biases.back().push_back(graph.Bias(node));
    }
  }
  // Both stop at the Levenberg-Marquardt tolerance: within micrometres
  // (GCC and clang round differently).
  Eigen::MatrixXd tilt_design(num_nodes, kTiltParams);
  Eigen::VectorXd height_error(num_nodes);
  for (int node = 0; node < num_nodes; ++node) {
    const size_t index = static_cast<size_t>(node);
    GR_EXPECT_LT(
        (solutions[0][index].translation - solutions[1][index].translation)
            .norm(),
        kSolverTolerance);
    GR_EXPECT_LT((biases[0][index] - biases[1][index]).norm(),
                 kSolverTolerance);
    tilt_design.row(node) << truth[index].translation.x(),
        truth[index].translation.y(), 1.0;
    height_error[node] =
        solutions[0][index].translation.z() - truth[index].translation.z();
  }
  // The floor tiles are free planes: they fix the height above the floor,
  // but the tilt of the whole drive comes from gravity (the IMU, whose
  // biases are estimated). Height errors are that tilt (about 0.006 deg,
  // 2 mm over the drive) plus a fraction of a millimetre.
  const Eigen::Vector3d tilt =
      tilt_design.colPivHouseholderQr().solve(height_error);
  GR_EXPECT_LT(std::atan(std::hypot(tilt.x(), tilt.y())),
               DegToRad(kMaxDriveTilt));
  GR_EXPECT_LT((height_error - tilt_design * tilt).cwiseAbs().maxCoeff(),
               kMaxHeightResidual);
}

}  // namespace
}  // namespace gr
