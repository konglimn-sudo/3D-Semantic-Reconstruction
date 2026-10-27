#include "gr/backend/pose_graph.h"

#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include "gr/sensors/imu_preintegration.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

Pose RandomPose(std::mt19937& rng, double scale) {
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  return Pose(So3Exp(Vec3(uniform(rng), uniform(rng), uniform(rng)) * 0.5),
              Vec3(uniform(rng), uniform(rng), uniform(rng)) * scale);
}

Pose PerturbRight(const Pose& pose, const Vec6& d) {
  return Pose(pose.rotation * So3Exp(d.tail<3>()),
              pose.translation + d.head<3>());
}

GR_TEST(PoseGraphTest, BetweenJacobiansMatchNumericDerivatives) {
  std::mt19937 rng(11);
  const Pose pose_i = RandomPose(rng, 3.0);
  const Pose pose_j = RandomPose(rng, 3.0);
  const Pose measurement = pose_i.Inverse() * pose_j * RandomPose(rng, 0.1);
  Mat6 jacobian_i, jacobian_j;
  const Vec6 e0 = PoseGraph::BetweenResidual(pose_i, pose_j, measurement,
                                             &jacobian_i, &jacobian_j);
  const double h = 1e-7;
  for (int a = 0; a < 6; ++a) {
    Vec6 d = Vec6::Zero();
    d[a] = h;
    const Vec6 ei = PoseGraph::BetweenResidual(PerturbRight(pose_i, d), pose_j,
                                               measurement);
    const Vec6 ej = PoseGraph::BetweenResidual(pose_i, PerturbRight(pose_j, d),
                                               measurement);
    GR_EXPECT_NEAR(((ei - e0) / h - jacobian_i.col(a)).norm(), 0.0, 1e-5);
    GR_EXPECT_NEAR(((ej - e0) / h - jacobian_j.col(a)).norm(), 0.0, 1e-5);
  }
}

GR_TEST(PoseGraphTest, LoopClosureReducesDrift) {
  // Square loop of 40 poses; odometry carries a yaw bias, one loop closure.
  std::vector<Pose> truth;
  const Vec3 corners[4] = {Vec3(0, 0, 0), Vec3(20, 0, 0), Vec3(20, 20, 0),
                           Vec3(0, 20, 0)};
  for (int k = 0; k < 40; ++k) {
    const int side = k / 10;
    const double yaw = side * kPi / 2.0;
    const Vec3 direction(std::cos(yaw), std::sin(yaw), 0.0);
    truth.emplace_back(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
                       corners[side] + direction * 2.0 * (k % 10));
  }
  PoseGraph graph;
  Pose current = truth[0];
  graph.AddNode(current, true);
  const Mat6 information = Mat6::Identity() * 100.0;
  for (size_t k = 1; k < truth.size(); ++k) {
    Pose measurement = truth[k - 1].Inverse() * truth[k];
    measurement.rotation = measurement.rotation * So3Exp(Vec3(0, 0, 0.01));
    current = current * measurement;
    graph.AddNode(current);
    graph.AddBetween(static_cast<int>(k) - 1, static_cast<int>(k), measurement,
                     information);
  }
  const double before =
      (graph.poses()[39].translation - truth[39].translation).norm();
  graph.AddBetween(39, 0, truth[39].Inverse() * truth[0], information * 10.0,
                   3.0);
  const PoseGraphSummary summary = graph.Optimize();
  const double after =
      (graph.poses()[39].translation - truth[39].translation).norm();
  GR_EXPECT_LT(summary.final_cost, summary.initial_cost);
  GR_EXPECT_GT(before, 1.0);
  GR_EXPECT_LT(after, 0.3 * before);
}

GR_TEST(PoseGraphTest, PositionFactorsAnchorTheGraph) {
  PoseGraph graph;
  graph.AddNode(Pose(Mat3::Identity(), Vec3(1, 1, 1)));
  graph.AddNode(Pose(Mat3::Identity(), Vec3(2, 1, 1)));
  graph.AddBetween(0, 1, Pose(Mat3::Identity(), Vec3(1, 0, 0)),
                   Mat6::Identity() * 100.0);
  graph.AddPosition(0, Vec3(0, 0, 0), Mat3::Identity() * 1e4);
  graph.AddPosition(1, Vec3(1, 0, 0), Mat3::Identity() * 1e4);
  graph.Optimize();
  GR_EXPECT_NEAR(graph.poses()[0].translation.norm(), 0.0, 1e-3);
  GR_EXPECT_NEAR((graph.poses()[1].translation - Vec3(1, 0, 0)).norm(), 0.0,
                 1e-3);
}

GR_TEST(PoseGraphTest, GravityFactorLevelsATiltedChain) {
  // A chain whose relative measurements carry a pitch bias drifts in z; one
  // gravity factor per node removes the tilt while yaw stays free.
  PoseGraph graph;
  Pose current;
  graph.AddNode(current, true);
  const Mat6 information = Mat6::Identity() * 100.0;
  for (int k = 1; k < 30; ++k) {
    const Pose measurement(So3Exp(Vec3(0, 0.003, 0)), Vec3(1, 0, 0));
    current = current * measurement;
    graph.AddNode(current);
    graph.AddBetween(k - 1, k, measurement, information);
  }
  const double z_before = std::abs(graph.poses().back().translation.z());
  for (int k = 0; k < 30; ++k) graph.AddGravity(k, Vec3::UnitZ(), 0.002);
  graph.Optimize();
  const Pose last = graph.poses().back();
  GR_EXPECT_GT(z_before, 1.0);
  GR_EXPECT_NEAR(std::abs(last.translation.z()), 0.0, 0.3 * z_before);
  GR_EXPECT_NEAR(
      (last.rotation.transpose() * Vec3::UnitZ() - Vec3::UnitZ()).norm(), 0.0,
      0.01);
}

GR_TEST(PoseGraphTest, ImuAndWheelFactorsRecoverTheImuBias) {
  // Vehicle on a circle (body = vehicle frame); graph nodes are LiDAR poses
  // with a lever arm. LiDAR-like relative poses fix the trajectory, so the IMU
  // factors observe the gyro and accelerometer biases.
  const double yaw_rate = 0.3;
  const double speed = 3.0;
  const auto body_pose = [&](double t) {
    const double yaw = yaw_rate * t;
    const double radius = speed / yaw_rate;
    return Pose(
        So3Exp(Vec3(0.0, 0.0, yaw)),
        Vec3(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0));
  };
  const auto body_velocity = [&](double t) -> Vec3 {
    return speed * Vec3(std::cos(yaw_rate * t), std::sin(yaw_rate * t), 0.0);
  };
  Vec6 bias;
  bias << 0.002, -0.003, 0.004, 0.04, -0.05, 0.03;
  ImuSamples imu;
  for (int i = 0; i <= 4400; ++i) {
    imu.stamps.push_back(0.005 * i);
    imu.gyro.push_back(Vec3(0.0, 0.0, yaw_rate) + bias.head<3>());
    imu.accel.push_back(Vec3(0.0, speed * yaw_rate, kGravity) + bias.tail<3>());
  }
  const Pose vehicle_from_lidar(So3Exp(Vec3(0.0, 0.0, 0.1)),
                                Vec3(1.0, 0.0, 1.8));
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();

  PoseGraph graph;
  std::vector<double> stamps;
  for (double t = 1.0; t <= 21.0; t += 0.5) stamps.push_back(t);
  const int n = static_cast<int>(stamps.size());
  for (int k = 0; k < n; ++k) {
    const double t = stamps[static_cast<size_t>(k)];
    graph.AddNode(body_pose(t) * vehicle_from_lidar);
    graph.AddInertialState(k, body_pose(t).rotation * Vec3(speed, 0, 0),
                           Vec6::Zero());
    graph.AddBodyVelocity(k, Vec3(speed, 0.0, 0.0), Vec3::Constant(0.02),
                          lidar_from_vehicle);
  }
  graph.AddPrior(0, graph.poses()[0], PoseInformation(0.001, 0.01));
  Vec6 walk;
  walk << Vec3::Constant(1e-5), Vec3::Constant(1e-4);
  Vec6 prior_sigma;
  prior_sigma << Vec3::Constant(0.02), Vec3::Constant(0.2);
  graph.AddBiasPrior(0, Vec6::Zero(), prior_sigma);
  for (int k = 1; k < n; ++k) {
    const Pose& a = graph.poses()[static_cast<size_t>(k - 1)];
    const Pose& b = graph.poses()[static_cast<size_t>(k)];
    graph.AddBetween(k - 1, k, a.Inverse() * b, PoseInformation(0.01, 0.05));
    const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
        imu, stamps[static_cast<size_t>(k - 1)], stamps[static_cast<size_t>(k)],
        Vec6::Zero(), ImuNoise());
    GR_ASSERT_TRUE(preintegration.has_value());
    graph.AddImu(k - 1, k, *preintegration, lidar_from_vehicle);
    graph.AddBiasBetween(k - 1, k, walk);
  }
  const PoseGraphSummary summary = graph.Optimize();
  GR_EXPECT_LT(summary.final_cost, 1e-3 * summary.initial_cost);
  for (int k : {5, n / 2, n - 5}) {
    const Vec6 error = graph.Bias(k) - bias;
    GR_EXPECT_LT(error.head<3>().norm(), 2e-4);  // rad/s
    GR_EXPECT_LT(error.tail<3>().norm(), 5e-3);  // m/s^2
    const double t = stamps[static_cast<size_t>(k)];
    GR_EXPECT_LT((graph.Velocity(k) - body_velocity(t)).norm(), 0.01);
  }
  // The trajectory itself is unchanged.
  const Pose last = graph.poses().back() * lidar_from_vehicle;
  GR_EXPECT_LT((last.translation - body_pose(stamps.back()).translation).norm(),
               0.01);
  for (double chi2 : graph.ImuChi2()) GR_EXPECT_LT(chi2, 1.0);
}

// Grid of points on the parallelogram origin + a * u + b * v (a in [0, la],
// b in [0, lb]) as seen from `world_from_node`, in the node frame, with a
// node-specific offset of the grid and noise along the plane normal.
PointCluster SampleRectangle(const Vec3& origin, const Vec3& u, const Vec3& v,
                             double la, double lb, const Pose& world_from_node,
                             std::mt19937& rng, double noise) {
  std::uniform_real_distribution<double> jitter(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 1.0);
  const Vec3 normal = u.cross(v).normalized();
  const Pose node_from_world = world_from_node.Inverse();
  PointCluster cluster;
  const double shift_a = jitter(rng);
  const double shift_b = jitter(rng);
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      const double a = la * (i + shift_a) / 8.0;
      const double b = lb * (j + shift_b) / 8.0;
      const Vec3 world = origin + a * u + b * v + noise * gauss(rng) * normal;
      cluster.Add(node_from_world * world);
    }
  }
  return cluster;
}

GR_TEST(PoseGraphTest, PlaneFactorsAlignTheScansOfARoom) {
  struct Rectangle {
    Vec3 origin, u, v;
    double la, lb;
  };
  const std::vector<Rectangle> planes = {
      {Vec3(0, 0, 0), Vec3::UnitX(), Vec3::UnitY(), 10, 6},  // Floor.
      {Vec3(0, 0, 3), Vec3::UnitX(), Vec3::UnitY(), 10, 6},  // Ceiling.
      {Vec3(0, 0, 0), Vec3::UnitY(), Vec3::UnitZ(), 6, 3},   // Walls.
      {Vec3(10, 0, 0), Vec3::UnitY(), Vec3::UnitZ(), 6, 3},
      {Vec3(0, 0, 0), Vec3::UnitX(), Vec3::UnitZ(), 10, 3},
      {Vec3(0, 6, 0), Vec3::UnitX(), Vec3::UnitZ(), 10, 3},
  };
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<Pose> truth;
  for (int k = 0; k < 8; ++k) {
    truth.emplace_back(
        So3Exp(Vec3(0.02 * uniform(rng), 0.02 * uniform(rng), 0.8 * k)),
        Vec3(1.5 + k, 3.0 + uniform(rng), 1.8));
  }
  for (const int dense_limit : {2500, 0}) {  // Dense and sparse solvers.
    PoseGraph graph;
    for (int k = 0; k < 8; ++k) {
      Vec6 delta;
      for (int a = 0; a < 6; ++a) delta[a] = uniform(rng);
      delta.head<3>() *= 0.05;
      delta.tail<3>() *= DegToRad(1.0);
      graph.AddNode(k == 0 ? truth[0] : PerturbRight(truth[k], delta), k == 0);
    }
    for (const Rectangle& r : planes) {
      std::vector<PlaneObservation> observations;
      for (int k = 0; k < 8; ++k) {
        observations.push_back(
            {k, SampleRectangle(r.origin, r.u, r.v, r.la, r.lb,
                                truth[static_cast<size_t>(k)], rng, 0.002)});
      }
      graph.AddPlane(observations, 0.005, 3.0);
    }
    GR_EXPECT_EQ(graph.num_planes(), planes.size());
    const std::vector<double> rms_before = graph.PlaneRms();
    PoseGraphOptions options;
    options.dense_max_pose_nodes = dense_limit;
    const PoseGraphSummary summary = graph.Optimize(options);
    GR_EXPECT_TRUE(summary.converged);
    for (size_t k = 0; k < truth.size(); ++k) {
      const Pose& pose = graph.poses()[k];
      GR_EXPECT_LT((pose.translation - truth[k].translation).norm(), 2e-3);
      GR_EXPECT_LT(RadToDeg(RotationAngle(pose.rotation.transpose() *
                                          truth[k].rotation)),
                   0.02);
    }
    const std::vector<double> rms_after = graph.PlaneRms();
    for (size_t p = 0; p < rms_after.size(); ++p) {
      GR_EXPECT_GT(rms_before[p], 0.01);
      GR_EXPECT_LT(rms_after[p], 0.003);
    }
  }
}

GR_TEST(PoseGraphTest, DenseAndSparseSolversAgreeWithInertialStates) {
  // A drive on a circle over a floor tiled into 4 m plane patches: plane,
  // IMU, wheel and bias factors together. The dense solver eliminates the
  // inertial states before factoring the pose block; both solvers must reach
  // the same optimum.
  const double yaw_rate = 0.3;
  const double speed = 3.0;
  const auto body_pose = [&](double t) {
    const double yaw = yaw_rate * t;
    const double radius = speed / yaw_rate;
    return Pose(
        So3Exp(Vec3(0.0, 0.0, yaw)),
        Vec3(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0));
  };
  ImuSamples imu;
  for (int i = 0; i <= 2400; ++i) {
    imu.stamps.push_back(0.005 * i);
    imu.gyro.push_back(Vec3(0.001, -0.002, yaw_rate));
    imu.accel.push_back(Vec3(0.02, speed * yaw_rate, kGravity));
  }
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  std::vector<double> stamps;
  for (double t = 1.0; t <= 11.0; t += 0.5) stamps.push_back(t);
  const int n = static_cast<int>(stamps.size());
  std::mt19937 rng(9);
  std::uniform_real_distribution<double> uniform(-1.0, 1.0);
  std::vector<Pose> truth;
  std::vector<Pose> start;
  for (double t : stamps) {
    truth.push_back(body_pose(t) * vehicle_from_lidar);
    Vec6 delta;
    for (int a = 0; a < 6; ++a) delta[a] = uniform(rng);
    delta.head<3>() *= 0.02;
    delta.tail<3>() *= DegToRad(0.3);
    start.push_back(PerturbRight(truth.back(), delta));
  }
  // Floor tiles at z = -0.3 seen by the nodes within 7 m.
  std::vector<std::vector<PlaneObservation>> tiles;
  for (double x = -12.0; x < 12.0; x += 4.0) {
    for (double y = -2.0; y < 22.0; y += 4.0) {
      std::vector<PlaneObservation> observations;
      for (int k = 0; k < n; ++k) {
        const Vec3 center(x + 2.0, y + 2.0, -0.3);
        const Pose& pose = truth[static_cast<size_t>(k)];
        if ((pose.translation - center).head<2>().norm() > 7.0) continue;
        observations.push_back(
            {k, SampleRectangle(Vec3(x, y, -0.3), Vec3::UnitX(), Vec3::UnitY(),
                                4.0, 4.0, pose, rng, 0.003)});
      }
      if (observations.size() >= 2) tiles.push_back(observations);
    }
  }
  GR_ASSERT_TRUE(tiles.size() > 5);
  std::vector<std::vector<Pose>> solutions;
  std::vector<std::vector<Vec6>> biases;
  for (const int dense_limit : {2500, 0}) {
    PoseGraph graph;
    for (int k = 0; k < n; ++k) {
      graph.AddNode(start[static_cast<size_t>(k)]);
      graph.AddInertialState(
          k,
          body_pose(stamps[static_cast<size_t>(k)]).rotation *
              Vec3(speed, 0.0, 0.0),
          Vec6::Zero());
      graph.AddBodyVelocity(k, Vec3(speed, 0.0, 0.0), Vec3(0.05, 0.3, 0.3),
                            lidar_from_vehicle);
    }
    graph.AddPrior(0, truth[0], PoseInformation(0.001, 0.01));
    Vec6 walk;
    walk << Vec3::Constant(1e-5), Vec3::Constant(1e-4);
    Vec6 prior_sigma;
    prior_sigma << Vec3::Constant(0.02), Vec3::Constant(0.2);
    graph.AddBiasPrior(0, Vec6::Zero(), prior_sigma);
    for (int k = 1; k < n; ++k) {
      const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
          imu, stamps[static_cast<size_t>(k - 1)],
          stamps[static_cast<size_t>(k)], Vec6::Zero(), ImuNoise());
      GR_ASSERT_TRUE(preintegration.has_value());
      graph.AddImu(k - 1, k, *preintegration, lidar_from_vehicle);
      graph.AddBiasBetween(k - 1, k, walk);
    }
    for (const std::vector<PlaneObservation>& tile : tiles) {
      graph.AddPlane(tile, 0.005);
    }
    PoseGraphOptions options;
    options.dense_max_pose_nodes = dense_limit;
    const PoseGraphSummary summary = graph.Optimize(options);
    GR_EXPECT_LT(summary.final_cost, summary.initial_cost);
    solutions.push_back(graph.poses());
    biases.emplace_back();
    for (int k = 0; k < n; ++k) biases.back().push_back(graph.Bias(k));
  }
  // Both stop at the Levenberg-Marquardt tolerance: within micrometres
  // (GCC and clang round differently).
  for (int k = 0; k < n; ++k) {
    const size_t kk = static_cast<size_t>(k);
    GR_EXPECT_LT(
        (solutions[0][kk].translation - solutions[1][kk].translation).norm(),
        1e-5);
    GR_EXPECT_LT((biases[0][kk] - biases[1][kk]).norm(), 1e-5);
    // The floor fixes height, roll and pitch.
    GR_EXPECT_NEAR(solutions[0][kk].translation.z(), truth[kk].translation.z(),
                   2e-3);
  }
}

}  // namespace
}  // namespace gr
