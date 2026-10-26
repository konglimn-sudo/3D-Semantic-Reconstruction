#include "gr/odometry/odometry.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::CorridorScene;
using ::gr::testing::CorridorSensorPose;
using ::gr::testing::GarageStructure;
using ::gr::testing::LidarModel;
using ::gr::testing::LoopRoute;
using ::gr::testing::MakeScene;
using ::gr::testing::ParkedCars;
using ::gr::testing::Proprioception;
using ::gr::testing::ScanFromPose;
using ::gr::testing::SimBox;
using ::gr::testing::SimulateProprioception;
using ::gr::testing::SimulateSweep;
using ::gr::testing::Sweep;

GR_TEST(OdometryTest, TracksACorridorDrive) {
  const BoxScene scene = CorridorScene();
  OdometryConfig config;
  config.voxel_size = 0.4;
  config.max_range = 40.0;
  Odometry odometry(config, CorridorSensorPose(0.0, 0.0));
  Pose last_truth;
  for (int k = 0; k < 120; ++k) {
    last_truth = CorridorSensorPose(0.28 * k, 0.08 * std::sin(0.1 * k));
    odometry.RegisterFrame(ScanFromPose(scene, last_truth).points_sensor, {},
                           0.2 * k);
  }
  const Pose estimate = odometry.poses().back();
  GR_EXPECT_NEAR((estimate.translation - last_truth.translation).norm(), 0.0,
                 0.15);
  GR_EXPECT_NEAR(
      So3Log(estimate.rotation.transpose() * last_truth.rotation).norm(), 0.0,
      0.01);
}

GR_TEST(OdometryTest, ConstantVelocityDeskew) {
  const Points points = {Vec3(1, 2, 3), Vec3(-1, 0, 2)};
  const Points still = DeskewConstantVelocity(points, {-0.05, -0.01},
                                              Vec3::Zero(), Vec3::Zero());
  GR_EXPECT_NEAR((still[0] - points[0]).norm(), 0.0, 1e-12);
  const Points moved =
      DeskewConstantVelocity(points, {-0.1, 0.0}, Vec3::Zero(), Vec3(10, 0, 0));
  GR_EXPECT_NEAR(moved[0].x(), 0.0, 1e-12);  // 1 m backwards in the end frame.
  GR_EXPECT_NEAR(moved[1].x(), -1.0, 1e-12);
}

GR_TEST(OdometryPipelineTest, SelectsKeyframesByDistanceAndAngle) {
  OdometryConfig config;
  KeyframeConfig keyframe_config;
  keyframe_config.distance = 1.0;
  keyframe_config.angle = 10.0;
  OdometryPipeline pipeline(config, keyframe_config, Pose());
  const Points sweep{Vec3(5, 0, 0), Vec3(0.5, 0, 0), Vec3(70, 0, 0)};
  const std::vector<float> intensity{0.1f, 0.2f, 0.3f};
  int num_keyframes = 0;
  for (int k = 0; k < 10; ++k) {  // 0.4 m per frame, then a 30 deg turn.
    const Pose external(
        Eigen::AngleAxisd(k == 9 ? DegToRad(30.0) : 0.0, Vec3::UnitZ())
            .toRotationMatrix(),
        Vec3(0.4 * std::min(k, 8), 0, 0));
    const OdometryFrame frame =
        pipeline.Process(sweep, intensity, {}, 0.1 * k, false, external);
    if (!frame.is_keyframe) continue;
    ++num_keyframes;
    GR_EXPECT_EQ(frame.points.size(), 1u);  // Cropped to [min, max] range.
    GR_EXPECT_NEAR(frame.intensity[0], 0.1f, 1e-7);
  }
  GR_EXPECT_TRUE((pipeline.keyframes() == std::vector<int>{0, 3, 6, 9}));
  GR_EXPECT_EQ(num_keyframes, 4);
}

GR_TEST(OdometryPipelineTest, TracksASimulatedGarageLoop) {
  std::vector<SimBox> boxes = GarageStructure();
  for (const SimBox& car : ParkedCars(false)) boxes.push_back(car);
  const BoxScene scene = MakeScene(boxes);
  const LoopRoute route;
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  Proprioception sensors = SimulateProprioception(route, -0.5, 12.0);
  OdometryConfig config;
  config.voxel_size = 0.4;
  config.min_range = 1.5;
  OdometryPipeline pipeline(
      config, KeyframeConfig(), vehicle_from_lidar, route.PoseAt(0.0),
      MotionPrior(std::move(sensors.imu), std::move(sensors.wheel)));
  GR_EXPECT_TRUE(pipeline.uses_prior());
  std::mt19937 rng(7);
  double worst = 0.0;
  // 11 s: straight, half-turn, back along the second aisle.
  for (int k = 0; k <= 110; ++k) {
    const double stamp = 0.1 * k;
    const Sweep sweep = SimulateSweep(scene, route, vehicle_from_lidar, stamp,
                                      LidarModel(), rng);
    const OdometryFrame frame = pipeline.Process(
        sweep.points, std::vector<float>(sweep.points.size(), 0.5f),
        sweep.times, stamp);
    worst =
        std::max(worst, (frame.pose.translation -
                         (route.PoseAt(stamp) * vehicle_from_lidar).translation)
                            .norm());
  }
  GR_EXPECT_LT(worst, 0.1);  // Open-loop drift over 44 m and a 180 deg turn.
  // 0.4 m per frame: a keyframe every 1.2 m.
  GR_EXPECT_GE(pipeline.keyframes().size(), 35u);
}

// A carried rig without wheel speed: walking pace along the loop while the
// rig swings +-15 deg in yaw at 1 Hz. The gyro predicts the swing that a
// constant-velocity model cannot follow (in-sweep and between sweeps).
double SwingingRigError(bool use_gyro, double* worst_yaw_deg) {
  std::vector<SimBox> boxes = GarageStructure();
  for (const SimBox& car : ParkedCars(false)) boxes.push_back(car);
  const BoxScene scene = MakeScene(boxes);
  const LoopRoute route(0.0, 1.4);
  const double amplitude = DegToRad(15.0);
  const double omega = 2.0 * kPi;
  const auto rig = [&](double t) {
    return route.PoseAt(t) *
           Pose(
               Eigen::AngleAxisd(amplitude * std::sin(omega * t), Vec3::UnitZ())
                   .toRotationMatrix(),
               Vec3::Zero());
  };
  ImuSamples imu;
  for (int i = 0; i <= 4400; ++i) {
    const double t = -0.5 + 0.005 * i;
    imu.stamps.push_back(t);
    imu.gyro.push_back(
        Vec3(0.0, 0.0,
             route.YawRateAt(t) + amplitude * omega * std::cos(omega * t)));
    imu.accel.push_back(Vec3(0.0, 0.0, 9.81));
  }
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(0.0, 0.0, 1.4));
  OdometryConfig config;
  config.voxel_size = 0.4;
  config.min_range = 1.5;
  config.use_gyro_prior = use_gyro;
  OdometryPipeline pipeline(config, KeyframeConfig(), vehicle_from_lidar,
                            rig(0.0),
                            MotionPrior(std::move(imu), WheelSamples()));
  GR_EXPECT_EQ(pipeline.uses_gyro_prior(), use_gyro);
  std::mt19937 rng(7);
  double worst = 0.0;
  *worst_yaw_deg = 0.0;
  for (int k = 0; k <= 200; ++k) {  // 20 s, 28 m.
    const double stamp = 0.1 * k;
    const Sweep sweep =
        SimulateSweep(scene, rig, vehicle_from_lidar, stamp, LidarModel(), rng);
    const OdometryFrame frame = pipeline.Process(
        sweep.points, std::vector<float>(sweep.points.size(), 0.5f),
        sweep.times, stamp);
    const Pose truth = rig(stamp) * vehicle_from_lidar;
    worst =
        std::max(worst, (frame.pose.translation - truth.translation).norm());
    *worst_yaw_deg = std::max(
        *worst_yaw_deg, RadToDeg(RotationAngle(truth.rotation.transpose() *
                                               frame.pose.rotation)));
  }
  return worst;
}

GR_TEST(OdometryPipelineTest, TheGyroFollowsASwingingRig) {
  double yaw_gyro = 0.0;
  double yaw_constant = 0.0;
  const double gyro = SwingingRigError(true, &yaw_gyro);
  const double constant = SwingingRigError(false, &yaw_constant);
  std::printf(
      "       gyro prior: %.3f m, %.2f deg; constant velocity: "
      "%.3f m, %.2f deg\n",
      gyro, yaw_gyro, constant, yaw_constant);
  GR_EXPECT_LT(gyro, 0.15);
  GR_EXPECT_LT(yaw_gyro, 0.5);
  GR_EXPECT_GT(constant, 1.0);  // Loses track without the gyro.
}

}  // namespace
}  // namespace gr
