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

// The simulated LiDARs sweep at 10 Hz.
constexpr double kSweepPeriod = 0.1;  // s
constexpr int kSeed = 7;
constexpr float kIntensity = 0.5f;
constexpr double kExactTolerance = 1e-12;

GR_TEST(OdometryTest, TracksACorridorDrive) {
  // 120 frames 0.28 m apart (5 Hz), weaving 8 cm sideways.
  constexpr int kFrames = 120;
  constexpr double kFramePeriod = 0.2;        // s
  constexpr double kStep = 0.28;              // m per frame
  constexpr double kWeave = 0.08;             // m
  constexpr double kWeaveRate = 0.1;          // rad per frame
  constexpr double kMaxPositionError = 0.15;  // m
  constexpr double kMaxRotationError = 0.01;  // rad
  const BoxScene scene = CorridorScene();
  OdometryConfig config;
  config.voxel_size = 0.4;
  config.max_range = 40.0;
  Odometry odometry(config, CorridorSensorPose(0.0, 0.0));
  Pose last_truth;
  for (int k = 0; k < kFrames; ++k) {
    last_truth =
        CorridorSensorPose(kStep * k, kWeave * std::sin(kWeaveRate * k));
    odometry.RegisterFrame(ScanFromPose(scene, last_truth).points_sensor, {},
                           kFramePeriod * k);
  }
  const Pose estimate = odometry.poses().back();
  GR_EXPECT_NEAR((estimate.translation - last_truth.translation).norm(), 0.0,
                 kMaxPositionError);
  GR_EXPECT_NEAR(
      So3Log(estimate.rotation.transpose() * last_truth.rotation).norm(), 0.0,
      kMaxRotationError);
}

GR_TEST(OdometryTest, ConstantVelocityDeskew) {
  const Points points = {Vec3(1, 2, 3), Vec3(-1, 0, 2)};
  const Points still = DeskewConstantVelocity(points, {-0.05, -0.01},
                                              Vec3::Zero(), Vec3::Zero());
  GR_EXPECT_NEAR((still[0] - points[0]).norm(), 0.0, kExactTolerance);
  // 10 m/s along x: the first point, 0.1 s before the end, moves 1 m back.
  const Points moved =
      DeskewConstantVelocity(points, {-0.1, 0.0}, Vec3::Zero(), Vec3(10, 0, 0));
  GR_EXPECT_NEAR(moved[0].x(), 0.0, kExactTolerance);
  GR_EXPECT_NEAR(moved[1].x(), -1.0, kExactTolerance);
}

GR_TEST(OdometryPipelineTest, SelectsKeyframesByDistanceAndAngle) {
  // 0.4 m per frame over 8 frames, then a 30 deg turn in place.
  constexpr int kFrames = 10;
  constexpr double kStep = 0.4;  // m per frame
  constexpr int kLastMovingFrame = 8;
  constexpr double kTurnDegrees = 30.0;
  constexpr float kIntensityTolerance = 1e-7f;
  OdometryConfig config;
  KeyframeConfig keyframe_config;
  keyframe_config.distance = 1.0;
  keyframe_config.angle = 10.0;
  OdometryPipeline pipeline(config, keyframe_config, Pose());
  // Only the 5 m point lies within [min_range, max_range].
  const Points sweep{Vec3(5, 0, 0), Vec3(0.5, 0, 0), Vec3(70, 0, 0)};
  const std::vector<float> intensity{0.1f, 0.2f, 0.3f};
  int num_keyframes = 0;
  for (int k = 0; k < kFrames; ++k) {
    const bool turned = k == kFrames - 1;
    const Pose external(
        Eigen::AngleAxisd(turned ? DegToRad(kTurnDegrees) : 0.0, Vec3::UnitZ())
            .toRotationMatrix(),
        Vec3(kStep * std::min(k, kLastMovingFrame), 0, 0));
    const OdometryFrame frame = pipeline.Process(
        sweep, intensity, {}, kSweepPeriod * k, false, external);
    if (!frame.is_keyframe) continue;
    ++num_keyframes;
    GR_EXPECT_EQ(frame.points.size(), 1u);
    GR_EXPECT_NEAR(frame.intensity[0], intensity[0], kIntensityTolerance);
  }
  // Every 1.2 m (three frames), and the turn.
  const std::vector<int> expected_keyframes{0, 3, 6, 9};
  GR_EXPECT_TRUE(pipeline.keyframes() == expected_keyframes);
  GR_EXPECT_EQ(num_keyframes, static_cast<int>(expected_keyframes.size()));
}

GR_TEST(OdometryPipelineTest, TracksASimulatedGarageLoop) {
  // 11 s: straight, half-turn, back along the second aisle (44 m).
  constexpr int kSweeps = 110;
  constexpr double kSensorStart = -0.5;  // s
  constexpr double kSensorEnd = 12.0;
  // Open-loop drift over 44 m and a 180 deg turn.
  constexpr double kMaxDrift = 0.1;  // m
  // 0.4 m per frame: a keyframe every 1.2 m.
  constexpr size_t kMinKeyframes = 35;
  std::vector<SimBox> boxes = GarageStructure();
  for (const SimBox& car : ParkedCars(false)) boxes.push_back(car);
  const BoxScene scene = MakeScene(boxes);
  const LoopRoute route;
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  Proprioception sensors =
      SimulateProprioception(route, kSensorStart, kSensorEnd);
  OdometryConfig config;
  config.voxel_size = 0.4;
  config.min_range = 1.5;
  OdometryPipeline pipeline(
      config, KeyframeConfig(), vehicle_from_lidar, route.PoseAt(0.0),
      MotionPrior(std::move(sensors.imu), std::move(sensors.wheel)));
  GR_EXPECT_TRUE(pipeline.uses_prior());
  std::mt19937 rng(kSeed);
  double worst = 0.0;
  for (int k = 0; k <= kSweeps; ++k) {
    const double stamp = kSweepPeriod * k;
    const Sweep sweep = SimulateSweep(scene, route, vehicle_from_lidar, stamp,
                                      LidarModel(), rng);
    const OdometryFrame frame = pipeline.Process(
        sweep.points, std::vector<float>(sweep.points.size(), kIntensity),
        sweep.times, stamp);
    worst =
        std::max(worst, (frame.pose.translation -
                         (route.PoseAt(stamp) * vehicle_from_lidar).translation)
                            .norm());
  }
  GR_EXPECT_LT(worst, kMaxDrift);
  GR_EXPECT_GE(pipeline.keyframes().size(), kMinKeyframes);
}

// A carried rig without wheel speed: walking pace along the loop while the
// rig swings +-15 deg in yaw at 1 Hz. The gyro predicts the swing that a
// constant-velocity model cannot follow (in-sweep and between sweeps).
double SwingingRigError(bool use_gyro, double* worst_yaw_deg) {
  constexpr double kWalkingSpeed = 1.4;  // m/s
  constexpr double kSwingDegrees = 15.0;
  constexpr double kSwingFrequency = 1.0;  // Hz
  // 200 Hz IMU from 0.5 s before the first sweep; 20 s of sweeps (28 m).
  constexpr int kImuSamples = 4400;
  constexpr double kImuStart = -0.5;    // s
  constexpr double kImuPeriod = 0.005;  // s
  constexpr int kSweeps = 200;
  std::vector<SimBox> boxes = GarageStructure();
  for (const SimBox& car : ParkedCars(false)) boxes.push_back(car);
  const BoxScene scene = MakeScene(boxes);
  const LoopRoute route(0.0, kWalkingSpeed);
  const double amplitude = DegToRad(kSwingDegrees);
  const double omega = 2.0 * kPi * kSwingFrequency;
  const auto rig = [&](double time) {
    return route.PoseAt(time) *
           Pose(Eigen::AngleAxisd(amplitude * std::sin(omega * time),
                                  Vec3::UnitZ())
                    .toRotationMatrix(),
                Vec3::Zero());
  };
  ImuSamples imu;
  for (int i = 0; i <= kImuSamples; ++i) {
    const double time = kImuStart + kImuPeriod * i;
    imu.stamps.push_back(time);
    imu.gyro.push_back(Vec3(
        0.0, 0.0,
        route.YawRateAt(time) + amplitude * omega * std::cos(omega * time)));
    imu.accel.push_back(Vec3(0.0, 0.0, kGravity));
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
  std::mt19937 rng(kSeed);
  double worst = 0.0;
  *worst_yaw_deg = 0.0;
  for (int k = 0; k <= kSweeps; ++k) {
    const double stamp = kSweepPeriod * k;
    const Sweep sweep =
        SimulateSweep(scene, rig, vehicle_from_lidar, stamp, LidarModel(), rng);
    const OdometryFrame frame = pipeline.Process(
        sweep.points, std::vector<float>(sweep.points.size(), kIntensity),
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
  constexpr double kMaxGyroError = 0.15;     // m
  constexpr double kMaxGyroYawError = 0.5;   // deg
  constexpr double kMinConstantError = 1.0;  // m: it loses track.
  double yaw_gyro = 0.0;
  double yaw_constant = 0.0;
  const double gyro = SwingingRigError(true, &yaw_gyro);
  const double constant = SwingingRigError(false, &yaw_constant);
  std::printf(
      "       gyro prior: %.3f m, %.2f deg; constant velocity: "
      "%.3f m, %.2f deg\n",
      gyro, yaw_gyro, constant, yaw_constant);
  GR_EXPECT_LT(gyro, kMaxGyroError);
  GR_EXPECT_LT(yaw_gyro, kMaxGyroYawError);
  GR_EXPECT_GT(constant, kMinConstantError);
}

}  // namespace
}  // namespace gr
