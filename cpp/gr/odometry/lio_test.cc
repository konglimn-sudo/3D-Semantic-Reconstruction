#include "gr/odometry/lio.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "gr/odometry/odometry.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::GarageStructure;
using ::gr::testing::LidarModel;
using ::gr::testing::MakeScene;
using ::gr::testing::ParkedCars;
using ::gr::testing::SimBox;
using ::gr::testing::SimulateSweep;
using ::gr::testing::Sweep;

constexpr int kSeed = 7;
constexpr float kIntensity = 0.5f;
// The IMU runs from 0.5 s before the first sweep to 0.5 s after the last.
constexpr double kImuMargin = 0.5;  // s

Mat3 Rpy(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Vec3::UnitZ()) *
          Eigen::AngleAxisd(pitch, Vec3::UnitY()) *
          Eigen::AngleAxisd(roll, Vec3::UnitX()))
      .toRotationMatrix();
}

// A handheld walk at 1.2 m/s along +x from (start_x, start_y): steps bob the
// device by 3 cm at 1.8 Hz, the body sways it in roll / pitch and the path
// weaves.
std::function<Pose(double)> Walk(double start_x, double start_y,
                                 double height) {
  constexpr double kSpeed = 1.2;      // m/s
  constexpr double kWeave = 0.5;      // m
  constexpr double kWeaveRate = 0.3;  // rad/s
  // The slope of the weave, kWeave * kWeaveRate (m/s) across the walk.
  constexpr double kWeaveSlope = 0.15;
  constexpr double kBob = 0.03;           // m
  constexpr double kStepFrequency = 1.8;  // Hz
  constexpr double kRollSway = 0.03;      // rad, at half the step frequency.
  constexpr double kSwayFrequency = 0.9;  // Hz
  constexpr double kPitchSway = 0.02;     // rad, at the step frequency.
  constexpr double kYawSway = 0.1;        // rad
  constexpr double kYawSwayRate = 0.5;    // rad/s
  return [=](double time) {
    const double x = start_x + kSpeed * time;
    const double y = start_y + kWeave * std::sin(kWeaveRate * time);
    const double z =
        height + kBob * std::sin(2.0 * kPi * kStepFrequency * time);
    const double heading =
        std::atan2(kWeaveSlope * std::cos(kWeaveRate * time), kSpeed);
    return Pose(Rpy(kRollSway * std::sin(2.0 * kPi * kSwayFrequency * time),
                    kPitchSway * std::sin(2.0 * kPi * kStepFrequency * time),
                    heading + kYawSway * std::sin(kYawSwayRate * time)),
                Vec3(x, y, z));
  };
}

// Ideal 200 Hz IMU (vehicle axes) of a trajectory, by central differences,
// measured at `position` in the vehicle frame.
ImuSamples SimulateImu(const std::function<Pose(double)>& trajectory,
                       double start_time, double end_time,
                       const Vec3& position = Vec3::Zero()) {
  constexpr double kPeriod = 0.005;  // s
  constexpr double kStep = 1e-3;     // s, of the central differences.
  ImuSamples imu;
  imu.position = position;
  for (double time = start_time; time <= end_time; time += kPeriod) {
    const Pose before = trajectory(time - kStep);
    const Pose now = trajectory(time);
    const Pose after = trajectory(time + kStep);
    imu.stamps.push_back(time);
    imu.gyro.push_back(So3Log(before.rotation.transpose() * after.rotation) /
                       (2 * kStep));
    const Vec3 acceleration =
        (after * position - 2.0 * (now * position) + before * position) /
        (kStep * kStep);
    imu.accel.push_back(now.rotation.transpose() *
                        (acceleration + Vec3(0, 0, kStandardGravity)));
  }
  return imu;
}

struct RunResult {
  double position_error = 0.0;  // m, at the end.
  double rotation_error = 0.0;  // deg, at the end.
  double max_position_error = 0.0;
};

// `imu_at`: the IMU (and, 5 cm off it, the LiDAR) in the vehicle frame;
// `imu_at_known`: whether the IMU samples say so (ImuSamples::position).
RunResult Run(const BoxScene& scene, const std::function<Pose(double)>& walk,
              const LidarModel& lidar, double duration, bool use_lio,
              const Vec3& imu_at = Vec3::Zero(), bool imu_at_known = true) {
  const Pose vehicle_from_lidar(Rpy(0.0, DegToRad(-20.0), 0.0),
                                Vec3(0.1, 0.0, 0.05) + imu_at);
  OdometryConfig config;
  config.voxel_size = 0.3;
  config.max_range = lidar.max_range;
  config.min_range = 0.5;
  config.use_lio = use_lio;
  config.lio.map_voxel = 0.3;
  config.lio.scan_voxel = 0.3;
  config.lio.insert_voxel = 0.15;
  // The simulated IMU is ideal: its noise model can be tight (real rigs
  // need the looser defaults).
  config.lio.gyro_noise = 5e-3;
  config.lio.accel_noise = 0.05;
  ImuSamples imu =
      SimulateImu(walk, -kImuMargin, duration + kImuMargin, imu_at);
  if (!imu_at_known) imu.position = Vec3::Zero();
  MotionPrior prior(imu, WheelSamples());
  OdometryPipeline pipeline(config, KeyframeConfig(), vehicle_from_lidar,
                            walk(0.0), prior);
  std::mt19937 rng(kSeed);
  RunResult result;
  for (int k = 0; k * lidar.sweep <= duration; ++k) {
    const double stamp = k * lidar.sweep;
    const Sweep sweep =
        SimulateSweep(scene, walk, vehicle_from_lidar, stamp, lidar, rng);
    const OdometryFrame frame = pipeline.Process(
        sweep.points, std::vector<float>(sweep.points.size(), kIntensity),
        sweep.times, stamp);
    const Pose truth = walk(stamp) * vehicle_from_lidar;
    const double error = (frame.pose.translation - truth.translation).norm();
    result.max_position_error = std::max(result.max_position_error, error);
    result.position_error = error;
    result.rotation_error = RadToDeg(
        So3Log(frame.pose.rotation.transpose() * truth.rotation).norm());
  }
  GR_EXPECT_EQ(pipeline.uses_lio(), use_lio);
  return result;
}

// An open deck: a floor and a few posts 12 m apart, seen by a wide but
// short-range LiDAR (OS0 in sunlight): mostly ground.
BoxScene OpenDeck() {
  std::vector<SimBox> boxes;
  boxes.push_back({Vec3(25, 0, -0.1), Vec3(80, 40, 0.2)});
  for (double x = 0.0; x <= 50.0; x += 12.0) {
    boxes.push_back({Vec3(x, 5.0, 1.0), Vec3(0.4, 0.4, 2.0)});
    boxes.push_back({Vec3(x + 6.0, -5.0, 1.0), Vec3(0.4, 0.4, 2.0)});
  }
  return MakeScene(boxes);
}

LidarModel WideShortRange() {
  LidarModel lidar;
  lidar.beams = 32;
  lidar.el_min = -45.0;
  lidar.el_max = 45.0;
  lidar.max_range = 8.0;
  return lidar;
}

// Walks hold the device at this height (m) and last this long (s) on the
// open deck.
constexpr double kDeviceHeight = 1.5;
constexpr double kDeckWalk = 30.0;

GR_TEST(LioTest, TracksAHandheldWalkThroughTheGarage) {
  constexpr double kAisleY = 5.0;             // m
  constexpr double kDuration = 20.0;          // s, 24 m.
  constexpr double kMaxPositionError = 0.10;  // m
  constexpr double kMaxRotationError = 0.3;   // deg
  std::vector<SimBox> boxes = GarageStructure();
  for (const SimBox& car : ParkedCars(false)) boxes.push_back(car);
  const BoxScene scene = MakeScene(boxes);
  const RunResult lio = Run(scene, Walk(0.0, kAisleY, kDeviceHeight),
                            LidarModel(), kDuration, true);
  std::printf("       garage walk 24 m: LIO end error %.3f m, %.3f deg\n",
              lio.position_error, lio.rotation_error);
  GR_EXPECT_LT(lio.max_position_error, kMaxPositionError);
  GR_EXPECT_LT(lio.rotation_error, kMaxRotationError);
}

GR_TEST(LioTest, KeepsTheImuMotionWhereTheGroundDominates) {
  constexpr double kMaxPositionError = 0.30;  // m
  constexpr double kMaxRotationError = 0.5;   // deg
  const BoxScene scene = OpenDeck();
  const RunResult icp = Run(scene, Walk(0.0, 0.0, kDeviceHeight),
                            WideShortRange(), kDeckWalk, false);
  const RunResult lio = Run(scene, Walk(0.0, 0.0, kDeviceHeight),
                            WideShortRange(), kDeckWalk, true);
  std::printf(
      "       open deck 36 m: ICP + gyro end error %.2f m (max %.2f), "
      "LIO %.3f m (max %.3f), %.3f deg\n",
      icp.position_error, icp.max_position_error, lio.position_error,
      lio.max_position_error, lio.rotation_error);
  GR_EXPECT_LT(lio.max_position_error, kMaxPositionError);
  GR_EXPECT_LT(lio.rotation_error, kMaxRotationError);
  GR_EXPECT_LT(lio.max_position_error, icp.max_position_error);
}

// Handheld profiles put the vehicle origin on the ground, 1.8 m below the
// IMU. The same device motion as KeepsTheImuMotionWhereTheGroundDominates,
// described with that origin: the filter integrating at the IMU tracks it
// exactly as before; integrating at the origin, where the device's sway
// (1-2 degrees here) reads as acceleration it does not have, is worse.
GR_TEST(LioTest, IntegratesTheImuWhereItIsMounted) {
  constexpr double kImuHeight = 1.8;  // m above the vehicle origin.
  // The known lever arm tracks as the device itself does; the ignored one
  // is worse by more than this (m).
  constexpr double kErrorTolerance = 0.02;
  constexpr double kMaxRotationError = 0.5;  // deg
  const BoxScene scene = OpenDeck();
  const Vec3 imu_at(0.0, 0.0, kImuHeight);
  const std::function<Pose(double)> device = Walk(0.0, 0.0, kDeviceHeight);
  const std::function<Pose(double)> origin = [&](double time) {
    return device(time) * Pose(Mat3::Identity(), -imu_at);
  };
  const RunResult baseline =
      Run(scene, device, WideShortRange(), kDeckWalk, true);
  const RunResult known =
      Run(scene, origin, WideShortRange(), kDeckWalk, true, imu_at, true);
  const RunResult ignored =
      Run(scene, origin, WideShortRange(), kDeckWalk, true, imu_at, false);
  std::printf(
      "       open deck, origin 1.8 m below the IMU: max error %.3f m "
      "(origin at the IMU %.3f m, lever arm ignored %.3f m)\n",
      known.max_position_error, baseline.max_position_error,
      ignored.max_position_error);
  GR_EXPECT_NEAR(known.max_position_error, baseline.max_position_error,
                 kErrorTolerance);
  GR_EXPECT_LT(known.rotation_error, kMaxRotationError);
  GR_EXPECT_GT(ignored.max_position_error,
               known.max_position_error + kErrorTolerance);
}

}  // namespace
}  // namespace gr
