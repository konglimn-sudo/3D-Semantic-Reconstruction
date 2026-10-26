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

constexpr double kGravity = 9.80665;

Mat3 Rpy(double roll, double pitch, double yaw) {
  return (Eigen::AngleAxisd(yaw, Vec3::UnitZ()) *
          Eigen::AngleAxisd(pitch, Vec3::UnitY()) *
          Eigen::AngleAxisd(roll, Vec3::UnitX()))
      .toRotationMatrix();
}

// A handheld walk at 1.2 m/s along +x from (x0, y0): steps bob the device by
// 3 cm at 1.8 Hz, the body sways it in roll / pitch and the path weaves.
std::function<Pose(double)> Walk(double x0, double y0, double height) {
  return [=](double t) {
    const double x = x0 + 1.2 * t;
    const double y = y0 + 0.5 * std::sin(0.3 * t);
    const double z = height + 0.03 * std::sin(2.0 * kPi * 1.8 * t);
    const double heading = std::atan2(0.15 * std::cos(0.3 * t), 1.2);
    return Pose(Rpy(0.03 * std::sin(2.0 * kPi * 0.9 * t),
                    0.02 * std::sin(2.0 * kPi * 1.8 * t),
                    heading + 0.1 * std::sin(0.5 * t)),
                Vec3(x, y, z));
  };
}

// Ideal 200 Hz IMU (vehicle axes) of a trajectory, by central differences,
// measured at `position` in the vehicle frame.
ImuSamples SimulateImu(const std::function<Pose(double)>& trajectory,
                       double t0, double t1,
                       const Vec3& position = Vec3::Zero()) {
  ImuSamples imu;
  imu.position = position;
  const double h = 1e-3;
  for (double t = t0; t <= t1; t += 0.005) {
    const Pose a = trajectory(t - h), b = trajectory(t), c = trajectory(t + h);
    imu.stamps.push_back(t);
    imu.gyro.push_back(So3Log(a.rotation.transpose() * c.rotation) / (2 * h));
    const Vec3 acceleration =
        (c * position - 2.0 * (b * position) + a * position) / (h * h);
    imu.accel.push_back(b.rotation.transpose() *
                        (acceleration + Vec3(0, 0, kGravity)));
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
  ImuSamples imu = SimulateImu(walk, -0.5, duration + 0.5, imu_at);
  if (!imu_at_known) imu.position = Vec3::Zero();
  MotionPrior prior(imu, WheelSamples());
  OdometryPipeline pipeline(config, KeyframeConfig(), vehicle_from_lidar,
                            walk(0.0), prior);
  std::mt19937 rng(7);
  RunResult result;
  for (int k = 0; k * lidar.sweep <= duration; ++k) {
    const double stamp = k * lidar.sweep;
    const Sweep sweep =
        SimulateSweep(scene, walk, vehicle_from_lidar, stamp, lidar, rng);
    const OdometryFrame frame = pipeline.Process(
        sweep.points, std::vector<float>(sweep.points.size(), 0.5f),
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

GR_TEST(LioTest, TracksAHandheldWalkThroughTheGarage) {
  std::vector<SimBox> boxes = GarageStructure();
  for (const SimBox& car : ParkedCars(false)) boxes.push_back(car);
  const BoxScene scene = MakeScene(boxes);
  const RunResult lio = Run(scene, Walk(0.0, 5.0, 1.5), LidarModel(), 20.0,
                            true);
  std::printf("       garage walk 24 m: LIO end error %.3f m, %.3f deg\n",
              lio.position_error, lio.rotation_error);
  GR_EXPECT_LT(lio.max_position_error, 0.10);
  GR_EXPECT_LT(lio.rotation_error, 0.3);
}

GR_TEST(LioTest, KeepsTheImuMotionWhereTheGroundDominates) {
  const BoxScene scene = OpenDeck();
  const RunResult icp =
      Run(scene, Walk(0.0, 0.0, 1.5), WideShortRange(), 30.0, false);
  const RunResult lio =
      Run(scene, Walk(0.0, 0.0, 1.5), WideShortRange(), 30.0, true);
  std::printf(
      "       open deck 36 m: ICP + gyro end error %.2f m (max %.2f), "
      "LIO %.3f m (max %.3f), %.3f deg\n",
      icp.position_error, icp.max_position_error, lio.position_error,
      lio.max_position_error, lio.rotation_error);
  GR_EXPECT_LT(lio.max_position_error, 0.30);
  GR_EXPECT_LT(lio.rotation_error, 0.5);
  GR_EXPECT_LT(lio.max_position_error, icp.max_position_error);
}

// Handheld profiles put the vehicle origin on the ground, 1.8 m below the
// IMU. The same device motion as KeepsTheImuMotionWhereTheGroundDominates,
// described with that origin: the filter integrating at the IMU tracks it
// exactly as before; integrating at the origin, where the device's sway
// (1-2 degrees here) reads as acceleration it does not have, is worse.
GR_TEST(LioTest, IntegratesTheImuWhereItIsMounted) {
  const BoxScene scene = OpenDeck();
  const Vec3 imu_at(0.0, 0.0, 1.8);
  const std::function<Pose(double)> device = Walk(0.0, 0.0, 1.5);
  const std::function<Pose(double)> origin = [&](double t) {
    return device(t) * Pose(Mat3::Identity(), -imu_at);
  };
  const RunResult baseline =
      Run(scene, device, WideShortRange(), 30.0, true);
  const RunResult known =
      Run(scene, origin, WideShortRange(), 30.0, true, imu_at, true);
  const RunResult ignored =
      Run(scene, origin, WideShortRange(), 30.0, true, imu_at, false);
  std::printf("       open deck, origin 1.8 m below the IMU: max error %.3f m "
              "(origin at the IMU %.3f m, lever arm ignored %.3f m)\n",
              known.max_position_error, baseline.max_position_error,
              ignored.max_position_error);
  GR_EXPECT_NEAR(known.max_position_error, baseline.max_position_error, 0.02);
  GR_EXPECT_LT(known.rotation_error, 0.5);
  GR_EXPECT_GT(ignored.max_position_error, known.max_position_error + 0.02);
}

}  // namespace
}  // namespace gr
