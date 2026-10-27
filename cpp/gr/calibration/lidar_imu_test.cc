#include "gr/calibration/lidar_imu.h"

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

struct Motion {
  std::function<Vec3(double)> position;
  std::function<Mat3(double)> rotation;  // world_from_lidar.
};

// LiDAR poses at 10 Hz and an IMU at 200 Hz with rotation `imu_from_lidar`,
// at `lever` (LiDAR axes), with biases, accelerometer scales and white
// noise.
void Simulate(const Motion& motion, double duration, const Mat3& r,
              const Vec3& gyro_bias, const Vec3& accel_bias,
              const Vec3& scale, std::vector<double>* stamps,
              std::vector<Pose>* poses, ImuSamples* imu,
              const Vec3& lever = Vec3::Zero()) {
  std::mt19937 rng(5);
  std::normal_distribution<double> noise(0.0, 1.0);
  for (double t = 0.0; t <= duration; t += 0.1) {
    stamps->push_back(t);
    poses->emplace_back(motion.rotation(t), motion.position(t));
  }
  const double dt = 1e-4;
  const Vec3 g(0.0, 0.0, -9.80665);
  const auto rate = [&](double t) {
    return Vec3(So3Log(motion.rotation(t - dt).transpose() *
                       motion.rotation(t + dt)) /
                (2.0 * dt));
  };
  for (double t = 0.0; t <= duration; t += 0.005) {
    const Mat3 rot = motion.rotation(t);
    const Vec3 w = rate(t);
    const Vec3 alpha = (rate(t + 1e-3) - rate(t - 1e-3)) / 2e-3;
    const Vec3 a = (motion.position(t + dt) - 2.0 * motion.position(t) +
                    motion.position(t - dt)) /
                   (dt * dt);
    const Vec3 f = rot.transpose() * (a - g) + alpha.cross(lever) +
                   w.cross(w.cross(lever));
    imu->stamps.push_back(t);
    imu->gyro.push_back(r * w + gyro_bias +
                        0.002 * Vec3(noise(rng), noise(rng), noise(rng)));
    imu->accel.push_back(scale.cwiseProduct(r * f) + accel_bias +
                         0.02 * Vec3(noise(rng), noise(rng), noise(rng)));
  }
}

double AngleDeg(const Mat3& a, const Mat3& b) {
  return RadToDeg(So3Log(a.transpose() * b).norm());
}

GR_TEST(LidarImuTest, RecoversRotationBiasesAndScales) {
  // A handheld-like motion: turns, tilts and accelerations on all axes.
  Motion motion;
  motion.position = [](double t) {
    return Vec3(3.0 * std::sin(0.3 * t) + 0.5 * t, 2.0 * std::sin(0.5 * t),
                0.4 * std::sin(0.7 * t));
  };
  motion.rotation = [](double t) {
    return Mat3(So3Exp(Vec3(0.3 * std::sin(0.9 * t),
                            0.25 * std::sin(0.6 * t + 1.0), 0.4 * t)));
  };
  const Mat3 truth = So3Exp(Vec3(0.02, -0.03, 0.5));
  std::vector<double> stamps;
  std::vector<Pose> poses;
  ImuSamples imu;
  Simulate(motion, 120.0, truth, Vec3(0.01, -0.02, 0.005),
           Vec3(0.1, -0.05, 0.2), Vec3(1.01, 0.99, 1.005), &stamps, &poses,
           &imu);
  const LidarImuResult result =
      CalibrateLidarImu(stamps, poses, imu, Pose());
  GR_ASSERT_TRUE(result.converged);
  GR_EXPECT_LT(AngleDeg(result.imu_from_lidar, truth), 0.1);
  GR_EXPECT_NEAR(result.time_offset, 0.0, 2e-3);
  GR_EXPECT_LT((result.gyro_bias - Vec3(0.01, -0.02, 0.005)).norm(), 1e-3);
  GR_EXPECT_LT((result.accel_bias - Vec3(0.1, -0.05, 0.2)).norm(), 0.03);
  GR_EXPECT_LT((result.accel_scale - Vec3(1.01, 0.99, 1.005)).norm(), 3e-3);
  GR_EXPECT_LT(result.sigma_rotation_deg.maxCoeff(), 0.05);
}

GR_TEST(LidarImuTest, ACarTurningOnFlatGroundStillFixesTheYaw) {
  // A car: yaw only, speed changes and turns on flat ground. The gyro sees
  // one axis; the accelerations fix the rest of the rotation. The vertical
  // scale and bias are not separable on flat ground (large sigma).
  Motion motion;
  motion.position = [](double t) {
    return Vec3(20.0 * std::sin(0.1 * t) + 2.5 * t, 8.0 * std::sin(0.17 * t),
                0.0);
  };
  motion.rotation = [](double t) {  // Heading along the velocity.
    const double dx = 2.0 * std::cos(0.1 * t) + 2.5;
    const double dy = 1.36 * std::cos(0.17 * t);
    return Mat3(So3Exp(Vec3(0.0, 0.0, std::atan2(dy, dx))));
  };
  const Mat3 truth = So3Exp(Vec3(0.01, 0.02, -0.3));
  std::vector<double> stamps;
  std::vector<Pose> poses;
  ImuSamples imu;
  Simulate(motion, 60.0, truth, Vec3(0.002, 0.001, -0.003),
           Vec3(0.05, 0.02, 0.0), Vec3::Ones(), &stamps, &poses, &imu);
  LidarImuConfig config;
  config.estimate_scale = false;
  config.lever_arm_sigma = 100.0;
  const LidarImuResult result =
      CalibrateLidarImu(stamps, poses, imu, Pose(), config);
  GR_ASSERT_TRUE(result.converged);
  GR_EXPECT_LT(AngleDeg(result.imu_from_lidar, truth), 0.3);
}

GR_TEST(LidarImuTest, EstimatesTheLeverArmOfARotatingRig) {
  // As the first test, with the IMU 0.3 m from the LiDAR; the rig's lever
  // arm is 5 cm off.
  Motion motion;
  motion.position = [](double t) {
    return Vec3(3.0 * std::sin(0.3 * t) + 0.5 * t, 2.0 * std::sin(0.5 * t),
                0.4 * std::sin(0.7 * t));
  };
  motion.rotation = [](double t) {
    return Mat3(So3Exp(Vec3(0.3 * std::sin(0.9 * t),
                            0.25 * std::sin(0.6 * t + 1.0), 0.4 * t)));
  };
  const Mat3 truth = So3Exp(Vec3(0.02, -0.03, 0.5));
  const Vec3 lever(0.1, -0.05, 0.25);
  std::vector<double> stamps;
  std::vector<Pose> poses;
  ImuSamples imu;
  Simulate(motion, 120.0, truth, Vec3::Zero(), Vec3::Zero(), Vec3::Ones(),
           &stamps, &poses, &imu, lever);
  // The rig's imu_from_lidar: rotation 0.8 deg off, lever arm 5 cm off
  // (the LiDAR at -R r in the IMU frame).
  const Mat3 rig = truth * So3Exp(Vec3(0.01, 0.0, -0.01));
  const Vec3 rig_lever = lever + Vec3(0.05, 0.0, 0.0);
  const LidarImuResult result =
      CalibrateLidarImu(stamps, poses, imu, Pose(rig, -(rig * rig_lever)));
  GR_ASSERT_TRUE(result.converged);
  GR_EXPECT_LT(AngleDeg(result.imu_from_lidar, truth), 0.1);
  GR_EXPECT_LT((result.lever_arm - lever).norm(), 0.02);
  GR_EXPECT_LT((result.accel_scale - Vec3::Ones()).norm(), 3e-3);
}

}  // namespace
}  // namespace gr
