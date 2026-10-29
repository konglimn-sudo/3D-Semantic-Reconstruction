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
void Simulate(const Motion& motion, double duration, const Mat3& imu_from_lidar,
              const Vec3& gyro_bias, const Vec3& accel_bias, const Vec3& scale,
              std::vector<double>* stamps, std::vector<Pose>* poses,
              ImuSamples* imu, const Vec3& lever = Vec3::Zero()) {
  constexpr int kSeed = 5;
  constexpr double kLidarPeriod = 0.1;  // s
  constexpr double kImuPeriod = 0.005;  // s
  constexpr double kStep = 1e-4;        // s, of the rate differences.
  constexpr double kRateStep = 1e-3;    // s, of the angular acceleration.
  constexpr double kGyroNoise = 0.002;  // rad/s
  constexpr double kAccelNoise = 0.02;  // m/s^2
  std::mt19937 rng(kSeed);
  std::normal_distribution<double> noise(0.0, 1.0);
  for (double time = 0.0; time <= duration; time += kLidarPeriod) {
    stamps->push_back(time);
    poses->emplace_back(motion.rotation(time), motion.position(time));
  }
  const Vec3 gravity(0.0, 0.0, -kStandardGravity);
  const auto rate = [&](double time) {
    return Vec3(So3Log(motion.rotation(time - kStep).transpose() *
                       motion.rotation(time + kStep)) /
                (2.0 * kStep));
  };
  for (double time = 0.0; time <= duration; time += kImuPeriod) {
    const Mat3 rotation = motion.rotation(time);
    const Vec3 angular_rate = rate(time);
    const Vec3 angular_acceleration =
        (rate(time + kRateStep) - rate(time - kRateStep)) / (2 * kRateStep);
    const Vec3 acceleration =
        (motion.position(time + kStep) - 2.0 * motion.position(time) +
         motion.position(time - kStep)) /
        (kStep * kStep);
    const Vec3 specific_force =
        rotation.transpose() * (acceleration - gravity) +
        angular_acceleration.cross(lever) +
        angular_rate.cross(angular_rate.cross(lever));
    imu->stamps.push_back(time);
    imu->gyro.push_back(imu_from_lidar * angular_rate + gyro_bias +
                        kGyroNoise * Vec3(noise(rng), noise(rng), noise(rng)));
    imu->accel.push_back(
        scale.cwiseProduct(imu_from_lidar * specific_force) + accel_bias +
        kAccelNoise * Vec3(noise(rng), noise(rng), noise(rng)));
  }
}

double AngleDeg(const Mat3& a, const Mat3& b) {
  return RadToDeg(So3Log(a.transpose() * b).norm());
}

// A handheld-like motion: turns, tilts and accelerations on all axes.
Motion HandheldMotion() {
  Motion motion;
  motion.position = [](double time) {
    return Vec3(3.0 * std::sin(0.3 * time) + 0.5 * time,
                2.0 * std::sin(0.5 * time), 0.4 * std::sin(0.7 * time));
  };
  motion.rotation = [](double time) {
    return Mat3(So3Exp(Vec3(0.3 * std::sin(0.9 * time),
                            0.25 * std::sin(0.6 * time + 1.0), 0.4 * time)));
  };
  return motion;
}

// The handheld recordings last two minutes.
constexpr double kHandheldDuration = 120.0;  // s
constexpr double kMaxRotationError = 0.1;    // deg
constexpr double kScaleTolerance = 3e-3;

GR_TEST(LidarImuTest, RecoversRotationBiasesAndScales) {
  constexpr double kTimeOffsetTolerance = 2e-3;  // s
  constexpr double kGyroBiasTolerance = 1e-3;    // rad/s
  constexpr double kAccelBiasTolerance = 0.03;   // m/s^2
  constexpr double kMaxRotationSigma = 0.05;     // deg
  const Motion motion = HandheldMotion();
  const Mat3 truth = So3Exp(Vec3(0.02, -0.03, 0.5));
  const Vec3 gyro_bias(0.01, -0.02, 0.005);
  const Vec3 accel_bias(0.1, -0.05, 0.2);
  const Vec3 scale(1.01, 0.99, 1.005);
  std::vector<double> stamps;
  std::vector<Pose> poses;
  ImuSamples imu;
  Simulate(motion, kHandheldDuration, truth, gyro_bias, accel_bias, scale,
           &stamps, &poses, &imu);
  const LidarImuResult result = CalibrateLidarImu(stamps, poses, imu, Pose());
  GR_ASSERT_TRUE(result.converged);
  GR_EXPECT_LT(AngleDeg(result.imu_from_lidar, truth), kMaxRotationError);
  GR_EXPECT_NEAR(result.time_offset, 0.0, kTimeOffsetTolerance);
  GR_EXPECT_LT((result.gyro_bias - gyro_bias).norm(), kGyroBiasTolerance);
  GR_EXPECT_LT((result.accel_bias - accel_bias).norm(), kAccelBiasTolerance);
  GR_EXPECT_LT((result.accel_scale - scale).norm(), kScaleTolerance);
  GR_EXPECT_LT(result.sigma_rotation_deg.maxCoeff(), kMaxRotationSigma);
}

GR_TEST(LidarImuTest, ACarTurningOnFlatGroundStillFixesTheYaw) {
  // A car: yaw only, speed changes and turns on flat ground. The gyro sees
  // one axis; the accelerations fix the rest of the rotation. The vertical
  // scale and bias are not separable on flat ground (large sigma).
  constexpr double kDuration = 60.0;  // s
  // The lever arm is free (a car does not observe its height).
  constexpr double kFreeLeverArmSigma = 100.0;  // m
  constexpr double kMaxCarRotationError = 0.3;  // deg
  Motion motion;
  motion.position = [](double time) {
    return Vec3(20.0 * std::sin(0.1 * time) + 2.5 * time,
                8.0 * std::sin(0.17 * time), 0.0);
  };
  motion.rotation = [](double time) {  // Heading along the velocity.
    const double dx = 2.0 * std::cos(0.1 * time) + 2.5;
    const double dy = 1.36 * std::cos(0.17 * time);
    return Mat3(So3Exp(Vec3(0.0, 0.0, std::atan2(dy, dx))));
  };
  const Mat3 truth = So3Exp(Vec3(0.01, 0.02, -0.3));
  std::vector<double> stamps;
  std::vector<Pose> poses;
  ImuSamples imu;
  Simulate(motion, kDuration, truth, Vec3(0.002, 0.001, -0.003),
           Vec3(0.05, 0.02, 0.0), Vec3::Ones(), &stamps, &poses, &imu);
  LidarImuConfig config;
  config.estimate_scale = false;
  config.lever_arm_sigma = kFreeLeverArmSigma;
  const LidarImuResult result =
      CalibrateLidarImu(stamps, poses, imu, Pose(), config);
  GR_ASSERT_TRUE(result.converged);
  GR_EXPECT_LT(AngleDeg(result.imu_from_lidar, truth), kMaxCarRotationError);
}

GR_TEST(LidarImuTest, EstimatesTheLeverArmOfARotatingRig) {
  // As the first test, with the IMU 0.3 m from the LiDAR; the rig's lever
  // arm is 5 cm off.
  constexpr double kLeverArmTolerance = 0.02;  // m
  const Motion motion = HandheldMotion();
  const Mat3 truth = So3Exp(Vec3(0.02, -0.03, 0.5));
  const Vec3 lever(0.1, -0.05, 0.25);
  std::vector<double> stamps;
  std::vector<Pose> poses;
  ImuSamples imu;
  Simulate(motion, kHandheldDuration, truth, Vec3::Zero(), Vec3::Zero(),
           Vec3::Ones(), &stamps, &poses, &imu, lever);
  // The rig's imu_from_lidar: rotation 0.8 deg off, lever arm 5 cm off
  // (the LiDAR at -R r in the IMU frame).
  const Mat3 rig = truth * So3Exp(Vec3(0.01, 0.0, -0.01));
  const Vec3 rig_lever = lever + Vec3(0.05, 0.0, 0.0);
  const LidarImuResult result =
      CalibrateLidarImu(stamps, poses, imu, Pose(rig, -(rig * rig_lever)));
  GR_ASSERT_TRUE(result.converged);
  GR_EXPECT_LT(AngleDeg(result.imu_from_lidar, truth), kMaxRotationError);
  GR_EXPECT_LT((result.lever_arm - lever).norm(), kLeverArmTolerance);
  GR_EXPECT_LT((result.accel_scale - Vec3::Ones()).norm(), kScaleTolerance);
}

}  // namespace
}  // namespace gr
