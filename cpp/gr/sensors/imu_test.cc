#include "gr/sensors/imu.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// The simulated drives last 10 s: IMU at 200 Hz, wheel speed at 100 Hz.
constexpr int kImuSamples = 2000;
constexpr double kImuPeriod = 0.005;  // s
constexpr int kWheelSamples = 1000;
constexpr double kWheelPeriod = 0.01;  // s

// A sweep of 100 points on a circle around the LiDAR, taken over the 0.1 s
// before the sweep stamp.
constexpr int kSweepPoints = 100;
constexpr double kAzimuthStep = 0.06;   // rad per point
constexpr double kPointHeight = 0.5;    // m
constexpr double kSweepStart = -0.1;    // s, relative to the stamp.
constexpr double kPointPeriod = 0.001;  // s
// Deskewing leaves at most this error (m).
constexpr double kMaxDeskewError = 2e-3;

constexpr double kTolerance = 1e-9;

// Constant yaw rate and speed (vehicle frame) sampled from t = 0 to 10 s.
MotionPrior CirclePrior(double yaw_rate, double speed) {
  ImuSamples imu;
  WheelSamples wheel;
  for (int i = 0; i <= kImuSamples; ++i) {
    imu.stamps.push_back(kImuPeriod * i);
    imu.gyro.push_back(Vec3(0, 0, yaw_rate));
    imu.accel.push_back(Vec3(0, speed * yaw_rate, kGravity));
  }
  for (int i = 0; i <= kWheelSamples; ++i) {
    wheel.stamps.push_back(kWheelPeriod * i);
    wheel.speed.push_back(speed);
  }
  return MotionPrior(std::move(imu), std::move(wheel));
}

// The sweep of points at `radius` (m) and their times.
void MakeSweep(double radius, Points* points, std::vector<double>* times) {
  for (int i = 0; i < kSweepPoints; ++i) {
    points->emplace_back(radius * std::cos(kAzimuthStep * i),
                         radius * std::sin(kAzimuthStep * i), kPointHeight);
    times->push_back(kSweepStart + kPointPeriod * i);
  }
}

GR_TEST(MotionPriorTest, IntegratesACircularArc) {
  constexpr double kYawRate = 0.3;  // rad/s
  constexpr double kSpeed = 5.0;    // m/s
  constexpr double kStart = 2.0;    // s; the arc lasts 1 s.
  constexpr double kEnd = 3.0;
  // Translation uses the heading at the start of each 5 ms IMU interval:
  // error ~ v * w * dt / 2 per second (4 mm here).
  constexpr double kTranslationTolerance = 6e-3;  // m
  const MotionPrior prior = CirclePrior(kYawRate, kSpeed);
  GR_ASSERT_TRUE(prior.valid());
  const std::optional<Pose> delta = prior.Delta(kStart, kEnd);
  GR_ASSERT_TRUE(delta.has_value());
  const double turned = kYawRate * (kEnd - kStart);
  GR_EXPECT_NEAR(RotationAngle(delta->rotation), turned, kTolerance);
  const double radius = kSpeed / kYawRate;
  GR_EXPECT_NEAR(delta->translation.x(), radius * std::sin(turned),
                 kTranslationTolerance);
  GR_EXPECT_NEAR(delta->translation.y(), radius * (1.0 - std::cos(turned)),
                 kTranslationTolerance);
  constexpr double kBeforeTheImu = -1.0;
  GR_EXPECT_FALSE(prior.Delta(kBeforeTheImu, kStart).has_value());
  GR_EXPECT_FALSE(prior.Delta(kEnd, kStart).has_value());  // Backwards.
}

GR_TEST(MotionPriorTest, DeskewCompensatesRotationWithLeverArm) {
  constexpr double kYawRate = 1.0;  // Pure rotation about the vehicle origin.
  constexpr double kSweepRadius = 10.0;  // m
  constexpr double kStamp = 5.0;         // s
  const MotionPrior prior = CirclePrior(kYawRate, 0.0);
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  Points points;
  std::vector<double> times;
  MakeSweep(kSweepRadius, &points, &times);
  const std::optional<Points> out =
      DeskewWithPrior(points, times, kStamp, prior, vehicle_from_lidar);
  GR_ASSERT_TRUE(out.has_value());
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  double worst = 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    // lidar(end) <- lidar(tau) for a vehicle turning about its origin.
    const Pose turn(Eigen::AngleAxisd(kYawRate * times[i], Vec3::UnitZ())
                        .toRotationMatrix(),
                    Vec3::Zero());
    const Vec3 expected =
        (lidar_from_vehicle * turn * vehicle_from_lidar) * points[i];
    worst = std::max(worst, ((*out)[i] - expected).norm());
  }
  GR_EXPECT_LT(worst, kMaxDeskewError);
}

// A rig without wheels: it turns at `kYawRate` while moving with a constant
// world velocity (a person walking with a handheld scanner).
GR_TEST(MotionPriorTest, GyroDeltaWithAConstantWorldVelocity) {
  constexpr double kYawRate = 0.8;  // rad/s
  constexpr double kStart = 2.0;    // s
  constexpr double kEnd = 2.3;
  constexpr double kSweepRadius = 8.0;  // m
  constexpr double kStamp = 5.0;        // s
  ImuSamples imu;
  for (int i = 0; i <= kImuSamples; ++i) {
    imu.stamps.push_back(kImuPeriod * i);
    imu.gyro.push_back(Vec3(0, 0, kYawRate));
    imu.accel.push_back(Vec3(0, 0, kGravity));
  }
  const MotionPrior prior(std::move(imu), WheelSamples());
  GR_EXPECT_FALSE(prior.valid());
  GR_ASSERT_TRUE(prior.has_gyro());
  GR_EXPECT_FALSE(prior.Delta(kStart, kEnd).has_value());
  const auto world_from_vehicle = [&](double time) {
    return Eigen::AngleAxisd(kYawRate * time, Vec3::UnitZ()).toRotationMatrix();
  };
  const Vec3 world_velocity(1.2, -0.4, 0.05);
  const std::optional<Pose> delta = prior.GyroDelta(
      kStart, kEnd, world_from_vehicle(kEnd).transpose() * world_velocity);
  GR_ASSERT_TRUE(delta.has_value());
  const Mat3 expected_rotation =
      world_from_vehicle(kStart).transpose() * world_from_vehicle(kEnd);
  GR_EXPECT_LT(RotationAngle(delta->rotation.transpose() * expected_rotation),
               kTolerance);
  const Vec3 expected_translation =
      world_from_vehicle(kStart).transpose() * world_velocity * (kEnd - kStart);
  GR_EXPECT_LT((delta->translation - expected_translation).norm(), kTolerance);
  // Beyond the end of the IMU samples.
  constexpr double kLateStart = 9.0;
  constexpr double kLateEnd = 11.0;
  GR_EXPECT_FALSE(
      prior.GyroDelta(kLateStart, kLateEnd, Vec3::Zero()).has_value());

  // Deskew with the gyro and the velocity: points of a static scene seen
  // during the sweep map to where they are seen at its end.
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(0.1, 0.0, 1.3));
  const Vec3 velocity_end =
      world_from_vehicle(kStamp).transpose() * world_velocity;
  Points points;
  std::vector<double> times;
  MakeSweep(kSweepRadius, &points, &times);
  const std::optional<Points> out =
      DeskewWithPrior(points, times, kStamp, prior, vehicle_from_lidar,
                      kDefaultDeskewSamples, velocity_end);
  GR_ASSERT_TRUE(out.has_value());
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  const auto world_vehicle = [&](double time) {
    return Pose(world_from_vehicle(time), world_velocity * time);
  };
  double worst = 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    const Pose end_from_tau =
        lidar_from_vehicle * world_vehicle(kStamp).Inverse() *
        world_vehicle(kStamp + times[i]) * vehicle_from_lidar;
    worst = std::max(worst, ((*out)[i] - end_from_tau * points[i]).norm());
  }
  GR_EXPECT_LT(worst, kMaxDeskewError);
}

GR_TEST(GravityTest, RecoversTiltAndSkipsTurns) {
  // 2 s at 200 Hz; the vehicle is pitched by 3 deg and turns at 10 deg/s
  // after 1 s.
  constexpr int kSamples = 400;
  constexpr double kPitchDegrees = 3.0;
  constexpr double kTurnStart = 1.0;  // s
  constexpr double kTurnRateDegrees = 10.0;
  constexpr double kDuration = 2.0;  // s
  constexpr double kLidarYaw = 0.2;  // rad
  // One stamp before the turn, one during it.
  constexpr double kStraightStamp = 0.5;
  constexpr double kTurningStamp = 1.6;
  constexpr double kSigmaTolerance = 1e-12;
  ImuSamples imu;
  WheelSamples wheel;
  const Mat3 world_from_vehicle =
      Eigen::AngleAxisd(DegToRad(kPitchDegrees), Vec3::UnitY())
          .toRotationMatrix();
  for (int i = 0; i <= kSamples; ++i) {
    const double time = kImuPeriod * i;
    imu.stamps.push_back(time);
    imu.gyro.push_back(
        Vec3(0, 0, time > kTurnStart ? DegToRad(kTurnRateDegrees) : 0.0));
    imu.accel.push_back(world_from_vehicle.transpose() * Vec3(0, 0, kGravity));
  }
  wheel.stamps = {0.0, kDuration};
  wheel.speed = {0.0, 0.0};
  const Mat3 lidar_from_vehicle =
      Eigen::AngleAxisd(kLidarYaw, Vec3::UnitZ()).toRotationMatrix();
  const GravityConfig config;
  const std::vector<GravityMeasurement> gravity = MeasureGravity(
      imu, wheel, lidar_from_vehicle, {kStraightStamp, kTurningStamp}, config);
  GR_ASSERT_TRUE(gravity.size() == 1);  // The stamp while turning is rejected.
  GR_EXPECT_EQ(gravity[0].index, 0);
  const Vec3 expected =
      lidar_from_vehicle * world_from_vehicle.transpose() * Vec3::UnitZ();
  GR_EXPECT_LT((gravity[0].up - expected).norm(), kTolerance);
  GR_EXPECT_NEAR(gravity[0].sigma_rad, DegToRad(config.sigma_deg),
                 kSigmaTolerance);
}

}  // namespace
}  // namespace gr
