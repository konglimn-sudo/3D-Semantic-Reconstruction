#include "gr/sensors/imu.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Constant yaw rate and speed (vehicle frame) sampled from t = 0 to 10 s.
MotionPrior CirclePrior(double yaw_rate, double speed) {
  ImuSamples imu;
  WheelSamples wheel;
  for (int i = 0; i <= 2000; ++i) {
    imu.stamps.push_back(0.005 * i);
    imu.gyro.push_back(Vec3(0, 0, yaw_rate));
    imu.accel.push_back(Vec3(0, speed * yaw_rate, 9.81));
  }
  for (int i = 0; i <= 1000; ++i) {
    wheel.stamps.push_back(0.01 * i);
    wheel.speed.push_back(speed);
  }
  return MotionPrior(std::move(imu), std::move(wheel));
}

GR_TEST(MotionPriorTest, IntegratesACircularArc) {
  const double yaw_rate = 0.3;
  const double speed = 5.0;
  const MotionPrior prior = CirclePrior(yaw_rate, speed);
  GR_ASSERT_TRUE(prior.valid());
  const std::optional<Pose> delta = prior.Delta(2.0, 3.0);
  GR_ASSERT_TRUE(delta.has_value());
  GR_EXPECT_NEAR(RotationAngle(delta->rotation), yaw_rate, 1e-9);
  // Translation uses the heading at the start of each 5 ms IMU interval:
  // error ~ v * w * dt / 2 per second (4 mm here).
  GR_EXPECT_NEAR(delta->translation.x(), speed / yaw_rate * std::sin(yaw_rate),
                 6e-3);
  GR_EXPECT_NEAR(delta->translation.y(),
                 speed / yaw_rate * (1.0 - std::cos(yaw_rate)), 6e-3);
  GR_EXPECT_FALSE(prior.Delta(-1.0, 0.5).has_value());  // Outside the span.
  GR_EXPECT_FALSE(prior.Delta(3.0, 2.0).has_value());
}

GR_TEST(MotionPriorTest, DeskewCompensatesRotationWithLeverArm) {
  const double yaw_rate = 1.0;  // Pure rotation about the vehicle origin.
  const MotionPrior prior = CirclePrior(yaw_rate, 0.0);
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
  Points points;
  std::vector<double> times;
  for (int i = 0; i < 100; ++i) {
    points.emplace_back(10.0 * std::cos(0.06 * i), 10.0 * std::sin(0.06 * i),
                        0.5);
    times.push_back(-0.1 + 0.001 * i);
  }
  const std::optional<Points> out =
      DeskewWithPrior(points, times, 5.0, prior, vehicle_from_lidar);
  GR_ASSERT_TRUE(out.has_value());
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  double worst = 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    // lidar(end) <- lidar(tau) for a vehicle turning about its origin.
    const Pose turn(Eigen::AngleAxisd(yaw_rate * times[i], Vec3::UnitZ())
                        .toRotationMatrix(),
                    Vec3::Zero());
    const Vec3 expected =
        (lidar_from_vehicle * turn * vehicle_from_lidar) * points[i];
    worst = std::max(worst, ((*out)[i] - expected).norm());
  }
  GR_EXPECT_LT(worst, 2e-3);
}

// A rig without wheels: it turns at `yaw_rate` while moving with a constant
// world velocity (a person walking with a handheld scanner).
GR_TEST(MotionPriorTest, GyroDeltaWithAConstantWorldVelocity) {
  const double yaw_rate = 0.8;
  ImuSamples imu;
  for (int i = 0; i <= 2000; ++i) {
    imu.stamps.push_back(0.005 * i);
    imu.gyro.push_back(Vec3(0, 0, yaw_rate));
    imu.accel.push_back(Vec3(0, 0, 9.81));
  }
  const MotionPrior prior(std::move(imu), WheelSamples());
  GR_EXPECT_FALSE(prior.valid());
  GR_ASSERT_TRUE(prior.has_gyro());
  GR_EXPECT_FALSE(prior.Delta(2.0, 2.3).has_value());
  const auto world_from_vehicle = [&](double t) {
    return Eigen::AngleAxisd(yaw_rate * t, Vec3::UnitZ()).toRotationMatrix();
  };
  const Vec3 world_velocity(1.2, -0.4, 0.05);
  const double t0 = 2.0;
  const double t1 = 2.3;
  const std::optional<Pose> delta = prior.GyroDelta(
      t0, t1, world_from_vehicle(t1).transpose() * world_velocity);
  GR_ASSERT_TRUE(delta.has_value());
  const Mat3 expected_rotation =
      world_from_vehicle(t0).transpose() * world_from_vehicle(t1);
  GR_EXPECT_LT(RotationAngle(delta->rotation.transpose() * expected_rotation),
               1e-9);
  const Vec3 expected_translation =
      world_from_vehicle(t0).transpose() * world_velocity * (t1 - t0);
  GR_EXPECT_LT((delta->translation - expected_translation).norm(), 1e-9);
  GR_EXPECT_FALSE(prior.GyroDelta(9.0, 11.0, Vec3::Zero()).has_value());

  // Deskew with the gyro and the velocity: points of a static scene seen
  // during the sweep map to where they are seen at its end.
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(0.1, 0.0, 1.3));
  const double stamp = 5.0;
  const Vec3 velocity_end =
      world_from_vehicle(stamp).transpose() * world_velocity;
  Points points;
  std::vector<double> times;
  for (int i = 0; i < 100; ++i) {
    points.emplace_back(8.0 * std::cos(0.06 * i), 8.0 * std::sin(0.06 * i),
                        0.5);
    times.push_back(-0.1 + 0.001 * i);
  }
  const std::optional<Points> out = DeskewWithPrior(
      points, times, stamp, prior, vehicle_from_lidar, 16, velocity_end);
  GR_ASSERT_TRUE(out.has_value());
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  const auto world_vehicle = [&](double t) {
    return Pose(world_from_vehicle(t), world_velocity * t);
  };
  double worst = 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    const Pose end_from_tau =
        lidar_from_vehicle * world_vehicle(stamp).Inverse() *
        world_vehicle(stamp + times[i]) * vehicle_from_lidar;
    worst = std::max(worst, ((*out)[i] - end_from_tau * points[i]).norm());
  }
  GR_EXPECT_LT(worst, 2e-3);
}

GR_TEST(GravityTest, RecoversTiltAndSkipsTurns) {
  ImuSamples imu;
  WheelSamples wheel;
  // Vehicle pitched by 3 deg; it starts turning after 1 s.
  const Mat3 world_from_vehicle =
      Eigen::AngleAxisd(DegToRad(3.0), Vec3::UnitY()).toRotationMatrix();
  for (int i = 0; i <= 400; ++i) {
    const double t = 0.005 * i;
    imu.stamps.push_back(t);
    imu.gyro.push_back(Vec3(0, 0, t > 1.0 ? DegToRad(10.0) : 0.0));
    imu.accel.push_back(world_from_vehicle.transpose() * Vec3(0, 0, 9.81));
  }
  wheel.stamps = {0.0, 2.0};
  wheel.speed = {0.0, 0.0};
  const Mat3 lidar_from_vehicle =
      Eigen::AngleAxisd(0.2, Vec3::UnitZ()).toRotationMatrix();
  const std::vector<GravityMeasurement> gravity = MeasureGravity(
      imu, wheel, lidar_from_vehicle, {0.5, 1.6}, GravityConfig());
  GR_ASSERT_TRUE(gravity.size() == 1);  // The stamp while turning is rejected.
  GR_EXPECT_EQ(gravity[0].index, 0);
  const Vec3 expected =
      lidar_from_vehicle * world_from_vehicle.transpose() * Vec3::UnitZ();
  GR_EXPECT_LT((gravity[0].up - expected).norm(), 1e-9);
  GR_EXPECT_NEAR(gravity[0].sigma_rad, DegToRad(0.2), 1e-12);
}

}  // namespace
}  // namespace gr
