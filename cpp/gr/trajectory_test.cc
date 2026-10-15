#include "gr/trajectory.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Vehicle on a horizontal circle (yaw rate w, speed v) starting at the origin
// heading +x; 200 Hz gyro and 100 Hz wheel speed over [0, 20] s.
struct CircleDrive {
  double yaw_rate = 0.5;
  double speed = 3.0;

  Pose PoseAt(double t) const {
    const double yaw = yaw_rate * t;
    const double radius = speed / yaw_rate;
    return Pose(
        So3Exp(Vec3(0.0, 0.0, yaw)),
        Vec3(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0));
  }
  ImuSamples Imu(const Vec3& gyro_bias) const {
    ImuSamples imu;
    for (int i = 0; i <= 4000; ++i) {
      imu.stamps.push_back(0.005 * i);
      imu.gyro.push_back(Vec3(0.0, 0.0, yaw_rate) + gyro_bias);
      imu.accel.push_back(Vec3(0.0, speed * yaw_rate, 9.81));
    }
    return imu;
  }
  WheelSamples Wheel() const {
    WheelSamples wheel;
    for (int i = 0; i <= 2000; ++i) {
      wheel.stamps.push_back(0.01 * i);
      wheel.speed.push_back(speed);
    }
    return wheel;
  }
  void Knots(double spacing, std::vector<double>& stamps,
             std::vector<Pose>& poses) const {
    for (double t = 1.0; t <= 19.0; t += spacing) {
      stamps.push_back(t);
      poses.push_back(PoseAt(t));
    }
  }
};

double MaxTranslationError(const ContinuousTrajectory& trajectory,
                           const CircleDrive& drive, double t0, double t1) {
  double worst = 0.0;
  for (double t = t0; t <= t1; t += 0.013) {
    worst = std::max(
        worst,
        (trajectory.At(t).translation - drive.PoseAt(t).translation).norm());
  }
  return worst;
}

GR_TEST(ContinuousTrajectoryTest, PassesThroughTheKnots) {
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(0.4, stamps, poses);
  const ContinuousTrajectory trajectory(stamps, poses);
  GR_EXPECT_NEAR(trajectory.start_time(), 1.0, 1e-12);
  for (size_t k = 0; k < stamps.size(); k += 7) {
    const Pose pose = trajectory.At(stamps[k]);
    GR_EXPECT_NEAR((pose.translation - poses[k].translation).norm(), 0.0, 1e-9);
    GR_EXPECT_NEAR(RotationAngle(pose.rotation.transpose() * poses[k].rotation),
                   0.0, 1e-9);
  }
  // Extrapolated with the knots' velocity for 0.2 s, clamped further out.
  const double speed = (poses[1].translation - poses[0].translation).norm() /
                       (stamps[1] - stamps[0]);
  GR_EXPECT_NEAR(
      (trajectory.At(-5.0).translation - poses[0].translation).norm(),
      0.2 * speed, 0.02 * speed);
  GR_EXPECT_NEAR((trajectory.At(-5.0).translation -
                  trajectory.At(stamps[0] - 0.2).translation)
                     .norm(),
                 0.0, 1e-9);
}

GR_TEST(ContinuousTrajectoryTest, GyroAndWheelFollowTheArcBetweenKnots) {
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(0.5, stamps, poses);  // 1.5 m and 14 degrees between knots.
  const ContinuousTrajectory knots_only(stamps, poses);
  const ContinuousTrajectory inertial(stamps, poses, drive.Imu(Vec3::Zero()),
                                      {}, {}, drive.Wheel());
  GR_EXPECT_NEAR(inertial.gyro_coverage(), 1.0, 1e-12);
  GR_EXPECT_NEAR(inertial.wheel_coverage(), 1.0, 1e-12);
  const double hermite_error =
      MaxTranslationError(knots_only, drive, 2.0, 18.0);
  const double inertial_error = MaxTranslationError(inertial, drive, 2.0, 18.0);
  GR_EXPECT_LT(inertial_error, 1e-3);
  GR_EXPECT_LT(inertial_error, 0.2 * hermite_error);
  // Dead reckoning beyond the first and last knot (a sweep before the first
  // LiDAR frame).
  for (double t : {stamps.front() - 0.15, stamps.back() + 0.15}) {
    GR_EXPECT_LT(
        (inertial.At(t).translation - drive.PoseAt(t).translation).norm(),
        1e-3);
  }
  // Rotation between knots follows the gyro exactly here.
  const double t = 7.23;
  GR_EXPECT_NEAR(RotationAngle(inertial.At(t).rotation.transpose() *
                               drive.PoseAt(t).rotation),
                 0.0, 1e-6);
}

GR_TEST(ContinuousTrajectoryTest, UsesTheEstimatedGyroBias) {
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(0.5, stamps, poses);
  // A roll-rate bias: its effect between knots is not linear in time (the
  // vehicle turns meanwhile), so the knot correction alone cannot remove it.
  const Vec3 bias(DegToRad(5.0), 0.0, 0.0);
  const ContinuousTrajectory corrected(
      stamps, poses, drive.Imu(bias), {0.0, 20.0}, {bias, bias}, drive.Wheel());
  const ContinuousTrajectory biased(stamps, poses, drive.Imu(bias), {}, {},
                                    drive.Wheel());
  const double t = 9.25;  // Middle of a knot interval.
  const double corrected_error = RotationAngle(
      corrected.At(t).rotation.transpose() * drive.PoseAt(t).rotation);
  const double biased_error = RotationAngle(biased.At(t).rotation.transpose() *
                                            drive.PoseAt(t).rotation);
  GR_EXPECT_LT(corrected_error, 1e-6);
  GR_EXPECT_GT(biased_error, 1e-4);
}

GR_TEST(DeskewSweepTest, MovesPointsIntoTheEndFrame) {
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(0.1, stamps, poses);
  const ContinuousTrajectory trajectory(stamps, poses, drive.Imu(Vec3::Zero()),
                                        {}, {}, drive.Wheel());
  const Pose vehicle_from_lidar(So3Exp(Vec3(0.0, 0.0, 0.3)),
                                Vec3(1.0, 0.2, 1.9));
  const double stamp = 10.0;
  Points raw;
  Points truth;
  std::vector<double> times;
  const Pose world_from_end = drive.PoseAt(stamp) * vehicle_from_lidar;
  for (int i = 0; i < 360; ++i) {
    const double tau = -0.1 + 0.1 * i / 360.0;
    const Vec3 world(5.0 * std::cos(0.0174 * i) + 9.0,
                     5.0 * std::sin(0.0174 * i), 1.0);
    const Pose world_from_lidar =
        drive.PoseAt(stamp + tau) * vehicle_from_lidar;
    raw.push_back(world_from_lidar.Inverse() * world);
    truth.push_back(world_from_end.Inverse() * world);
    times.push_back(tau);
  }
  const Points out =
      DeskewSweep(raw, times, stamp, trajectory, vehicle_from_lidar);
  double worst = 0.0;
  double raw_worst = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    worst = std::max(worst, (out[i] - truth[i]).norm());
    raw_worst = std::max(raw_worst, (raw[i] - truth[i]).norm());
  }
  GR_EXPECT_GT(raw_worst, 0.2);
  GR_EXPECT_LT(worst, 1e-3);
  const SweepSamples kept = DeskewKeyframeSweep(
      raw, std::vector<float>(raw.size(), 0.5f), times, stamp, trajectory,
      vehicle_from_lidar, 1.0, 100.0, 0.05);
  GR_EXPECT_EQ(kept.points.size(), kept.intensity.size());
  GR_EXPECT_GT(kept.points.size(), 100u);
}

}  // namespace
}  // namespace gr
