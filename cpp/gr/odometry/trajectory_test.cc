#include "gr/odometry/trajectory.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Errors the trajectory reproduces exactly, and what dead reckoning at
// 200 Hz leaves.
constexpr double kExactTolerance = 1e-12;
constexpr double kKnotTolerance = 1e-9;
constexpr double kDeadReckoningTolerance = 1e-3;  // m
constexpr double kGyroRotationTolerance = 1e-6;   // rad

// Vehicle on a horizontal circle (yaw rate w, speed v) starting at the origin
// heading +x; 200 Hz gyro and 100 Hz wheel speed over [0, 20] s.
struct CircleDrive {
  static constexpr int kImuSamples = 4000;
  static constexpr double kImuPeriod = 0.005;  // s
  static constexpr int kWheelSamples = 2000;
  static constexpr double kWheelPeriod = 0.01;  // s
  // Knots from 1 s to 19 s: the drive extends 1 s beyond them.
  static constexpr double kFirstKnot = 1.0;
  static constexpr double kLastKnot = 19.0;

  double yaw_rate = 0.5;  // rad/s
  double speed = 3.0;     // m/s

  Pose PoseAt(double time) const {
    const double yaw = yaw_rate * time;
    const double radius = speed / yaw_rate;
    return Pose(
        So3Exp(Vec3(0.0, 0.0, yaw)),
        Vec3(radius * std::sin(yaw), radius * (1.0 - std::cos(yaw)), 0.0));
  }
  ImuSamples Imu(const Vec3& gyro_bias) const {
    ImuSamples imu;
    for (int i = 0; i <= kImuSamples; ++i) {
      imu.stamps.push_back(kImuPeriod * i);
      imu.gyro.push_back(Vec3(0.0, 0.0, yaw_rate) + gyro_bias);
      imu.accel.push_back(Vec3(0.0, speed * yaw_rate, kGravity));
    }
    return imu;
  }
  WheelSamples Wheel() const {
    WheelSamples wheel;
    for (int i = 0; i <= kWheelSamples; ++i) {
      wheel.stamps.push_back(kWheelPeriod * i);
      wheel.speed.push_back(speed);
    }
    return wheel;
  }
  void Knots(double spacing, std::vector<double>* stamps,
             std::vector<Pose>* poses) const {
    for (double time = kFirstKnot; time <= kLastKnot; time += spacing) {
      stamps->push_back(time);
      poses->push_back(PoseAt(time));
    }
  }
};

double MaxTranslationError(const ContinuousTrajectory& trajectory,
                           const CircleDrive& drive, double start_time,
                           double end_time) {
  // Off the knot and table grids.
  constexpr double kProbeStep = 0.013;  // s
  double worst = 0.0;
  for (double time = start_time; time <= end_time; time += kProbeStep) {
    worst = std::max(worst, (trajectory.At(time).translation -
                             drive.PoseAt(time).translation)
                                .norm());
  }
  return worst;
}

GR_TEST(ContinuousTrajectoryTest, PassesThroughTheKnots) {
  constexpr double kKnotSpacing = 0.4;  // s
  constexpr size_t kKnotStride = 7;     // Knots checked.
  // TrajectoryOptions::extrapolation, and how well constant velocity
  // extrapolates the arc over it (a fraction of the distance).
  constexpr double kExtrapolation = 0.2;  // s
  constexpr double kExtrapolationTolerance = 0.02;
  constexpr double kFarBefore = -5.0;  // s, far before the first knot.
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(kKnotSpacing, &stamps, &poses);
  const ContinuousTrajectory trajectory(stamps, poses);
  GR_EXPECT_NEAR(trajectory.start_time(), CircleDrive::kFirstKnot,
                 kExactTolerance);
  for (size_t k = 0; k < stamps.size(); k += kKnotStride) {
    const Pose pose = trajectory.At(stamps[k]);
    GR_EXPECT_NEAR((pose.translation - poses[k].translation).norm(), 0.0,
                   kKnotTolerance);
    GR_EXPECT_NEAR(RotationAngle(pose.rotation.transpose() * poses[k].rotation),
                   0.0, kKnotTolerance);
  }
  // Extrapolated with the knots' velocity for 0.2 s, clamped further out.
  const double speed = (poses[1].translation - poses[0].translation).norm() /
                       (stamps[1] - stamps[0]);
  GR_EXPECT_NEAR(
      (trajectory.At(kFarBefore).translation - poses[0].translation).norm(),
      kExtrapolation * speed, kExtrapolationTolerance * speed);
  GR_EXPECT_NEAR((trajectory.At(kFarBefore).translation -
                  trajectory.At(stamps[0] - kExtrapolation).translation)
                     .norm(),
                 0.0, kKnotTolerance);
}

GR_TEST(ContinuousTrajectoryTest, GyroAndWheelFollowTheArcBetweenKnots) {
  // 1.5 m and 14 degrees between knots.
  constexpr double kKnotSpacing = 0.5;  // s
  constexpr double kProbeStart = 2.0;   // s
  constexpr double kProbeEnd = 18.0;
  // The inertial curve beats the Hermite spline by at least 5x.
  constexpr double kMaxErrorRatio = 0.2;
  // A sweep before the first LiDAR frame (and after the last).
  constexpr double kBeyondKnots = 0.15;   // s
  constexpr double kBetweenKnots = 7.23;  // s
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(kKnotSpacing, &stamps, &poses);
  const ContinuousTrajectory knots_only(stamps, poses);
  const ContinuousTrajectory inertial(stamps, poses, drive.Imu(Vec3::Zero()),
                                      {}, {}, drive.Wheel());
  GR_EXPECT_NEAR(inertial.gyro_coverage(), 1.0, kExactTolerance);
  GR_EXPECT_NEAR(inertial.wheel_coverage(), 1.0, kExactTolerance);
  const double hermite_error =
      MaxTranslationError(knots_only, drive, kProbeStart, kProbeEnd);
  const double inertial_error =
      MaxTranslationError(inertial, drive, kProbeStart, kProbeEnd);
  GR_EXPECT_LT(inertial_error, kDeadReckoningTolerance);
  GR_EXPECT_LT(inertial_error, kMaxErrorRatio * hermite_error);
  for (double time :
       {stamps.front() - kBeyondKnots, stamps.back() + kBeyondKnots}) {
    GR_EXPECT_LT(
        (inertial.At(time).translation - drive.PoseAt(time).translation).norm(),
        kDeadReckoningTolerance);
  }
  // Rotation between knots follows the gyro exactly here.
  GR_EXPECT_NEAR(RotationAngle(inertial.At(kBetweenKnots).rotation.transpose() *
                               drive.PoseAt(kBetweenKnots).rotation),
                 0.0, kGyroRotationTolerance);
}

GR_TEST(ContinuousTrajectoryTest, UsesTheEstimatedGyroBias) {
  constexpr double kKnotSpacing = 0.5;        // s
  constexpr double kRollBiasDegrees = 5.0;    // deg/s
  constexpr double kMiddleOfInterval = 9.25;  // s
  // Without the bias the rotation is off by at least this (rad).
  constexpr double kMinBiasedError = 1e-4;
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(kKnotSpacing, &stamps, &poses);
  // A roll-rate bias: its effect between knots is not linear in time (the
  // vehicle turns meanwhile), so the knot correction alone cannot remove it.
  const Vec3 bias(DegToRad(kRollBiasDegrees), 0.0, 0.0);
  const ContinuousTrajectory corrected(
      stamps, poses, drive.Imu(bias), {0.0, 20.0}, {bias, bias}, drive.Wheel());
  const ContinuousTrajectory biased(stamps, poses, drive.Imu(bias), {}, {},
                                    drive.Wheel());
  const double corrected_error =
      RotationAngle(corrected.At(kMiddleOfInterval).rotation.transpose() *
                    drive.PoseAt(kMiddleOfInterval).rotation);
  const double biased_error =
      RotationAngle(biased.At(kMiddleOfInterval).rotation.transpose() *
                    drive.PoseAt(kMiddleOfInterval).rotation);
  GR_EXPECT_LT(corrected_error, kGyroRotationTolerance);
  GR_EXPECT_GT(biased_error, kMinBiasedError);
}

GR_TEST(DeskewSweepTest, MovesPointsIntoTheEndFrame) {
  constexpr double kKnotSpacing = 0.1;  // s
  constexpr double kStamp = 10.0;       // s
  // 360 points over the 0.1 s before the stamp, on a 5 m circle around
  // (9, 0, 1), a degree apart.
  constexpr int kPoints = 360;
  constexpr double kSweepDuration = 0.1;   // s
  constexpr double kCircleRadius = 5.0;    // m
  constexpr double kCircleCenterX = 9.0;   // m
  constexpr double kCircleHeight = 1.0;    // m
  constexpr double kAzimuthStep = 0.0174;  // rad
  // The vehicle moves 0.3 m during the sweep: the raw points are far off.
  constexpr double kMinRawError = 0.2;  // m
  // DeskewKeyframeSweep: ranges and voxel (m), and points it keeps.
  constexpr double kMinRange = 1.0;
  constexpr double kMaxRange = 100.0;
  constexpr double kVoxel = 0.05;
  constexpr size_t kMinKept = 100;
  constexpr float kIntensity = 0.5f;
  const CircleDrive drive;
  std::vector<double> stamps;
  std::vector<Pose> poses;
  drive.Knots(kKnotSpacing, &stamps, &poses);
  const ContinuousTrajectory trajectory(stamps, poses, drive.Imu(Vec3::Zero()),
                                        {}, {}, drive.Wheel());
  const Pose vehicle_from_lidar(So3Exp(Vec3(0.0, 0.0, 0.3)),
                                Vec3(1.0, 0.2, 1.9));
  Points raw;
  Points truth;
  std::vector<double> times;
  const Pose world_from_end = drive.PoseAt(kStamp) * vehicle_from_lidar;
  for (int i = 0; i < kPoints; ++i) {
    const double tau =
        -kSweepDuration + kSweepDuration * i / static_cast<double>(kPoints);
    const Vec3 world(
        kCircleRadius * std::cos(kAzimuthStep * i) + kCircleCenterX,
        kCircleRadius * std::sin(kAzimuthStep * i), kCircleHeight);
    const Pose world_from_lidar =
        drive.PoseAt(kStamp + tau) * vehicle_from_lidar;
    raw.push_back(world_from_lidar.Inverse() * world);
    truth.push_back(world_from_end.Inverse() * world);
    times.push_back(tau);
  }
  const Points out =
      DeskewSweep(raw, times, kStamp, trajectory, vehicle_from_lidar);
  double worst = 0.0;
  double raw_worst = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    worst = std::max(worst, (out[i] - truth[i]).norm());
    raw_worst = std::max(raw_worst, (raw[i] - truth[i]).norm());
  }
  GR_EXPECT_GT(raw_worst, kMinRawError);
  GR_EXPECT_LT(worst, kDeadReckoningTolerance);
  const SweepSamples kept = DeskewKeyframeSweep(
      raw, std::vector<float>(raw.size(), kIntensity), times, kStamp,
      trajectory, vehicle_from_lidar, kMinRange, kMaxRange, kVoxel);
  GR_EXPECT_EQ(kept.points.size(), kept.intensity.size());
  GR_EXPECT_GT(kept.points.size(), kMinKept);
}

}  // namespace
}  // namespace gr
