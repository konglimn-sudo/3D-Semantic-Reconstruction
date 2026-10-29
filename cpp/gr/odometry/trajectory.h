// Continuous-time trajectory of one drive: world_from_vehicle(t).
//
// The knots are the optimised poses of every LiDAR frame. Between two knots
// the rotation follows the bias-corrected gyro and the position follows the
// wheel speed (along the vehicle x axis); both are corrected linearly in time
// so that the curve passes exactly through the knots. Without IMU or wheel
// data the rotation is slerped and the position is a cubic Hermite spline.
// Before the first and after the last knot the curve is extended by dead
// reckoning (gyro and wheel, else constant velocity) for
// TrajectoryOptions::extrapolation: the first sweep of a drive ends at the
// first knot, so all of it lies before it.
//
// This is the single source of poses after the back-end: LiDAR sweeps are
// deskewed with it, and every camera row takes its pose from it at the row's
// exposure time.

#ifndef GARAGE_RECON_GR_ODOMETRY_TRAJECTORY_H_
#define GARAGE_RECON_GR_ODOMETRY_TRAJECTORY_H_

#include <utility>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/sensors/imu.h"

namespace gr {

struct TrajectoryOptions {
  double table_step = 0.005;   // s between samples of the evaluation table.
  double extrapolation = 0.2;  // s beyond the first / last knot.
};

class ContinuousTrajectory {
 public:
  ContinuousTrajectory() = default;
  // `stamps` strictly increasing (at least one), `poses` world_from_vehicle.
  // `imu` / `wheel` may be empty. The gyro bias is piecewise linear between
  // `bias_stamps` (empty: zero bias).
  ContinuousTrajectory(std::vector<double> stamps, std::vector<Pose> poses,
                       const ImuSamples& imu,
                       const std::vector<double>& bias_stamps,
                       const std::vector<Vec3>& gyro_biases,
                       const WheelSamples& wheel,
                       const TrajectoryOptions& options = {});
  // Knots only: slerp and Hermite interpolation.
  ContinuousTrajectory(std::vector<double> stamps, std::vector<Pose> poses);

  ContinuousTrajectory(const ContinuousTrajectory&) = default;
  ContinuousTrajectory& operator=(const ContinuousTrajectory&) = default;
  ContinuousTrajectory(ContinuousTrajectory&&) = default;
  ContinuousTrajectory& operator=(ContinuousTrajectory&&) = default;

  bool empty() const { return knot_stamps_.empty(); }
  double start_time() const { return knot_stamps_.front(); }
  double end_time() const { return knot_stamps_.back(); }
  const std::vector<double>& knot_stamps() const { return knot_stamps_; }
  const std::vector<Pose>& knot_poses() const { return knot_poses_; }
  // Fraction of knot intervals that used the gyro / the wheel speed.
  double gyro_coverage() const { return gyro_coverage_; }
  double wheel_coverage() const { return wheel_coverage_; }

  // world_from_vehicle at `time`; extrapolated up to `extrapolation` beyond
  // [start_time(), end_time()] and clamped further out.
  Pose At(double time) const;
  std::vector<Pose> PosesAt(const std::vector<double>& times) const;

 private:
  // Extrapolate's directions.
  static constexpr int kBackward = -1;
  static constexpr int kForward = 1;

  // Appends the samples of knot interval `interval` to the table (without
  // its end) and says whether they used the gyro and the wheel speed.
  void SampleInterval(size_t interval, const ImuSamples& imu,
                      const std::vector<double>& bias_stamps,
                      const std::vector<Vec3>& gyro_biases,
                      const WheelSamples& wheel, double step, bool* used_gyro,
                      bool* used_wheel);
  void Build(const ImuSamples& imu, const std::vector<double>& bias_stamps,
             const std::vector<Vec3>& gyro_biases, const WheelSamples& wheel,
             const TrajectoryOptions& options);
  // Table samples over `duration` before the first knot (direction
  // kBackward) or after the last one (kForward), in time order.
  std::vector<std::pair<double, Pose>> Extrapolate(
      int direction, const ImuSamples& imu,
      const std::vector<double>& bias_stamps,
      const std::vector<Vec3>& gyro_biases, const WheelSamples& wheel,
      double duration, double step) const;

  std::vector<double> knot_stamps_;
  std::vector<Pose> knot_poses_;
  std::vector<Vec3> tangents_;  // Hermite velocities at the knots (world).
  std::vector<double> table_stamps_;
  std::vector<Pose> table_poses_;
  double gyro_coverage_ = 0.0;
  double wheel_coverage_ = 0.0;
};

// Moves every point of a sweep into the LiDAR frame at `stamp` (the end of
// the sweep) with the trajectory. `times` are relative to `stamp` (<= 0).
// DeskewSweep evaluates the trajectory at this many instants by default,
// and at least at kMinDeskewSamples (the start and the end of the sweep).
inline constexpr int kDefaultTrajectoryDeskewSamples = 32;
inline constexpr int kMinDeskewSamples = 2;

Points DeskewSweep(const Points& points, const std::vector<double>& times,
                   double stamp, const ContinuousTrajectory& trajectory,
                   const Pose& vehicle_from_lidar,
                   int samples = kDefaultTrajectoryDeskewSamples);

struct SweepSamples {
  Points points;
  std::vector<float> intensity;
};

// A keyframe sweep for the map: deskewed with the trajectory, cropped to
// [min_range, max_range] and voxel-downsampled (first point per voxel), like
// the sweeps OdometryPipeline keeps.
SweepSamples DeskewKeyframeSweep(const Points& points,
                                 const std::vector<float>& intensity,
                                 const std::vector<double>& times, double stamp,
                                 const ContinuousTrajectory& trajectory,
                                 const Pose& vehicle_from_lidar,
                                 double min_range, double max_range,
                                 double voxel);

}  // namespace gr

#endif  // GARAGE_RECON_GR_ODOMETRY_TRAJECTORY_H_
