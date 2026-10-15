// Proprioception: wheel-speed + gyro dead reckoning (registration prior and
// in-sweep motion compensation, as in LIO systems) and accelerometer gravity
// measurements for the roll / pitch factors of the back-end.

#ifndef GARAGE_RECON_GR_IMU_H_
#define GARAGE_RECON_GR_IMU_H_

#include <optional>
#include <vector>

#include "gr/pose.h"
#include "gr/types.h"

namespace gr {

// IMU samples expressed in the vehicle frame.
struct ImuSamples {
  std::vector<double> stamps;
  std::vector<Vec3> gyro;   // rad/s
  std::vector<Vec3> accel;  // Specific force, m/s^2.

  size_t size() const { return stamps.size(); }
  bool empty() const { return stamps.empty(); }
  // Rows [t, gx, gy, gz, ax, ay, az] in the IMU frame (imu.csv); requires at
  // least 7 columns when there are rows.
  static ImuSamples FromRows(const Eigen::MatrixXd& rows,
                             const Mat3& vehicle_from_imu_rotation);
};

// Linear interpolation of a vector-valued series per component, clamped at
// the ends (numpy.interp). `stamps` must be ascending, non-empty and as long
// as `values`.
Vec3 InterpolateSeries(const std::vector<double>& stamps,
                       const std::vector<Vec3>& values, double t);

struct WheelSamples {
  std::vector<double> stamps;
  std::vector<double> speed;  // m/s along the vehicle x axis, scale applied.

  size_t size() const { return stamps.size(); }
  // Rows [t, speed] (wheel.csv); requires at least 2 columns when there are
  // rows.
  static WheelSamples FromRows(const Eigen::MatrixXd& rows,
                               double speed_scale = 1.0);
};

// Dead reckoning from wheel speed (vehicle x axis) and gyro rates.
// Underground there is no GNSS, and on ramps and long featureless aisles the
// LiDAR alone can degenerate, so this prior seeds every registration.
// Rigs without wheels (handheld, backpack) still get the rotation from the
// gyro (`GyroDelta`), with a velocity estimated elsewhere.
class MotionPrior {
 public:
  MotionPrior() = default;
  MotionPrior(ImuSamples imu, WheelSamples wheel);
  MotionPrior(const MotionPrior&) = default;
  MotionPrior& operator=(const MotionPrior&) = default;
  MotionPrior(MotionPrior&&) = default;
  MotionPrior& operator=(MotionPrior&&) = default;

  // Wheel speed and gyro: `Delta` is available.
  bool valid() const { return valid_; }
  // Gyro only: `GyroDelta` is available.
  bool has_gyro() const { return imu_.size() > 1; }
  // Vehicle-frame motion T_v(t0)^-1 T_v(t1); nullopt if [t0, t1] reaches
  // more than kEdgeTolerance beyond the IMU span (within it, the first /
  // last rates are held: the first sweep of a drive starts with the IMU).
  std::optional<Pose> Delta(double t0, double t1) const;
  // Like `Delta`, with the rotation from the gyro and the translation from a
  // velocity that is constant in the world, given in the vehicle frame at
  // t1. Needs only the gyro.
  std::optional<Pose> GyroDelta(double t0, double t1,
                                const Vec3& velocity_t1) const;

  static constexpr double kEdgeTolerance = 0.1;  // s

  const ImuSamples& imu() const { return imu_; }
  const WheelSamples& wheel() const { return wheel_; }

 private:
  ImuSamples imu_;
  WheelSamples wheel_;
  bool valid_ = false;
};

// Motion compensation of one sweep with the prior: the sensor motion within
// the sweep is integrated at `samples` instants and interpolated per point.
// `times` are seconds relative to `stamp` (<= 0, the stamp ends the sweep).
// With `velocity` (vehicle frame at the end of the sweep) the motion comes
// from `GyroDelta` instead of the wheel speed. Returns nullopt when the prior
// does not cover the sweep.
std::optional<Points> DeskewWithPrior(
    const Points& points, const std::vector<double>& times, double stamp,
    const MotionPrior& prior, const Pose& vehicle_from_lidar, int samples = 16,
    const std::optional<Vec3>& velocity = std::nullopt);

struct GravityConfig {
  double window = 0.25;       // s, averaging window around each stamp.
  double max_dev = 0.3;       // m/s^2; rejects unexplained acceleration.
  double max_yaw_rate = 3.0;  // deg/s; turning side-slip spoils compensation.
  double sigma_deg = 0.2;
};

struct GravityMeasurement {
  int index = 0;            // Into the queried stamps (or keyframes).
  Vec3 up = Vec3::UnitZ();  // World +z expressed in the sensor frame.
  double sigma_rad = 0.0;
};

// World +z in the sensor frame around each stamp: the specific force minus
// the motion-induced acceleration (forward: wheel-speed derivative; lateral /
// vertical: speed x yaw / pitch rate) is gravity. This makes roll and pitch
// observable on every level, which keeps levels connected only by ramps from
// tilting against each other.
std::vector<GravityMeasurement> MeasureGravity(
    const ImuSamples& imu, const WheelSamples& wheel,
    const Mat3& sensor_from_vehicle_rotation, const std::vector<double>& stamps,
    const GravityConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IMU_H_
