// Proprioception: wheel-speed + gyro dead reckoning (registration prior and
// in-sweep motion compensation, as in LIO systems) and accelerometer gravity
// measurements for the roll / pitch factors of the back-end.

#ifndef GARAGE_RECON_GR_SENSORS_IMU_H_
#define GARAGE_RECON_GR_SENSORS_IMU_H_

#include <optional>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"

namespace gr {

inline constexpr double kGravity = 9.81;  // m/s^2, along world -z.
// Standard gravity g_n (m/s^2), where a model wants the exact value.
inline constexpr double kStandardGravity = 9.80665;

// DeskewWithPrior integrates the motion at this many instants by default.
inline constexpr int kDefaultDeskewSamples = 16;

// IMU samples expressed in the vehicle frame.
struct ImuSamples {
  // Columns of imu.csv: [t, gx, gy, gz, ax, ay, az].
  static constexpr Eigen::Index kStampColumn = 0;
  static constexpr Eigen::Index kGyroColumn = kStampColumn + 1;
  static constexpr Eigen::Index kAccelColumn = kGyroColumn + kSpaceDim;
  static constexpr Eigen::Index kCsvColumns = kAccelColumn + kSpaceDim;

  std::vector<double> stamps;
  std::vector<Vec3> gyro;   // rad/s
  std::vector<Vec3> accel;  // Specific force, m/s^2.
  // The IMU origin in the vehicle frame. The accelerometer measures there:
  // where the vehicle origin is far from it (a handheld rig's lies on the
  // ground, 1.8 m below), rotation alone reads as acceleration (w x (w x r)),
  // so inertial factors use the IMU point as their body origin.
  Vec3 position = Vec3::Zero();

  size_t size() const { return stamps.size(); }
  bool empty() const { return stamps.empty(); }
  // Rows of imu.csv (IMU frame); requires all kCsvColumns columns when
  // there are rows. Stored in the vehicle axes.
  static ImuSamples FromRows(const Eigen::MatrixXd& rows,
                             const Mat3& vehicle_from_imu_rotation,
                             const Vec3& position = Vec3::Zero());
  // Vehicle axes at the IMU origin: the body frame of inertial factors.
  Pose VehicleFromBody() const { return Pose(Mat3::Identity(), position); }
};

// Linear interpolation of a vector-valued series per component, clamped at
// the ends (numpy.interp). `stamps` must be ascending, non-empty and as long
// as `values`.
Vec3 InterpolateSeries(const std::vector<double>& stamps,
                       const std::vector<Vec3>& values, double time);

// The knots that split [start_time, end_time] at every stamp strictly
// inside it: start_time, those stamps, end_time. `stamps` must be ascending.
std::vector<double> SplitAtStamps(const std::vector<double>& stamps,
                                  double start_time, double end_time);

struct WheelSamples {
  // Columns of wheel.csv: [t, speed].
  static constexpr Eigen::Index kStampColumn = 0;
  static constexpr Eigen::Index kSpeedColumn = 1;
  static constexpr Eigen::Index kCsvColumns = 2;

  std::vector<double> stamps;
  std::vector<double> speed;  // m/s along the vehicle x axis, scale applied.

  size_t size() const { return stamps.size(); }
  // Rows [t, speed] (wheel.csv); requires both columns when there are rows.
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
  // Delta and GyroDelta hold the first / last rates this far (s) beyond the
  // IMU span: the first sweep of a drive starts with the IMU.
  static constexpr double kEdgeTolerance = 0.1;

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
  // Vehicle-frame motion T_v(start_time)^-1 T_v(end_time); nullopt if the
  // interval reaches more than kEdgeTolerance beyond the IMU span.
  std::optional<Pose> Delta(double start_time, double end_time) const;
  // Like `Delta`, with the rotation from the gyro and the translation from a
  // velocity that is constant in the world, given in the vehicle frame at
  // end_time. Needs only the gyro.
  std::optional<Pose> GyroDelta(double start_time, double end_time,
                                const Vec3& velocity_at_end) const;

  const ImuSamples& imu() const { return imu_; }
  const WheelSamples& wheel() const { return wheel_; }

 private:
  // Whether [start_time, end_time] is empty, backwards or reaches more than
  // kEdgeTolerance beyond the IMU span.
  bool OutsideSpan(double start_time, double end_time) const;

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
    const MotionPrior& prior, const Pose& vehicle_from_lidar,
    int samples = kDefaultDeskewSamples,
    const std::optional<Vec3>& velocity = std::nullopt);

struct GravityConfig {
  double window = 0.25;       // s, averaging window around each stamp.
  double max_dev = 0.3;       // m/s^2; rejects unexplained acceleration.
  double max_yaw_rate = 3.0;  // deg/s; turning side-slip spoils compensation.
  double sigma_deg = 0.2;     // deg, the tilt uncertainty of a measurement.
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

#endif  // GARAGE_RECON_GR_SENSORS_IMU_H_
