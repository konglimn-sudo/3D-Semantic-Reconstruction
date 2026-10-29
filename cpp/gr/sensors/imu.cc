#include "gr/sensors/imu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

// The vector stored in `rows` from column `first` on.
Vec3 VectorAt(const Eigen::MatrixXd& rows, Eigen::Index row,
              Eigen::Index first) {
  return Vec3(rows(row, first), rows(row, first + 1), rows(row, first + 2));
}

// Deskewing needs the motion at the start and the end of the sweep at least.
constexpr int kMinDeskewSamples = 2;
// Sweeps with more points than this are deskewed by several threads.
constexpr size_t kMinPointsForThreads = 20000;

// A gravity estimate averages at least this many samples.
constexpr size_t kMinGravitySamples = 3;
// Stamps this close (s) to the edge of a window fall inside it.
constexpr double kStampTolerance = 1e-9;

}  // namespace

Vec3 InterpolateSeries(const std::vector<double>& stamps,
                       const std::vector<Vec3>& values, double time) {
  GR_CHECK(!stamps.empty() && stamps.size() == values.size());
  if (time <= stamps.front()) return values.front();
  if (time >= stamps.back()) return values.back();
  const size_t after = static_cast<size_t>(
      std::upper_bound(stamps.begin(), stamps.end(), time) - stamps.begin());
  const size_t before = after - 1;
  const double interval = stamps[after] - stamps[before];
  Vec3 out;
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    out[axis] =
        values[before][axis] + (values[after][axis] - values[before][axis]) /
                                   interval * (time - stamps[before]);
  }
  return out;
}

std::vector<double> SplitAtStamps(const std::vector<double>& stamps,
                                  double start_time, double end_time) {
  std::vector<double> knots{start_time};
  for (auto stamp = std::upper_bound(stamps.begin(), stamps.end(), start_time);
       stamp != stamps.end() && *stamp < end_time; ++stamp) {
    knots.push_back(*stamp);
  }
  knots.push_back(end_time);
  return knots;
}

ImuSamples ImuSamples::FromRows(const Eigen::MatrixXd& rows,
                                const Mat3& vehicle_from_imu_rotation,
                                const Vec3& position) {
  GR_CHECK(rows.rows() == 0 || rows.cols() >= kCsvColumns);
  ImuSamples samples;
  samples.position = position;
  const size_t count = static_cast<size_t>(rows.rows());
  samples.stamps.resize(count);
  samples.gyro.resize(count);
  samples.accel.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const Eigen::Index row = static_cast<Eigen::Index>(i);
    samples.stamps[i] = rows(row, kStampColumn);
    samples.gyro[i] =
        vehicle_from_imu_rotation * VectorAt(rows, row, kGyroColumn);
    samples.accel[i] =
        vehicle_from_imu_rotation * VectorAt(rows, row, kAccelColumn);
  }
  return samples;
}

WheelSamples WheelSamples::FromRows(const Eigen::MatrixXd& rows,
                                    double speed_scale) {
  GR_CHECK(rows.rows() == 0 || rows.cols() >= kCsvColumns);
  WheelSamples samples;
  const size_t count = static_cast<size_t>(rows.rows());
  samples.stamps.resize(count);
  samples.speed.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const Eigen::Index row = static_cast<Eigen::Index>(i);
    samples.stamps[i] = rows(row, kStampColumn);
    samples.speed[i] = rows(row, kSpeedColumn) * speed_scale;
  }
  return samples;
}

MotionPrior::MotionPrior(ImuSamples imu, WheelSamples wheel)
    : imu_(std::move(imu)), wheel_(std::move(wheel)) {
  GR_CHECK(imu_.gyro.size() == imu_.stamps.size());
  GR_CHECK(wheel_.speed.size() == wheel_.stamps.size());
  valid_ = imu_.size() > 1 && wheel_.size() > 1;
}

bool MotionPrior::OutsideSpan(double start_time, double end_time) const {
  return end_time <= start_time ||
         start_time < imu_.stamps.front() - kEdgeTolerance ||
         end_time > imu_.stamps.back() + kEdgeTolerance;
}

std::optional<Pose> MotionPrior::Delta(double start_time,
                                       double end_time) const {
  if (!valid_ || OutsideSpan(start_time, end_time)) return std::nullopt;
  // Rates are taken at the midpoints of the intervals between the knots.
  const std::vector<double> knots =
      SplitAtStamps(imu_.stamps, start_time, end_time);
  Mat3 rotation = Mat3::Identity();
  Vec3 position = Vec3::Zero();
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    const double step = knots[k + 1] - knots[k];
    const double midpoint = 0.5 * (knots[k] + knots[k + 1]);
    const Vec3 rate = InterpolateSeries(imu_.stamps, imu_.gyro, midpoint);
    const double speed = Interp1(wheel_.stamps, wheel_.speed, midpoint);
    position += rotation * Vec3(speed * step, 0.0, 0.0);
    rotation = rotation * So3Exp(rate * step);
  }
  return Pose(rotation, position);
}

std::optional<Pose> MotionPrior::GyroDelta(double start_time, double end_time,
                                           const Vec3& velocity_at_end) const {
  if (!has_gyro() || OutsideSpan(start_time, end_time)) return std::nullopt;
  const std::vector<double> knots =
      SplitAtStamps(imu_.stamps, start_time, end_time);
  Mat3 rotation = Mat3::Identity();
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    const Vec3 rate = InterpolateSeries(imu_.stamps, imu_.gyro,
                                        0.5 * (knots[k] + knots[k + 1]));
    rotation = rotation * So3Exp(rate * (knots[k + 1] - knots[k]));
  }
  // The velocity is constant in the world: in the frame at start_time it is
  // R(start)^T R(end) v_end.
  return Pose(rotation, rotation * velocity_at_end * (end_time - start_time));
}

std::optional<Points> DeskewWithPrior(const Points& points,
                                      const std::vector<double>& times,
                                      double stamp, const MotionPrior& prior,
                                      const Pose& vehicle_from_lidar,
                                      int samples,
                                      const std::optional<Vec3>& velocity) {
  if (times.size() != points.size() || times.empty()) return std::nullopt;
  const double earliest = *std::min_element(times.begin(), times.end());
  if (earliest >= 0.0) return points;
  samples = std::max(samples, kMinDeskewSamples);
  const size_t sample_count = static_cast<size_t>(samples);
  // numpy.linspace(earliest, 0, samples) and the sample indices.
  std::vector<double> sample_times(sample_count);
  std::vector<double> sample_indices(sample_count);
  const double step = (0.0 - earliest) / (samples - 1);
  for (size_t k = 0; k < sample_count; ++k) {
    sample_times[k] = static_cast<double>(k) * step + earliest;
    sample_indices[k] = static_cast<double>(k);
  }
  sample_times.back() = 0.0;
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  std::vector<Vec3> rotations(sample_count, Vec3::Zero());
  std::vector<Vec3> translations(sample_count, Vec3::Zero());
  for (size_t k = 0; k + 1 < sample_count; ++k) {
    // vehicle(tau) -> vehicle(end)
    const std::optional<Pose> delta =
        velocity ? prior.GyroDelta(stamp + sample_times[k], stamp, *velocity)
                 : prior.Delta(stamp + sample_times[k], stamp);
    if (!delta) return std::nullopt;
    // lidar(end) <- lidar(tau)
    const Pose end_from_tau =
        lidar_from_vehicle * delta->Inverse() * vehicle_from_lidar;
    rotations[k] = So3Log(end_from_tau.rotation);
    translations[k] = end_from_tau.translation;
  }
  const int last_interval = samples - 2;
  Points out(points.size());
  [[maybe_unused]] const bool use_threads =
      points.size() > kMinPointsForThreads;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (use_threads)
#endif
  for (int64_t signed_index = 0;
       signed_index < static_cast<int64_t>(points.size()); ++signed_index) {
    const size_t point_index = static_cast<size_t>(signed_index);
    // Interpolate the (small) motion between neighbouring samples.
    const double position =
        Interp1(sample_times, sample_indices, times[point_index]);
    const size_t before = static_cast<size_t>(
        std::clamp(static_cast<int>(std::floor(position)), 0, last_interval));
    const double weight = position - static_cast<double>(before);
    const Vec3 rotation =
        (1.0 - weight) * rotations[before] + weight * rotations[before + 1];
    const Vec3 translation = (1.0 - weight) * translations[before] +
                             weight * translations[before + 1];
    out[point_index] = So3Exp(rotation) * points[point_index] + translation;
  }
  return out;
}

std::vector<GravityMeasurement> MeasureGravity(
    const ImuSamples& imu, const WheelSamples& wheel,
    const Mat3& sensor_from_vehicle_rotation, const std::vector<double>& stamps,
    const GravityConfig& config) {
  std::vector<GravityMeasurement> out;
  const size_t sample_count = imu.size();
  if (sample_count < kMinGravitySamples) return out;
  std::vector<double> speed(sample_count, 0.0);
  std::vector<double> speed_rate(sample_count, 0.0);
  if (wheel.size() > 1) {
    for (size_t i = 0; i < sample_count; ++i) {
      speed[i] = Interp1(wheel.stamps, wheel.speed, imu.stamps[i]);
    }
    speed_rate = Gradient(speed, imu.stamps);
  }
  std::vector<Vec3> gravity(sample_count);
  for (size_t i = 0; i < sample_count; ++i) {
    gravity[i] = imu.accel[i] - Vec3(speed_rate[i], speed[i] * imu.gyro[i].z(),
                                     -speed[i] * imu.gyro[i].y());
  }
  for (size_t k = 0; k < stamps.size(); ++k) {
    const double stamp = stamps[k];
    Vec3 sum = Vec3::Zero();
    size_t count = 0;
    double max_yaw_rate = 0.0;
    for (auto sample_stamp =
             std::lower_bound(imu.stamps.begin(), imu.stamps.end(),
                              stamp - config.window - kStampTolerance);
         sample_stamp != imu.stamps.end() &&
         *sample_stamp <= stamp + config.window + kStampTolerance;
         ++sample_stamp) {
      if (std::abs(*sample_stamp - stamp) > config.window) continue;
      const size_t sample =
          static_cast<size_t>(sample_stamp - imu.stamps.begin());
      sum += gravity[sample];
      max_yaw_rate = std::max(max_yaw_rate, std::abs(imu.gyro[sample].z()));
      ++count;
    }
    if (count < kMinGravitySamples) continue;
    // While turning, side-slip makes the compensation unreliable.
    if (RadToDeg(max_yaw_rate) > config.max_yaw_rate) continue;
    const Vec3 mean = sum / static_cast<double>(count);
    const double norm = mean.norm();
    if (std::abs(norm - kGravity) > config.max_dev) continue;
    GravityMeasurement measurement;
    measurement.index = static_cast<int>(k);
    measurement.up = sensor_from_vehicle_rotation * (mean / norm);
    measurement.sigma_rad = DegToRad(config.sigma_deg);
    out.push_back(measurement);
  }
  return out;
}

}  // namespace gr
