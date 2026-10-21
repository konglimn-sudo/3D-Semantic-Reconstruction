#include "gr/imu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gr/check.h"
#include "gr/numeric.h"

namespace gr {

Vec3 InterpolateSeries(const std::vector<double>& stamps,
                       const std::vector<Vec3>& values, double t) {
  GR_CHECK(!stamps.empty() && stamps.size() == values.size());
  if (t <= stamps.front()) return values.front();
  if (t >= stamps.back()) return values.back();
  const size_t j = static_cast<size_t>(
      std::upper_bound(stamps.begin(), stamps.end(), t) - stamps.begin());
  const double dx = stamps[j] - stamps[j - 1];
  Vec3 out;
  for (int a = 0; a < 3; ++a) {
    out[a] = values[j - 1][a] +
             (values[j][a] - values[j - 1][a]) / dx * (t - stamps[j - 1]);
  }
  return out;
}

ImuSamples ImuSamples::FromRows(const Eigen::MatrixXd& rows,
                                const Mat3& vehicle_from_imu_rotation,
                                const Vec3& position) {
  GR_CHECK(rows.rows() == 0 || rows.cols() >= 7);
  ImuSamples samples;
  samples.position = position;
  const size_t n = static_cast<size_t>(rows.rows());
  samples.stamps.resize(n);
  samples.gyro.resize(n);
  samples.accel.resize(n);
  for (size_t i = 0; i < n; ++i) {
    const Eigen::Index r = static_cast<Eigen::Index>(i);
    samples.stamps[i] = rows(r, 0);
    samples.gyro[i] =
        vehicle_from_imu_rotation * Vec3(rows(r, 1), rows(r, 2), rows(r, 3));
    samples.accel[i] =
        vehicle_from_imu_rotation * Vec3(rows(r, 4), rows(r, 5), rows(r, 6));
  }
  return samples;
}

WheelSamples WheelSamples::FromRows(const Eigen::MatrixXd& rows,
                                    double speed_scale) {
  GR_CHECK(rows.rows() == 0 || rows.cols() >= 2);
  WheelSamples samples;
  const size_t n = static_cast<size_t>(rows.rows());
  samples.stamps.resize(n);
  samples.speed.resize(n);
  for (size_t i = 0; i < n; ++i) {
    samples.stamps[i] = rows(static_cast<Eigen::Index>(i), 0);
    samples.speed[i] = rows(static_cast<Eigen::Index>(i), 1) * speed_scale;
  }
  return samples;
}

MotionPrior::MotionPrior(ImuSamples imu, WheelSamples wheel)
    : imu_(std::move(imu)), wheel_(std::move(wheel)) {
  GR_CHECK(imu_.gyro.size() == imu_.stamps.size());
  GR_CHECK(wheel_.speed.size() == wheel_.stamps.size());
  valid_ = imu_.size() > 1 && wheel_.size() > 1;
}

std::optional<Pose> MotionPrior::Delta(double t0, double t1) const {
  if (!valid_ || t1 <= t0 || t0 < imu_.stamps.front() - kEdgeTolerance ||
      t1 > imu_.stamps.back() + kEdgeTolerance) {
    return std::nullopt;
  }
  // Knots: t0, the IMU stamps strictly inside (t0, t1), t1. Rates are taken
  // at the interval midpoints.
  std::vector<double> knots{t0};
  for (auto it = std::upper_bound(imu_.stamps.begin(), imu_.stamps.end(), t0);
       it != imu_.stamps.end() && *it < t1; ++it) {
    knots.push_back(*it);
  }
  knots.push_back(t1);
  Mat3 rotation = Mat3::Identity();
  Vec3 position = Vec3::Zero();
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    const double dt = knots[k + 1] - knots[k];
    const double mid = 0.5 * (knots[k] + knots[k + 1]);
    const Vec3 rate = InterpolateSeries(imu_.stamps, imu_.gyro, mid);
    const double speed = Interp1(wheel_.stamps, wheel_.speed, mid);
    position += rotation * Vec3(speed * dt, 0.0, 0.0);
    rotation = rotation * So3Exp(rate * dt);
  }
  return Pose(rotation, position);
}

std::optional<Pose> MotionPrior::GyroDelta(double t0, double t1,
                                           const Vec3& velocity_t1) const {
  if (!has_gyro() || t1 <= t0 || t0 < imu_.stamps.front() - kEdgeTolerance ||
      t1 > imu_.stamps.back() + kEdgeTolerance) {
    return std::nullopt;
  }
  std::vector<double> knots{t0};
  for (auto it = std::upper_bound(imu_.stamps.begin(), imu_.stamps.end(), t0);
       it != imu_.stamps.end() && *it < t1; ++it) {
    knots.push_back(*it);
  }
  knots.push_back(t1);
  Mat3 rotation = Mat3::Identity();
  for (size_t k = 0; k + 1 < knots.size(); ++k) {
    const Vec3 rate = InterpolateSeries(imu_.stamps, imu_.gyro,
                                        0.5 * (knots[k] + knots[k + 1]));
    rotation = rotation * So3Exp(rate * (knots[k + 1] - knots[k]));
  }
  // The velocity is constant in the world: in the frame at t0 it is
  // R(t0)^T R(t1) v_t1.
  return Pose(rotation, rotation * velocity_t1 * (t1 - t0));
}

std::optional<Points> DeskewWithPrior(const Points& points,
                                      const std::vector<double>& times,
                                      double stamp, const MotionPrior& prior,
                                      const Pose& vehicle_from_lidar,
                                      int samples,
                                      const std::optional<Vec3>& velocity) {
  if (times.size() != points.size() || times.empty()) return std::nullopt;
  const double t_min = *std::min_element(times.begin(), times.end());
  if (t_min >= 0.0) return points;
  samples = std::max(samples, 2);
  const size_t n_samples = static_cast<size_t>(samples);
  // numpy.linspace(t_min, 0, samples) and the sample indices.
  std::vector<double> taus(n_samples);
  std::vector<double> knots(n_samples);
  const double step = (0.0 - t_min) / (samples - 1);
  for (size_t k = 0; k < n_samples; ++k) {
    taus[k] = static_cast<double>(k) * step + t_min;
    knots[k] = static_cast<double>(k);
  }
  taus.back() = 0.0;
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  std::vector<Vec3> rotations(n_samples, Vec3::Zero());
  std::vector<Vec3> translations(n_samples, Vec3::Zero());
  for (size_t k = 0; k + 1 < n_samples; ++k) {
    // vehicle(tau) -> vehicle(end)
    const std::optional<Pose> delta =
        velocity ? prior.GyroDelta(stamp + taus[k], stamp, *velocity)
                 : prior.Delta(stamp + taus[k], stamp);
    if (!delta) return std::nullopt;
    // lidar(end) <- lidar(tau)
    const Pose end_from_tau =
        lidar_from_vehicle * delta->Inverse() * vehicle_from_lidar;
    rotations[k] = So3Log(end_from_tau.rotation);
    translations[k] = end_from_tau.translation;
  }
  Points out(points.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (points.size() > 20000)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(points.size()); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    // Interpolate the (small) motion between neighbouring samples.
    const double position = Interp1(taus, knots, times[i]);
    const size_t i0 = static_cast<size_t>(
        std::clamp(static_cast<int>(std::floor(position)), 0, samples - 2));
    const double w = position - static_cast<double>(i0);
    const Vec3 r = (1.0 - w) * rotations[i0] + w * rotations[i0 + 1];
    const Vec3 t = (1.0 - w) * translations[i0] + w * translations[i0 + 1];
    out[i] = So3Exp(r) * points[i] + t;
  }
  return out;
}

std::vector<GravityMeasurement> MeasureGravity(
    const ImuSamples& imu, const WheelSamples& wheel,
    const Mat3& sensor_from_vehicle_rotation, const std::vector<double>& stamps,
    const GravityConfig& config) {
  std::vector<GravityMeasurement> out;
  const size_t n = imu.size();
  if (n < 3) return out;
  std::vector<double> speed(n, 0.0);
  std::vector<double> speed_rate(n, 0.0);
  if (wheel.size() > 1) {
    for (size_t i = 0; i < n; ++i) {
      speed[i] = Interp1(wheel.stamps, wheel.speed, imu.stamps[i]);
    }
    speed_rate = Gradient(speed, imu.stamps);
  }
  std::vector<Vec3> gravity(n);
  for (size_t i = 0; i < n; ++i) {
    gravity[i] = imu.accel[i] - Vec3(speed_rate[i], speed[i] * imu.gyro[i].z(),
                                     -speed[i] * imu.gyro[i].y());
  }
  constexpr double kEps = 1e-9;
  for (size_t k = 0; k < stamps.size(); ++k) {
    const double s = stamps[k];
    Vec3 sum = Vec3::Zero();
    int count = 0;
    double max_rate = 0.0;
    for (auto it = std::lower_bound(imu.stamps.begin(), imu.stamps.end(),
                                    s - config.window - kEps);
         it != imu.stamps.end() && *it <= s + config.window + kEps; ++it) {
      if (std::abs(*it - s) > config.window) continue;
      const size_t i = static_cast<size_t>(it - imu.stamps.begin());
      sum += gravity[i];
      max_rate = std::max(max_rate, std::abs(imu.gyro[i].z()));
      ++count;
    }
    if (count < 3) continue;
    // While turning, side-slip makes the compensation unreliable.
    if (RadToDeg(max_rate) > config.max_yaw_rate) continue;
    const Vec3 mean = sum / count;
    const double norm = mean.norm();
    if (std::abs(norm - 9.81) > config.max_dev) continue;
    GravityMeasurement measurement;
    measurement.index = static_cast<int>(k);
    measurement.up = sensor_from_vehicle_rotation * (mean / norm);
    measurement.sigma_rad = DegToRad(config.sigma_deg);
    out.push_back(measurement);
  }
  return out;
}

}  // namespace gr
