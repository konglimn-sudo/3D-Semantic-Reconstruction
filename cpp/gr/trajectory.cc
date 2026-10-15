#include "gr/trajectory.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gr/check.h"
#include "gr/numeric.h"
#include "gr/voxel.h"

namespace gr {
namespace {

// Gyro bias at `t`: piecewise linear between the bias knots, zero without.
Vec3 GyroBiasAt(const std::vector<double>& stamps,
                const std::vector<Vec3>& biases, double t) {
  if (stamps.empty()) return Vec3::Zero();
  return InterpolateSeries(stamps, biases, t);
}

bool Covers(const std::vector<double>& stamps, double t0, double t1) {
  return stamps.size() > 1 && stamps.front() <= t0 && stamps.back() >= t1;
}

// Cubic Hermite spline on one interval, s in [0, 1].
Vec3 Hermite(const Vec3& p0, const Vec3& v0, const Vec3& p1, const Vec3& v1,
             double duration, double s) {
  const double s2 = s * s;
  const double s3 = s2 * s;
  return (2.0 * s3 - 3.0 * s2 + 1.0) * p0 +
         (s3 - 2.0 * s2 + s) * duration * v0 + (-2.0 * s3 + 3.0 * s2) * p1 +
         (s3 - s2) * duration * v1;
}

}  // namespace

ContinuousTrajectory::ContinuousTrajectory(
    std::vector<double> stamps, std::vector<Pose> poses, const ImuSamples& imu,
    const std::vector<double>& bias_stamps,
    const std::vector<Vec3>& gyro_biases, const WheelSamples& wheel,
    const TrajectoryOptions& options)
    : knot_stamps_(std::move(stamps)), knot_poses_(std::move(poses)) {
  Build(imu, bias_stamps, gyro_biases, wheel, options);
}

ContinuousTrajectory::ContinuousTrajectory(std::vector<double> stamps,
                                           std::vector<Pose> poses)
    : knot_stamps_(std::move(stamps)), knot_poses_(std::move(poses)) {
  Build(ImuSamples(), {}, {}, WheelSamples(), TrajectoryOptions());
}

void ContinuousTrajectory::Build(const ImuSamples& imu,
                                 const std::vector<double>& bias_stamps,
                                 const std::vector<Vec3>& gyro_biases,
                                 const WheelSamples& wheel,
                                 const TrajectoryOptions& options) {
  GR_CHECK(!knot_stamps_.empty() && knot_stamps_.size() == knot_poses_.size());
  GR_CHECK(bias_stamps.size() == gyro_biases.size());
  GR_CHECK(imu.gyro.size() == imu.stamps.size());
  GR_CHECK(wheel.speed.size() == wheel.stamps.size());
  GR_CHECK(options.table_step > 0.0);
  const size_t n = knot_stamps_.size();
  for (size_t k = 1; k < n; ++k) {
    GR_CHECK(knot_stamps_[k] > knot_stamps_[k - 1]);
  }
  tangents_.assign(n, Vec3::Zero());
  for (size_t k = 0; k < n && n > 1; ++k) {
    const size_t a = k == 0 ? 0 : k - 1;
    const size_t b = k + 1 == n ? n - 1 : k + 1;
    tangents_[k] = (knot_poses_[b].translation - knot_poses_[a].translation) /
                   (knot_stamps_[b] - knot_stamps_[a]);
  }
  table_stamps_.clear();
  table_poses_.clear();
  for (const auto& [t, pose] :
       Extrapolate(-1, imu, bias_stamps, gyro_biases, wheel,
                   options.extrapolation, options.table_step)) {
    table_stamps_.push_back(t);
    table_poses_.push_back(pose);
  }
  int gyro_intervals = 0;
  int wheel_intervals = 0;
  for (size_t k = 0; k + 1 < n; ++k) {
    bool used_gyro = false;
    bool used_wheel = false;
    SampleInterval(k, imu, bias_stamps, gyro_biases, wheel, options.table_step,
                   used_gyro, used_wheel);
    gyro_intervals += used_gyro ? 1 : 0;
    wheel_intervals += used_wheel ? 1 : 0;
  }
  table_stamps_.push_back(knot_stamps_.back());
  table_poses_.push_back(knot_poses_.back());
  for (const auto& [t, pose] :
       Extrapolate(+1, imu, bias_stamps, gyro_biases, wheel,
                   options.extrapolation, options.table_step)) {
    table_stamps_.push_back(t);
    table_poses_.push_back(pose);
  }
  if (n > 1) {
    gyro_coverage_ = static_cast<double>(gyro_intervals) / (n - 1);
    wheel_coverage_ = static_cast<double>(wheel_intervals) / (n - 1);
  }
}

std::vector<std::pair<double, Pose>> ContinuousTrajectory::Extrapolate(
    int direction, const ImuSamples& imu,
    const std::vector<double>& bias_stamps,
    const std::vector<Vec3>& gyro_biases, const WheelSamples& wheel,
    double duration, double step) const {
  std::vector<std::pair<double, Pose>> out;
  if (duration <= 0.0) return out;
  const size_t n = knot_stamps_.size();
  const size_t end = direction < 0 ? 0 : n - 1;
  const double t_end = knot_stamps_[end];
  const Pose& pose_end = knot_poses_[end];
  // Constant-velocity fallback from the two outermost knots.
  Vec3 rate = Vec3::Zero();  // Body frame.
  Vec3 velocity = tangents_[end];
  if (n > 1) {
    const size_t inner = direction < 0 ? 1 : n - 2;
    const size_t a = std::min(end, inner);
    const size_t b = std::max(end, inner);
    rate =
        So3Log(knot_poses_[a].rotation.transpose() * knot_poses_[b].rotation) /
        (knot_stamps_[b] - knot_stamps_[a]);
  }
  const double t_far = t_end + direction * duration;
  const double lo = std::min(t_end, t_far);
  const double hi = std::max(t_end, t_far);
  const bool gyro = imu.stamps.size() > 1 &&
                    imu.stamps.front() <= lo + MotionPrior::kEdgeTolerance &&
                    imu.stamps.back() >= hi - MotionPrior::kEdgeTolerance;
  const bool use_wheel =
      gyro && wheel.stamps.size() > 1 &&
      wheel.stamps.front() <= lo + MotionPrior::kEdgeTolerance &&
      wheel.stamps.back() >= hi - MotionPrior::kEdgeTolerance;
  // Integrate outwards from the knot in steps of `step`.
  const int num_steps =
      std::max(1, static_cast<int>(std::ceil(duration / step)));
  const double dt = duration / num_steps;
  Pose pose = pose_end;
  for (int m = 1; m <= num_steps; ++m) {
    const double t_prev = t_end + direction * (m - 1) * dt;
    const double mid = t_prev + direction * 0.5 * dt;
    const Vec3 body_rate =
        gyro ? Vec3(InterpolateSeries(imu.stamps, imu.gyro, mid) -
                    GyroBiasAt(bias_stamps, gyro_biases, mid))
             : rate;
    const Mat3 half = So3Exp(body_rate * (direction * 0.5 * dt));
    if (use_wheel) {
      const double speed = Interp1(wheel.stamps, wheel.speed, mid);
      pose.translation +=
          pose.rotation * half * Vec3(direction * speed * dt, 0.0, 0.0);
    } else {
      pose.translation += direction * dt * velocity;
    }
    pose.rotation = pose.rotation * half * half;
    pose.Normalize();
    out.emplace_back(t_end + direction * m * dt, pose);
  }
  if (direction < 0) std::reverse(out.begin(), out.end());
  return out;
}

void ContinuousTrajectory::SampleInterval(
    size_t k, const ImuSamples& imu, const std::vector<double>& bias_stamps,
    const std::vector<Vec3>& gyro_biases, const WheelSamples& wheel,
    double step, bool& used_gyro, bool& used_wheel) {
  const double t0 = knot_stamps_[k];
  const double t1 = knot_stamps_[k + 1];
  const Pose& pose0 = knot_poses_[k];
  const Pose& pose1 = knot_poses_[k + 1];
  const double duration = t1 - t0;
  const int num_samples =
      std::max(1, static_cast<int>(std::ceil(duration / step)));
  std::vector<double> sample_times(static_cast<size_t>(num_samples));
  for (int m = 0; m < num_samples; ++m) {
    sample_times[static_cast<size_t>(m)] = t0 + duration * m / num_samples;
  }
  used_gyro = Covers(imu.stamps, t0, t1);
  used_wheel = used_gyro && Covers(wheel.stamps, t0, t1);
  const auto hermite = [&](double s) -> Vec3 {
    return Hermite(pose0.translation, tangents_[k], pose1.translation,
                   tangents_[k + 1], duration, s);
  };
  if (!used_gyro) {
    for (double t : sample_times) {
      const double s = (t - t0) / duration;
      Pose pose = Interpolate(pose0, pose1, s);
      pose.translation = hermite(s);
      table_stamps_.push_back(t);
      table_poses_.push_back(pose);
    }
    return;
  }

  // Integration pieces end at every output sample and IMU stamp.
  std::vector<double> knots = sample_times;
  for (auto it = std::upper_bound(imu.stamps.begin(), imu.stamps.end(), t0);
       it != imu.stamps.end() && *it < t1; ++it) {
    knots.push_back(*it);
  }
  knots.push_back(t1);
  std::sort(knots.begin(), knots.end());
  knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
  // Explicit return type: an Eigen expression would outlive its operands.
  const auto rate_at = [&](double t) -> Vec3 {
    return InterpolateSeries(imu.stamps, imu.gyro, t) -
           GyroBiasAt(bias_stamps, gyro_biases, t);
  };

  // Pass 1: the gyro rotation over the interval and its mismatch with the
  // end knot, distributed linearly in time.
  Mat3 gyro_rotation = Mat3::Identity();
  for (size_t p = 0; p + 1 < knots.size(); ++p) {
    const double dt = knots[p + 1] - knots[p];
    gyro_rotation = gyro_rotation * So3Exp(rate_at(knots[p] + 0.5 * dt) * dt);
  }
  const Vec3 rotation_correction = So3Log(
      gyro_rotation.transpose() * pose0.rotation.transpose() * pose1.rotation);

  // Pass 2: corrected rotations at the samples; wheel dead reckoning.
  gyro_rotation = Mat3::Identity();
  Vec3 position = pose0.translation;
  const size_t first = table_stamps_.size();
  size_t next_sample = 0;
  for (size_t p = 0; p < knots.size(); ++p) {
    const double t = knots[p];
    if (next_sample < sample_times.size() && t == sample_times[next_sample]) {
      const double s = (t - t0) / duration;
      table_stamps_.push_back(t);
      table_poses_.emplace_back(
          pose0.rotation * gyro_rotation * So3Exp(s * rotation_correction),
          position);
      ++next_sample;
    }
    if (p + 1 == knots.size()) break;
    const double dt = knots[p + 1] - t;
    const double mid = t + 0.5 * dt;
    const Vec3 rate = rate_at(mid);
    if (used_wheel) {
      const Mat3 rotation_mid =
          pose0.rotation * gyro_rotation * So3Exp(rate * (0.5 * dt)) *
          So3Exp(((mid - t0) / duration) * rotation_correction);
      position += rotation_mid *
                  Vec3(Interp1(wheel.stamps, wheel.speed, mid) * dt, 0.0, 0.0);
    }
    gyro_rotation = gyro_rotation * So3Exp(rate * dt);
  }

  // Pass 3: positions, corrected to end exactly at the next knot.
  const Vec3 position_correction = pose1.translation - position;
  for (size_t m = first; m < table_stamps_.size(); ++m) {
    const double s = (table_stamps_[m] - t0) / duration;
    table_poses_[m].Normalize();
    if (used_wheel) {
      table_poses_[m].translation += s * position_correction;
    } else {
      table_poses_[m].translation = hermite(s);
    }
  }
}

Pose ContinuousTrajectory::At(double t) const {
  GR_CHECK(!empty() && !table_stamps_.empty());
  if (t <= table_stamps_.front()) return table_poses_.front();
  if (t >= table_stamps_.back()) return table_poses_.back();
  const size_t j = static_cast<size_t>(
      std::upper_bound(table_stamps_.begin(), table_stamps_.end(), t) -
      table_stamps_.begin());
  const double dt = table_stamps_[j] - table_stamps_[j - 1];
  return Interpolate(table_poses_[j - 1], table_poses_[j],
                     (t - table_stamps_[j - 1]) / dt);
}

std::vector<Pose> ContinuousTrajectory::PosesAt(
    const std::vector<double>& times) const {
  std::vector<Pose> out;
  out.reserve(times.size());
  for (double t : times) out.push_back(At(t));
  return out;
}

Points DeskewSweep(const Points& points, const std::vector<double>& times,
                   double stamp, const ContinuousTrajectory& trajectory,
                   const Pose& vehicle_from_lidar, int samples) {
  GR_CHECK(times.size() == points.size());
  if (points.empty()) return points;
  const double t_min = *std::min_element(times.begin(), times.end());
  if (t_min >= 0.0) return points;
  samples = std::max(samples, 2);
  const size_t n_samples = static_cast<size_t>(samples);
  const Pose end_from_world =
      (trajectory.At(stamp) * vehicle_from_lidar).Inverse();
  const double step = -t_min / (samples - 1);
  std::vector<Vec3> rotations(n_samples);
  std::vector<Vec3> translations(n_samples);
  for (size_t k = 0; k < n_samples; ++k) {
    const double tau = k + 1 == n_samples ? 0.0 : t_min + step * k;
    const Pose end_from_tau =
        end_from_world * trajectory.At(stamp + tau) * vehicle_from_lidar;
    rotations[k] = So3Log(end_from_tau.rotation);
    translations[k] = end_from_tau.translation;
  }
  Points out(points.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (points.size() > 20000)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(points.size()); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    const double position =
        std::clamp((times[i] - t_min) / step, 0.0, samples - 1.0);
    const size_t i0 =
        static_cast<size_t>(std::min(static_cast<int>(position), samples - 2));
    const double w = position - static_cast<double>(i0);
    const Vec3 r = (1.0 - w) * rotations[i0] + w * rotations[i0 + 1];
    const Vec3 t = (1.0 - w) * translations[i0] + w * translations[i0 + 1];
    out[i] = So3Exp(r) * points[i] + t;
  }
  return out;
}

SweepSamples DeskewKeyframeSweep(const Points& points,
                                 const std::vector<float>& intensity,
                                 const std::vector<double>& times, double stamp,
                                 const ContinuousTrajectory& trajectory,
                                 const Pose& vehicle_from_lidar,
                                 double min_range, double max_range,
                                 double voxel) {
  const Points sweep =
      times.size() == points.size()
          ? DeskewSweep(points, times, stamp, trajectory, vehicle_from_lidar)
          : points;
  Points kept;
  std::vector<float> kept_intensity;
  kept.reserve(sweep.size());
  kept_intensity.reserve(sweep.size());
  for (size_t i = 0; i < sweep.size(); ++i) {
    const double r = sweep[i].norm();
    if (r < min_range || r > max_range) continue;
    kept.push_back(sweep[i]);
    kept_intensity.push_back(i < intensity.size() ? intensity[i] : 0.0f);
  }
  SweepSamples out;
  const std::vector<int> indices = VoxelDownsampleIndices(kept, voxel);
  out.points.reserve(indices.size());
  out.intensity.reserve(indices.size());
  for (int i : indices) {
    out.points.push_back(kept[static_cast<size_t>(i)]);
    out.intensity.push_back(kept_intensity[static_cast<size_t>(i)]);
  }
  return out;
}

}  // namespace gr
