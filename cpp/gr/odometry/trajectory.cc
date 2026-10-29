#include "gr/odometry/trajectory.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/numeric.h"
#include "gr/core/voxel.h"

namespace gr {
namespace {

// Sweeps with more points than this are deskewed by several threads.
constexpr size_t kMinPointsForThreads = 20000;

// Gyro bias at `time`: piecewise linear between the bias knots, zero
// without.
Vec3 GyroBiasAt(const std::vector<double>& stamps,
                const std::vector<Vec3>& biases, double time) {
  if (stamps.empty()) return Vec3::Zero();
  return InterpolateSeries(stamps, biases, time);
}

// Whether samples at `stamps` cover [start_time, end_time].
bool Covers(const std::vector<double>& stamps, double start_time,
            double end_time) {
  return stamps.size() > 1 && stamps.front() <= start_time &&
         stamps.back() >= end_time;
}

// Cubic Hermite spline from `start` (velocity `start_velocity`) to `end`
// over `duration`, at `fraction` in [0, 1] (the standard basis h00, h10,
// h01, h11).
Vec3 Hermite(const Vec3& start, const Vec3& start_velocity, const Vec3& end,
             const Vec3& end_velocity, double duration, double fraction) {
  const double s = fraction;
  const double s2 = s * s;
  const double s3 = s2 * s;
  return (2.0 * s3 - 3.0 * s2 + 1.0) * start +
         (s3 - 2.0 * s2 + s) * duration * start_velocity +
         (-2.0 * s3 + 3.0 * s2) * end + (s3 - s2) * duration * end_velocity;
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
  const size_t knot_count = knot_stamps_.size();
  for (size_t k = 1; k < knot_count; ++k) {
    GR_CHECK(knot_stamps_[k] > knot_stamps_[k - 1]);
  }
  // Central differences inside, one-sided at the ends.
  tangents_.assign(knot_count, Vec3::Zero());
  for (size_t k = 0; k < knot_count && knot_count > 1; ++k) {
    const size_t before = k == 0 ? 0 : k - 1;
    const size_t after = k + 1 == knot_count ? knot_count - 1 : k + 1;
    tangents_[k] =
        (knot_poses_[after].translation - knot_poses_[before].translation) /
        (knot_stamps_[after] - knot_stamps_[before]);
  }
  table_stamps_.clear();
  table_poses_.clear();
  for (const auto& [time, pose] :
       Extrapolate(kBackward, imu, bias_stamps, gyro_biases, wheel,
                   options.extrapolation, options.table_step)) {
    table_stamps_.push_back(time);
    table_poses_.push_back(pose);
  }
  int gyro_intervals = 0;
  int wheel_intervals = 0;
  for (size_t interval = 0; interval + 1 < knot_count; ++interval) {
    bool used_gyro = false;
    bool used_wheel = false;
    SampleInterval(interval, imu, bias_stamps, gyro_biases, wheel,
                   options.table_step, &used_gyro, &used_wheel);
    gyro_intervals += used_gyro ? 1 : 0;
    wheel_intervals += used_wheel ? 1 : 0;
  }
  table_stamps_.push_back(knot_stamps_.back());
  table_poses_.push_back(knot_poses_.back());
  for (const auto& [time, pose] :
       Extrapolate(kForward, imu, bias_stamps, gyro_biases, wheel,
                   options.extrapolation, options.table_step)) {
    table_stamps_.push_back(time);
    table_poses_.push_back(pose);
  }
  if (knot_count > 1) {
    gyro_coverage_ = static_cast<double>(gyro_intervals) / (knot_count - 1);
    wheel_coverage_ = static_cast<double>(wheel_intervals) / (knot_count - 1);
  }
}

std::vector<std::pair<double, Pose>> ContinuousTrajectory::Extrapolate(
    int direction, const ImuSamples& imu,
    const std::vector<double>& bias_stamps,
    const std::vector<Vec3>& gyro_biases, const WheelSamples& wheel,
    double duration, double step) const {
  std::vector<std::pair<double, Pose>> out;
  if (duration <= 0.0) return out;
  const size_t knot_count = knot_stamps_.size();
  const size_t end = direction < 0 ? 0 : knot_count - 1;
  const double end_time = knot_stamps_[end];
  const Pose& end_pose = knot_poses_[end];
  // Constant-velocity fallback from the two outermost knots.
  Vec3 rate = Vec3::Zero();  // Body frame.
  Vec3 velocity = tangents_[end];
  if (knot_count > 1) {
    const size_t inner = direction < 0 ? 1 : knot_count - 2;
    const size_t first = std::min(end, inner);
    const size_t second = std::max(end, inner);
    rate = So3Log(knot_poses_[first].rotation.transpose() *
                  knot_poses_[second].rotation) /
           (knot_stamps_[second] - knot_stamps_[first]);
  }
  const double far_time = end_time + direction * duration;
  const double earliest = std::min(end_time, far_time);
  const double latest = std::max(end_time, far_time);
  const bool use_gyro =
      imu.stamps.size() > 1 &&
      imu.stamps.front() <= earliest + MotionPrior::kEdgeTolerance &&
      imu.stamps.back() >= latest - MotionPrior::kEdgeTolerance;
  const bool use_wheel =
      use_gyro && wheel.stamps.size() > 1 &&
      wheel.stamps.front() <= earliest + MotionPrior::kEdgeTolerance &&
      wheel.stamps.back() >= latest - MotionPrior::kEdgeTolerance;
  // Integrate outwards from the knot in steps of `step`.
  const int num_steps =
      std::max(1, static_cast<int>(std::ceil(duration / step)));
  const double dt = duration / num_steps;
  Pose pose = end_pose;
  for (int m = 1; m <= num_steps; ++m) {
    const double previous_time = end_time + direction * (m - 1) * dt;
    const double midpoint = previous_time + direction * 0.5 * dt;
    const Vec3 body_rate =
        use_gyro ? Vec3(InterpolateSeries(imu.stamps, imu.gyro, midpoint) -
                        GyroBiasAt(bias_stamps, gyro_biases, midpoint))
                 : rate;
    // Half the step's rotation: the translation uses the midpoint heading.
    const Mat3 half_turn = So3Exp(body_rate * (direction * 0.5 * dt));
    if (use_wheel) {
      const double speed = Interp1(wheel.stamps, wheel.speed, midpoint);
      pose.translation +=
          pose.rotation * half_turn * Vec3(direction * speed * dt, 0.0, 0.0);
    } else {
      pose.translation += direction * dt * velocity;
    }
    pose.rotation = pose.rotation * half_turn * half_turn;
    pose.Normalize();
    out.emplace_back(end_time + direction * m * dt, pose);
  }
  if (direction < 0) std::reverse(out.begin(), out.end());
  return out;
}

void ContinuousTrajectory::SampleInterval(
    size_t interval, const ImuSamples& imu,
    const std::vector<double>& bias_stamps,
    const std::vector<Vec3>& gyro_biases, const WheelSamples& wheel,
    double step, bool* used_gyro, bool* used_wheel) {
  const double start_time = knot_stamps_[interval];
  const double end_time = knot_stamps_[interval + 1];
  const Pose& start_pose = knot_poses_[interval];
  const Pose& end_pose = knot_poses_[interval + 1];
  const double duration = end_time - start_time;
  const int num_samples =
      std::max(1, static_cast<int>(std::ceil(duration / step)));
  std::vector<double> sample_times(static_cast<size_t>(num_samples));
  for (int m = 0; m < num_samples; ++m) {
    sample_times[static_cast<size_t>(m)] =
        start_time + duration * m / num_samples;
  }
  *used_gyro = Covers(imu.stamps, start_time, end_time);
  *used_wheel = *used_gyro && Covers(wheel.stamps, start_time, end_time);
  const auto hermite = [&](double fraction) -> Vec3 {
    return Hermite(start_pose.translation, tangents_[interval],
                   end_pose.translation, tangents_[interval + 1], duration,
                   fraction);
  };
  if (!*used_gyro) {
    for (double time : sample_times) {
      const double fraction = (time - start_time) / duration;
      Pose pose = Interpolate(start_pose, end_pose, fraction);
      pose.translation = hermite(fraction);
      table_stamps_.push_back(time);
      table_poses_.push_back(pose);
    }
    return;
  }

  // Integration pieces end at every output sample and IMU stamp.
  std::vector<double> knots = sample_times;
  for (auto stamp =
           std::upper_bound(imu.stamps.begin(), imu.stamps.end(), start_time);
       stamp != imu.stamps.end() && *stamp < end_time; ++stamp) {
    knots.push_back(*stamp);
  }
  knots.push_back(end_time);
  std::sort(knots.begin(), knots.end());
  knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
  // Explicit return type: an Eigen expression would outlive its operands.
  const auto rate_at = [&](double time) -> Vec3 {
    return InterpolateSeries(imu.stamps, imu.gyro, time) -
           GyroBiasAt(bias_stamps, gyro_biases, time);
  };

  // Pass 1: the gyro rotation over the interval and its mismatch with the
  // end knot, distributed linearly in time.
  Mat3 gyro_rotation = Mat3::Identity();
  for (size_t piece = 0; piece + 1 < knots.size(); ++piece) {
    const double dt = knots[piece + 1] - knots[piece];
    gyro_rotation =
        gyro_rotation * So3Exp(rate_at(knots[piece] + 0.5 * dt) * dt);
  }
  const Vec3 rotation_correction =
      So3Log(gyro_rotation.transpose() * start_pose.rotation.transpose() *
             end_pose.rotation);

  // Pass 2: corrected rotations at the samples; wheel dead reckoning.
  gyro_rotation = Mat3::Identity();
  Vec3 position = start_pose.translation;
  const size_t first = table_stamps_.size();
  size_t next_sample = 0;
  for (size_t piece = 0; piece < knots.size(); ++piece) {
    const double time = knots[piece];
    if (next_sample < sample_times.size() &&
        time == sample_times[next_sample]) {
      const double fraction = (time - start_time) / duration;
      table_stamps_.push_back(time);
      table_poses_.emplace_back(start_pose.rotation * gyro_rotation *
                                    So3Exp(fraction * rotation_correction),
                                position);
      ++next_sample;
    }
    if (piece + 1 == knots.size()) break;
    const double dt = knots[piece + 1] - time;
    const double midpoint = time + 0.5 * dt;
    const Vec3 rate = rate_at(midpoint);
    if (*used_wheel) {
      const Mat3 midpoint_rotation =
          start_pose.rotation * gyro_rotation * So3Exp(rate * (0.5 * dt)) *
          So3Exp(((midpoint - start_time) / duration) * rotation_correction);
      position +=
          midpoint_rotation *
          Vec3(Interp1(wheel.stamps, wheel.speed, midpoint) * dt, 0.0, 0.0);
    }
    gyro_rotation = gyro_rotation * So3Exp(rate * dt);
  }

  // Pass 3: positions, corrected to end exactly at the next knot.
  const Vec3 position_correction = end_pose.translation - position;
  for (size_t m = first; m < table_stamps_.size(); ++m) {
    const double fraction = (table_stamps_[m] - start_time) / duration;
    table_poses_[m].Normalize();
    if (*used_wheel) {
      table_poses_[m].translation += fraction * position_correction;
    } else {
      table_poses_[m].translation = hermite(fraction);
    }
  }
}

Pose ContinuousTrajectory::At(double time) const {
  GR_CHECK(!empty() && !table_stamps_.empty());
  if (time <= table_stamps_.front()) return table_poses_.front();
  if (time >= table_stamps_.back()) return table_poses_.back();
  const size_t after = static_cast<size_t>(
      std::upper_bound(table_stamps_.begin(), table_stamps_.end(), time) -
      table_stamps_.begin());
  const size_t before = after - 1;
  const double dt = table_stamps_[after] - table_stamps_[before];
  return Interpolate(table_poses_[before], table_poses_[after],
                     (time - table_stamps_[before]) / dt);
}

std::vector<Pose> ContinuousTrajectory::PosesAt(
    const std::vector<double>& times) const {
  std::vector<Pose> out;
  out.reserve(times.size());
  for (double time : times) out.push_back(At(time));
  return out;
}

Points DeskewSweep(const Points& points, const std::vector<double>& times,
                   double stamp, const ContinuousTrajectory& trajectory,
                   const Pose& vehicle_from_lidar, int samples) {
  GR_CHECK(times.size() == points.size());
  if (points.empty()) return points;
  const double earliest = *std::min_element(times.begin(), times.end());
  if (earliest >= 0.0) return points;
  samples = std::max(samples, kMinDeskewSamples);
  const size_t sample_count = static_cast<size_t>(samples);
  const Pose end_from_world =
      (trajectory.At(stamp) * vehicle_from_lidar).Inverse();
  const double step = -earliest / (samples - 1);
  std::vector<Vec3> rotations(sample_count);
  std::vector<Vec3> translations(sample_count);
  for (size_t k = 0; k < sample_count; ++k) {
    const double tau = k + 1 == sample_count ? 0.0 : earliest + step * k;
    const Pose end_from_tau =
        end_from_world * trajectory.At(stamp + tau) * vehicle_from_lidar;
    rotations[k] = So3Log(end_from_tau.rotation);
    translations[k] = end_from_tau.translation;
  }
  const double last_sample = samples - 1.0;
  const int last_interval = samples - 2;
  Points out(points.size());
  [[maybe_unused]] const bool use_threads =
      points.size() > kMinPointsForThreads;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (use_threads)
#endif
  for (int64_t signed_index = 0;
       signed_index < static_cast<int64_t>(points.size()); ++signed_index) {
    const size_t i = static_cast<size_t>(signed_index);
    const double position =
        std::clamp((times[i] - earliest) / step, 0.0, last_sample);
    const size_t before = static_cast<size_t>(
        std::min(static_cast<int>(position), last_interval));
    const double weight = position - static_cast<double>(before);
    const Vec3 rotation =
        (1.0 - weight) * rotations[before] + weight * rotations[before + 1];
    const Vec3 translation = (1.0 - weight) * translations[before] +
                             weight * translations[before + 1];
    out[i] = So3Exp(rotation) * points[i] + translation;
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
    const double range = sweep[i].norm();
    if (range < min_range || range > max_range) continue;
    kept.push_back(sweep[i]);
    kept_intensity.push_back(i < intensity.size() ? intensity[i] : 0.0f);
  }
  SweepSamples out;
  const std::vector<int> indices = VoxelDownsampleIndices(kept, voxel);
  out.points.reserve(indices.size());
  out.intensity.reserve(indices.size());
  for (int index : indices) {
    out.points.push_back(kept[static_cast<size_t>(index)]);
    out.intensity.push_back(kept_intensity[static_cast<size_t>(index)]);
  }
  return out;
}

}  // namespace gr
