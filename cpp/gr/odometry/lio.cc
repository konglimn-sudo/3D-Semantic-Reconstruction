#include "gr/odometry/lio.h"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "gr/core/check.h"
#include "gr/core/voxel.h"

namespace gr {
namespace {

using Mat15 = LioOdometry::Mat15;
using Vec15 = LioOdometry::Vec15;
using PoseJacobian = Eigen::Matrix<double, kSpaceDim, kPoseDim>;
using PoseHessian = Eigen::Matrix<double, kPoseDim, kPoseDim>;
using PoseGradient = Eigen::Matrix<double, kPoseDim, 1>;

// The LiDAR constrains the pose: the states from the rotation on, in the
// order of PoseJacobian's columns (rotation, then position).
constexpr int kPoseState = LioOdometry::kRotationState;

// Yaw and position define the frame: their initial variance is tiny.
constexpr double kGaugeVariance = 1e-8;
// A zero-velocity test needs this many IMU samples in its window.
constexpr size_t kMinStationarySamples = 5;

// What a sweep point was matched to.
enum class Match : uint8_t { kNone, kPlane, kPoint };

// Re-orthonormalised rotation.
Mat3 Orthonormal(const Mat3& rotation) {
  Pose pose(rotation, Vec3::Zero());
  pose.Normalize();
  return pose.rotation;
}

// Point-to-plane correspondence of a world point, as in AlignToMap: PCA of
// the map within a radius that doubles until the neighbourhood is planar.
bool FindPlane(const VoxelHashMap& map, const Vec3& query,
               const LioConfig& config, Vec3* normal, Vec3* centroid) {
  double radius =
      config.plane_radius > 0.0 ? config.plane_radius : config.map_voxel;
  while (true) {
    const auto plane =
        map.FitLocalPlane(query, radius, config.min_plane_points);
    if (plane && plane->planarity >= config.min_planarity &&
        plane->thickness <= config.max_plane_thickness) {
      *normal = plane->normal;
      *centroid = plane->centroid;
      return true;
    }
    if (radius >= config.max_plane_radius) return false;
    radius = std::min(2.0 * radius, config.max_plane_radius);
  }
}

// The 3 x 3 block of `matrix` at the states `row` and `column`.
auto StateBlock(Mat15& matrix, int row, int column) {
  return matrix.block<kSpaceDim, kSpaceDim>(row, column);
}

}  // namespace

LioOdometry::LioOdometry(const LioConfig& config, ImuSamples imu,
                         const Pose& vehicle_from_lidar,
                         const Pose& world_from_vehicle0)
    : config_(config),
      imu_(std::move(imu)),
      body_from_vehicle_(imu_.VehicleFromBody().Inverse()),
      body_from_lidar_(body_from_vehicle_ * vehicle_from_lidar),
      lidar_from_body_(body_from_lidar_.Inverse()),
      map_(config.map_voxel, config.map_radius, config.max_points_per_voxel) {
  GR_CHECK(imu_.size() > 1);
  const Pose world_from_body0 = world_from_vehicle0 * imu_.VehicleFromBody();
  rotation_ = world_from_body0.rotation;
  position_ = world_from_body0.translation;
  covariance_.setZero();
  const double tilt = DegToRad(config.init_tilt_sigma_deg);
  // Roll / pitch from gravity; yaw and position define the frame.
  StateBlock(covariance_, kRotationState, kRotationState) =
      Vec3(tilt * tilt, tilt * tilt, kGaugeVariance).asDiagonal();
  StateBlock(covariance_, kPositionState, kPositionState) =
      Mat3::Identity() * kGaugeVariance;
  StateBlock(covariance_, kVelocityState, kVelocityState) =
      Mat3::Identity() * config.init_velocity_sigma *
      config.init_velocity_sigma;
  StateBlock(covariance_, kGyroBiasState, kGyroBiasState) =
      Mat3::Identity() * config.init_gyro_bias_sigma *
      config.init_gyro_bias_sigma;
  StateBlock(covariance_, kAccelBiasState, kAccelBiasState) =
      Mat3::Identity() * config.init_accel_bias_sigma *
      config.init_accel_bias_sigma;
}

Pose LioOdometry::world_from_vehicle() const {
  return Pose(rotation_, position_) * body_from_vehicle_;
}

void LioOdometry::Propagate(double time) {
  samples_.clear();
  samples_.push_back({time_, rotation_, position_});
  const Vec3 gravity(0.0, 0.0, -config_.gravity);
  // Noise variances per second.
  const double gyro_variance = config_.gyro_noise * config_.gyro_noise;
  const double accel_variance = config_.accel_noise * config_.accel_noise;
  const double gyro_bias_variance =
      config_.gyro_bias_walk * config_.gyro_bias_walk;
  const double accel_bias_variance =
      config_.accel_bias_walk * config_.accel_bias_walk;
  const size_t sample_count = imu_.size();
  while (time_ < time) {
    // The sample in force over [time_, next): the last one at or before.
    while (imu_index_ < sample_count && imu_.stamps[imu_index_] <= time_) {
      ++imu_index_;
    }
    const size_t sample = imu_index_ == 0 ? 0 : imu_index_ - 1;
    const double next = imu_index_ < sample_count
                            ? std::min(imu_.stamps[imu_index_], time)
                            : time;  // Hold the last rate.
    const double dt = next - time_;
    if (dt <= 0.0) {
      time_ = next;
      continue;
    }
    const Vec3 rate = imu_.gyro[sample] - gyro_bias_;
    const Vec3 specific_force = imu_.accel[sample] - accel_bias_;
    const Mat3 step_rotation = So3Exp(rate * dt);
    const Vec3 acceleration = rotation_ * specific_force + gravity;
    // Error-state transition (right perturbation of the rotation).
    Mat15 transition = Mat15::Identity();
    StateBlock(transition, kRotationState, kRotationState) =
        step_rotation.transpose();
    StateBlock(transition, kRotationState, kGyroBiasState) =
        -Mat3::Identity() * dt;
    StateBlock(transition, kPositionState, kVelocityState) =
        Mat3::Identity() * dt;
    StateBlock(transition, kVelocityState, kRotationState) =
        -rotation_ * Hat(specific_force) * dt;
    StateBlock(transition, kVelocityState, kAccelBiasState) = -rotation_ * dt;
    Mat15 process_noise = Mat15::Zero();
    StateBlock(process_noise, kRotationState, kRotationState) =
        Mat3::Identity() * gyro_variance * dt;
    StateBlock(process_noise, kVelocityState, kVelocityState) =
        Mat3::Identity() * accel_variance * dt;
    StateBlock(process_noise, kGyroBiasState, kGyroBiasState) =
        Mat3::Identity() * gyro_bias_variance * dt;
    StateBlock(process_noise, kAccelBiasState, kAccelBiasState) =
        Mat3::Identity() * accel_bias_variance * dt;
    covariance_ =
        transition * covariance_ * transition.transpose() + process_noise;
    position_ += velocity_ * dt + 0.5 * acceleration * dt * dt;
    velocity_ += acceleration * dt;
    rotation_ = Orthonormal(rotation_ * step_rotation);
    time_ = next;
    samples_.push_back({time_, rotation_, position_});
  }
}

void LioOdometry::BackfillFirstSweep(double first_time, double stamp) {
  // Gyro only, backwards from the stamp, at rest (the velocity is unknown).
  std::vector<Sample> backwards{{stamp, rotation_, position_}};
  Mat3 rotation = rotation_;
  double time = stamp;
  // The first IMU sample at or after `time`.
  size_t next_sample = static_cast<size_t>(
      std::lower_bound(imu_.stamps.begin(), imu_.stamps.end(), stamp) -
      imu_.stamps.begin());
  while (time > first_time) {
    const size_t sample = next_sample == 0 ? 0 : next_sample - 1;
    const double previous = std::max(
        first_time, next_sample == 0 ? first_time : imu_.stamps[sample]);
    const double dt = time - previous;
    if (dt > 0.0) {
      rotation = Orthonormal(rotation *
                             So3Exp(-(imu_.gyro[sample] - gyro_bias_) * dt));
      backwards.push_back({previous, rotation, position_});
    }
    time = previous;
    if (next_sample == 0) break;
    --next_sample;
  }
  std::reverse(backwards.begin(), backwards.end());
  samples_ = std::move(backwards);
}

void LioOdometry::PoseAt(double time, Mat3* rotation, Vec3* position) const {
  if (samples_.empty() || time >= samples_.back().time) {
    *rotation = samples_.empty() ? rotation_ : samples_.back().rotation;
    *position = samples_.empty() ? position_ : samples_.back().position;
    return;
  }
  if (time <= samples_.front().time) {
    *rotation = samples_.front().rotation;
    *position = samples_.front().position;
    return;
  }
  const auto after_it = std::upper_bound(
      samples_.begin(), samples_.end(), time,
      [](double value, const Sample& sample) { return value < sample.time; });
  const Sample& after = *after_it;
  const Sample& before = *(after_it - 1);
  const double span = after.time - before.time;
  const double fraction = span > 0.0 ? (time - before.time) / span : 0.0;
  *rotation =
      before.rotation *
      So3Exp(fraction * So3Log(before.rotation.transpose() * after.rotation));
  *position = before.position + fraction * (after.position - before.position);
}

Points LioOdometry::Deskew(const Points& points,
                           const std::vector<double>& times,
                           double stamp) const {
  Points out(points.size());
  const Mat3 stamp_from_world_rotation = rotation_.transpose();
  for (size_t i = 0; i < points.size(); ++i) {
    const Vec3 body = body_from_lidar_ * points[i];
    if (times.size() != points.size()) {
      out[i] = points[i];
      continue;
    }
    Mat3 rotation;
    Vec3 position;
    PoseAt(stamp + times[i], &rotation, &position);
    const Vec3 world = rotation * body + position;
    out[i] =
        lidar_from_body_ * (stamp_from_world_rotation * (world - position_));
  }
  return out;
}

void LioOdometry::Update(const Points& sweep_body, bool stationary) {
  const Mat3 rotation_prior = rotation_;
  const Vec3 position_prior = position_;
  const Vec3 velocity_prior = velocity_;
  const Vec3 gyro_bias_prior = gyro_bias_;
  const Vec3 accel_bias_prior = accel_bias_;
  const Mat15 prior_information = covariance_.inverse();
  const double plane_weight = 1.0 / (config_.point_sigma * config_.point_sigma);
  const double squared_robust_scale =
      config_.robust_scale * config_.robust_scale;
  Mat15 information = prior_information;
  stats_ = LioStats();
  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const int64_t point_count = static_cast<int64_t>(sweep_body.size());
    // Per point: one row for a plane, all three for a point fallback.
    std::vector<PoseJacobian> jacobians(sweep_body.size());
    std::vector<Vec3> residuals(sweep_body.size(), Vec3::Zero());
    std::vector<Match> matches(sweep_body.size(), Match::kNone);
#ifdef _OPENMP
    constexpr int kChunkSize = 64;  // Points per OpenMP work chunk.
#pragma omp parallel for schedule(dynamic, kChunkSize)
#endif
    for (int64_t signed_index = 0; signed_index < point_count; ++signed_index) {
      const size_t k = static_cast<size_t>(signed_index);
      const Vec3& body_point = sweep_body[k];
      const Vec3 world_point = rotation_ * body_point + position_;
      const Mat3 world_point_by_rotation = -rotation_ * Hat(body_point);
      Vec3 normal;
      Vec3 centroid;
      if (FindPlane(map_, world_point, config_, &normal, &centroid)) {
        const double residual = normal.dot(world_point - centroid);
        if (std::abs(residual) > config_.max_residual) continue;
        jacobians[k].row(0).leftCols<kSpaceDim>() =
            normal.transpose() * world_point_by_rotation;
        jacobians[k].row(0).rightCols<kSpaceDim>() = normal.transpose();
        residuals[k](0) = residual;
        matches[k] = Match::kPlane;
      } else if (config_.point_weight > 0.0) {
        const auto nearest = map_.Nearest(world_point, config_.max_residual);
        if (!nearest) continue;
        jacobians[k].leftCols<kSpaceDim>() = world_point_by_rotation;
        jacobians[k].rightCols<kSpaceDim>() = Mat3::Identity();
        residuals[k] = world_point - nearest->point;
        matches[k] = Match::kPoint;
      }
    }
    PoseHessian hessian = PoseHessian::Zero();
    PoseGradient gradient = PoseGradient::Zero();
    int plane_count = 0;
    int point_residual_count = 0;
    double squared_error_sum = 0.0;
    for (size_t i = 0; i < sweep_body.size(); ++i) {
      if (matches[i] == Match::kPlane) {
        // Cauchy weight of the residual.
        const double residual = residuals[i](0);
        const double weight =
            plane_weight / (1.0 + residual * residual / squared_robust_scale);
        const Eigen::Matrix<double, 1, kPoseDim> jacobian = jacobians[i].row(0);
        hessian += weight * jacobian.transpose() * jacobian;
        gradient += weight * jacobian.transpose() * residual;
        squared_error_sum += residual * residual;
        ++plane_count;
      } else if (matches[i] == Match::kPoint) {
        const Vec3& residual = residuals[i];
        const double weight =
            config_.point_weight * plane_weight /
            (1.0 + residual.squaredNorm() / squared_robust_scale);
        hessian += weight * jacobians[i].transpose() * jacobians[i];
        gradient += weight * jacobians[i].transpose() * residual;
        ++point_residual_count;
      }
    }
    // Error of the current iterate from the prediction.
    Vec15 error;
    error.segment<kSpaceDim>(kRotationState) =
        So3Log(rotation_prior.transpose() * rotation_);
    error.segment<kSpaceDim>(kPositionState) = position_ - position_prior;
    error.segment<kSpaceDim>(kVelocityState) = velocity_ - velocity_prior;
    error.segment<kSpaceDim>(kGyroBiasState) = gyro_bias_ - gyro_bias_prior;
    error.segment<kSpaceDim>(kAccelBiasState) = accel_bias_ - accel_bias_prior;
    information = prior_information;
    information.block<kPoseDim, kPoseDim>(kPoseState, kPoseState) += hessian;
    Vec15 rhs = -prior_information * error;
    rhs.segment<kPoseDim>(kPoseState) -= gradient;
    if (config_.max_speed > 0.0 && velocity_prior.norm() > config_.max_speed) {
      // Pulled back to the speed limit, along the predicted direction.
      const Vec3 limit =
          velocity_prior * (config_.max_speed / velocity_prior.norm());
      const double speed_weight =
          1.0 / (config_.speed_sigma * config_.speed_sigma);
      StateBlock(information, kVelocityState, kVelocityState) +=
          speed_weight * Mat3::Identity();
      rhs.segment<kSpaceDim>(kVelocityState) -=
          speed_weight * (velocity_ - limit);
    }
    if (stationary) {  // Velocity measured as zero.
      const double zupt_weight =
          1.0 / (config_.zupt_sigma * config_.zupt_sigma);
      StateBlock(information, kVelocityState, kVelocityState) +=
          zupt_weight * Mat3::Identity();
      rhs.segment<kSpaceDim>(kVelocityState) -= zupt_weight * velocity_;
    }
    const Vec15 step = information.ldlt().solve(rhs);
    rotation_ = Orthonormal(rotation_ *
                            So3Exp(step.segment<kSpaceDim>(kRotationState)));
    position_ += step.segment<kSpaceDim>(kPositionState);
    velocity_ += step.segment<kSpaceDim>(kVelocityState);
    gyro_bias_ += step.segment<kSpaceDim>(kGyroBiasState);
    accel_bias_ += step.segment<kSpaceDim>(kAccelBiasState);
    stats_.iterations = iteration + 1;
    stats_.correspondences = plane_count;
    stats_.point_residuals = point_residual_count;
    stats_.rmse =
        plane_count > 0 ? std::sqrt(squared_error_sum / plane_count) : 0.0;
    // The translation block of the pose Hessian (after the rotation).
    const Eigen::SelfAdjointEigenSolver<Mat3> eigen(
        hessian.block<kSpaceDim, kSpaceDim>(kSpaceDim, kSpaceDim));
    stats_.min_translation_information = eigen.eigenvalues()(0);
    if (step.segment<kPoseDim>(kPoseState).norm() < config_.convergence) {
      break;
    }
  }
  covariance_ = information.inverse();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose());
  stats_.stationary = stationary;
  stats_.velocity = velocity_;
}

bool LioOdometry::Stationary(double time) const {
  if (config_.zupt_sigma <= 0.0) return false;
  const auto begin = std::lower_bound(imu_.stamps.begin(), imu_.stamps.end(),
                                      time - config_.zupt_window);
  const auto end =
      std::upper_bound(imu_.stamps.begin(), imu_.stamps.end(), time);
  const size_t first = static_cast<size_t>(begin - imu_.stamps.begin());
  const size_t last_end = static_cast<size_t>(end - imu_.stamps.begin());
  if (last_end < first + kMinStationarySamples) return false;
  Vec3 mean = Vec3::Zero();
  for (size_t i = first; i < last_end; ++i) {
    if ((imu_.gyro[i] - gyro_bias_).norm() > config_.zupt_max_gyro) {
      return false;
    }
    mean += imu_.accel[i];
  }
  mean /= static_cast<double>(last_end - first);
  for (size_t i = first; i < last_end; ++i) {
    if ((imu_.accel[i] - mean).norm() > config_.zupt_max_accel) return false;
  }
  return true;
}

Pose LioOdometry::Process(const Points& points,
                          const std::vector<double>& times, double stamp) {
  if (!initialized_) {
    time_ = stamp;
    imu_index_ = static_cast<size_t>(
        std::upper_bound(imu_.stamps.begin(), imu_.stamps.end(), stamp) -
        imu_.stamps.begin());
    double first = stamp;
    for (double offset : times) first = std::min(first, stamp + offset);
    BackfillFirstSweep(first, stamp);
    initialized_ = true;
  } else {
    Propagate(stamp);
  }
  deskewed_ = Deskew(points, times, stamp);

  Points kept;
  kept.reserve(deskewed_.size());
  for (const Vec3& point : deskewed_) {
    const double range = point.norm();
    if (range >= config_.min_range && range <= config_.max_range) {
      kept.push_back(point);
    }
  }
  Points body = VoxelDownsample(kept, config_.scan_voxel);
  for (Vec3& point : body) point = body_from_lidar_ * point;
  if (!map_.empty()) Update(body, Stationary(stamp));

  Points insert = VoxelDownsample(kept, config_.insert_voxel);
  const Pose world_from_lidar = Pose(rotation_, position_) * body_from_lidar_;
  for (Vec3& point : insert) point = world_from_lidar * point;
  map_.AddPoints(insert);
  map_.RemoveFar(position_);
  return world_from_lidar;
}

}  // namespace gr
