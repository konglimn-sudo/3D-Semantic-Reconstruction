#include "gr/lio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>

#include "gr/check.h"
#include "gr/voxel.h"

namespace gr {
namespace {

using Mat15 = LioOdometry::Mat15;
using Vec15 = LioOdometry::Vec15;

// Re-orthonormalised rotation.
Mat3 Orthonormal(const Mat3& rotation) {
  Pose pose(rotation, Vec3::Zero());
  pose.Normalize();
  return pose.rotation;
}

// Point-to-plane correspondence of a world point, as in AlignToMap: PCA of
// the map within a radius that doubles until the neighbourhood is planar.
bool FindPlane(const VoxelHashMap& map, const Vec3& query,
               const LioConfig& config, Vec3& normal, Vec3& centroid) {
  double radius =
      config.plane_radius > 0.0 ? config.plane_radius : config.map_voxel;
  while (true) {
    const auto plane =
        map.FitLocalPlane(query, radius, config.min_plane_points);
    if (plane && plane->planarity >= config.min_planarity &&
        plane->thickness <= config.max_plane_thickness) {
      normal = plane->normal;
      centroid = plane->centroid;
      return true;
    }
    if (radius >= config.max_plane_radius) return false;
    radius = std::min(2.0 * radius, config.max_plane_radius);
  }
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
  const Pose world_from_body0 =
      world_from_vehicle0 * imu_.VehicleFromBody();
  rotation_ = world_from_body0.rotation;
  position_ = world_from_body0.translation;
  covariance_.setZero();
  const double tilt = DegToRad(config.init_tilt_sigma_deg);
  // Yaw and position define the frame: tiny; roll / pitch from gravity.
  covariance_.block<3, 3>(0, 0) =
      Vec3(tilt * tilt, tilt * tilt, 1e-8).asDiagonal();
  covariance_.block<3, 3>(3, 3) = Mat3::Identity() * 1e-8;
  covariance_.block<3, 3>(6, 6) = Mat3::Identity() *
                                  config.init_velocity_sigma *
                                  config.init_velocity_sigma;
  covariance_.block<3, 3>(9, 9) = Mat3::Identity() *
                                  config.init_gyro_bias_sigma *
                                  config.init_gyro_bias_sigma;
  covariance_.block<3, 3>(12, 12) = Mat3::Identity() *
                                    config.init_accel_bias_sigma *
                                    config.init_accel_bias_sigma;
}

Pose LioOdometry::world_from_vehicle() const {
  return Pose(rotation_, position_) * body_from_vehicle_;
}

void LioOdometry::Propagate(double t) {
  samples_.clear();
  samples_.push_back({time_, rotation_, position_});
  const Vec3 gravity(0.0, 0.0, -config_.gravity);
  const double qg = config_.gyro_noise * config_.gyro_noise;
  const double qa = config_.accel_noise * config_.accel_noise;
  const double qbg = config_.gyro_bias_walk * config_.gyro_bias_walk;
  const double qba = config_.accel_bias_walk * config_.accel_bias_walk;
  const size_t n = imu_.size();
  while (time_ < t) {
    // The sample in force over [time_, next): the last one at or before.
    while (imu_index_ < n && imu_.stamps[imu_index_] <= time_) ++imu_index_;
    const size_t k = imu_index_ == 0 ? 0 : imu_index_ - 1;
    const double next = imu_index_ < n ? std::min(imu_.stamps[imu_index_], t)
                                       : t;  // Hold the last rate.
    const double dt = next - time_;
    if (dt <= 0.0) {
      time_ = next;
      continue;
    }
    const Vec3 omega = imu_.gyro[k] - gyro_bias_;
    const Vec3 specific = imu_.accel[k] - accel_bias_;
    const Mat3 delta = So3Exp(omega * dt);
    const Vec3 acceleration = rotation_ * specific + gravity;
    // Error-state transition (right perturbation of the rotation).
    Mat15 F = Mat15::Identity();
    F.block<3, 3>(0, 0) = delta.transpose();
    F.block<3, 3>(0, 9) = -Mat3::Identity() * dt;
    F.block<3, 3>(3, 6) = Mat3::Identity() * dt;
    F.block<3, 3>(6, 0) = -rotation_ * Hat(specific) * dt;
    F.block<3, 3>(6, 12) = -rotation_ * dt;
    Mat15 Q = Mat15::Zero();
    Q.block<3, 3>(0, 0) = Mat3::Identity() * qg * dt;
    Q.block<3, 3>(6, 6) = Mat3::Identity() * qa * dt;
    Q.block<3, 3>(9, 9) = Mat3::Identity() * qbg * dt;
    Q.block<3, 3>(12, 12) = Mat3::Identity() * qba * dt;
    covariance_ = F * covariance_ * F.transpose() + Q;
    position_ += velocity_ * dt + 0.5 * acceleration * dt * dt;
    velocity_ += acceleration * dt;
    rotation_ = Orthonormal(rotation_ * delta);
    time_ = next;
    samples_.push_back({time_, rotation_, position_});
  }
}

void LioOdometry::BackfillFirstSweep(double t_first, double stamp) {
  // Gyro only, backwards from the stamp, at rest (the velocity is unknown).
  std::vector<Sample> back{{stamp, rotation_, position_}};
  Mat3 rotation = rotation_;
  double t = stamp;
  size_t k = static_cast<size_t>(
      std::lower_bound(imu_.stamps.begin(), imu_.stamps.end(), stamp) -
      imu_.stamps.begin());
  while (t > t_first) {
    const size_t j = k == 0 ? 0 : k - 1;
    const double previous =
        std::max(t_first, k == 0 ? t_first : imu_.stamps[j]);
    const double dt = t - previous;
    if (dt > 0.0) {
      rotation = Orthonormal(
          rotation * So3Exp(-(imu_.gyro[j] - gyro_bias_) * dt));
      back.push_back({previous, rotation, position_});
    }
    t = previous;
    if (k == 0) break;
    --k;
  }
  std::reverse(back.begin(), back.end());
  samples_ = std::move(back);
}

void LioOdometry::PoseAt(double t, Mat3& rotation, Vec3& position) const {
  if (samples_.empty() || t >= samples_.back().t) {
    rotation = samples_.empty() ? rotation_ : samples_.back().rotation;
    position = samples_.empty() ? position_ : samples_.back().position;
    return;
  }
  if (t <= samples_.front().t) {
    rotation = samples_.front().rotation;
    position = samples_.front().position;
    return;
  }
  const auto it = std::upper_bound(
      samples_.begin(), samples_.end(), t,
      [](double value, const Sample& s) { return value < s.t; });
  const Sample& b = *it;
  const Sample& a = *(it - 1);
  const double span = b.t - a.t;
  const double alpha = span > 0.0 ? (t - a.t) / span : 0.0;
  rotation = a.rotation *
             So3Exp(alpha * So3Log(a.rotation.transpose() * b.rotation));
  position = a.position + alpha * (b.position - a.position);
}

Points LioOdometry::Deskew(const Points& points,
                           const std::vector<double>& times,
                           double stamp) const {
  Points out(points.size());
  const Mat3 rotation_t = rotation_.transpose();
  for (size_t i = 0; i < points.size(); ++i) {
    const Vec3 body = body_from_lidar_ * points[i];
    if (times.size() != points.size()) {
      out[i] = points[i];
      continue;
    }
    Mat3 rotation;
    Vec3 position;
    PoseAt(stamp + times[i], rotation, position);
    const Vec3 world = rotation * body + position;
    out[i] = lidar_from_body_ * (rotation_t * (world - position_));
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
  const double w0 = 1.0 / (config_.point_sigma * config_.point_sigma);
  const double c2 = config_.robust_scale * config_.robust_scale;
  Mat15 information = prior_information;
  stats_ = LioStats();
  for (int iteration = 0; iteration < config_.max_iterations; ++iteration) {
    const int64_t n = static_cast<int64_t>(sweep_body.size());
    // Per point: up to 3 rows (1 for a plane, 3 for a point fallback).
    std::vector<Eigen::Matrix<double, 3, 6>> jacobians(sweep_body.size());
    std::vector<Vec3> residuals(sweep_body.size(), Vec3::Zero());
    std::vector<uint8_t> found(sweep_body.size(), 0);  // 1 plane, 2 point
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 64)
#endif
    for (int64_t i = 0; i < n; ++i) {
      const size_t k = static_cast<size_t>(i);
      const Vec3& y = sweep_body[k];
      const Vec3 q = rotation_ * y + position_;
      const Mat3 dq_dtheta = -rotation_ * Hat(y);
      Vec3 normal, centroid;
      if (FindPlane(map_, q, config_, normal, centroid)) {
        const double r = normal.dot(q - centroid);
        if (std::abs(r) > config_.max_residual) continue;
        jacobians[k].row(0).leftCols<3>() = normal.transpose() * dq_dtheta;
        jacobians[k].row(0).rightCols<3>() = normal.transpose();
        residuals[k](0) = r;
        found[k] = 1;
      } else if (config_.point_weight > 0.0) {
        const auto nearest = map_.Nearest(q, config_.max_residual);
        if (!nearest) continue;
        jacobians[k].leftCols<3>() = dq_dtheta;
        jacobians[k].rightCols<3>() = Mat3::Identity();
        residuals[k] = q - nearest->point;
        found[k] = 2;
      }
    }
    Eigen::Matrix<double, 6, 6> hessian = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> gradient = Eigen::Matrix<double, 6, 1>::Zero();
    int count = 0, points = 0;
    double sse = 0.0;
    for (size_t i = 0; i < sweep_body.size(); ++i) {
      if (found[i] == 1) {
        const double r = residuals[i](0);
        const double w = w0 / (1.0 + r * r / c2);
        const Eigen::Matrix<double, 1, 6> J = jacobians[i].row(0);
        hessian += w * J.transpose() * J;
        gradient += w * J.transpose() * r;
        sse += r * r;
        ++count;
      } else if (found[i] == 2) {
        const Vec3& r = residuals[i];
        const double w =
            config_.point_weight * w0 / (1.0 + r.squaredNorm() / c2);
        hessian += w * jacobians[i].transpose() * jacobians[i];
        gradient += w * jacobians[i].transpose() * r;
        ++points;
      }
    }
    // Error of the current iterate from the prediction.
    Vec15 error;
    error.segment<3>(0) = So3Log(rotation_prior.transpose() * rotation_);
    error.segment<3>(3) = position_ - position_prior;
    error.segment<3>(6) = velocity_ - velocity_prior;
    error.segment<3>(9) = gyro_bias_ - gyro_bias_prior;
    error.segment<3>(12) = accel_bias_ - accel_bias_prior;
    information = prior_information;
    information.block<6, 6>(0, 0) += hessian;
    Vec15 rhs = -prior_information * error;
    rhs.segment<6>(0) -= gradient;
    if (config_.max_speed > 0.0 && velocity_prior.norm() > config_.max_speed) {
      // Pulled back to the speed limit, along the predicted direction.
      const Vec3 limit =
          velocity_prior * (config_.max_speed / velocity_prior.norm());
      const double ws = 1.0 / (config_.speed_sigma * config_.speed_sigma);
      information.block<3, 3>(6, 6) += ws * Mat3::Identity();
      rhs.segment<3>(6) -= ws * (velocity_ - limit);
    }
    if (stationary) {  // Velocity measured as zero.
      const double wv = 1.0 / (config_.zupt_sigma * config_.zupt_sigma);
      information.block<3, 3>(6, 6) += wv * Mat3::Identity();
      rhs.segment<3>(6) -= wv * velocity_;
    }
    const Vec15 step = information.ldlt().solve(rhs);
    rotation_ = Orthonormal(rotation_ * So3Exp(step.segment<3>(0)));
    position_ += step.segment<3>(3);
    velocity_ += step.segment<3>(6);
    gyro_bias_ += step.segment<3>(9);
    accel_bias_ += step.segment<3>(12);
    stats_.iterations = iteration + 1;
    stats_.correspondences = count;
    stats_.point_residuals = points;
    stats_.rmse = count > 0 ? std::sqrt(sse / count) : 0.0;
    const Eigen::SelfAdjointEigenSolver<Mat3> eigen(
        hessian.block<3, 3>(3, 3));
    stats_.min_translation_information = eigen.eigenvalues()(0);
    if (step.segment<6>(0).norm() < config_.convergence) break;
  }
  covariance_ = information.inverse();
  covariance_ = 0.5 * (covariance_ + covariance_.transpose());
  stats_.stationary = stationary;
  stats_.velocity = velocity_;
}

bool LioOdometry::Stationary(double t) const {
  if (config_.zupt_sigma <= 0.0) return false;
  const auto begin = std::lower_bound(imu_.stamps.begin(), imu_.stamps.end(),
                                      t - config_.zupt_window);
  const auto end = std::upper_bound(imu_.stamps.begin(), imu_.stamps.end(), t);
  const size_t a = static_cast<size_t>(begin - imu_.stamps.begin());
  const size_t b = static_cast<size_t>(end - imu_.stamps.begin());
  if (b < a + 5) return false;
  Vec3 mean = Vec3::Zero();
  for (size_t i = a; i < b; ++i) {
    if ((imu_.gyro[i] - gyro_bias_).norm() > config_.zupt_max_gyro) {
      return false;
    }
    mean += imu_.accel[i];
  }
  mean /= static_cast<double>(b - a);
  for (size_t i = a; i < b; ++i) {
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
    for (double t : times) first = std::min(first, stamp + t);
    BackfillFirstSweep(first, stamp);
    initialized_ = true;
  } else {
    Propagate(stamp);
  }
  deskewed_ = Deskew(points, times, stamp);

  Points kept;
  kept.reserve(deskewed_.size());
  for (const Vec3& p : deskewed_) {
    const double range = p.norm();
    if (range >= config_.min_range && range <= config_.max_range) {
      kept.push_back(p);
    }
  }
  Points body = VoxelDownsample(kept, config_.scan_voxel);
  for (Vec3& p : body) p = body_from_lidar_ * p;
  if (!map_.empty()) Update(body, Stationary(stamp));

  Points insert = VoxelDownsample(kept, config_.insert_voxel);
  const Pose world_from_lidar = Pose(rotation_, position_) * body_from_lidar_;
  for (Vec3& p : insert) p = world_from_lidar * p;
  map_.AddPoints(insert);
  map_.RemoveFar(position_);
  return world_from_lidar;
}

}  // namespace gr
