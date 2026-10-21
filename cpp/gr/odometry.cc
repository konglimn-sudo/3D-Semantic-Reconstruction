#include "gr/odometry.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace gr {

Points DeskewConstantVelocity(const Points& points,
                              const std::vector<double>& times,
                              const Vec3& angular_velocity,
                              const Vec3& linear_velocity) {
  if (times.size() != points.size()) return points;
  Points out(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    const double tau = times[i];
    out[i] = So3Exp(angular_velocity * tau) * points[i] + linear_velocity * tau;
  }
  return out;
}

Odometry::Odometry(const OdometryConfig& config, const Pose& initial_pose)
    : config_(config),
      initial_pose_(initial_pose),
      map_(config.voxel_size, config.max_range, config.max_points_per_voxel) {}

double Odometry::AdaptiveThreshold() const {
  if (num_samples_ < 1) return config_.initial_threshold;
  return std::sqrt(model_sse_ / num_samples_);
}

Pose Odometry::RegisterFrame(const Points& points,
                             const std::vector<double>& times, double stamp,
                             const std::optional<Pose>& predicted_motion) {
  // Velocity / prediction from the prior or from the last two poses (body
  // frame).
  Vec3 angular_velocity = Vec3::Zero();
  Vec3 linear_velocity = Vec3::Zero();
  Pose predicted;  // Identity.
  const double dt_new = stamps_.empty() ? 0.0 : stamp - stamps_.back();
  if (predicted_motion && dt_new > 1e-6) {
    predicted = *predicted_motion;
    angular_velocity = So3Log(predicted.rotation) / dt_new;
    linear_velocity = predicted.translation / dt_new;
  } else if (poses_.size() >= 2) {
    const size_t n = poses_.size();
    const double dt = stamps_[n - 1] - stamps_[n - 2];
    if (dt > 1e-6) {
      const Pose delta = poses_[n - 2].Inverse() * poses_[n - 1];
      angular_velocity = So3Log(delta.rotation) / dt;
      linear_velocity = delta.translation / dt;
      predicted =
          Pose(So3Exp(angular_velocity * dt_new), linear_velocity * dt_new);
    }
  }

  // Deskew and range crop.
  const Points deskewed =
      (config_.deskew && !times.empty())
          ? DeskewConstantVelocity(points, times, angular_velocity,
                                   linear_velocity)
          : points;
  Points cropped;
  cropped.reserve(deskewed.size());
  const double min_r2 = config_.min_range * config_.min_range;
  const double max_r2 = config_.max_range * config_.max_range;
  for (const Vec3& p : deskewed) {
    const double r2 = p.squaredNorm();
    if (r2 >= min_r2 && r2 <= max_r2) cropped.push_back(p);
  }
  last_deskewed_ = cropped;

  const Points frame = VoxelDownsample(cropped, 0.5 * config_.voxel_size);
  const Points source = VoxelDownsample(frame, 1.5 * config_.voxel_size);
  const Pose guess = poses_.empty() ? initial_pose_ : poses_.back() * predicted;
  Pose pose = guess;

  const double sigma = AdaptiveThreshold();
  last_sigma_ = sigma;
  if (!map_.empty()) {
    IcpOptions options;
    options.max_corr_dist = 3.0 * sigma;
    options.kernel = sigma / 3.0;
    options.max_iters = config_.max_icp_iters;
    options.conv_eps = config_.icp_conv;
    last_icp_ = AlignToMap(source, map_, guess, options);
    pose = last_icp_.pose;

    // Model deviation for the adaptive threshold.
    const Pose deviation = guess.Inverse() * pose;
    const double cos_angle =
        std::clamp((deviation.rotation.trace() - 1.0) / 2.0, -1.0, 1.0);
    const double angle = std::acos(cos_angle);
    const double error = 2.0 * config_.max_range * std::sin(angle / 2.0) +
                         deviation.translation.norm();
    if (error > config_.min_motion_th) {
      model_sse_ += error * error;
      ++num_samples_;
    }
  }

  map_.AddPoints(TransformPoints(frame, pose));
  map_.RemoveFar(pose.translation);
  poses_.push_back(pose);
  stamps_.push_back(stamp);
  return pose;
}

OdometryPipeline::OdometryPipeline(const OdometryConfig& config,
                                   const KeyframeConfig& keyframe_config,
                                   const Pose& vehicle_from_lidar,
                                   const Pose& world_from_vehicle0,
                                   MotionPrior prior)
    : config_(config),
      keyframe_config_(keyframe_config),
      vehicle_from_lidar_(vehicle_from_lidar),
      lidar_from_vehicle_(vehicle_from_lidar.Inverse()),
      prior_(std::move(prior)),
      odometry_(config, world_from_vehicle0 * vehicle_from_lidar) {
  use_prior_ = config_.use_wheel_prior && prior_.valid();
  use_gyro_ = !use_prior_ && config_.use_gyro_prior && prior_.has_gyro();
  if (config_.use_lio && prior_.has_gyro()) {
    LioConfig lio = config_.lio;
    lio.min_range = config_.min_range;
    lio.max_range = config_.max_range;
    lio_.emplace(lio, prior_.imu(), vehicle_from_lidar, world_from_vehicle0);
    use_prior_ = use_gyro_ = false;
  }
}

bool OdometryPipeline::IsKeyframe(const Pose& pose) const {
  const Pose relative = last_keyframe_pose_.Inverse() * pose;
  return relative.translation.norm() >= keyframe_config_.distance ||
         RadToDeg(RotationAngle(relative.rotation)) >= keyframe_config_.angle;
}

void OdometryPipeline::FillKeyframe(const Points& points,
                                    const std::vector<float>& intensity,
                                    const std::vector<double>& times,
                                    double stamp,
                                    std::optional<Points> imu_deskewed,
                                    OdometryFrame& frame) const {
  Points sweep;
  if (imu_deskewed) {
    sweep = std::move(*imu_deskewed);
  } else if (config_.deskew && !poses_.empty() &&
             times.size() == points.size()) {
    // Constant velocity from the registered motion.
    const Pose delta = poses_.back().Inverse() * frame.pose;
    const double dt = stamp - stamps_.back();
    sweep = dt > 1e-6 ? DeskewConstantVelocity(points, times,
                                               So3Log(delta.rotation) / dt,
                                               delta.translation / dt)
                      : points;
  } else {
    sweep = points;
  }
  Points kept;
  std::vector<float> kept_intensity;
  kept.reserve(sweep.size());
  kept_intensity.reserve(sweep.size());
  for (size_t i = 0; i < sweep.size(); ++i) {
    const double r = sweep[i].norm();
    if (r < config_.min_range || r > config_.max_range) continue;
    kept.push_back(sweep[i]);
    kept_intensity.push_back(i < intensity.size() ? intensity[i] : 0.0f);
  }
  const std::vector<int> indices =
      VoxelDownsampleIndices(kept, keyframe_config_.voxel);
  frame.points.reserve(indices.size());
  frame.intensity.reserve(indices.size());
  for (int i : indices) {
    frame.points.push_back(kept[static_cast<size_t>(i)]);
    frame.intensity.push_back(kept_intensity[static_cast<size_t>(i)]);
  }
}

OdometryFrame OdometryPipeline::Process(
    const Points& points, const std::vector<float>& intensity,
    const std::vector<double>& times, double stamp, bool force_keyframe,
    const std::optional<Pose>& external_pose) {
  const size_t k = poses_.size();
  // Wheel / gyro prediction of the motion since the previous sweep (LiDAR
  // frame).
  std::optional<Pose> predicted;
  if (use_prior_ && k > 0) {
    if (const std::optional<Pose> delta = prior_.Delta(stamps_.back(), stamp)) {
      predicted = lidar_from_vehicle_ * *delta * vehicle_from_lidar_;
    }
  }
  // Without wheels: gyro rotation, velocity of the last two sweeps (constant
  // in the world, zero before), in the vehicle frame at this sweep. The
  // first sweep is compensated too: everything after registers to it.
  std::optional<Vec3> velocity;
  if (use_gyro_) {
    Vec3 world_velocity = Vec3::Zero();
    Mat3 world_from_previous = Mat3::Identity();
    if (k > 0) {
      const Pose previous = poses_[k - 1] * lidar_from_vehicle_;
      world_from_previous = previous.rotation;
      if (k > 1 && stamps_[k - 1] - stamps_[k - 2] > 1e-6) {
        const Pose before = poses_[k - 2] * lidar_from_vehicle_;
        world_velocity = (previous.translation - before.translation) /
                         (stamps_[k - 1] - stamps_[k - 2]);
      }
    }
    const std::optional<Pose> turn =
        k > 0 ? prior_.GyroDelta(stamps_.back(), stamp, Vec3::Zero())
              : prior_.GyroDelta(stamp - 0.1, stamp, Vec3::Zero());
    if (turn) {
      velocity =
          (world_from_previous * turn->rotation).transpose() * world_velocity;
      if (k > 0) {
        predicted = lidar_from_vehicle_ *
                    *prior_.GyroDelta(stamps_.back(), stamp, *velocity) *
                    vehicle_from_lidar_;
      }
    }
  }
  // In-sweep motion from the IMU: the mean angular rate between two sweeps
  // differs strongly from the rate during the sweep when turning.
  std::optional<Points> imu_deskewed;
  if (config_.deskew && (use_prior_ || velocity)) {
    imu_deskewed = DeskewWithPrior(points, times, stamp, prior_,
                                   vehicle_from_lidar_, 16, velocity);
  }

  OdometryFrame frame;
  if (lio_) {
    const Pose pose = lio_->Process(points, times, stamp);
    frame.pose = external_pose ? *external_pose : pose;
    imu_deskewed = lio_->last_deskewed();
  } else if (external_pose) {
    frame.pose = *external_pose;
  } else if (imu_deskewed) {  // Compensated: no constant-velocity deskew.
    frame.pose = odometry_.RegisterFrame(*imu_deskewed, {}, stamp, predicted);
  } else {
    frame.pose = odometry_.RegisterFrame(points, times, stamp, predicted);
  }

  frame.is_keyframe = k == 0 || force_keyframe || IsKeyframe(frame.pose);
  if (frame.is_keyframe) {
    FillKeyframe(points, intensity, times, stamp, std::move(imu_deskewed),
                 frame);
    last_keyframe_pose_ = frame.pose;
    keyframes_.push_back(static_cast<int>(k));
  }
  poses_.push_back(frame.pose);
  stamps_.push_back(stamp);
  return frame;
}

}  // namespace gr
