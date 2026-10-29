#include "gr/odometry/odometry.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace gr {
namespace {

// Intervals shorter than this (s) give no velocity.
constexpr double kMinInterval = 1e-6;
// KISS-ICP: each sweep is downsampled to half a map voxel for the map and
// to 1.5 voxels for registration; correspondences reach 3 sigma of the
// motion model and the robust kernel scale is sigma / 3.
constexpr double kFrameVoxelFactor = 0.5;
constexpr double kSourceVoxelFactor = 1.5;
constexpr double kCorrespondenceSigmas = 3.0;
constexpr double kKernelSigmaDivisor = 3.0;
// The first sweep of a gyro-only rig is compensated over this span (s).
constexpr double kFirstSweepDuration = 0.1;

}  // namespace

Points DeskewConstantVelocity(const Points& points,
                              const std::vector<double>& times,
                              const Vec3& angular_velocity,
                              const Vec3& linear_velocity) {
  if (times.size() != points.size()) return points;
  Points out(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    const double time = times[i];
    out[i] =
        So3Exp(angular_velocity * time) * points[i] + linear_velocity * time;
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
  const double interval = stamps_.empty() ? 0.0 : stamp - stamps_.back();
  if (predicted_motion && interval > kMinInterval) {
    predicted = *predicted_motion;
    angular_velocity = So3Log(predicted.rotation) / interval;
    linear_velocity = predicted.translation / interval;
  } else if (poses_.size() >= 2) {
    const size_t last = poses_.size() - 1;
    const double last_interval = stamps_[last] - stamps_[last - 1];
    if (last_interval > kMinInterval) {
      const Pose delta = poses_[last - 1].Inverse() * poses_[last];
      angular_velocity = So3Log(delta.rotation) / last_interval;
      linear_velocity = delta.translation / last_interval;
      predicted =
          Pose(So3Exp(angular_velocity * interval), linear_velocity * interval);
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
  const double min_squared_range = config_.min_range * config_.min_range;
  const double max_squared_range = config_.max_range * config_.max_range;
  for (const Vec3& point : deskewed) {
    const double squared_range = point.squaredNorm();
    if (squared_range >= min_squared_range &&
        squared_range <= max_squared_range) {
      cropped.push_back(point);
    }
  }
  last_deskewed_ = cropped;

  const Points frame =
      VoxelDownsample(cropped, kFrameVoxelFactor * config_.voxel_size);
  const Points source =
      VoxelDownsample(frame, kSourceVoxelFactor * config_.voxel_size);
  const Pose guess = poses_.empty() ? initial_pose_ : poses_.back() * predicted;
  Pose pose = guess;

  const double sigma = AdaptiveThreshold();
  last_sigma_ = sigma;
  if (!map_.empty()) {
    IcpOptions options;
    options.max_corr_dist = kCorrespondenceSigmas * sigma;
    options.kernel = sigma / kKernelSigmaDivisor;
    options.max_iters = config_.max_icp_iters;
    options.conv_eps = config_.icp_conv;
    last_icp_ = AlignToMap(source, map_, guess, options);
    pose = last_icp_.pose;

    // Model deviation for the adaptive threshold: the rotation angle
    // (cos = (trace - 1) / 2) moves a point at max_range by the chord
    // 2 max_range sin(angle / 2).
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
                                    OdometryFrame* frame) const {
  Points sweep;
  if (imu_deskewed) {
    sweep = std::move(*imu_deskewed);
  } else if (config_.deskew && !poses_.empty() &&
             times.size() == points.size()) {
    // Constant velocity from the registered motion.
    const Pose delta = poses_.back().Inverse() * frame->pose;
    const double interval = stamp - stamps_.back();
    sweep = interval > kMinInterval
                ? DeskewConstantVelocity(points, times,
                                         So3Log(delta.rotation) / interval,
                                         delta.translation / interval)
                : points;
  } else {
    sweep = points;
  }
  Points kept;
  std::vector<float> kept_intensity;
  kept.reserve(sweep.size());
  kept_intensity.reserve(sweep.size());
  for (size_t i = 0; i < sweep.size(); ++i) {
    const double range = sweep[i].norm();
    if (range < config_.min_range || range > config_.max_range) continue;
    kept.push_back(sweep[i]);
    kept_intensity.push_back(i < intensity.size() ? intensity[i] : 0.0f);
  }
  const std::vector<int> indices =
      VoxelDownsampleIndices(kept, keyframe_config_.voxel);
  frame->points.reserve(indices.size());
  frame->intensity.reserve(indices.size());
  for (int index : indices) {
    frame->points.push_back(kept[static_cast<size_t>(index)]);
    frame->intensity.push_back(kept_intensity[static_cast<size_t>(index)]);
  }
}

OdometryFrame OdometryPipeline::Process(
    const Points& points, const std::vector<float>& intensity,
    const std::vector<double>& times, double stamp, bool force_keyframe,
    const std::optional<Pose>& external_pose, bool fill_sweep) {
  const size_t sweep_index = poses_.size();
  // Wheel / gyro prediction of the motion since the previous sweep (LiDAR
  // frame).
  std::optional<Pose> predicted;
  if (use_prior_ && sweep_index > 0) {
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
    if (sweep_index > 0) {
      const size_t previous_index = sweep_index - 1;
      const Pose previous = poses_[previous_index] * lidar_from_vehicle_;
      world_from_previous = previous.rotation;
      if (sweep_index > 1 &&
          stamps_[previous_index] - stamps_[previous_index - 1] >
              kMinInterval) {
        const Pose before = poses_[previous_index - 1] * lidar_from_vehicle_;
        world_velocity =
            (previous.translation - before.translation) /
            (stamps_[previous_index] - stamps_[previous_index - 1]);
      }
    }
    const std::optional<Pose> turn =
        sweep_index > 0 ? prior_.GyroDelta(stamps_.back(), stamp, Vec3::Zero())
                        : prior_.GyroDelta(stamp - kFirstSweepDuration, stamp,
                                           Vec3::Zero());
    if (turn) {
      velocity =
          (world_from_previous * turn->rotation).transpose() * world_velocity;
      if (sweep_index > 0) {
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
    imu_deskewed =
        DeskewWithPrior(points, times, stamp, prior_, vehicle_from_lidar_,
                        kDefaultDeskewSamples, velocity);
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

  frame.is_keyframe =
      sweep_index == 0 || force_keyframe || IsKeyframe(frame.pose);
  if (frame.is_keyframe || fill_sweep) {
    FillKeyframe(points, intensity, times, stamp, std::move(imu_deskewed),
                 &frame);
  }
  if (frame.is_keyframe) {
    last_keyframe_pose_ = frame.pose;
    keyframes_.push_back(static_cast<int>(sweep_index));
  }
  poses_.push_back(frame.pose);
  stamps_.push_back(stamp);
  return frame;
}

}  // namespace gr
