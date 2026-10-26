#include "gr/nav/localization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Eigenvalues>

#include "gr/core/check.h"
#include "gr/core/voxel.h"
#include "gr/odometry/odometry.h"

namespace gr {
namespace {

// Keeps the components of `delta` along the eigenvectors of `block` whose
// eigenvalue reaches `ratio` of the largest; returns the number dropped and
// the smallest eigenvalue ratio.
struct Projection {
  Vec3 delta;
  int dropped = 0;
  double min_ratio = 0.0;
  Mat3 degenerate = Mat3::Zero();  // Sum of v v^T over dropped directions.
};

Projection ProjectWellConstrained(const Mat3& block, const Vec3& delta,
                                  double ratio) {
  Eigen::SelfAdjointEigenSolver<Mat3> solver(block);
  const Vec3 values = solver.eigenvalues().cwiseMax(0.0);
  const double largest = values.maxCoeff();
  Projection out;
  out.delta = Vec3::Zero();
  out.min_ratio = largest > 0.0 ? values.minCoeff() / largest : 0.0;
  for (int i = 0; i < 3; ++i) {
    const Vec3 v = solver.eigenvectors().col(i);
    if (largest > 0.0 && values(i) >= ratio * largest) {
      out.delta += v * v.dot(delta);
    } else {
      ++out.dropped;
      out.degenerate += v * v.transpose();
    }
  }
  return out;
}

// The registration's correction of `from`, about the sensor: the sensor
// moves by dt and turns by dphi (world axes) about its own position,
// icp.pose = Pose(Exp(dphi) R, t + dt). The ICP Hessian is for the left
// perturbation about the world origin, x0 = A x with A = [I [t]x; 0 I];
// about the sensor it is A^T H A. Steps along directions the sweep does not
// constrain are dropped.
struct ConstrainedStep {
  Pose pose;
  Projection t;
  Projection r;
  Mat6 hessian;  // About the sensor, order [dt, dphi].
};

ConstrainedStep ConstrainStep(const Pose& from, const IcpResult& icp,
                              double ratio) {
  Mat6 a = Mat6::Identity();
  a.topRightCorner<3, 3>() = Hat(from.translation);
  ConstrainedStep out;
  out.hessian = a.transpose() * icp.hessian * a;
  const Vec3 dt = icp.pose.translation - from.translation;
  const Vec3 dphi = So3Log(icp.pose.rotation * from.rotation.transpose());
  out.t = ProjectWellConstrained(out.hessian.topLeftCorner<3, 3>(), dt, ratio);
  out.r = ProjectWellConstrained(out.hessian.bottomRightCorner<3, 3>(), dphi,
                                 ratio);
  out.pose = Pose(So3Exp(out.r.delta) * from.rotation,
                  from.translation + out.t.delta);
  return out;
}

}  // namespace

Localizer::Localizer(const Points& map_points, const LocalizerConfig& config)
    : config_(config),
      map_points_(VoxelDownsample(map_points, config.map_voxel)),
      coarse_map_(VoxelDownsample(map_points, config.placement.voxel)),
      map_(config.hash_voxel, 1e9, 40) {
  GR_CHECK(!config_.gates.empty());
  map_.AddPoints(map_points_);
}

void Localizer::SetPrior(const Pose& map_from_sensor) {
  prior_ = map_from_sensor;
  status_ = LocalizationStatus::kInitializing;
  failures_ = 0;
}

Points Localizer::PrepareSweep(const Points& points,
                               const std::vector<double>& times, double stamp,
                               const Pose& odom_from_sensor) const {
  Points sweep;
  if (config_.deskew && last_odom_ && times.size() == points.size()) {
    const Pose delta = last_odom_->Inverse() * odom_from_sensor;
    const double dt = stamp - last_stamp_;
    if (dt > 1e-6 && dt < 1.0) {
      sweep = DeskewConstantVelocity(points, times,
                                     So3Log(delta.rotation) / dt,
                                     delta.translation / dt);
    }
  }
  if (sweep.empty()) sweep = points;
  Points kept;
  kept.reserve(sweep.size());
  for (const Vec3& p : sweep) {
    const double r = p.norm();
    if (r >= config_.min_range && r <= config_.max_range) kept.push_back(p);
  }
  return VoxelDownsample(kept, config_.scan_voxel);
}

void Localizer::AddToSubmap(const Points& sweep,
                            const Pose& odom_from_sensor) {
  if (!submap_.empty() &&
      travelled_ - submap_.back().travelled < config_.init_spacing) {
    return;
  }
  submap_.push_back({odom_from_sensor,
                     VoxelDownsample(sweep, config_.init_voxel), travelled_});
  while (submap_.size() > 1 &&
         travelled_ - submap_.front().travelled > config_.init_max_distance) {
    submap_.pop_front();
  }
}

bool Localizer::TryPrior(const Points& sweep, const Pose& odom_from_sensor,
                         LocalizationResult& result) {
  const Pose guess = *prior_;
  prior_.reset();
  const IcpResult icp =
      AlignMultiScale(sweep, map_points_, guess, config_.prior_schedule,
                      config_.init_voxel, config_.icp_iters);
  result.fitness = icp.fitness;
  result.rmse = icp.rmse;
  if (icp.fitness < config_.min_fitness || icp.rmse > config_.max_rmse) {
    return false;
  }
  // In a featureless corridor the prior stands along the corridor axis.
  const ConstrainedStep step =
      ConstrainStep(guess, icp, config_.degenerate_ratio);
  map_from_odom_ = step.pose * odom_from_sensor.Inverse();
  return true;
}

bool Localizer::TryGlobal(const Points& sweep, const Pose& odom_from_sensor,
                          LocalizationResult& result) {
  if (submap_.empty() ||
      travelled_ - submap_.front().travelled < config_.init_distance ||
      travelled_ - last_attempt_ < config_.init_retry) {
    return false;
  }
  last_attempt_ = travelled_;
  Points merged;
  for (const SubmapSweep& s : submap_) {
    const Points odom = TransformPoints(s.points, s.odom_from_sensor);
    merged.insert(merged.end(), odom.begin(), odom.end());
  }
  ++placement_attempts_;
  last_placement_ = PlaceGravityAligned(
      VoxelDownsample(merged, config_.placement.voxel), coarse_map_,
      config_.placement);
  result.fitness = last_placement_.fitness;
  result.rmse = last_placement_.rmse;
  if (!last_placement_.placed) return false;
  map_from_odom_ = last_placement_.reference_from_source;
  return true;
}

void Localizer::Track(const Points& sweep, const Pose& odom_from_sensor,
                      LocalizationResult& result) {
  const Pose predicted = map_from_odom_ * odom_from_sensor;
  IcpOptions options;
  options.max_iters = config_.icp_iters;
  IcpResult icp;
  icp.pose = predicted;
  for (double gate : config_.gates) {
    options.max_corr_dist = gate;
    options.kernel = gate / 3.0;
    icp = AlignToMap(sweep, map_, icp.pose, options);
  }
  const ConstrainedStep step =
      ConstrainStep(predicted, icp, config_.degenerate_ratio);
  const Projection& t = step.t;
  const Projection& r = step.r;
  const Pose& corrected = step.pose;
  result.fitness = icp.fitness;
  result.rmse = icp.rmse;
  result.degenerate_t = t.dropped;
  result.degenerate_r = r.dropped;
  result.min_eigen_ratio_t = t.min_ratio;
  const bool fits = icp.inliers > 0 && icp.fitness >= config_.min_fitness &&
                    icp.rmse <= config_.max_rmse;
  result.step_t = t.delta.norm();
  result.step_r = RadToDeg(r.delta.norm());
  const bool jumps =
      result.step_t > config_.max_jump_t || result.step_r > config_.max_jump_r;
  if (fits && !jumps) {
    map_from_odom_ = corrected * odom_from_sensor.Inverse();
    failures_ = 0;
    result.registered = true;
    result.pose = corrected;
    const double sigma = std::max(icp.rmse, 0.01);
    const double regularizer = 1e-9 * std::max(step.hessian.trace(), 1.0);
    Mat6 covariance =
        sigma * sigma *
        (step.hessian + regularizer * Mat6::Identity()).inverse();
    // Degenerate directions keep the prediction: report the odometry's
    // uncertainty there instead of the (meaningless) ICP one.
    for (int block = 0; block < 2; ++block) {
      const Mat3& dropped = block == 0 ? t.degenerate : r.degenerate;
      if (dropped.isZero()) continue;
      const double s = block == 0 ? config_.degenerate_sigma_t
                                  : DegToRad(config_.degenerate_sigma_r);
      const Mat3 kept = Mat3::Identity() - dropped;
      Mat3 b = covariance.block<3, 3>(3 * block, 3 * block);
      b = kept * b * kept + s * s * dropped;
      covariance.block<3, 3>(3 * block, 3 * block) = b;
    }
    result.covariance = covariance;
  } else {
    ++failures_;
    result.pose = predicted;
    if (failures_ >= config_.max_failures) {
      status_ = LocalizationStatus::kLost;
      last_attempt_ = -std::numeric_limits<double>::infinity();
    }
  }
}

LocalizationResult Localizer::Process(const Points& points,
                                      const std::vector<double>& times,
                                      double stamp,
                                      const Pose& odom_from_sensor) {
  LocalizationResult result;
  const Points sweep = PrepareSweep(points, times, stamp, odom_from_sensor);
  if (last_odom_) {
    travelled_ += (odom_from_sensor.translation - last_odom_->translation)
                      .norm();
  }
  last_odom_ = odom_from_sensor;
  last_stamp_ = stamp;
  AddToSubmap(sweep, odom_from_sensor);
  if (status_ != LocalizationStatus::kTracking) {
    const bool placed = prior_ ? TryPrior(sweep, odom_from_sensor, result)
                               : TryGlobal(sweep, odom_from_sensor, result);
    if (!placed) {
      result.status = status_;
      result.failures = failures_;
      if (status_ == LocalizationStatus::kLost) {
        result.pose = map_from_odom_ * odom_from_sensor;
      }
      return result;
    }
    // Verified by the current sweep before it is trusted.
    const LocalizationStatus before = status_;
    const Pose previous = map_from_odom_;
    status_ = LocalizationStatus::kTracking;
    failures_ = 0;
    Track(sweep, odom_from_sensor, result);
    if (!result.registered) {
      status_ = before;
      failures_ = 0;
      map_from_odom_ = previous;
      result.status = status_;
      result.pose = Pose();
      return result;
    }
    result.initialized = true;
    result.status = status_;
    result.failures = failures_;
    return result;
  }
  Track(sweep, odom_from_sensor, result);
  result.status = status_;
  result.failures = failures_;
  return result;
}

}  // namespace gr
