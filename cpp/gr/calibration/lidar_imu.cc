#include "gr/calibration/lidar_imu.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace gr {
namespace {

// The entries of a 3 x 3 matrix, column by column (vec(A)).
constexpr int kMatrixEntries = kSpaceDim * kSpaceDim;

// One window around a LiDAR stamp, for a time offset of the IMU.
//
// Gyro: the hat-weighted mean of the IMU rates against the same mean of the
// LiDAR rates (piecewise constant between sweeps), so that an IMU sample
// entering or leaving the window as the offset changes has no weight.
//
// Accelerometer: the divided second difference of the LiDAR positions is
// the hat-weighted mean of the world acceleration over the window,
//   a_W = sum_i w_i (R_WL(t_i) A f_i) - B A b_a + g_W,
// with A = R^T diag(1 / scale) and B = sum_i w_i R_WL(t_i): linear in A;
// K = sum_i w_i (f_i^T kron R_WL(t_i)) gives the first sum as K vec(A).
// The IMU sits at r (LiDAR axes) and also reads the rotation's
// acceleration there: the mean adds M r with
// M = sum_i w_i R_WL(t_i) ([alpha_i]x + [w_i]x^2), the rates from the gyro
// (in LiDAR axes with the rig's rotation; a correction term).
struct Sample {
  Vec3 lidar_rate = Vec3::Zero();
  Vec3 imu_rate = Vec3::Zero();
  Vec3 world_acceleration = Vec3::Zero();  // a_W of the LiDAR positions.
  Eigen::Matrix<double, kSpaceDim, kMatrixEntries> force_sum =
      Eigen::Matrix<double, kSpaceDim, kMatrixEntries>::Zero();  // K
  Mat3 attitude_sum = Mat3::Zero();                              // B
  Mat3 lever_matrix = Mat3::Zero();                              // M
};

// The parameters: the rotation correction, the gyro and accelerometer
// biases, the two tilt angles of gravity, the accelerometer scales (when
// estimated), then the lever arm and the time offset (when estimated).
constexpr int kTiltAngles = 2;
constexpr int kRotationParam = 0;
constexpr int kGyroBiasParam = kRotationParam + kSpaceDim;
constexpr int kAccelBiasParam = kGyroBiasParam + kSpaceDim;
constexpr int kGravityParam = kAccelBiasParam + kSpaceDim;
constexpr int kScaleParam = kGravityParam + kTiltAngles;

// The residuals of a window: gyro, then accelerometer.
constexpr int kGyroResidual = 0;
constexpr int kAccelResidual = kGyroResidual + kSpaceDim;
constexpr int kResidualsPerWindow = kAccelResidual + kSpaceDim;

// The sweep period is at least this (s); windows keep this much (s) more
// than the time offset from the ends of the IMU record.
constexpr double kMinSweepPeriod = 1e-3;
constexpr double kWindowMargin = 0.05;
// LiDAR rates use intervals of at least this (s).
constexpr double kMinInterval = 1e-6;
// Central-difference steps of the time offset (s) and the others.
constexpr double kTimeOffsetStep = 1e-4;
constexpr double kParameterStep = 1e-6;
// Window samples of at most this many offsets are kept.
constexpr size_t kMaxCachedOffsets = 16;
// Window weights sum to at least this.
constexpr double kMinWeightSum = 1e-12;
// Calibration needs ten windows.
constexpr size_t kMinWindows = 10;
// Levenberg-Marquardt: the damping's start, decrease, increase and floor,
// the attempts per iteration, and the cost and step below which the solve
// has converged.
constexpr double kInitialDamping = 1e-3;
constexpr double kDampingDecrease = 3.0;
constexpr double kDampingIncrease = 5.0;
constexpr double kMinDamping = 1e-9;
constexpr double kMinDampedDiagonal = 1e-9;
constexpr int kMaxDampingAttempts = 10;
constexpr double kMinCost = 1e-12;
constexpr double kMinStep = 1e-12;

class Problem {
 public:
  // time_index() when the time offset is not estimated.
  static constexpr int kNoParameter = -1;

  Problem(const std::vector<double>& stamps, const std::vector<Pose>& poses,
          const ImuSamples& imu, const Pose& initial,
          const LidarImuConfig& config)
      : stamps_(stamps),
        poses_(poses),
        imu_(imu),
        initial_(initial.rotation),
        lever_arm_(-(initial.rotation.transpose() * initial.translation)),
        config_(config) {
    // Windows on the LiDAR stamps (positions not interpolated): `reach`
    // sweeps to either side, about `half_window` seconds, far enough from
    // the ends of the IMU record for any offset tried.
    std::vector<double> steps;
    for (size_t k = 1; k < stamps.size(); ++k) {
      steps.push_back(stamps[k] - stamps[k - 1]);
    }
    std::nth_element(steps.begin(), steps.begin() + steps.size() / 2,
                     steps.end());
    const double period = std::max(steps[steps.size() / 2], kMinSweepPeriod);
    reach_ = static_cast<size_t>(
        std::max(1.0, std::round(config.half_window / period)));
    const double margin = config.max_time_offset + kWindowMargin;
    for (size_t k = reach_; k + reach_ < stamps.size();
         k += static_cast<size_t>(std::max(config.stride, 1))) {
      if (stamps[k - reach_] - margin >= imu.stamps.front() &&
          stamps[k + reach_] + margin <= imu.stamps.back()) {
        windows_.push_back(k);
      }
    }
    // LiDAR rates between consecutive sweeps (body axes).
    for (size_t k = 0; k + 1 < stamps.size(); ++k) {
      rates_.push_back(
          So3Log(poses[k].rotation.transpose() * poses[k + 1].rotation) /
          std::max(stamps[k + 1] - stamps[k], kMinInterval));
    }
  }

  int num_parameters() const {
    return lever_index() + (config_.estimate_lever_arm ? kSpaceDim : 0) +
           (config_.estimate_time_offset ? 1 : 0);
  }
  int lever_index() const {
    return kScaleParam + (config_.estimate_scale ? kSpaceDim : 0);
  }
  // The time offset's parameter; kNoParameter when it is not estimated.
  int time_index() const {
    return config_.estimate_time_offset ? num_parameters() - 1 : kNoParameter;
  }
  size_t size() const { return windows_.size(); }
  // Residuals: kResidualsPerWindow per window, then the lever arm prior.
  int num_residuals() const {
    return kResidualsPerWindow * static_cast<int>(size()) +
           (config_.estimate_lever_arm ? kSpaceDim : 0);
  }

  Mat3 Rotation(const Eigen::VectorXd& x) const {
    return initial_ * So3Exp(x.segment<kSpaceDim>(kRotationParam));
  }
  Vec3 Gravity(const Eigen::VectorXd& x) const {
    return So3Exp(Vec3(x[kGravityParam], x[kGravityParam + 1], 0.0)) *
           Vec3(0.0, 0.0, -config_.gravity);
  }
  // 1 / scale of the accelerometer axes.
  Vec3 InverseScale(const Eigen::VectorXd& x) const {
    return config_.estimate_scale
               ? Vec3(Vec3::Ones() + x.segment<kSpaceDim>(kScaleParam))
               : Vec3(Vec3::Ones());
  }
  // The IMU origin in LiDAR axes.
  Vec3 LeverArm(const Eigen::VectorXd& x) const {
    return config_.estimate_lever_arm
               ? Vec3(lever_arm_ + x.segment<kSpaceDim>(lever_index()))
               : lever_arm_;
  }
  // IMU time = LiDAR time + offset.
  double TimeOffset(const Eigen::VectorXd& x) const {
    return config_.estimate_time_offset ? x[time_index()] : 0.0;
  }

  // Whitened residuals: per window [gyro, accel].
  Eigen::VectorXd Residuals(const Eigen::VectorXd& x) const {
    const std::vector<Sample>& samples = SamplesAt(TimeOffset(x));
    const Mat3 rotation = Rotation(x);
    const Vec3 gravity = Gravity(x);
    // A = R^T diag(1 / scale): raw specific force -> LiDAR axes.
    const Mat3 force_to_lidar =
        rotation.transpose() * InverseScale(x).asDiagonal();
    const Eigen::Map<const Eigen::Matrix<double, kMatrixEntries, 1>>
        force_to_lidar_entries(force_to_lidar.data());
    const Vec3 scaled_bias =
        force_to_lidar * x.segment<kSpaceDim>(kAccelBiasParam);
    const Vec3 lever = LeverArm(x);
    const bool prior = config_.estimate_lever_arm;
    Eigen::VectorXd out(kResidualsPerWindow * samples.size() +
                        (prior ? kSpaceDim : 0));
    for (size_t k = 0; k < samples.size(); ++k) {
      const Sample& sample = samples[k];
      const Vec3 imu_rate =
          rotation * sample.lidar_rate + x.segment<kSpaceDim>(kGyroBiasParam);
      const Vec3 predicted = sample.force_sum * force_to_lidar_entries -
                             sample.attitude_sum * scaled_bias + gravity;
      out.segment<kSpaceDim>(kResidualsPerWindow * k + kGyroResidual) =
          (sample.imu_rate - imu_rate) / config_.gyro_sigma;
      out.segment<kSpaceDim>(kResidualsPerWindow * k + kAccelResidual) =
          (sample.world_acceleration + sample.lever_matrix * lever -
           predicted) /
          config_.accel_sigma;
    }
    if (prior) {  // Weak: the rig's lever arm.
      out.tail<kSpaceDim>() = (lever - lever_arm_) / config_.lever_arm_sigma;
    }
    return out;
  }

  // Huber weights per 3-vector.
  Eigen::VectorXd Weights(const Eigen::VectorXd& residuals) const {
    Eigen::VectorXd weights(residuals.size());
    for (Eigen::Index k = 0; k < residuals.size(); k += kSpaceDim) {
      const double norm = residuals.segment<kSpaceDim>(k).norm();
      weights.segment<kSpaceDim>(k).setConstant(
          norm <= config_.huber ? 1.0 : config_.huber / norm);
    }
    return weights;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd jacobian(num_residuals(), num_parameters());
    for (int parameter = 0; parameter < num_parameters(); ++parameter) {
      const double step =
          parameter == time_index() ? kTimeOffsetStep : kParameterStep;
      Eigen::VectorXd plus = x;
      Eigen::VectorXd minus = x;
      plus[parameter] += step;
      minus[parameter] -= step;
      jacobian.col(parameter) =
          (Residuals(plus) - Residuals(minus)) / (2.0 * step);
    }
    return jacobian;
  }

 private:
  const std::vector<Sample>& SamplesAt(double offset) const {
    const auto found = cache_.find(offset);
    if (found != cache_.end()) return found->second;
    if (cache_.size() > kMaxCachedOffsets) cache_.clear();
    return cache_[offset] = BuildSamples(offset);
  }

  std::vector<Sample> BuildSamples(double offset) const {
    // The IMU samples of every window, in LiDAR time.
    std::vector<double> times;
    std::vector<std::pair<size_t, size_t>> ranges;
    size_t first = 0;
    for (size_t k : windows_) {
      const double start = stamps_[k - reach_] + offset;
      const double end = stamps_[k + reach_] + offset;
      while (first < imu_.size() && imu_.stamps[first] < start) ++first;
      size_t last = first;
      while (last < imu_.size() && imu_.stamps[last] <= end) ++last;
      ranges.emplace_back(first, last);
      for (size_t i = first; i < last; ++i) {
        times.push_back(imu_.stamps[i] - offset);
      }
    }
    const std::vector<Pose> attitudes =
        InterpolatePoses(stamps_, poses_, times);
    std::vector<Sample> out;
    out.reserve(windows_.size());
    size_t imu_index = 0;
    const Mat3 lidar_from_imu = initial_.transpose();
    for (size_t window = 0; window < windows_.size(); ++window) {
      const size_t k = windows_[window];
      const double center = stamps_[k];
      const double before_span = center - stamps_[k - reach_];
      const double after_span = stamps_[k + reach_] - center;
      const Pose& before = poses_[k - reach_];
      const Pose& now = poses_[k];
      const Pose& after = poses_[k + reach_];
      Sample sample;
      sample.world_acceleration =
          2.0 *
          ((after.translation - now.translation) / after_span -
           (now.translation - before.translation) / before_span) /
          (before_span + after_span);
      double weights = 0.0;
      size_t interval = k - reach_;
      for (size_t i = ranges[window].first; i < ranges[window].second;
           ++i, ++imu_index) {
        const double time = times[imu_index];
        // The hat: 1 at the center, 0 at the window ends.
        const double weight =
            time < center ? std::max(0.0, 1.0 - (center - time) / before_span)
                          : std::max(0.0, 1.0 - (time - center) / after_span);
        // The rate interval [stamps_[interval], stamps_[interval + 1]) of
        // `time`; the last one is the one before the last stamp.
        while (interval + 2 < stamps_.size() && stamps_[interval + 1] <= time) {
          ++interval;
        }
        const Mat3& rotation = attitudes[imu_index].rotation;
        for (int column = 0; column < kSpaceDim; ++column) {
          sample.force_sum.block<kSpaceDim, kSpaceDim>(0, kSpaceDim * column) +=
              weight * imu_.accel[i][column] * rotation;
        }
        sample.attitude_sum += weight * rotation;
        // The rotation's acceleration at the IMU (LiDAR axes).
        const size_t previous = i > 0 ? i - 1 : i;
        const size_t next = i + 1 < imu_.size() ? i + 1 : i;
        const double span = imu_.stamps[next] - imu_.stamps[previous];
        const Vec3 rate = lidar_from_imu * imu_.gyro[i];
        const Vec3 angular_acceleration =
            span > 0.0 ? Vec3(lidar_from_imu *
                              (imu_.gyro[next] - imu_.gyro[previous]) / span)
                       : Vec3(Vec3::Zero());
        sample.lever_matrix +=
            weight * rotation *
            (Hat(angular_acceleration) + Hat(rate) * Hat(rate));
        sample.imu_rate += weight * imu_.gyro[i];
        sample.lidar_rate += weight * rates_[interval];
        weights += weight;
      }
      const double norm = std::max(weights, kMinWeightSum);
      sample.force_sum /= norm;
      sample.attitude_sum /= norm;
      sample.lever_matrix /= norm;
      sample.imu_rate /= norm;
      sample.lidar_rate /= norm;
      out.push_back(sample);
    }
    return out;
  }

  const std::vector<double>& stamps_;
  const std::vector<Pose>& poses_;
  const ImuSamples& imu_;
  Mat3 initial_;
  Vec3 lever_arm_;  // The rig's.
  LidarImuConfig config_;
  size_t reach_ = 1;
  std::vector<size_t> windows_;
  std::vector<Vec3> rates_;
  mutable std::map<double, std::vector<Sample>> cache_;
};

}  // namespace

LidarImuResult CalibrateLidarImu(const std::vector<double>& stamps,
                                 const std::vector<Pose>& world_from_lidar,
                                 const ImuSamples& imu, const Pose& initial,
                                 const LidarImuConfig& config) {
  LidarImuResult result;
  if (stamps.size() < kMinLidarImuSamples ||
      imu.size() < kMinLidarImuSamples) {
    return result;
  }
  Problem problem(stamps, world_from_lidar, imu, initial, config);
  result.samples = static_cast<int>(problem.size());
  if (problem.size() < kMinWindows) return result;
  const int parameter_count = problem.num_parameters();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(parameter_count);
  // The gyro bias starts at its mean.
  {
    const Eigen::VectorXd residuals = problem.Residuals(x);
    Vec3 gyro = Vec3::Zero();
    for (size_t k = 0; k < problem.size(); ++k) {
      gyro += residuals.segment<kSpaceDim>(kResidualsPerWindow * k +
                                           kGyroResidual) *
              config.gyro_sigma;
    }
    x.segment<kSpaceDim>(kGyroBiasParam) = gyro / problem.size();
  }
  const auto cost = [&](const Eigen::VectorXd& parameters) {
    const Eigen::VectorXd residuals = problem.Residuals(parameters);
    const Eigen::VectorXd weights = problem.Weights(residuals);
    return (weights.array() * residuals.array().square()).sum();
  };
  const auto clamp = [&](Eigen::VectorXd* parameters) {
    if (problem.time_index() != Problem::kNoParameter) {
      double& offset = (*parameters)[problem.time_index()];
      offset =
          std::clamp(offset, -config.max_time_offset, config.max_time_offset);
    }
  };
  double lambda = kInitialDamping;
  double current = cost(x);
  for (int iteration = 0; iteration < config.max_iterations; ++iteration) {
    const Eigen::VectorXd residuals = problem.Residuals(x);
    const Eigen::VectorXd weights = problem.Weights(residuals);
    const Eigen::MatrixXd jacobian = problem.Jacobian(x);
    const Eigen::MatrixXd weighted_transpose =
        jacobian.transpose() * weights.asDiagonal();
    const Eigen::MatrixXd hessian = weighted_transpose * jacobian;
    const Eigen::VectorXd gradient = weighted_transpose * residuals;
    bool improved = false;
    for (int attempt = 0; attempt < kMaxDampingAttempts && !improved;
         ++attempt) {
      Eigen::MatrixXd damped = hessian;
      damped.diagonal() +=
          lambda * hessian.diagonal().cwiseMax(kMinDampedDiagonal);
      const Eigen::VectorXd delta = damped.ldlt().solve(-gradient);
      Eigen::VectorXd candidate = x + delta;
      clamp(&candidate);
      const double next = cost(candidate);
      if (next < current) {
        improved = true;
        const double gain = (current - next) / std::max(current, kMinCost);
        x = candidate;
        current = next;
        lambda = std::max(lambda / kDampingDecrease, kMinDamping);
        if (gain < config.tolerance || delta.norm() < kMinStep) {
          result.converged = true;
        }
      } else {
        lambda *= kDampingIncrease;
      }
    }
    if (!improved || result.converged) {
      result.converged = true;
      break;
    }
  }
  result.imu_from_lidar = problem.Rotation(x);
  result.gyro_bias = x.segment<kSpaceDim>(kGyroBiasParam);
  result.accel_bias = x.segment<kSpaceDim>(kAccelBiasParam);
  result.accel_scale = problem.InverseScale(x).cwiseInverse();
  result.gravity_world = problem.Gravity(x);
  result.time_offset = problem.TimeOffset(x);
  result.lever_arm = problem.LeverArm(x);
  const Eigen::VectorXd residuals = problem.Residuals(x);
  const Eigen::VectorXd weights = problem.Weights(residuals);
  double gyro = 0.0;
  double accel = 0.0;
  for (size_t k = 0; k < problem.size(); ++k) {
    gyro +=
        residuals.segment<kSpaceDim>(kResidualsPerWindow * k + kGyroResidual)
            .squaredNorm();
    accel +=
        residuals.segment<kSpaceDim>(kResidualsPerWindow * k + kAccelResidual)
            .squaredNorm();
  }  // (The lever arm prior at the end is left out.)
  const double components = static_cast<double>(kSpaceDim) * problem.size();
  result.gyro_rms = std::sqrt(gyro / components) * config.gyro_sigma;
  result.accel_rms = std::sqrt(accel / components) * config.accel_sigma;
  // Covariance from the normal matrix, scaled by the residual variance.
  const Eigen::MatrixXd jacobian = problem.Jacobian(x);
  const Eigen::MatrixXd hessian =
      jacobian.transpose() * weights.asDiagonal() * jacobian;
  const double dof = std::max<double>(
      static_cast<double>(residuals.size()) - parameter_count, 1.0);
  const double variance =
      (weights.array() * residuals.array().square()).sum() / dof;
  const Eigen::MatrixXd covariance =
      hessian.completeOrthogonalDecomposition().pseudoInverse() * variance;
  const Eigen::VectorXd sigma = covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
  result.sigma_rotation_deg =
      sigma.segment<kSpaceDim>(kRotationParam) * (kDegreesPerHalfTurn / kPi);
  result.sigma_gyro_bias = sigma.segment<kSpaceDim>(kGyroBiasParam);
  result.sigma_accel_bias = sigma.segment<kSpaceDim>(kAccelBiasParam);
  if (config.estimate_scale) {  // d(1 / s) = -ds / s^2 with s ~ 1.
    result.sigma_accel_scale = sigma.segment<kSpaceDim>(kScaleParam);
  }
  if (problem.time_index() != Problem::kNoParameter) {
    result.sigma_time_offset = sigma[problem.time_index()];
  }
  if (config.estimate_lever_arm) {
    result.sigma_lever_arm = sigma.segment<kSpaceDim>(problem.lever_index());
  }
  return result;
}

}  // namespace gr
