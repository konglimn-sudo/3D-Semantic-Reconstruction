#include "gr/calibration/lidar_imu.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace gr {
namespace {

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
  Vec3 w_lidar;
  Vec3 w_imu;
  Vec3 a_world;
  Eigen::Matrix<double, 3, 9> k;
  Mat3 b;
  Mat3 m;
};

constexpr int kRotation = 0;
constexpr int kGyroBias = 3;
constexpr int kAccelBias = 6;
constexpr int kGravity = 9;  // Two tilt angles.
constexpr int kScale = 11;   // Then the lever arm and the time offset.

class Problem {
 public:
  Problem(const std::vector<double>& stamps, const std::vector<Pose>& poses,
          const ImuSamples& imu, const Pose& initial,
          const LidarImuConfig& config)
      : stamps_(stamps), poses_(poses), imu_(imu),
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
    const double period = std::max(steps[steps.size() / 2], 1e-3);
    reach_ = static_cast<size_t>(
        std::max(1.0, std::round(config.half_window / period)));
    const double margin = config.max_time_offset + 0.05;
    for (size_t k = reach_; k + reach_ < stamps.size();
         k += static_cast<size_t>(std::max(config.stride, 1))) {
      if (stamps[k - reach_] - margin >= imu.stamps.front() &&
          stamps[k + reach_] + margin <= imu.stamps.back()) {
        windows_.push_back(k);
      }
    }
    // LiDAR rates between consecutive sweeps (body axes).
    for (size_t k = 0; k + 1 < stamps.size(); ++k) {
      rates_.push_back(So3Log(poses[k].rotation.transpose() *
                              poses[k + 1].rotation) /
                       std::max(stamps[k + 1] - stamps[k], 1e-6));
    }
  }

  int num_parameters() const {
    return lever_index() + (config_.estimate_lever_arm ? 3 : 0) +
           (config_.estimate_time_offset ? 1 : 0);
  }
  int lever_index() const {
    return kScale + (config_.estimate_scale ? 3 : 0);
  }
  int time_index() const {
    return config_.estimate_time_offset ? num_parameters() - 1 : -1;
  }
  size_t size() const { return windows_.size(); }

  Mat3 Rotation(const Eigen::VectorXd& x) const {
    return initial_ * So3Exp(x.segment<3>(kRotation));
  }
  Vec3 Gravity(const Eigen::VectorXd& x) const {
    return So3Exp(Vec3(x[kGravity], x[kGravity + 1], 0.0)) *
           Vec3(0.0, 0.0, -config_.gravity);
  }
  // 1 / scale of the accelerometer axes.
  Vec3 InverseScale(const Eigen::VectorXd& x) const {
    return config_.estimate_scale
               ? Vec3(Vec3::Ones() + x.segment<3>(kScale))
               : Vec3(Vec3::Ones());
  }
  // The IMU origin in LiDAR axes.
  Vec3 LeverArm(const Eigen::VectorXd& x) const {
    return config_.estimate_lever_arm
               ? Vec3(lever_arm_ + x.segment<3>(lever_index()))
               : lever_arm_;
  }
  // IMU time = LiDAR time + offset.
  double TimeOffset(const Eigen::VectorXd& x) const {
    return config_.estimate_time_offset ? x[time_index()] : 0.0;
  }

  // Whitened residuals: per window [gyro (3), accel (3)].
  Eigen::VectorXd Residuals(const Eigen::VectorXd& x) const {
    const std::vector<Sample>& samples = SamplesAt(TimeOffset(x));
    const Mat3 r = Rotation(x);
    const Vec3 g = Gravity(x);
    const Mat3 a = r.transpose() * InverseScale(x).asDiagonal();
    const Eigen::Map<const Eigen::Matrix<double, 9, 1>> vec_a(a.data());
    const Vec3 a_bias = a * x.segment<3>(kAccelBias);
    const Vec3 lever = LeverArm(x);
    const bool prior = config_.estimate_lever_arm;
    Eigen::VectorXd out(6 * samples.size() + (prior ? 3 : 0));
    for (size_t k = 0; k < samples.size(); ++k) {
      const Sample& m = samples[k];
      const Vec3 w = r * m.w_lidar + x.segment<3>(kGyroBias);
      const Vec3 predicted = m.k * vec_a - m.b * a_bias + g;
      out.segment<3>(6 * k) = (m.w_imu - w) / config_.gyro_sigma;
      out.segment<3>(6 * k + 3) =
          (m.a_world + m.m * lever - predicted) / config_.accel_sigma;
    }
    if (prior) {  // Weak: the rig's lever arm.
      out.tail<3>() = (lever - lever_arm_) / config_.lever_arm_sigma;
    }
    return out;
  }

  // Huber weights per 3-vector.
  Eigen::VectorXd Weights(const Eigen::VectorXd& residuals) const {
    Eigen::VectorXd w(residuals.size());
    for (Eigen::Index k = 0; k < residuals.size(); k += 3) {
      const double norm = residuals.segment<3>(k).norm();
      w.segment<3>(k).setConstant(
          norm <= config_.huber ? 1.0 : config_.huber / norm);
    }
    return w;
  }

  Eigen::MatrixXd Jacobian(const Eigen::VectorXd& x) const {
    Eigen::MatrixXd j(6 * size() + (config_.estimate_lever_arm ? 3 : 0),
                      num_parameters());
    for (int p = 0; p < num_parameters(); ++p) {
      const double step = p == time_index() ? 1e-4 : 1e-6;
      Eigen::VectorXd plus = x;
      Eigen::VectorXd minus = x;
      plus[p] += step;
      minus[p] -= step;
      j.col(p) = (Residuals(plus) - Residuals(minus)) / (2.0 * step);
    }
    return j;
  }

 private:
  const std::vector<Sample>& SamplesAt(double offset) const {
    const auto found = cache_.find(offset);
    if (found != cache_.end()) return found->second;
    if (cache_.size() > 16) cache_.clear();
    return cache_[offset] = BuildSamples(offset);
  }

  std::vector<Sample> BuildSamples(double offset) const {
    // The IMU samples of every window, in LiDAR time.
    std::vector<double> times;
    std::vector<std::pair<size_t, size_t>> ranges;
    size_t first = 0;
    for (size_t k : windows_) {
      const double t0 = stamps_[k - reach_] + offset;
      const double t1 = stamps_[k + reach_] + offset;
      while (first < imu_.size() && imu_.stamps[first] < t0) ++first;
      size_t last = first;
      while (last < imu_.size() && imu_.stamps[last] <= t1) ++last;
      ranges.emplace_back(first, last);
      for (size_t i = first; i < last; ++i) {
        times.push_back(imu_.stamps[i] - offset);
      }
    }
    const std::vector<Pose> attitudes =
        InterpolatePoses(stamps_, poses_, times);
    std::vector<Sample> out;
    out.reserve(windows_.size());
    size_t at = 0;
    const Mat3 lidar_from_imu = initial_.transpose();
    for (size_t w = 0; w < windows_.size(); ++w) {
      const size_t k = windows_[w];
      const double t = stamps_[k];
      const double h0 = t - stamps_[k - reach_];
      const double h1 = stamps_[k + reach_] - t;
      const Pose& before = poses_[k - reach_];
      const Pose& now = poses_[k];
      const Pose& after = poses_[k + reach_];
      Sample s;
      s.a_world = 2.0 *
                  ((after.translation - now.translation) / h1 -
                   (now.translation - before.translation) / h0) /
                  (h0 + h1);
      s.k.setZero();
      s.b.setZero();
      s.m.setZero();
      s.w_lidar.setZero();
      s.w_imu.setZero();
      double weights = 0.0;
      size_t interval = k - reach_;
      for (size_t i = ranges[w].first; i < ranges[w].second; ++i, ++at) {
        const double tau = times[at];
        const double weight = tau < t ? std::max(0.0, 1.0 - (t - tau) / h0)
                                      : std::max(0.0, 1.0 - (tau - t) / h1);
        while (interval + 2 < stamps_.size() &&
               stamps_[interval + 1] <= tau) {
          ++interval;
        }
        const Mat3& rotation = attitudes[at].rotation;
        for (int col = 0; col < 3; ++col) {
          s.k.block<3, 3>(0, 3 * col) +=
              weight * imu_.accel[i][col] * rotation;
        }
        s.b += weight * rotation;
        // The rotation's acceleration at the IMU (LiDAR axes).
        const size_t i0 = i > 0 ? i - 1 : i;
        const size_t i1 = i + 1 < imu_.size() ? i + 1 : i;
        const double span = imu_.stamps[i1] - imu_.stamps[i0];
        const Vec3 w = lidar_from_imu * imu_.gyro[i];
        const Vec3 alpha =
            span > 0.0
                ? Vec3(lidar_from_imu * (imu_.gyro[i1] - imu_.gyro[i0]) /
                       span)
                : Vec3(Vec3::Zero());
        s.m += weight * rotation * (Hat(alpha) + Hat(w) * Hat(w));
        s.w_imu += weight * imu_.gyro[i];
        s.w_lidar += weight * rates_[interval];
        weights += weight;
      }
      const double norm = std::max(weights, 1e-12);
      s.k /= norm;
      s.b /= norm;
      s.m /= norm;
      s.w_imu /= norm;
      s.w_lidar /= norm;
      out.push_back(s);
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
  if (stamps.size() < 3 || imu.size() < 3) return result;
  Problem problem(stamps, world_from_lidar, imu, initial, config);
  result.samples = static_cast<int>(problem.size());
  if (problem.size() < 10) return result;
  const int n = problem.num_parameters();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  // The gyro bias starts at its mean.
  {
    const Eigen::VectorXd r = problem.Residuals(x);
    Vec3 gyro = Vec3::Zero();
    for (size_t k = 0; k < problem.size(); ++k) {
      gyro += r.segment<3>(6 * k) * config.gyro_sigma;
    }
    x.segment<3>(kGyroBias) = gyro / problem.size();
  }
  const auto cost = [&](const Eigen::VectorXd& p) {
    const Eigen::VectorXd r = problem.Residuals(p);
    const Eigen::VectorXd w = problem.Weights(r);
    return (w.array() * r.array().square()).sum();
  };
  const auto clamp = [&](Eigen::VectorXd* p) {
    if (problem.time_index() >= 0) {
      double& offset = (*p)[problem.time_index()];
      offset = std::clamp(offset, -config.max_time_offset,
                          config.max_time_offset);
    }
  };
  double lambda = 1e-3;
  double current = cost(x);
  for (int it = 0; it < config.max_iterations; ++it) {
    const Eigen::VectorXd r = problem.Residuals(x);
    const Eigen::VectorXd w = problem.Weights(r);
    const Eigen::MatrixXd j = problem.Jacobian(x);
    const Eigen::MatrixXd jtw = j.transpose() * w.asDiagonal();
    const Eigen::MatrixXd hessian = jtw * j;
    const Eigen::VectorXd gradient = jtw * r;
    bool improved = false;
    for (int tries = 0; tries < 10 && !improved; ++tries) {
      Eigen::MatrixXd damped = hessian;
      damped.diagonal() += lambda * hessian.diagonal().cwiseMax(1e-9);
      const Eigen::VectorXd delta = damped.ldlt().solve(-gradient);
      Eigen::VectorXd candidate = x + delta;
      clamp(&candidate);
      const double next = cost(candidate);
      if (next < current) {
        improved = true;
        const double gain = (current - next) / std::max(current, 1e-12);
        x = candidate;
        current = next;
        lambda = std::max(lambda / 3.0, 1e-9);
        if (gain < config.tolerance || delta.norm() < 1e-12) {
          result.converged = true;
        }
      } else {
        lambda *= 5.0;
      }
    }
    if (!improved || result.converged) {
      result.converged = true;
      break;
    }
  }
  result.imu_from_lidar = problem.Rotation(x);
  result.gyro_bias = x.segment<3>(kGyroBias);
  result.accel_bias = x.segment<3>(kAccelBias);
  result.accel_scale = problem.InverseScale(x).cwiseInverse();
  result.gravity_world = problem.Gravity(x);
  result.time_offset = problem.TimeOffset(x);
  result.lever_arm = problem.LeverArm(x);
  const Eigen::VectorXd r = problem.Residuals(x);
  const Eigen::VectorXd w = problem.Weights(r);
  double gyro = 0.0;
  double accel = 0.0;
  for (size_t k = 0; k < problem.size(); ++k) {
    gyro += r.segment<3>(6 * k).squaredNorm();
    accel += r.segment<3>(6 * k + 3).squaredNorm();
  }  // (The lever arm prior at the end is left out.)
  result.gyro_rms =
      std::sqrt(gyro / (3.0 * problem.size())) * config.gyro_sigma;
  result.accel_rms =
      std::sqrt(accel / (3.0 * problem.size())) * config.accel_sigma;
  // Covariance from the normal matrix, scaled by the residual variance.
  const Eigen::MatrixXd j = problem.Jacobian(x);
  const Eigen::MatrixXd hessian = j.transpose() * w.asDiagonal() * j;
  const double dof =
      std::max<double>(static_cast<double>(r.size()) - n, 1.0);
  const double variance = (w.array() * r.array().square()).sum() / dof;
  const Eigen::MatrixXd covariance =
      hessian.completeOrthogonalDecomposition().pseudoInverse() * variance;
  const Eigen::VectorXd sigma =
      covariance.diagonal().cwiseMax(0.0).cwiseSqrt();
  result.sigma_rotation_deg = sigma.segment<3>(kRotation) * (180.0 / kPi);
  result.sigma_gyro_bias = sigma.segment<3>(kGyroBias);
  result.sigma_accel_bias = sigma.segment<3>(kAccelBias);
  if (config.estimate_scale) {  // d(1 / s) = -ds / s^2 with s ~ 1.
    result.sigma_accel_scale = sigma.segment<3>(kScale);
  }
  if (problem.time_index() >= 0) {
    result.sigma_time_offset = sigma[problem.time_index()];
  }
  if (config.estimate_lever_arm) {
    result.sigma_lever_arm = sigma.segment<3>(problem.lever_index());
  }
  return result;
}

}  // namespace gr
