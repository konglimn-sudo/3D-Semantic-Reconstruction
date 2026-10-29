#include "gr/calibration/clock_sync.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "Eigen/SVD"

#include "gr/core/check.h"
#include "gr/core/numeric.h"
#include "gr/sensors/imu.h"

namespace gr {
namespace {

// Rates are smoothed over at least this half width (s).
constexpr double kMinSmoothing = 1e-4;
// A window must turn: the 90th percentile of its rates reaches min_rate.
constexpr double kRatePercentile = 90.0;
// The robust line through the window lags: fit, drop the windows beyond
// 3 sigma (from the median absolute deviation, at least 2 ms), fit again.
constexpr int kRobustFitPasses = 2;
constexpr double kInlierSigmas = 3.0;
constexpr double kMadToSigma = 1.4826;     // For normally distributed lags.
constexpr double kMinInlierLimit = 0.002;  // s
// A drift needs two windows spread over more than this (s^2 of spread).
constexpr double kMinWindowsForDrift = 2.0;
constexpr double kMinTimeSpread = 1e-9;

struct LagFit {
  double residual = 0.0;
  Mat3 rotation = Mat3::Identity();
};

// Running integral of a rate series (trapezoids), same stamps.
std::vector<Vec3> Integral(const std::vector<double>& stamps,
                           const std::vector<Vec3>& rates) {
  std::vector<Vec3> out(stamps.size(), Vec3::Zero());
  for (size_t i = 1; i < stamps.size(); ++i) {
    out[i] = out[i - 1] +
             0.5 * (rates[i] + rates[i - 1]) * (stamps[i] - stamps[i - 1]);
  }
  return out;
}

// Mean rate over [time - half, time + half] from the integral.
Vec3 MeanRate(const std::vector<double>& stamps,
              const std::vector<Vec3>& integral, double time, double half) {
  return (InterpolateSeries(stamps, integral, time + half) -
          InterpolateSeries(stamps, integral, time - half)) /
         (2.0 * half);
}

// Rotation (reference <- other) and RMS rate residual at one lag over the
// `other` samples [first, last); `other_means` are the other's mean rates.
LagFit FitAtLag(const std::vector<double>& reference_stamps,
                const std::vector<Vec3>& reference_integral,
                const std::vector<double>& other_stamps,
                const std::vector<Vec3>& other_means, size_t first, size_t last,
                double lag, double half) {
  const size_t count = last - first;
  std::vector<Vec3> reference(count);
  Mat3 cross = Mat3::Zero();
  for (size_t i = 0; i < count; ++i) {
    reference[i] = MeanRate(reference_stamps, reference_integral,
                            other_stamps[first + i] + lag, half);
    cross += other_means[first + i] * reference[i].transpose();
  }
  // Kabsch: the rotation of the SVD, its last axis flipped for a
  // reflection.
  const Eigen::JacobiSVD<Mat3> svd(cross,
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
  constexpr int kLastAxis = kSpaceDim - 1;
  Mat3 sign = Mat3::Identity();
  sign(kLastAxis, kLastAxis) =
      (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0 ? -1.0
                                                                      : 1.0;
  LagFit fit;
  fit.rotation = svd.matrixV() * sign * svd.matrixU().transpose();
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    sum += (fit.rotation * other_means[first + i] - reference[i]).squaredNorm();
  }
  fit.residual = std::sqrt(sum / static_cast<double>(count));
  return fit;
}

}  // namespace

ClockSyncResult EstimateClockOffset(const std::vector<double>& reference_stamps,
                                    const std::vector<Vec3>& reference_rates,
                                    const std::vector<double>& other_stamps,
                                    const std::vector<Vec3>& other_rates,
                                    const ClockSyncConfig& config) {
  GR_CHECK(reference_stamps.size() == reference_rates.size());
  GR_CHECK(other_stamps.size() == other_rates.size());
  GR_CHECK(config.window > 0.0 && config.max_lag > 0.0 && config.step > 0.0);
  ClockSyncResult result;
  if (reference_stamps.size() < 2 || other_stamps.size() < 2) return result;
  const int num_lags =
      2 * static_cast<int>(std::lround(config.max_lag / config.step)) + 1;
  const double half = std::max(config.smoothing, kMinSmoothing);
  const std::vector<Vec3> reference_integral =
      Integral(reference_stamps, reference_rates);
  const std::vector<Vec3> other_integral = Integral(other_stamps, other_rates);
  std::vector<Vec3> other_means(other_stamps.size());
  for (size_t i = 0; i < other_stamps.size(); ++i) {
    other_means[i] =
        MeanRate(other_stamps, other_integral, other_stamps[i], half);
  }
  // Samples whose shifted window may leave either span are not used.
  const double first_usable =
      std::max(reference_stamps.front() + config.max_lag,
               other_stamps.front()) +
      half;
  const double last_usable =
      std::min(reference_stamps.back() - config.max_lag, other_stamps.back()) -
      half;
  ClockSyncWindow best_window;
  double best_residual = -1.0;  // None yet.
  for (double start = other_stamps.front(); start < other_stamps.back();
       start += config.window) {
    const double end = start + config.window;
    const size_t first = static_cast<size_t>(
        std::lower_bound(other_stamps.begin(), other_stamps.end(),
                         std::max(start, first_usable)) -
        other_stamps.begin());
    const size_t last = static_cast<size_t>(
        std::lower_bound(other_stamps.begin(), other_stamps.end(),
                         std::min(end, last_usable)) -
        other_stamps.begin());
    if (last <= first || static_cast<int>(last - first) < config.min_samples) {
      continue;
    }
    std::vector<double> rates;
    rates.reserve(last - first);
    for (size_t i = first; i < last; ++i) {
      rates.push_back(other_means[i].norm());
    }
    if (Percentile(rates, kRatePercentile) < config.min_rate) continue;

    std::vector<double> residuals(static_cast<size_t>(num_lags));
    for (int lag_index = 0; lag_index < num_lags; ++lag_index) {
      const double lag = -config.max_lag + lag_index * config.step;
      residuals[static_cast<size_t>(lag_index)] =
          FitAtLag(reference_stamps, reference_integral, other_stamps,
                   other_means, first, last, lag, half)
              .residual;
    }
    const int best =
        static_cast<int>(std::min_element(residuals.begin(), residuals.end()) -
                         residuals.begin());
    double lag = -config.max_lag + best * config.step;
    if (best > 0 && best + 1 < num_lags) {
      // The vertex of the parabola through the best lag and its neighbours.
      const double before = residuals[static_cast<size_t>(best - 1)];
      const double at = residuals[static_cast<size_t>(best)];
      const double after = residuals[static_cast<size_t>(best + 1)];
      const double curvature = before - 2.0 * at + after;
      if (curvature > 0.0) {
        lag += 0.5 * (before - after) / curvature * config.step;
      }
    }
    const LagFit fit =
        FitAtLag(reference_stamps, reference_integral, other_stamps,
                 other_means, first, last, lag, half);
    if (fit.residual > config.max_residual) continue;
    ClockSyncWindow window;
    window.center = 0.5 * (other_stamps[first] + other_stamps[last - 1]);
    window.lag = lag;
    window.residual = fit.residual;
    window.samples = static_cast<int>(last - first);
    result.windows.push_back(window);
    if (best_residual < 0.0 || fit.residual < best_residual) {
      best_residual = fit.residual;
      result.rotation = fit.rotation;
      best_window = window;
    }
  }
  if (result.windows.empty()) return result;

  // Robust straight line through the window lags.
  std::vector<bool> inlier(result.windows.size(), true);
  double sum_centers = 0.0;
  for (const ClockSyncWindow& window : result.windows) {
    sum_centers += window.center;
  }
  result.reference_time =
      sum_centers / static_cast<double>(result.windows.size());
  for (int pass = 0; pass < kRobustFitPasses; ++pass) {
    // Least squares lag = offset + drift * time over the inliers.
    double count = 0.0;
    double sum_time = 0.0;
    double sum_lag = 0.0;
    double sum_time_time = 0.0;
    double sum_time_lag = 0.0;
    for (size_t i = 0; i < result.windows.size(); ++i) {
      if (!inlier[i]) continue;
      const double time = result.windows[i].center - result.reference_time;
      const double lag = result.windows[i].lag;
      count += 1.0;
      sum_time += time;
      sum_lag += lag;
      sum_time_time += time * time;
      sum_time_lag += time * lag;
    }
    const double denominator = count * sum_time_time - sum_time * sum_time;
    if (count >= kMinWindowsForDrift && denominator > kMinTimeSpread) {
      result.drift = (count * sum_time_lag - sum_time * sum_lag) / denominator;
      result.offset = (sum_lag - result.drift * sum_time) / count;
    } else {
      result.drift = 0.0;
      result.offset = count > 0.0 ? sum_lag / count : best_window.lag;
    }
    std::vector<double> deviations;
    for (const ClockSyncWindow& window : result.windows) {
      deviations.push_back(
          std::abs(window.lag - result.offset -
                   result.drift * (window.center - result.reference_time)));
    }
    const double limit = std::max(
        kInlierSigmas * kMadToSigma * Median(deviations), kMinInlierLimit);
    for (size_t i = 0; i < result.windows.size(); ++i) {
      inlier[i] = deviations[i] <= limit;
    }
  }
  result.valid = true;
  return result;
}

}  // namespace gr
