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

// Mean rate over [t - half, t + half] from the integral.
Vec3 MeanRate(const std::vector<double>& stamps,
              const std::vector<Vec3>& integral, double t, double half) {
  return (InterpolateSeries(stamps, integral, t + half) -
          InterpolateSeries(stamps, integral, t - half)) /
         (2.0 * half);
}

// Rotation (reference <- other) and RMS rate residual at one lag over the
// `other` samples [first, last); `other_means` are the other's mean rates.
LagFit FitAtLag(const std::vector<double>& reference_stamps,
                const std::vector<Vec3>& reference_integral,
                const std::vector<double>& other_stamps,
                const std::vector<Vec3>& other_means, size_t first, size_t last,
                double lag, double half) {
  const size_t n = last - first;
  std::vector<Vec3> reference(n);
  Mat3 cross = Mat3::Zero();
  for (size_t i = 0; i < n; ++i) {
    reference[i] = MeanRate(reference_stamps, reference_integral,
                            other_stamps[first + i] + lag, half);
    cross += other_means[first + i] * reference[i].transpose();
  }
  const Eigen::JacobiSVD<Mat3> svd(cross,
                                   Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat3 sign = Mat3::Identity();
  sign(2, 2) = (svd.matrixV() * svd.matrixU().transpose()).determinant() < 0.0
                   ? -1.0
                   : 1.0;
  LagFit fit;
  fit.rotation = svd.matrixV() * sign * svd.matrixU().transpose();
  double sum = 0.0;
  for (size_t i = 0; i < n; ++i) {
    sum += (fit.rotation * other_means[first + i] - reference[i]).squaredNorm();
  }
  fit.residual = std::sqrt(sum / static_cast<double>(n));
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
  const double half = std::max(config.smoothing, 1e-4);
  const std::vector<Vec3> reference_integral =
      Integral(reference_stamps, reference_rates);
  const std::vector<Vec3> other_integral = Integral(other_stamps, other_rates);
  std::vector<Vec3> other_means(other_stamps.size());
  for (size_t i = 0; i < other_stamps.size(); ++i) {
    other_means[i] =
        MeanRate(other_stamps, other_integral, other_stamps[i], half);
  }
  // Samples whose shifted window may leave either span are not used.
  const double lo = std::max(reference_stamps.front() + config.max_lag,
                             other_stamps.front()) +
                    half;
  const double hi =
      std::min(reference_stamps.back() - config.max_lag, other_stamps.back()) -
      half;
  ClockSyncWindow best_window;
  double best_residual = -1.0;
  for (double start = other_stamps.front(); start < other_stamps.back();
       start += config.window) {
    const double end = start + config.window;
    const size_t first = static_cast<size_t>(
        std::lower_bound(other_stamps.begin(), other_stamps.end(),
                         std::max(start, lo)) -
        other_stamps.begin());
    const size_t last = static_cast<size_t>(
        std::lower_bound(other_stamps.begin(), other_stamps.end(),
                         std::min(end, hi)) -
        other_stamps.begin());
    if (last <= first || static_cast<int>(last - first) < config.min_samples) {
      continue;
    }
    std::vector<double> rates;
    rates.reserve(last - first);
    for (size_t i = first; i < last; ++i)
      rates.push_back(other_means[i].norm());
    if (Percentile(rates, 90.0) < config.min_rate) continue;

    std::vector<double> residuals(static_cast<size_t>(num_lags));
    for (int k = 0; k < num_lags; ++k) {
      const double lag = -config.max_lag + k * config.step;
      residuals[static_cast<size_t>(k)] =
          FitAtLag(reference_stamps, reference_integral, other_stamps,
                   other_means, first, last, lag, half)
              .residual;
    }
    const int k =
        static_cast<int>(std::min_element(residuals.begin(), residuals.end()) -
                         residuals.begin());
    double lag = -config.max_lag + k * config.step;
    if (k > 0 && k + 1 < num_lags) {
      const double y0 = residuals[static_cast<size_t>(k - 1)];
      const double y1 = residuals[static_cast<size_t>(k)];
      const double y2 = residuals[static_cast<size_t>(k + 1)];
      const double curvature = y0 - 2.0 * y1 + y2;
      if (curvature > 0.0) lag += 0.5 * (y0 - y2) / curvature * config.step;
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
  double sum_t = 0.0;
  for (const ClockSyncWindow& w : result.windows) sum_t += w.center;
  result.reference_time = sum_t / static_cast<double>(result.windows.size());
  for (int pass = 0; pass < 2; ++pass) {
    double n = 0.0, st = 0.0, sl = 0.0, stt = 0.0, stl = 0.0;
    for (size_t i = 0; i < result.windows.size(); ++i) {
      if (!inlier[i]) continue;
      const double t = result.windows[i].center - result.reference_time;
      const double l = result.windows[i].lag;
      n += 1.0;
      st += t;
      sl += l;
      stt += t * t;
      stl += t * l;
    }
    const double denominator = n * stt - st * st;
    if (n >= 2.0 && denominator > 1e-9) {
      result.drift = (n * stl - st * sl) / denominator;
      result.offset = (sl - result.drift * st) / n;
    } else {
      result.drift = 0.0;
      result.offset = n > 0.0 ? sl / n : best_window.lag;
    }
    std::vector<double> deviations;
    for (const ClockSyncWindow& w : result.windows) {
      deviations.push_back(
          std::abs(w.lag - result.offset -
                   result.drift * (w.center - result.reference_time)));
    }
    const double limit = std::max(3.0 * 1.4826 * Median(deviations), 0.002);
    for (size_t i = 0; i < result.windows.size(); ++i) {
      inlier[i] = deviations[i] <= limit;
    }
  }
  result.valid = true;
  return result;
}

}  // namespace gr
