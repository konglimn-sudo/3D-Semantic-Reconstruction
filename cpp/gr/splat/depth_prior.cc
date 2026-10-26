#include "gr/splat/depth_prior.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "gr/core/check.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

bool InRange(double value, const DepthAlignConfig& config) {
  return std::isfinite(value) && value >= config.min_range &&
         value <= config.max_range;
}

struct Pair {
  int u;  // Pixel of the sample (for the cells).
  int v;
  double prior;
  double lidar;
};

// Prior at a sub-pixel position: bilinear if the four neighbours are valid
// and within 10 % of each other (not across a depth edge), else NaN.
double SamplePrior(const std::vector<float>& prior, int height, int width,
                   double u, double v) {
  const int u0 = static_cast<int>(std::floor(u));
  const int v0 = static_cast<int>(std::floor(v));
  if (u0 < 0 || v0 < 0 || u0 + 1 >= width || v0 + 1 >= height) {
    const int ui = static_cast<int>(std::lround(u));
    const int vi = static_cast<int>(std::lround(v));
    if (ui < 0 || vi < 0 || ui >= width || vi >= height) return kNaN;
    return prior[static_cast<size_t>(vi) * width + ui];
  }
  const double a = u - u0, b = v - v0;
  const size_t i = static_cast<size_t>(v0) * width + u0;
  const double p00 = prior[i], p01 = prior[i + 1];
  const double p10 = prior[i + width], p11 = prior[i + width + 1];
  if (!std::isfinite(p00) || !std::isfinite(p01) || !std::isfinite(p10) ||
      !std::isfinite(p11)) {
    return kNaN;
  }
  const double lo = std::min({p00, p01, p10, p11});
  const double hi = std::max({p00, p01, p10, p11});
  if (hi > 1.1 * lo) return kNaN;
  return (1 - a) * (1 - b) * p00 + a * (1 - b) * p01 + (1 - a) * b * p10 +
         a * b * p11;
}

// Weighted least squares of l = s p + b on the relative residual
// (l - s p - b) / l, i.e. rows (p / l, 1 / l) -> 1, weights w.
void SolveScaleShift(const std::vector<Pair>& pairs,
                     const std::vector<double>& weights, bool fit_shift,
                     double& scale, double& shift) {
  double a00 = 0.0, a01 = 0.0, a11 = 0.0, b0 = 0.0, b1 = 0.0;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const double w = weights[i];
    const double x0 = pairs[i].prior / pairs[i].lidar;
    const double x1 = 1.0 / pairs[i].lidar;
    a00 += w * x0 * x0;
    a01 += w * x0 * x1;
    a11 += w * x1 * x1;
    b0 += w * x0;
    b1 += w * x1;
  }
  const double det = a00 * a11 - a01 * a01;
  if (!fit_shift || std::abs(det) <= 1e-12 * std::max(1.0, a00 * a11)) {
    if (a00 > 0.0) scale = b0 / a00;
    shift = 0.0;
    return;
  }
  scale = (b0 * a11 - b1 * a01) / det;
  shift = (a00 * b1 - a01 * b0) / det;
}

double RelativeResidual(const Pair& pair, double scale, double shift) {
  return std::abs(pair.lidar - scale * pair.prior - shift) / pair.lidar;
}

// Pixels within `radius` (Chebyshev) of a set pixel: separable max filter.
std::vector<uint8_t> Dilate(const std::vector<uint8_t>& mask, int height,
                            int width, int radius) {
  if (radius <= 0) return mask;
  std::vector<uint8_t> rows(mask.size(), 0), out(mask.size(), 0);
  for (int r = 0; r < height; ++r) {
    int last = -1000000;  // Last set column seen.
    std::vector<int> next(width, 1000000);
    for (int c = width - 1, seen = 1000000; c >= 0; --c) {
      if (mask[static_cast<size_t>(r) * width + c]) seen = c;
      next[c] = seen;
    }
    for (int c = 0; c < width; ++c) {
      if (mask[static_cast<size_t>(r) * width + c]) last = c;
      rows[static_cast<size_t>(r) * width + c] =
          (c - last <= radius || next[c] - c <= radius) ? 1 : 0;
    }
  }
  for (int c = 0; c < width; ++c) {
    int last = -1000000;
    std::vector<int> next(height, 1000000);
    for (int r = height - 1, seen = 1000000; r >= 0; --r) {
      if (rows[static_cast<size_t>(r) * width + c]) seen = r;
      next[r] = seen;
    }
    for (int r = 0; r < height; ++r) {
      if (rows[static_cast<size_t>(r) * width + c]) last = r;
      out[static_cast<size_t>(r) * width + c] =
          (r - last <= radius || next[r] - r <= radius) ? 1 : 0;
    }
  }
  return out;
}

}  // namespace

FusedDepth FuseDepthPrior(const std::vector<float>& prior,
                          const std::vector<float>& prior_confidence,
                          const std::vector<float>& lidar,
                          const std::vector<DepthSample>& samples, int height,
                          int width, const DepthAlignConfig& config) {
  GR_CHECK(height > 0 && width > 0);
  const size_t n = static_cast<size_t>(height) * static_cast<size_t>(width);
  GR_CHECK(prior.size() == n && lidar.size() == n);
  GR_CHECK(prior_confidence.empty() || prior_confidence.size() == n);
  GR_CHECK(config.cell > 0);

  FusedDepth out;
  out.range.assign(n, kNaN);
  out.confidence.assign(n, 0.0f);
  out.source.assign(n, kDepthNone);
  std::vector<uint8_t> measured(n, 0);
  for (size_t i = 0; i < n; ++i) {
    if (InRange(lidar[i], config)) {
      out.range[i] = lidar[i];
      out.confidence[i] = 1.0f;
      out.source[i] = kDepthLidar;
      measured[i] = 1;
    }
  }

  std::vector<Pair> pairs;
  if (samples.empty()) {
    for (size_t i = 0; i < n; ++i) {
      if (measured[i] && InRange(prior[i], config)) {
        pairs.push_back({static_cast<int>(i % width),
                         static_cast<int>(i / width), prior[i], lidar[i]});
      }
    }
  } else {
    for (const DepthSample& s : samples) {
      if (!InRange(s.range, config)) continue;
      const double p = SamplePrior(prior, height, width, s.u, s.v);
      if (!InRange(p, config)) continue;
      pairs.push_back({static_cast<int>(std::lround(s.u)),
                       static_cast<int>(std::lround(s.v)), p, s.range});
    }
  }
  DepthAlignment& alignment = out.alignment;
  alignment.pixels = static_cast<int>(pairs.size());
  if (alignment.pixels < config.min_pixels) return out;

  // Start from the median ratio, then IRLS with a Cauchy weight.
  std::vector<double> ratios;
  ratios.reserve(pairs.size());
  for (const Pair& p : pairs) ratios.push_back(p.lidar / p.prior);
  double scale = Median(ratios);
  alignment.raw_ratio = scale;
  double shift = 0.0;
  std::vector<double> weights(pairs.size(), 1.0);
  const double c2 = config.robust_scale * config.robust_scale;
  for (int iteration = 0; iteration < config.iterations; ++iteration) {
    for (size_t i = 0; i < pairs.size(); ++i) {
      const double e = RelativeResidual(pairs[i], scale, shift);
      weights[i] = 1.0 / (1.0 + e * e / c2);
    }
    SolveScaleShift(pairs, weights, config.fit_shift, scale, shift);
  }
  std::vector<double> residuals(pairs.size());
  int inliers = 0;
  for (size_t i = 0; i < pairs.size(); ++i) {
    residuals[i] = RelativeResidual(pairs[i], scale, shift);
    if (residuals[i] < config.robust_scale) ++inliers;
  }
  alignment.scale = scale;
  alignment.shift = shift;
  alignment.residual_median = Median(residuals);
  alignment.residual_p90 = Percentile(residuals, 90.0);
  alignment.inlier_fraction =
      static_cast<double>(inliers) / static_cast<double>(pairs.size());
  alignment.valid = alignment.residual_median <= config.max_residual &&
                    scale >= config.min_scale && scale <= config.max_scale;
  if (!alignment.valid) return out;

  // Per-cell residuals of the fitted prior.
  const int cells_x = (width + config.cell - 1) / config.cell;
  const int cells_y = (height + config.cell - 1) / config.cell;
  const size_t num_cells =
      static_cast<size_t>(cells_x) * static_cast<size_t>(cells_y);
  std::vector<std::vector<double>> cell_residuals(num_cells);
  for (size_t i = 0; i < pairs.size(); ++i) {
    const int col = std::clamp(pairs[i].u, 0, width - 1);
    const int row = std::clamp(pairs[i].v, 0, height - 1);
    cell_residuals[static_cast<size_t>(row / config.cell) * cells_x +
                   col / config.cell]
        .push_back(residuals[i]);
  }
  std::vector<double> cell_error(num_cells, alignment.residual_median);
  std::vector<uint8_t> bad(num_cells, 0);
  for (size_t c = 0; c < num_cells; ++c) {
    if (static_cast<int>(cell_residuals[c].size()) < config.min_cell_pixels) {
      continue;
    }
    ++alignment.cells_checked;
    cell_error[c] = Median(cell_residuals[c]);
    if (cell_error[c] > config.max_cell_residual) bad[c] = 1;
  }
  std::vector<uint8_t> rejected(num_cells, 0);
  for (int cy = 0; cy < cells_y; ++cy) {
    for (int cx = 0; cx < cells_x; ++cx) {
      if (!bad[static_cast<size_t>(cy) * cells_x + cx]) continue;
      ++alignment.cells_rejected;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          const int x = cx + dx, y = cy + dy;
          if (x < 0 || y < 0 || x >= cells_x || y >= cells_y) continue;
          rejected[static_cast<size_t>(y) * cells_x + x] = 1;
        }
      }
    }
  }

  const std::vector<uint8_t> covered =
      Dilate(measured, height, width, config.hole_radius);
  const double t2 = config.tolerance * config.tolerance;
  for (size_t i = 0; i < n; ++i) {
    if (covered[i] || !InRange(prior[i], config)) continue;
    const int row = static_cast<int>(i / width);
    const int col = static_cast<int>(i % width);
    const size_t c =
        static_cast<size_t>(row / config.cell) * cells_x + col / config.cell;
    if (rejected[c]) continue;
    const double own =
        prior_confidence.empty()
            ? 1.0
            : std::clamp(static_cast<double>(prior_confidence[i]), 0.0, 1.0);
    const double e = cell_error[c];
    const double confidence = own * std::exp(-0.5 * e * e / t2);
    if (confidence < config.min_confidence) continue;
    const double range = scale * prior[i] + shift;
    if (!(range > 0.0)) continue;
    out.range[i] = static_cast<float>(range);
    out.confidence[i] = static_cast<float>(confidence);
    out.source[i] = kDepthPrior;
    ++out.prior_pixels;
  }
  return out;
}

}  // namespace gr
