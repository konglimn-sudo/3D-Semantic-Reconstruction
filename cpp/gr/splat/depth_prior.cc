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
// The four prior pixels around a sample are interpolated only if the largest
// is at most this factor of the smallest (not across a depth edge).
constexpr double kMaxBilinearSpread = 1.1;
// Normal equations with a relative determinant below this are singular.
constexpr double kSingularDeterminant = 1e-12;
// Distance (pixels) to a set pixel when there is none in a row / column.
constexpr int kNoSetPixel = 1000000;
// Confidence of the measured LiDAR pixels.
constexpr float kLidarConfidence = 1.0f;
// Upper percentile of the residuals reported.
constexpr double kResidualPercentile = 90.0;
// A rejected cell also rejects its neighbours this many cells around.
constexpr int kNeighbourhoodRadius = 1;

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
// and within kMaxBilinearSpread of each other, else NaN; at the image border
// the nearest pixel.
double SamplePrior(const std::vector<float>& prior, int height, int width,
                   double u, double v) {
  const int left = static_cast<int>(std::floor(u));
  const int top = static_cast<int>(std::floor(v));
  if (left < 0 || top < 0 || left + 1 >= width || top + 1 >= height) {
    const int nearest_u = static_cast<int>(std::lround(u));
    const int nearest_v = static_cast<int>(std::lround(v));
    if (nearest_u < 0 || nearest_v < 0 || nearest_u >= width ||
        nearest_v >= height) {
      return kNaN;
    }
    return prior[static_cast<size_t>(nearest_v) * width + nearest_u];
  }
  const double fraction_u = u - left;
  const double fraction_v = v - top;
  const size_t pixel = static_cast<size_t>(top) * width + left;
  const double top_left = prior[pixel];
  const double top_right = prior[pixel + 1];
  const double bottom_left = prior[pixel + width];
  const double bottom_right = prior[pixel + width + 1];
  if (!std::isfinite(top_left) || !std::isfinite(top_right) ||
      !std::isfinite(bottom_left) || !std::isfinite(bottom_right)) {
    return kNaN;
  }
  const double lowest =
      std::min({top_left, top_right, bottom_left, bottom_right});
  const double highest =
      std::max({top_left, top_right, bottom_left, bottom_right});
  if (highest > kMaxBilinearSpread * lowest) return kNaN;
  return (1 - fraction_u) * (1 - fraction_v) * top_left +
         fraction_u * (1 - fraction_v) * top_right +
         (1 - fraction_u) * fraction_v * bottom_left +
         fraction_u * fraction_v * bottom_right;
}

// Weighted least squares of l = s p + b on the relative residual
// (l - s p - b) / l, i.e. rows (p / l, 1 / l) -> 1, weights w: the normal
// equations [a00 a01; a01 a11] (s, b) = (b0, b1).
void SolveScaleShift(const std::vector<Pair>& pairs,
                     const std::vector<double>& weights, bool fit_shift,
                     double& scale, double& shift) {
  double a00 = 0.0;
  double a01 = 0.0;
  double a11 = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const double weight = weights[i];
    const double scale_row = pairs[i].prior / pairs[i].lidar;
    const double shift_row = 1.0 / pairs[i].lidar;
    a00 += weight * scale_row * scale_row;
    a01 += weight * scale_row * shift_row;
    a11 += weight * shift_row * shift_row;
    b0 += weight * scale_row;
    b1 += weight * shift_row;
  }
  const double determinant = a00 * a11 - a01 * a01;
  if (!fit_shift || std::abs(determinant) <=
                        kSingularDeterminant * std::max(1.0, a00 * a11)) {
    if (a00 > 0.0) scale = b0 / a00;
    shift = 0.0;
    return;
  }
  scale = (b0 * a11 - b1 * a01) / determinant;
  shift = (a00 * b1 - a01 * b0) / determinant;
}

double RelativeResidual(const Pair& pair, double scale, double shift) {
  return std::abs(pair.lidar - scale * pair.prior - shift) / pair.lidar;
}

// Pixels within `radius` (Chebyshev) of a set pixel: separable max filter.
std::vector<uint8_t> Dilate(const std::vector<uint8_t>& mask, int height,
                            int width, int radius) {
  if (radius <= 0) return mask;
  std::vector<uint8_t> horizontal(mask.size(), 0);
  std::vector<uint8_t> out(mask.size(), 0);
  for (int row = 0; row < height; ++row) {
    int last = -kNoSetPixel;  // Last set column seen.
    std::vector<int> next(width, kNoSetPixel);
    for (int column = width - 1, seen = kNoSetPixel; column >= 0; --column) {
      if (mask[static_cast<size_t>(row) * width + column]) seen = column;
      next[column] = seen;
    }
    for (int column = 0; column < width; ++column) {
      if (mask[static_cast<size_t>(row) * width + column]) last = column;
      horizontal[static_cast<size_t>(row) * width + column] =
          (column - last <= radius || next[column] - column <= radius) ? 1
                                                                       : 0;
    }
  }
  for (int column = 0; column < width; ++column) {
    int last = -kNoSetPixel;  // Last set row seen.
    std::vector<int> next(height, kNoSetPixel);
    for (int row = height - 1, seen = kNoSetPixel; row >= 0; --row) {
      if (horizontal[static_cast<size_t>(row) * width + column]) seen = row;
      next[row] = seen;
    }
    for (int row = 0; row < height; ++row) {
      if (horizontal[static_cast<size_t>(row) * width + column]) last = row;
      out[static_cast<size_t>(row) * width + column] =
          (row - last <= radius || next[row] - row <= radius) ? 1 : 0;
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
  const size_t pixels =
      static_cast<size_t>(height) * static_cast<size_t>(width);
  GR_CHECK(prior.size() == pixels && lidar.size() == pixels);
  GR_CHECK(prior_confidence.empty() || prior_confidence.size() == pixels);
  GR_CHECK(config.cell > 0);

  FusedDepth out;
  out.range.assign(pixels, kNaN);
  out.confidence.assign(pixels, 0.0f);
  out.source.assign(pixels, kDepthNone);
  std::vector<uint8_t> measured(pixels, 0);
  for (size_t pixel = 0; pixel < pixels; ++pixel) {
    if (InRange(lidar[pixel], config)) {
      out.range[pixel] = lidar[pixel];
      out.confidence[pixel] = kLidarConfidence;
      out.source[pixel] = kDepthLidar;
      measured[pixel] = 1;
    }
  }

  std::vector<Pair> pairs;
  if (samples.empty()) {
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
      if (measured[pixel] && InRange(prior[pixel], config)) {
        pairs.push_back({static_cast<int>(pixel % width),
                         static_cast<int>(pixel / width), prior[pixel],
                         lidar[pixel]});
      }
    }
  } else {
    for (const DepthSample& sample : samples) {
      if (!InRange(sample.range, config)) continue;
      const double prior_range =
          SamplePrior(prior, height, width, sample.u, sample.v);
      if (!InRange(prior_range, config)) continue;
      pairs.push_back({static_cast<int>(std::lround(sample.u)),
                       static_cast<int>(std::lround(sample.v)), prior_range,
                       sample.range});
    }
  }
  DepthAlignment& alignment = out.alignment;
  alignment.pixels = static_cast<int>(pairs.size());
  if (alignment.pixels < config.min_pixels) return out;

  // Start from the median ratio, then IRLS with a Cauchy weight.
  std::vector<double> ratios;
  ratios.reserve(pairs.size());
  for (const Pair& pair : pairs) ratios.push_back(pair.lidar / pair.prior);
  double scale = Median(ratios);
  alignment.raw_ratio = scale;
  double shift = 0.0;
  std::vector<double> weights(pairs.size(), 1.0);
  const double squared_scale = config.robust_scale * config.robust_scale;
  for (int iteration = 0; iteration < config.iterations; ++iteration) {
    for (size_t i = 0; i < pairs.size(); ++i) {
      const double residual = RelativeResidual(pairs[i], scale, shift);
      weights[i] = 1.0 / (1.0 + residual * residual / squared_scale);
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
  alignment.residual_p90 = Percentile(residuals, kResidualPercentile);
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
    const int column = std::clamp(pairs[i].u, 0, width - 1);
    const int row = std::clamp(pairs[i].v, 0, height - 1);
    cell_residuals[static_cast<size_t>(row / config.cell) * cells_x +
                   column / config.cell]
        .push_back(residuals[i]);
  }
  std::vector<double> cell_error(num_cells, alignment.residual_median);
  std::vector<uint8_t> bad(num_cells, 0);
  for (size_t cell = 0; cell < num_cells; ++cell) {
    if (static_cast<int>(cell_residuals[cell].size()) <
        config.min_cell_pixels) {
      continue;
    }
    ++alignment.cells_checked;
    cell_error[cell] = Median(cell_residuals[cell]);
    if (cell_error[cell] > config.max_cell_residual) bad[cell] = 1;
  }
  std::vector<uint8_t> rejected(num_cells, 0);
  for (int cell_y = 0; cell_y < cells_y; ++cell_y) {
    for (int cell_x = 0; cell_x < cells_x; ++cell_x) {
      if (!bad[static_cast<size_t>(cell_y) * cells_x + cell_x]) continue;
      ++alignment.cells_rejected;
      for (int offset_y = -kNeighbourhoodRadius;
           offset_y <= kNeighbourhoodRadius; ++offset_y) {
        for (int offset_x = -kNeighbourhoodRadius;
             offset_x <= kNeighbourhoodRadius; ++offset_x) {
          const int x = cell_x + offset_x;
          const int y = cell_y + offset_y;
          if (x < 0 || y < 0 || x >= cells_x || y >= cells_y) continue;
          rejected[static_cast<size_t>(y) * cells_x + x] = 1;
        }
      }
    }
  }

  const std::vector<uint8_t> covered =
      Dilate(measured, height, width, config.hole_radius);
  const double squared_tolerance = config.tolerance * config.tolerance;
  for (size_t pixel = 0; pixel < pixels; ++pixel) {
    if (covered[pixel] || !InRange(prior[pixel], config)) continue;
    const int row = static_cast<int>(pixel / width);
    const int column = static_cast<int>(pixel % width);
    const size_t cell = static_cast<size_t>(row / config.cell) * cells_x +
                        column / config.cell;
    if (rejected[cell]) continue;
    const double own =
        prior_confidence.empty()
            ? 1.0
            : std::clamp(static_cast<double>(prior_confidence[pixel]), 0.0,
                         1.0);
    const double error = cell_error[cell];
    const double confidence =
        own * std::exp(-0.5 * error * error / squared_tolerance);
    if (confidence < config.min_confidence) continue;
    const double range = scale * prior[pixel] + shift;
    if (!(range > 0.0)) continue;
    out.range[pixel] = static_cast<float>(range);
    out.confidence[pixel] = static_cast<float>(confidence);
    out.source[pixel] = kDepthPrior;
    ++out.prior_pixels;
  }
  return out;
}

}  // namespace gr
