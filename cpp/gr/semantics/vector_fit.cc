#include "gr/semantics/vector_fit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

#include "gr/core/numeric.h"

namespace gr {
namespace {

// numpy.arange(start, stop, step).
std::vector<double> Arange(double start, double stop, double step) {
  std::vector<double> out;
  const long n = static_cast<long>(std::ceil((stop - start) / step));
  for (long k = 0; k < n; ++k) out.push_back(start + k * step);
  return out;
}

// numpy.linspace(start, stop, n).
std::vector<double> Linspace(double start, double stop, int n) {
  std::vector<double> out(static_cast<size_t>(n));
  const double step = n > 1 ? (stop - start) / (n - 1) : 0.0;
  for (int k = 0; k < n; ++k) out[static_cast<size_t>(k)] = start + k * step;
  if (n > 1) out.back() = stop;
  return out;
}

// Bilinear sample of a raster at fractional (row, col); outside cells take
// the edge value (`extend`) or `fill`.
double Bilinear(const std::vector<double>& values, int width, int height,
                double row, double col, bool extend, double fill) {
  const double r0 = std::floor(row);
  const double c0 = std::floor(col);
  const double fr = row - r0;
  const double fc = col - c0;
  auto at = [&](long r, long c) {
    if (extend) {
      r = std::clamp<long>(r, 0, height - 1);
      c = std::clamp<long>(c, 0, width - 1);
    } else if (r < 0 || c < 0 || r >= height || c >= width) {
      return fill;
    }
    return values[static_cast<size_t>(r * width + c)];
  };
  const long r = static_cast<long>(r0);
  const long c = static_cast<long>(c0);
  return (1.0 - fr) * ((1.0 - fc) * at(r, c) + fc * at(r, c + 1)) +
         fr * ((1.0 - fc) * at(r + 1, c) + fc * at(r + 1, c + 1));
}

// (centre, width) of the bright stripe in a profile across a line. The
// stripe must be observed; of the floor band on each side at least half.
std::optional<std::pair<double, double>> StripeCenter(
    const std::vector<double>& profile, const std::vector<uint8_t>& seen,
    const std::vector<double>& offsets, const LineFitConfig& config) {
  const size_t n = profile.size();
  std::vector<double> floor;
  int sides_ok = 0;
  for (int side : {-1, 1}) {
    int zone = 0;
    int observed = 0;
    for (size_t k = 0; k < n; ++k) {
      const double a = std::abs(offsets[k]);
      if (a >= config.floor_band_inner && a <= config.floor_band_outer &&
          side * offsets[k] > 0) {
        ++zone;
        observed += seen[k];
      }
    }
    if (zone > 0 && 2 * observed >= zone) ++sides_ok;
    if (zone == 0) return std::nullopt;  // numpy: mean of nothing is NaN.
  }
  if (sides_ok < 2) return std::nullopt;
  int peak = -1;
  for (size_t k = 0; k < n; ++k) {
    const double a = std::abs(offsets[k]);
    if (a < config.floor_band_inner) {
      if (!seen[k]) return std::nullopt;
      if (peak < 0 || profile[k] > profile[static_cast<size_t>(peak)]) {
        peak = static_cast<int>(k);
      }
    } else if (a <= config.floor_band_outer && seen[k]) {
      floor.push_back(profile[k]);
    }
  }
  if (peak < 0 || floor.empty()) return std::nullopt;
  const double floor_level = Median(floor);
  const double paint = profile[static_cast<size_t>(peak)];
  if (paint - floor_level < config.min_contrast) return std::nullopt;
  const double half = 0.5 * (paint + floor_level);
  int left = peak;
  while (left > 0 && profile[static_cast<size_t>(left - 1)] > half) --left;
  int right = peak;
  while (right + 1 < static_cast<int>(n) &&
         profile[static_cast<size_t>(right + 1)] > half) {
    ++right;
  }
  if (left == 0 || right == static_cast<int>(n) - 1) return std::nullopt;
  auto crossing = [&](int outside, int inside) {
    const double a = profile[static_cast<size_t>(outside)];
    const double b = profile[static_cast<size_t>(inside)];
    const double t = b != a ? (half - a) / (b - a) : 0.5;
    return offsets[static_cast<size_t>(outside)] +
           t * (offsets[static_cast<size_t>(inside)] -
                offsets[static_cast<size_t>(outside)]);
  };
  const double lo = crossing(left - 1, left);
  const double hi = crossing(right + 1, right);
  return std::make_pair(0.5 * (lo + hi), hi - lo);
}

// Position of one end along the centre line (`sign` +1: beyond p1) from
// `origin`, and whether it is reliable.
std::pair<double, bool> End(const FloorImage& image, const Vec2& origin,
                            const Vec2& direction, double start, double sign,
                            double paint, const LineFitConfig& config) {
  std::vector<double> s;
  for (double v : Arange(-0.2, config.end_search + 1e-9, config.sample_step)) {
    s.push_back(start + sign * v);
  }
  std::vector<double> profile;
  for (double v : s) {
    profile.push_back(image.Brightness(origin + v * direction));
  }
  std::vector<Vec2> beyond;
  for (double v : Arange(config.end_floor_near, config.end_floor_far + 1e-9,
                         config.sample_step)) {
    beyond.push_back(origin + (start + sign * v) * direction);
  }
  // The floor around the continuation must be observed: a column or a car
  // next to the line hides its end even if the axis itself looks free.
  const Vec2 normal(-direction.y(), direction.x());
  int patch = 0;
  int patch_seen = 0;
  for (double w : {-config.end_floor_width, 0.0, config.end_floor_width}) {
    for (const Vec2& p : beyond) {
      ++patch;
      patch_seen += image.Observed(p + w * normal) >= config.min_valid;
    }
  }
  if (static_cast<double>(patch_seen) / patch < config.min_valid) {
    return {start, false};
  }
  std::vector<double> floor;
  for (const Vec2& p : beyond) {
    if (image.Observed(p) >= config.min_valid) {
      floor.push_back(image.Brightness(p));
    }
  }
  if (floor.empty()) return {start, false};
  // Lower quartile: robust to paint that reaches into the band.
  const double floor_level = Percentile(floor, 25.0);
  if (paint - floor_level < config.min_contrast) return {start, false};
  const double half = 0.5 * (paint + floor_level);
  size_t k = 0;
  while (k < profile.size() && !(profile[k] < half)) ++k;
  if (k == profile.size() || k == 0) return {start, false};
  const double a = profile[k - 1];
  const double b = profile[k];
  const double t = a != b ? (a - half) / (a - b) : 0.5;
  const double position = s[k - 1] + t * (s[k] - s[k - 1]);
  if (std::abs(position - start) > config.max_end_shift) {
    return {start, false};
  }
  return {position, true};
}

// numpy.polyfit(x, y, 1) on the flagged samples: (slope, intercept).
std::pair<double, double> FitLinear(const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    const std::vector<uint8_t>& use) {
  double sx = 0.0;
  double sy = 0.0;
  double sxx = 0.0;
  double sxy = 0.0;
  double n = 0.0;
  for (size_t k = 0; k < x.size(); ++k) {
    if (!use[k]) continue;
    sx += x[k];
    sy += y[k];
    sxx += x[k] * x[k];
    sxy += x[k] * y[k];
    n += 1.0;
  }
  const double mx = sx / n;
  const double my = sy / n;
  const double var = sxx / n - mx * mx;
  const double slope = var > 0.0 ? (sxy / n - mx * my) / var : 0.0;
  return {slope, my - slope * mx};
}

}  // namespace

double FloorImage::Brightness(const Vec2& xy) const {
  return Bilinear(brightness, width, height, (xy.y() - y0) / res - 0.5,
                  (xy.x() - x0) / res - 0.5, true, 0.0);
}

double FloorImage::Observed(const Vec2& xy) const {
  return Bilinear(valid, width, height, (xy.y() - y0) / res - 0.5,
                  (xy.x() - x0) / res - 0.5, false, 0.0);
}

std::optional<LineFit> FitLine(const Vec2& p0, const Vec2& p1,
                               const FloorImage& image,
                               const LineFitConfig& config) {
  const double length = (p1 - p0).norm();
  if (length < 0.3) return std::nullopt;
  const Vec2 direction = (p1 - p0) / length;
  const Vec2 normal(-direction.y(), direction.x());
  const std::vector<double> offsets =
      Arange(-config.half_span, config.half_span + 1e-9, config.sample_step);
  std::vector<double> centers;
  std::vector<double> widths;
  std::vector<double> used;
  std::vector<double> profile(offsets.size());
  std::vector<uint8_t> seen(offsets.size());
  for (double s : Arange(0.1, length - 0.1 + 1e-9, config.station_step)) {
    for (size_t k = 0; k < offsets.size(); ++k) {
      const Vec2 xy = p0 + s * direction + offsets[k] * normal;
      profile[k] = image.Brightness(xy);
      seen[k] = image.Observed(xy) >= config.min_valid;
    }
    const auto stripe = StripeCenter(profile, seen, offsets, config);
    if (!stripe) continue;
    centers.push_back(stripe->first);
    widths.push_back(stripe->second);
    used.push_back(s);
  }
  if (used.size() < 4) return std::nullopt;
  std::vector<uint8_t> keep(used.size(), 1);
  std::vector<double> residual(used.size());
  double slope = 0.0;
  double intercept = 0.0;
  for (int round = 0; round < 3; ++round) {  // Centre offset linear in s.
    std::tie(slope, intercept) = FitLinear(used, centers, keep);
    std::vector<double> kept;
    for (size_t k = 0; k < used.size(); ++k) {
      residual[k] = centers[k] - (slope * used[k] + intercept);
      if (keep[k]) kept.push_back(std::abs(residual[k]));
    }
    const double mad = 1.4826 * Median(kept) + 1e-3;
    int count = 0;
    for (size_t k = 0; k < used.size(); ++k) {
      keep[k] = std::abs(residual[k]) <= 3.0 * mad;
      count += keep[k];
    }
    if (count < 4) return std::nullopt;
  }
  const Vec2 origin = p0 + intercept * normal;
  Vec2 axis = direction + slope * normal;
  const double scale = axis.norm();
  axis /= scale;
  // Stations are measured along `direction`; on the fitted axis a point at
  // s lies at s * scale.
  const double start0 = 0.0;
  const double end0 = length * scale;
  std::vector<double> paint_samples;
  for (double v : Linspace(0.2, std::max(end0 - 0.2, 0.2), 20)) {
    paint_samples.push_back(image.Brightness(origin + v * axis));
  }
  const double paint = Median(paint_samples);
  const auto [s0, ok0] = End(image, origin, axis, start0, -1.0, paint, config);
  const auto [s1, ok1] = End(image, origin, axis, end0, 1.0, paint, config);
  LineFit fit;
  fit.p0 = origin + s0 * axis;
  fit.p1 = origin + s1 * axis;
  std::vector<double> kept_widths;
  double squared = 0.0;
  int count = 0;
  for (size_t k = 0; k < used.size(); ++k) {
    if (!keep[k]) continue;
    kept_widths.push_back(widths[k]);
    squared += residual[k] * residual[k];
    ++count;
  }
  fit.width = Median(kept_widths);
  fit.end_ok[0] = ok0;
  fit.end_ok[1] = ok1;
  fit.stations = count;
  fit.rms = std::sqrt(squared / count);
  return fit;
}

}  // namespace gr
