#include "gr/semantics/vector_fit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <utility>

#include "gr/core/numeric.h"

namespace gr {
namespace {

// Ranges include their end up to this rounding slack (m).
constexpr double kRangeEndSlack = 1e-9;
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;
// A profile has two floor bands, of which at least 1 / kBandShareDivisor
// must be observed.
constexpr int kProfileSides = 2;
constexpr int kBandShareDivisor = 2;
// No sample found.
constexpr int kNoSample = -1;
// A crossing between two equal samples lies half way.
constexpr double kFlatCrossing = 0.5;
// End profiles start this far (m) inside the line.
constexpr double kEndBacktrack = 0.2;
// The floor beyond an end: this percentile (robust to paint reaching in).
constexpr double kFloorPercentile = 25.0;
// Lines shorter than this (m) are not re-measured.
constexpr double kMinLineLength = 0.3;
// Stations start and stop this far (m) in from the skeleton ends.
constexpr double kStationInset = 0.1;
// At least this many stations must remain inliers.
constexpr size_t kMinStations = 4;
// Rounds of the robust centre-line fit; stations within kInlierSigmas
// robust sigmas (MAD x kMadToSigma, plus kMinSigma m) stay inliers.
constexpr int kFitRounds = 3;
constexpr double kMadToSigma = 1.4826;
constexpr double kMinSigma = 1e-3;
constexpr double kInlierSigmas = 3.0;
// The paint brightness: the median of kPaintSamples samples along the
// fitted axis, kPaintInset (m) in from the ends.
constexpr int kPaintSamples = 20;
constexpr double kPaintInset = 0.2;

// numpy.arange(start, stop, step).
std::vector<double> Arange(double start, double stop, double step) {
  std::vector<double> out;
  const long count = static_cast<long>(std::ceil((stop - start) / step));
  for (long k = 0; k < count; ++k) out.push_back(start + k * step);
  return out;
}

// numpy.linspace(start, stop, count).
std::vector<double> Linspace(double start, double stop, int count) {
  std::vector<double> out(static_cast<size_t>(count));
  const double step = count > 1 ? (stop - start) / (count - 1) : 0.0;
  for (int k = 0; k < count; ++k) {
    out[static_cast<size_t>(k)] = start + k * step;
  }
  if (count > 1) out.back() = stop;
  return out;
}

// Bilinear sample of a raster at fractional (row, column); outside cells take
// the edge value (`extend`) or `fill`.
double Bilinear(const std::vector<double>& values, int width, int height,
                double row, double column, bool extend, double fill) {
  const double row_floor = std::floor(row);
  const double column_floor = std::floor(column);
  const double row_fraction = row - row_floor;
  const double column_fraction = column - column_floor;
  const auto at = [&](long cell_row, long cell_column) {
    if (extend) {
      cell_row = std::clamp<long>(cell_row, 0, height - 1);
      cell_column = std::clamp<long>(cell_column, 0, width - 1);
    } else if (cell_row < 0 || cell_column < 0 || cell_row >= height ||
               cell_column >= width) {
      return fill;
    }
    return values[static_cast<size_t>(cell_row * width + cell_column)];
  };
  const long top = static_cast<long>(row_floor);
  const long left = static_cast<long>(column_floor);
  return (1.0 - row_fraction) * ((1.0 - column_fraction) * at(top, left) +
                                 column_fraction * at(top, left + 1)) +
         row_fraction * ((1.0 - column_fraction) * at(top + 1, left) +
                         column_fraction * at(top + 1, left + 1));
}

// (centre, width) of the bright stripe in a profile across a line. The
// stripe must be observed; of the floor band on each side at least half.
std::optional<std::pair<double, double>> StripeCenter(
    const std::vector<double>& profile, const std::vector<uint8_t>& seen,
    const std::vector<double>& offsets, const LineFitConfig& config) {
  const size_t samples = profile.size();
  std::vector<double> floor;
  int sides_observed = 0;
  for (int side : {-1, 1}) {
    int band = 0;
    int observed = 0;
    for (size_t k = 0; k < samples; ++k) {
      const double distance = std::abs(offsets[k]);
      if (distance >= config.floor_band_inner &&
          distance <= config.floor_band_outer && side * offsets[k] > 0) {
        ++band;
        observed += seen[k];
      }
    }
    if (band > 0 && kBandShareDivisor * observed >= band) ++sides_observed;
    if (band == 0) return std::nullopt;  // numpy: mean of nothing is NaN.
  }
  if (sides_observed < kProfileSides) return std::nullopt;
  int peak = kNoSample;
  for (size_t k = 0; k < samples; ++k) {
    const double distance = std::abs(offsets[k]);
    if (distance < config.floor_band_inner) {
      if (!seen[k]) return std::nullopt;
      if (peak == kNoSample ||
          profile[k] > profile[static_cast<size_t>(peak)]) {
        peak = static_cast<int>(k);
      }
    } else if (distance <= config.floor_band_outer && seen[k]) {
      floor.push_back(profile[k]);
    }
  }
  if (peak == kNoSample || floor.empty()) return std::nullopt;
  const double floor_level = Median(floor);
  const double paint = profile[static_cast<size_t>(peak)];
  if (paint - floor_level < config.min_contrast) return std::nullopt;
  const double half = 0.5 * (paint + floor_level);
  int left = peak;
  while (left > 0 && profile[static_cast<size_t>(left - 1)] > half) --left;
  int right = peak;
  while (right + 1 < static_cast<int>(samples) &&
         profile[static_cast<size_t>(right + 1)] > half) {
    ++right;
  }
  if (left == 0 || right == static_cast<int>(samples) - 1) {
    return std::nullopt;
  }
  const auto crossing = [&](int outside, int inside) {
    const double outside_value = profile[static_cast<size_t>(outside)];
    const double inside_value = profile[static_cast<size_t>(inside)];
    const double fraction =
        inside_value != outside_value
            ? (half - outside_value) / (inside_value - outside_value)
            : kFlatCrossing;
    return offsets[static_cast<size_t>(outside)] +
           fraction * (offsets[static_cast<size_t>(inside)] -
                       offsets[static_cast<size_t>(outside)]);
  };
  const double left_edge = crossing(left - 1, left);
  const double right_edge = crossing(right + 1, right);
  return std::make_pair(0.5 * (left_edge + right_edge),
                        right_edge - left_edge);
}

// Position of one end along the centre line (`sign` +1: beyond p1) from
// `origin`, and whether it is reliable.
std::pair<double, bool> MeasureEnd(const FloorImage& image, const Vec2& origin,
                                   const Vec2& direction, double start,
                                   double sign, double paint,
                                   const LineFitConfig& config) {
  std::vector<double> positions;
  for (double offset : Arange(-kEndBacktrack,
                              config.end_search + kRangeEndSlack,
                              config.sample_step)) {
    positions.push_back(start + sign * offset);
  }
  std::vector<double> profile;
  for (double position : positions) {
    profile.push_back(image.Brightness(origin + position * direction));
  }
  std::vector<Vec2> beyond;
  for (double offset :
       Arange(config.end_floor_near, config.end_floor_far + kRangeEndSlack,
              config.sample_step)) {
    beyond.push_back(origin + (start + sign * offset) * direction);
  }
  // The floor around the continuation must be observed: a column or a car
  // next to the line hides its end even if the axis itself looks free.
  const Vec2 normal(-direction.y(), direction.x());
  int patch = 0;
  int patch_seen = 0;
  for (double side_offset :
       {-config.end_floor_width, 0.0, config.end_floor_width}) {
    for (const Vec2& point : beyond) {
      ++patch;
      patch_seen +=
          image.Observed(point + side_offset * normal) >= config.min_valid;
    }
  }
  if (static_cast<double>(patch_seen) / patch < config.min_valid) {
    return {start, false};
  }
  std::vector<double> floor;
  for (const Vec2& point : beyond) {
    if (image.Observed(point) >= config.min_valid) {
      floor.push_back(image.Brightness(point));
    }
  }
  if (floor.empty()) return {start, false};
  // Lower quartile: robust to paint that reaches into the band.
  const double floor_level = Percentile(floor, kFloorPercentile);
  if (paint - floor_level < config.min_contrast) return {start, false};
  const double half = 0.5 * (paint + floor_level);
  size_t below = 0;  // The first sample below half way.
  while (below < profile.size() && !(profile[below] < half)) ++below;
  if (below == profile.size() || below == 0) return {start, false};
  const double before = profile[below - 1];
  const double after = profile[below];
  const double fraction =
      before != after ? (before - half) / (before - after) : kFlatCrossing;
  const double position =
      positions[below - 1] +
      fraction * (positions[below] - positions[below - 1]);
  if (std::abs(position - start) > config.max_end_shift) {
    return {start, false};
  }
  return {position, true};
}

// numpy.polyfit(x, y, 1) on the flagged samples: (slope, intercept).
std::pair<double, double> FitLinear(const std::vector<double>& x,
                                    const std::vector<double>& y,
                                    const std::vector<uint8_t>& use) {
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xx = 0.0;
  double sum_xy = 0.0;
  double count = 0.0;
  for (size_t k = 0; k < x.size(); ++k) {
    if (!use[k]) continue;
    sum_x += x[k];
    sum_y += y[k];
    sum_xx += x[k] * x[k];
    sum_xy += x[k] * y[k];
    count += 1.0;
  }
  const double mean_x = sum_x / count;
  const double mean_y = sum_y / count;
  const double variance = sum_xx / count - mean_x * mean_x;
  const double slope =
      variance > 0.0 ? (sum_xy / count - mean_x * mean_y) / variance : 0.0;
  return {slope, mean_y - slope * mean_x};
}

}  // namespace

double FloorImage::Brightness(const Vec2& xy) const {
  return Bilinear(brightness, width, height, (xy.y() - y0) / res - kCellCentre,
                  (xy.x() - x0) / res - kCellCentre, true, 0.0);
}

double FloorImage::Observed(const Vec2& xy) const {
  return Bilinear(valid, width, height, (xy.y() - y0) / res - kCellCentre,
                  (xy.x() - x0) / res - kCellCentre, false, 0.0);
}

std::optional<LineFit> FitLine(const Vec2& p0, const Vec2& p1,
                               const FloorImage& image,
                               const LineFitConfig& config) {
  const double length = (p1 - p0).norm();
  if (length < kMinLineLength) return std::nullopt;
  const Vec2 direction = (p1 - p0) / length;
  const Vec2 normal(-direction.y(), direction.x());
  const std::vector<double> offsets =
      Arange(-config.half_span, config.half_span + kRangeEndSlack,
             config.sample_step);
  std::vector<double> centers;
  std::vector<double> widths;
  std::vector<double> stations;
  std::vector<double> profile(offsets.size());
  std::vector<uint8_t> seen(offsets.size());
  for (double station :
       Arange(kStationInset, length - kStationInset + kRangeEndSlack,
              config.station_step)) {
    for (size_t k = 0; k < offsets.size(); ++k) {
      const Vec2 xy = p0 + station * direction + offsets[k] * normal;
      profile[k] = image.Brightness(xy);
      seen[k] = image.Observed(xy) >= config.min_valid;
    }
    const auto stripe = StripeCenter(profile, seen, offsets, config);
    if (!stripe) continue;
    centers.push_back(stripe->first);
    widths.push_back(stripe->second);
    stations.push_back(station);
  }
  if (stations.size() < kMinStations) return std::nullopt;
  std::vector<uint8_t> keep(stations.size(), 1);
  std::vector<double> residual(stations.size());
  double slope = 0.0;
  double intercept = 0.0;
  // Centre offset linear in the station.
  for (int round = 0; round < kFitRounds; ++round) {
    std::tie(slope, intercept) = FitLinear(stations, centers, keep);
    std::vector<double> kept;
    for (size_t k = 0; k < stations.size(); ++k) {
      residual[k] = centers[k] - (slope * stations[k] + intercept);
      if (keep[k]) kept.push_back(std::abs(residual[k]));
    }
    const double sigma = kMadToSigma * Median(kept) + kMinSigma;
    size_t inliers = 0;
    for (size_t k = 0; k < stations.size(); ++k) {
      keep[k] = std::abs(residual[k]) <= kInlierSigmas * sigma;
      inliers += keep[k];
    }
    if (inliers < kMinStations) return std::nullopt;
  }
  const Vec2 origin = p0 + intercept * normal;
  Vec2 axis = direction + slope * normal;
  const double scale = axis.norm();
  axis /= scale;
  // Stations are measured along `direction`; on the fitted axis a point at
  // s lies at s * scale.
  const double skeleton_start = 0.0;
  const double skeleton_end = length * scale;
  std::vector<double> paint_samples;
  for (double position :
       Linspace(kPaintInset, std::max(skeleton_end - kPaintInset, kPaintInset),
                kPaintSamples)) {
    paint_samples.push_back(image.Brightness(origin + position * axis));
  }
  const double paint = Median(paint_samples);
  const auto [start, start_ok] =
      MeasureEnd(image, origin, axis, skeleton_start, -1.0, paint, config);
  const auto [end, end_ok] =
      MeasureEnd(image, origin, axis, skeleton_end, 1.0, paint, config);
  LineFit fit;
  fit.p0 = origin + start * axis;
  fit.p1 = origin + end * axis;
  std::vector<double> kept_widths;
  double squared = 0.0;
  int count = 0;
  for (size_t k = 0; k < stations.size(); ++k) {
    if (!keep[k]) continue;
    kept_widths.push_back(widths[k]);
    squared += residual[k] * residual[k];
    ++count;
  }
  fit.width = Median(kept_widths);
  fit.end_ok[0] = start_ok;
  fit.end_ok[1] = end_ok;
  fit.stations = count;
  fit.rms = std::sqrt(squared / count);
  return fit;
}

}  // namespace gr
