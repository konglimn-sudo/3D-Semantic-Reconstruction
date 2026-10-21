#include "gr/cloud_compare.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "gr/check.h"
#include "gr/geometry.h"
#include "gr/registration.h"
#include "gr/voxel.h"

namespace gr {
namespace {

constexpr double kQuarterTurn = kPi / 2.0;

struct Structure {
  Points vertical;                 // Points on vertical surfaces.
  std::vector<double> azimuth;     // Their normal azimuth mod 90 deg (rad).
  std::vector<double> horizontal;  // Heights of horizontal-surface points.
};

Structure Analyze(const Points& cloud, const CoarseAlignConfig& config) {
  const Points points = VoxelDownsample(cloud, config.voxel);
  const NormalsResult normals =
      EstimateNormals(points, config.normal_radius, 8, 32);
  Structure out;
  for (size_t i = 0; i < points.size(); ++i) {
    const Vec3& n = normals.normals[i];
    if (n.isZero() || normals.planarity[i] < 0.5) continue;
    const double nz = std::abs(n.z());
    if (nz < config.vertical_max_nz) {
      out.vertical.push_back(points[i]);
      double azimuth = std::fmod(std::atan2(n.y(), n.x()), kQuarterTurn);
      if (azimuth < 0.0) azimuth += kQuarterTurn;
      out.azimuth.push_back(azimuth);
    } else if (nz > config.horizontal_min_nz) {
      out.horizontal.push_back(points[i].z());
    }
  }
  return out;
}

// Smoothed circular histogram of angles in [0, period).
std::vector<double> CircularHistogram(const std::vector<double>& angles,
                                      double period, int bins) {
  std::vector<double> histogram(static_cast<size_t>(bins), 0.0);
  for (double a : angles) {
    const int b = std::clamp(static_cast<int>(a / period * bins), 0, bins - 1);
    histogram[static_cast<size_t>(b)] += 1.0;
  }
  std::vector<double> smoothed(histogram.size(), 0.0);
  constexpr int kHalf = 2;
  for (int b = 0; b < bins; ++b) {
    for (int d = -kHalf; d <= kHalf; ++d) {
      const int k = ((b + d) % bins + bins) % bins;
      smoothed[static_cast<size_t>(b)] +=
          histogram[static_cast<size_t>(k)] * std::exp(-0.5 * d * d);
    }
  }
  return smoothed;
}

using Cell = std::pair<int, int>;

std::vector<Cell> UniqueCells(const Points& points, double cell,
                              const Mat3& rotation) {
  std::vector<Cell> cells;
  cells.reserve(points.size());
  for (const Vec3& p : points) {
    const Vec3 q = rotation * p;
    cells.emplace_back(static_cast<int>(std::floor(q.x() / cell)),
                       static_cast<int>(std::floor(q.y() / cell)));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

// Dense occupancy image of the reference structure, dilated by one cell.
class OccupancyImage {
 public:
  OccupancyImage(const std::vector<Cell>& cells) {
    GR_CHECK(!cells.empty());
    x0_ = y0_ = std::numeric_limits<int>::max();
    int x1 = std::numeric_limits<int>::min();
    int y1 = std::numeric_limits<int>::min();
    for (const Cell& c : cells) {
      x0_ = std::min(x0_, c.first - 1);
      y0_ = std::min(y0_, c.second - 1);
      x1 = std::max(x1, c.first + 1);
      y1 = std::max(y1, c.second + 1);
    }
    width_ = x1 - x0_ + 1;
    height_ = y1 - y0_ + 1;
    occupied_.assign(static_cast<size_t>(width_) * height_, 0);
    for (const Cell& c : cells) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          occupied_[Index(c.first + dx, c.second + dy)] = 1;
        }
      }
    }
  }

  bool At(int x, int y) const {
    if (x < x0_ || y < y0_ || x >= x0_ + width_ || y >= y0_ + height_) {
      return false;
    }
    return occupied_[Index(x, y)] != 0;
  }
  int x0() const { return x0_; }
  int y0() const { return y0_; }
  int width() const { return width_; }
  int height() const { return height_; }

 private:
  size_t Index(int x, int y) const {
    return static_cast<size_t>(y - y0_) * static_cast<size_t>(width_) +
           static_cast<size_t>(x - x0_);
  }

  int x0_ = 0;
  int y0_ = 0;
  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> occupied_;
};

double Score(const std::vector<Cell>& source, const OccupancyImage& reference,
             int dx, int dy) {
  int hits = 0;
  for (const Cell& c : source)
    hits += reference.At(c.first + dx, c.second + dy);
  return static_cast<double>(hits) / static_cast<double>(source.size());
}

struct Offset {
  int dx = 0;
  int dy = 0;
  double score = -1.0;
};

// Exhaustive search over the offsets that make the two images overlap.
Offset SearchAll(const std::vector<Cell>& source,
                 const OccupancyImage& reference) {
  int sx0 = std::numeric_limits<int>::max();
  int sy0 = std::numeric_limits<int>::max();
  int sx1 = std::numeric_limits<int>::min();
  int sy1 = std::numeric_limits<int>::min();
  for (const Cell& c : source) {
    sx0 = std::min(sx0, c.first);
    sy0 = std::min(sy0, c.second);
    sx1 = std::max(sx1, c.first);
    sy1 = std::max(sy1, c.second);
  }
  const int dx_min = reference.x0() - sx1;
  const int dx_max = reference.x0() + reference.width() - sx0;
  const int dy_min = reference.y0() - sy1;
  const int dy_max = reference.y0() + reference.height() - sy0;
  std::vector<Offset> per_row(static_cast<size_t>(dy_max - dy_min + 1));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int dy = dy_min; dy <= dy_max; ++dy) {
    Offset& best = per_row[static_cast<size_t>(dy - dy_min)];
    for (int dx = dx_min; dx <= dx_max; ++dx) {
      const double score = Score(source, reference, dx, dy);
      if (score > best.score) best = {dx, dy, score};
    }
  }
  Offset best;
  for (const Offset& o : per_row) {
    if (o.score > best.score) best = o;
  }
  return best;
}

Offset SearchAround(const std::vector<Cell>& source,
                    const OccupancyImage& reference, int cx, int cy,
                    int reach) {
  Offset best;
  for (int dy = cy - reach; dy <= cy + reach; ++dy) {
    for (int dx = cx - reach; dx <= cx + reach; ++dx) {
      const double score = Score(source, reference, dx, dy);
      if (score > best.score) best = {dx, dy, score};
    }
  }
  return best;
}

// Shifts (in bins) of `b` that match `a` (a[i] ~ b[i - shift]): the local
// maxima of the correlation reaching `ratio` of the best one, best first
// (the best alone with ratio >= 1).
std::vector<int> PeakShifts(const std::vector<double>& a,
                            const std::vector<double>& b, int min_shift,
                            int max_shift, double ratio) {
  std::vector<double> correlation;
  for (int s = min_shift; s <= max_shift; ++s) {
    double sum = 0.0;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
      const int j = i - s;
      if (j < 0 || j >= static_cast<int>(b.size())) continue;
      sum += a[static_cast<size_t>(i)] * b[static_cast<size_t>(j)];
    }
    correlation.push_back(sum);
  }
  const auto best = std::max_element(correlation.begin(), correlation.end());
  std::vector<std::pair<double, int>> peaks = {
      {*best, min_shift + static_cast<int>(best - correlation.begin())}};
  if (ratio < 1.0) {
    // Peaks on the 3-bin sum: a surface pair straddling a bin boundary
    // splits its peak over two shifts.
    const int n = static_cast<int>(correlation.size());
    std::vector<double> window(correlation.size(), 0.0);
    for (int k = 0; k < n; ++k) {
      for (int d = std::max(k - 1, 0); d <= std::min(k + 1, n - 1); ++d) {
        window[static_cast<size_t>(k)] += correlation[static_cast<size_t>(d)];
      }
    }
    const double top = *std::max_element(window.begin(), window.end());
    peaks.clear();
    for (int k = 0; k < n; ++k) {
      const double c = window[static_cast<size_t>(k)];
      const bool left = k == 0 || c >= window[static_cast<size_t>(k - 1)];
      const bool right = k + 1 == n || c > window[static_cast<size_t>(k + 1)];
      if (left && right && c >= ratio * top) {
        peaks.emplace_back(c, min_shift + k);
      }
    }
    std::sort(peaks.begin(), peaks.end(),
              [](const auto& x, const auto& y) { return x.first > y.first; });
  }
  std::vector<int> out;
  for (const auto& peak : peaks) out.push_back(peak.second);
  return out;
}

}  // namespace

std::optional<CoarseAlignment> AlignGravityAligned(
    const Points& source, const Points& reference,
    const CoarseAlignConfig& config) {
  const std::vector<CoarseAlignment> candidates =
      AlignGravityAlignedCandidates(source, reference, config, 1.0);
  if (candidates.empty()) return std::nullopt;
  return candidates.front();
}

std::vector<CoarseAlignment> AlignGravityAlignedCandidates(
    const Points& source, const Points& reference,
    const CoarseAlignConfig& config, double z_peak_ratio) {
  GR_CHECK(config.coarse_cell > config.fine_cell && config.fine_cell > 0.0);
  const Structure s = Analyze(source, config);
  const Structure r = Analyze(reference, config);
  constexpr size_t kMinVertical = 200;
  if (s.vertical.size() < kMinVertical || r.vertical.size() < kMinVertical) {
    return {};
  }
  // Yaw modulo 90 deg from the wall directions: rotating the source by psi
  // adds psi to its azimuths.
  const int bins =
      std::max(4, static_cast<int>(std::lround(90.0 / config.yaw_bin_deg)));
  const std::vector<double> hs =
      CircularHistogram(s.azimuth, kQuarterTurn, bins);
  const std::vector<double> hr =
      CircularHistogram(r.azimuth, kQuarterTurn, bins);
  int best_bin = 0;
  double best_corr = -1.0;
  for (int shift = 0; shift < bins; ++shift) {
    double corr = 0.0;
    for (int b = 0; b < bins; ++b) {
      corr += hs[static_cast<size_t>(b)] *
              hr[static_cast<size_t>((b + shift) % bins)];
    }
    if (corr > best_corr) {
      best_corr = corr;
      best_bin = shift;
    }
  }
  const double base_yaw = kQuarterTurn * best_bin / bins;

  const OccupancyImage coarse_reference(
      UniqueCells(r.vertical, config.coarse_cell, Mat3::Identity()));
  const OccupancyImage fine_reference(
      UniqueCells(r.vertical, config.fine_cell, Mat3::Identity()));
  const double ratio = config.coarse_cell / config.fine_cell;
  const int reach = static_cast<int>(std::ceil(ratio)) + 1;
  std::vector<std::pair<double, Pose>> candidates;
  for (int k = 0; k < 4; ++k) {
    const double yaw = base_yaw + k * kQuarterTurn;
    const Mat3 rotation = So3Exp(Vec3(0.0, 0.0, yaw));
    const Offset coarse =
        SearchAll(UniqueCells(s.vertical, config.coarse_cell, rotation),
                  coarse_reference);
    const Offset fine = SearchAround(
        UniqueCells(s.vertical, config.fine_cell, rotation), fine_reference,
        static_cast<int>(std::lround(coarse.dx * ratio)),
        static_cast<int>(std::lround(coarse.dy * ratio)), reach);
    candidates.emplace_back(
        fine.score, Pose(rotation, Vec3(fine.dx * config.fine_cell,
                                        fine.dy * config.fine_cell, 0.0)));
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  const double ambiguity = candidates[0].first > 0.0
                               ? candidates[1].first / candidates[0].first
                               : 1.0;

  // Height from the horizontal surfaces (floors, ceilings); levels of the
  // same layout give one correlation peak per level spacing.
  std::vector<double> heights = {0.0};
  if (!s.horizontal.empty() && !r.horizontal.empty()) {
    const auto [s_lo, s_hi] =
        std::minmax_element(s.horizontal.begin(), s.horizontal.end());
    const auto [r_lo, r_hi] =
        std::minmax_element(r.horizontal.begin(), r.horizontal.end());
    const double z0 = std::min(*s_lo, *r_lo);
    const int z_bins = static_cast<int>(std::ceil(
                           (std::max(*s_hi, *r_hi) - z0) / config.z_bin)) +
                       1;
    std::vector<double> zs(static_cast<size_t>(z_bins), 0.0);
    std::vector<double> zr(static_cast<size_t>(z_bins), 0.0);
    for (double z : s.horizontal) {
      zs[static_cast<size_t>((z - z0) / config.z_bin)] += 1.0;
    }
    for (double z : r.horizontal) {
      zr[static_cast<size_t>((z - z0) / config.z_bin)] += 1.0;
    }
    heights.clear();
    for (int shift :
         PeakShifts(zr, zs, -(z_bins - 1), z_bins - 1, z_peak_ratio)) {
      heights.push_back(shift * config.z_bin);
    }
  }
  std::vector<CoarseAlignment> out;
  for (const auto& [score, pose] : candidates) {
    for (double z : heights) {
      CoarseAlignment alignment;
      alignment.score = score;
      alignment.ambiguity = ambiguity;
      alignment.reference_from_source = pose;
      alignment.reference_from_source.translation.z() = z;
      out.push_back(alignment);
    }
  }
  return out;
}

SurfaceDistances MeasureSurfaceDistance(const Points& query,
                                        const Points& reference, double radius,
                                        double max_distance) {
  GR_CHECK(radius > 0.0 && max_distance > 0.0);
  const double voxel = std::max(radius, 0.5 * max_distance);
  VoxelHashMap map(voxel, 1e9, 60);
  map.AddPoints(reference);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  SurfaceDistances out;
  out.signed_distance.assign(query.size(), nan);
  out.nearest.assign(query.size(), nan);
  out.normals.assign(query.size(), Vec3::Zero());
  const int64_t n = static_cast<int64_t>(query.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
  for (int64_t ii = 0; ii < n; ++ii) {
    const size_t i = static_cast<size_t>(ii);
    const std::optional<VoxelHashMap::Neighbor> nearest =
        map.Nearest(query[i], max_distance);
    if (!nearest) continue;
    out.nearest[i] = std::sqrt(nearest->distance_squared);
    const std::optional<VoxelHashMap::LocalPlane> plane =
        map.FitLocalPlane(nearest->point, radius, 6);
    if (!plane || plane->planarity < 0.3) continue;
    Vec3 normal = plane->normal;
    if (normal.z() < 0.0) normal = -normal;  // Horizontal surfaces face up.
    out.normals[i] = normal;
    out.signed_distance[i] = normal.dot(query[i] - plane->centroid);
  }
  return out;
}

}  // namespace gr
