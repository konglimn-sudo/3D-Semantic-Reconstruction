#include "gr/registration/cloud_compare.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/geometry.h"
#include "gr/core/voxel.h"
#include "gr/registration/registration.h"

namespace gr {
namespace {

constexpr double kQuarterTurn = kPi / 2.0;
constexpr double kQuarterTurnDegrees = 90.0;
// Wall directions fix the yaw modulo a quarter turn: four candidates.
constexpr int kYawCandidates = 4;
// Normals of the downsampled clouds: from the closest 8 to 32 neighbours,
// and kept where the neighbourhood is this planar.
constexpr int kMinNormalNeighbors = 8;
constexpr int kMaxNormalNeighbors = 32;
constexpr double kMinSurfacePlanarity = 0.5;
// The circular histogram is smoothed by a Gaussian of one bin, cut off
// this many bins away.
constexpr int kSmoothingHalfWidth = 2;
// The reference occupancy image is dilated by this many cells.
constexpr int kDilationCells = 1;
// Clouds with fewer vertical-surface points carry no layout.
constexpr size_t kMinVertical = 200;
// At a placement voxel coarser than CoarseAlignConfig's 5 cm, the normal
// radius spans this many voxels (about 20 points).
constexpr double kPlacementNormalRadiusVoxels = 2.5;
// MeasureSurfaceDistance: the map voxel is at least half the distance gate
// and holds 60 points; a local plane needs 6 points and this planarity.
constexpr double kSurfaceVoxelPerGate = 0.5;
constexpr int kSurfacePointsPerVoxel = 60;
constexpr int kSurfacePlanePoints = 6;
constexpr double kSurfaceMinPlanarity = 0.3;

struct Structure {
  Points vertical;                 // Points on vertical surfaces.
  std::vector<double> azimuth;     // Their normal azimuth mod 90 deg (rad).
  std::vector<double> horizontal;  // Heights of horizontal-surface points.
};

Structure Analyze(const Points& cloud, const CoarseAlignConfig& config) {
  const Points points = VoxelDownsample(cloud, config.voxel);
  const NormalsResult normals = EstimateNormals(
      points, config.normal_radius, kMinNormalNeighbors, kMaxNormalNeighbors);
  Structure out;
  for (size_t i = 0; i < points.size(); ++i) {
    const Vec3& normal = normals.normals[i];
    if (normal.isZero() || normals.planarity[i] < kMinSurfacePlanarity) {
      continue;
    }
    const double nz = std::abs(normal.z());
    if (nz < config.vertical_max_nz) {
      out.vertical.push_back(points[i]);
      double azimuth =
          std::fmod(std::atan2(normal.y(), normal.x()), kQuarterTurn);
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
  for (double angle : angles) {
    const int bin =
        std::clamp(static_cast<int>(angle / period * bins), 0, bins - 1);
    histogram[static_cast<size_t>(bin)] += 1.0;
  }
  std::vector<double> smoothed(histogram.size(), 0.0);
  for (int bin = 0; bin < bins; ++bin) {
    for (int offset = -kSmoothingHalfWidth; offset <= kSmoothingHalfWidth;
         ++offset) {
      const int wrapped = ((bin + offset) % bins + bins) % bins;
      smoothed[static_cast<size_t>(bin)] +=
          histogram[static_cast<size_t>(wrapped)] *
          std::exp(-0.5 * offset * offset);
    }
  }
  return smoothed;
}

using Cell = std::pair<int, int>;

std::vector<Cell> UniqueCells(const Points& points, double cell,
                              const Mat3& rotation) {
  std::vector<Cell> cells;
  cells.reserve(points.size());
  for (const Vec3& point : points) {
    const Vec3 rotated = rotation * point;
    cells.emplace_back(static_cast<int>(std::floor(rotated.x() / cell)),
                       static_cast<int>(std::floor(rotated.y() / cell)));
  }
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  return cells;
}

// Dense occupancy image of the reference structure, dilated by
// kDilationCells. x0 and y0 are its first cell.
class OccupancyImage {
 public:
  explicit OccupancyImage(const std::vector<Cell>& cells) {
    GR_CHECK(!cells.empty());
    x0_ = y0_ = std::numeric_limits<int>::max();
    int last_x = std::numeric_limits<int>::min();
    int last_y = std::numeric_limits<int>::min();
    for (const Cell& cell : cells) {
      x0_ = std::min(x0_, cell.first - kDilationCells);
      y0_ = std::min(y0_, cell.second - kDilationCells);
      last_x = std::max(last_x, cell.first + kDilationCells);
      last_y = std::max(last_y, cell.second + kDilationCells);
    }
    width_ = last_x - x0_ + 1;
    height_ = last_y - y0_ + 1;
    occupied_.assign(static_cast<size_t>(width_) * height_, 0);
    for (const Cell& cell : cells) {
      for (int dx = -kDilationCells; dx <= kDilationCells; ++dx) {
        for (int dy = -kDilationCells; dy <= kDilationCells; ++dy) {
          occupied_[Index(cell.first + dx, cell.second + dy)] = 1;
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
  for (const Cell& cell : source) {
    hits += reference.At(cell.first + dx, cell.second + dy);
  }
  return static_cast<double>(hits) / static_cast<double>(source.size());
}

struct Offset {
  int dx = 0;
  int dy = 0;
  double score = -1.0;  // Below any score: the first offset wins.
};

// Exhaustive search over the offsets that make the two images overlap.
Offset SearchAll(const std::vector<Cell>& source,
                 const OccupancyImage& reference) {
  // The bounds of the source cells.
  int min_x = std::numeric_limits<int>::max();
  int min_y = std::numeric_limits<int>::max();
  int max_x = std::numeric_limits<int>::min();
  int max_y = std::numeric_limits<int>::min();
  for (const Cell& cell : source) {
    min_x = std::min(min_x, cell.first);
    min_y = std::min(min_y, cell.second);
    max_x = std::max(max_x, cell.first);
    max_y = std::max(max_y, cell.second);
  }
  const int dx_min = reference.x0() - max_x;
  const int dx_max = reference.x0() + reference.width() - min_x;
  const int dy_min = reference.y0() - max_y;
  const int dy_max = reference.y0() + reference.height() - min_y;
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
  for (const Offset& row_best : per_row) {
    if (row_best.score > best.score) best = row_best;
  }
  return best;
}

// The best offset within `reach` of (center_dx, center_dy).
Offset SearchAround(const std::vector<Cell>& source,
                    const OccupancyImage& reference, int center_dx,
                    int center_dy, int reach) {
  Offset best;
  for (int dy = center_dy - reach; dy <= center_dy + reach; ++dy) {
    for (int dx = center_dx - reach; dx <= center_dx + reach; ++dx) {
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
  for (int shift = min_shift; shift <= max_shift; ++shift) {
    double sum = 0.0;
    for (int i = 0; i < static_cast<int>(a.size()); ++i) {
      const int j = i - shift;
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
    const int count = static_cast<int>(correlation.size());
    std::vector<double> window(correlation.size(), 0.0);
    for (int k = 0; k < count; ++k) {
      for (int neighbor = std::max(k - 1, 0);
           neighbor <= std::min(k + 1, count - 1); ++neighbor) {
        window[static_cast<size_t>(k)] +=
            correlation[static_cast<size_t>(neighbor)];
      }
    }
    const double top = *std::max_element(window.begin(), window.end());
    peaks.clear();
    for (int k = 0; k < count; ++k) {
      const double value = window[static_cast<size_t>(k)];
      const bool left = k == 0 || value >= window[static_cast<size_t>(k - 1)];
      const bool right =
          k + 1 == count || value > window[static_cast<size_t>(k + 1)];
      if (left && right && value >= ratio * top) {
        peaks.emplace_back(value, min_shift + k);
      }
    }
    std::sort(peaks.begin(), peaks.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first > rhs.first;
    });
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
      AlignGravityAlignedCandidates(source, reference, config, kBestPeakOnly);
  if (candidates.empty()) return std::nullopt;
  return candidates.front();
}

std::vector<CoarseAlignment> AlignGravityAlignedCandidates(
    const Points& source, const Points& reference,
    const CoarseAlignConfig& config, double z_peak_ratio) {
  GR_CHECK(config.coarse_cell > config.fine_cell && config.fine_cell > 0.0);
  const Structure source_structure = Analyze(source, config);
  const Structure reference_structure = Analyze(reference, config);
  if (source_structure.vertical.size() < kMinVertical ||
      reference_structure.vertical.size() < kMinVertical) {
    return {};
  }
  // Yaw modulo 90 deg from the wall directions: rotating the source by psi
  // adds psi to its azimuths.
  const int bins = std::max(
      kYawCandidates,
      static_cast<int>(std::lround(kQuarterTurnDegrees / config.yaw_bin_deg)));
  const std::vector<double> source_histogram =
      CircularHistogram(source_structure.azimuth, kQuarterTurn, bins);
  const std::vector<double> reference_histogram =
      CircularHistogram(reference_structure.azimuth, kQuarterTurn, bins);
  int best_bin = 0;
  double best_correlation = -1.0;  // Below any correlation.
  for (int shift = 0; shift < bins; ++shift) {
    double correlation = 0.0;
    for (int bin = 0; bin < bins; ++bin) {
      correlation +=
          source_histogram[static_cast<size_t>(bin)] *
          reference_histogram[static_cast<size_t>((bin + shift) % bins)];
    }
    if (correlation > best_correlation) {
      best_correlation = correlation;
      best_bin = shift;
    }
  }
  const double base_yaw = kQuarterTurn * best_bin / bins;

  const OccupancyImage coarse_reference(UniqueCells(
      reference_structure.vertical, config.coarse_cell, Mat3::Identity()));
  const OccupancyImage fine_reference(UniqueCells(
      reference_structure.vertical, config.fine_cell, Mat3::Identity()));
  const double ratio = config.coarse_cell / config.fine_cell;
  // The fine search covers a coarse cell and one fine cell of margin.
  const int reach = static_cast<int>(std::ceil(ratio)) + 1;
  std::vector<std::pair<double, Pose>> candidates;
  for (int candidate = 0; candidate < kYawCandidates; ++candidate) {
    const double yaw = base_yaw + candidate * kQuarterTurn;
    const Mat3 rotation = So3Exp(Vec3(0.0, 0.0, yaw));
    const Offset coarse = SearchAll(
        UniqueCells(source_structure.vertical, config.coarse_cell, rotation),
        coarse_reference);
    const Offset fine = SearchAround(
        UniqueCells(source_structure.vertical, config.fine_cell, rotation),
        fine_reference, static_cast<int>(std::lround(coarse.dx * ratio)),
        static_cast<int>(std::lround(coarse.dy * ratio)), reach);
    candidates.emplace_back(
        fine.score, Pose(rotation, Vec3(fine.dx * config.fine_cell,
                                        fine.dy * config.fine_cell, 0.0)));
  }
  std::sort(
      candidates.begin(), candidates.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
  // The runner-up yaw against the best one.
  const double ambiguity = candidates[0].first > 0.0
                               ? candidates[1].first / candidates[0].first
                               : 1.0;

  // Height from the horizontal surfaces (floors, ceilings); levels of the
  // same layout give one correlation peak per level spacing.
  std::vector<double> heights = {0.0};
  const std::vector<double>& source_heights = source_structure.horizontal;
  const std::vector<double>& reference_heights = reference_structure.horizontal;
  if (!source_heights.empty() && !reference_heights.empty()) {
    const auto [source_lowest, source_highest] =
        std::minmax_element(source_heights.begin(), source_heights.end());
    const auto [reference_lowest, reference_highest] =
        std::minmax_element(reference_heights.begin(), reference_heights.end());
    const double z_min = std::min(*source_lowest, *reference_lowest);
    const int z_bins =
        static_cast<int>(
            std::ceil((std::max(*source_highest, *reference_highest) - z_min) /
                      config.z_bin)) +
        1;
    std::vector<double> source_counts(static_cast<size_t>(z_bins), 0.0);
    std::vector<double> reference_counts(static_cast<size_t>(z_bins), 0.0);
    for (double z : source_heights) {
      source_counts[static_cast<size_t>((z - z_min) / config.z_bin)] += 1.0;
    }
    for (double z : reference_heights) {
      reference_counts[static_cast<size_t>((z - z_min) / config.z_bin)] += 1.0;
    }
    heights.clear();
    for (int shift : PeakShifts(reference_counts, source_counts, -(z_bins - 1),
                                z_bins - 1, z_peak_ratio)) {
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

Placement PlaceGravityAligned(const Points& source, const Points& reference,
                              const PlacementConfig& config) {
  Placement placement;
  struct Fit {
    double score = 0.0;  // Coarse.
    IcpResult icp;
  };
  std::vector<Fit> fits;
  // The defaults of CoarseAlignConfig suit 5 cm maps; at a coarser voxel a
  // plane needs a wider neighbourhood for its normal (about 20 points).
  CoarseAlignConfig coarse_config;
  coarse_config.voxel = config.voxel;
  coarse_config.normal_radius = kPlacementNormalRadiusVoxels * config.voxel;
  for (const CoarseAlignment& coarse : AlignGravityAlignedCandidates(
           source, reference, coarse_config, config.z_peak_ratio)) {
    if (static_cast<int>(fits.size()) >= config.max_candidates) break;
    if (coarse.score < config.min_score) continue;
    fits.push_back(
        {coarse.score,
         AlignMultiScale(source, reference, coarse.reference_from_source,
                         config.icp_schedule, config.voxel, config.icp_iters)});
  }
  placement.hypotheses = static_cast<int>(fits.size());
  if (fits.empty()) return placement;
  const Fit& best = *std::max_element(
      fits.begin(), fits.end(), [](const Fit& lhs, const Fit& rhs) {
        return lhs.icp.fitness < rhs.icp.fitness;
      });
  for (const Fit& fit : fits) {
    const Pose delta = best.icp.pose.Inverse() * fit.icp.pose;
    const bool distinct =
        delta.translation.norm() > config.distinct_t ||
        RadToDeg(RotationAngle(delta.rotation)) > config.distinct_r;
    if (distinct) {
      placement.runner_up = std::max(placement.runner_up, fit.icp.fitness);
    }
  }
  placement.score = best.score;
  placement.fitness = best.icp.fitness;
  placement.rmse = best.icp.rmse;
  placement.reference_from_source = best.icp.pose;
  if (best.icp.fitness < config.min_fitness ||
      best.icp.rmse > config.max_rmse) {
    placement.verdict = "ICP rejected";
  } else if (best.icp.fitness - placement.runner_up < config.min_fitness_gap) {
    placement.verdict = "ambiguous";
  } else {
    placement.placed = true;
    placement.verdict = "placed";
  }
  return placement;
}

SurfaceDistances MeasureSurfaceDistance(const Points& query,
                                        const Points& reference, double radius,
                                        double max_distance) {
  GR_CHECK(radius > 0.0 && max_distance > 0.0);
  const double voxel = std::max(radius, kSurfaceVoxelPerGate * max_distance);
  VoxelHashMap map(voxel, VoxelHashMap::kUnbounded, kSurfacePointsPerVoxel);
  map.AddPoints(reference);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  SurfaceDistances out;
  out.signed_distance.assign(query.size(), nan);
  out.nearest.assign(query.size(), nan);
  out.normals.assign(query.size(), Vec3::Zero());
  const int64_t query_count = static_cast<int64_t>(query.size());
#ifdef _OPENMP
  constexpr int kSurfaceChunk = 1024;  // Query points per OpenMP work chunk.
#pragma omp parallel for schedule(dynamic, kSurfaceChunk)
#endif
  for (int64_t signed_index = 0; signed_index < query_count; ++signed_index) {
    const size_t i = static_cast<size_t>(signed_index);
    const std::optional<VoxelHashMap::Neighbor> nearest =
        map.Nearest(query[i], max_distance);
    if (!nearest) continue;
    out.nearest[i] = std::sqrt(nearest->distance_squared);
    const std::optional<VoxelHashMap::LocalPlane> plane =
        map.FitLocalPlane(nearest->point, radius, kSurfacePlanePoints);
    if (!plane || plane->planarity < kSurfaceMinPlanarity) continue;
    Vec3 normal = plane->normal;
    if (normal.z() < 0.0) normal = -normal;  // Horizontal surfaces face up.
    out.normals[i] = normal;
    out.signed_distance[i] = normal.dot(query[i] - plane->centroid);
  }
  return out;
}

}  // namespace gr
