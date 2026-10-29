#include "gr/mapping/static_map.h"

#include <algorithm>
#include <cmath>

#include "gr/core/check.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

// A voxel's centre, in voxels from its corner.
constexpr double kVoxelCentre = 0.5;
// The keyframe takes the bits above the point index in a sequence number.
constexpr int kSequenceKeyframeShift = 32;
// Ranges below this (m) carry no elevation.
constexpr double kMinRange = 1e-6;
// Elevations are rounded to 1 / this of a degree before beams are found.
constexpr double kElevationStepsPerDegree = 10.0;
// Elevation gaps wider than this (deg) separate two beams.
constexpr double kBeamGap = 0.35;
// Beams needed to measure their spacing, and the spacing (deg) without, and
// its plausible range.
constexpr size_t kMinBeamsForSpacing = 3;
constexpr double kDefaultBeamSpacing = 1.0;
constexpr double kMinBeamSpacing = 0.3;
constexpr double kMaxBeamSpacing = 4.0;
// No path sample is near a map point.
constexpr int kNoSample = -1;
// Path samples within the ego radius of a point lie in its cell or in the
// cells this many steps around it (cells of the ego radius).
constexpr int kCellNeighbourhood = 1;

// The centre of a voxel of size `voxel`.
Vec3 VoxelCentre(const VoxelKey& key, double voxel) {
  return (Vec3(key.x, key.y, key.z) + Vec3::Constant(kVoxelCentre)) * voxel;
}

}  // namespace

MapAccumulator::MapAccumulator(double voxel, double max_range,
                               const MapFusionConfig& fusion)
    : voxel_(voxel),
      inverse_voxel_(1.0 / voxel),
      max_range_(max_range),
      fusion_(fusion) {
  GR_CHECK(voxel > 0.0);
}

void MapAccumulator::AddRanges(const Pose& world_from_lidar,
                               const Points& sweep) {
  for (const Vec3& point : sweep) {
    const double range = point.norm();
    if (range > max_range_) continue;
    const VoxelKey key = VoxelOf(world_from_lidar * point, inverse_voxel_);
    if (region_ && !region_->Contains(key)) continue;
    const auto [entry, inserted] =
        nearest_.try_emplace(key, static_cast<float>(range));
    if (!inserted) {
      entry->second = std::min(entry->second, static_cast<float>(range));
    }
  }
}

void MapAccumulator::Add(int keyframe, const Pose& world_from_lidar,
                         const Points& sweep,
                         const std::vector<float>& intensity) {
  const double angle_sigma = DegToRad(fusion_.angle_sigma_deg);
  const double range_variance = fusion_.range_sigma * fusion_.range_sigma;
  for (size_t i = 0; i < sweep.size(); ++i) {
    const double range = sweep[i].norm();
    if (range > max_range_) continue;
    const Vec3 point = world_from_lidar * sweep[i];
    const VoxelKey key = VoxelOf(point, inverse_voxel_);
    if (region_ && !region_->Contains(key)) continue;
    const float value = i < intensity.size() ? intensity[i] : 0.0f;
    Cell& cell = cells_[key];
    if (cell.keyframe == kNoKeyframe || range < cell.range) {
      cell.range = range;
      // Increases with every Add (keyframes in order) and point, whatever
      // the region.
      cell.sequence =
          (static_cast<int64_t>(keyframe) << kSequenceKeyframeShift) |
          static_cast<int64_t>(i);
      cell.point = point;
      cell.intensity = value;
      cell.keyframe = keyframe;
    }
    if (!fusion_.fuse) continue;
    const auto nearest = nearest_.find(key);
    const double nearest_range =
        nearest == nearest_.end() ? range : nearest->second;
    if (range > fusion_.range_ratio * nearest_range + fusion_.range_margin) {
      continue;
    }
    const double angular_sigma = range * angle_sigma;
    const float weight = static_cast<float>(
        range_variance / (range_variance + angular_sigma * angular_sigma));
    const Vec3 offset = point - VoxelCentre(key, voxel_);
    cell.weight += weight;
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      cell.sum[axis] += weight * static_cast<float>(offset[axis]);
    }
    int term = 0;
    for (int row = 0; row < kSpaceDim; ++row) {
      for (int column = row; column < kSpaceDim; ++column) {
        cell.outer[term++] +=
            weight * static_cast<float>(offset[row] * offset[column]);
      }
    }
    cell.weighted_intensity += weight * value;
    ++cell.count;
  }
}

MapAccumulator::Cloud MapAccumulator::Collect() const {
  std::vector<std::pair<const VoxelKey*, const Cell*>> cells;
  cells.reserve(cells_.size());
  for (const auto& [key, cell] : cells_) cells.emplace_back(&key, &cell);
  std::sort(cells.begin(), cells.end(), [](const auto& a, const auto& b) {
    return a.second->range != b.second->range
               ? a.second->range < b.second->range
               : a.second->sequence < b.second->sequence;
  });
  Cloud cloud;
  cloud.points.reserve(cells.size());
  cloud.intensity.reserve(cells.size());
  cloud.keyframe.reserve(cells.size());
  cloud.observations.reserve(cells.size());
  cloud.spread.reserve(cells.size());
  cloud.model_sigma.reserve(cells.size());
  cloud.range.reserve(cells.size());
  cloud.sequence.reserve(cells.size());
  cloud.voxel.reserve(cells.size());
  const double angle_sigma = DegToRad(fusion_.angle_sigma_deg);
  for (const auto& [key, cell] : cells) {
    cloud.keyframe.push_back(cell->keyframe);
    cloud.range.push_back(cell->range);
    cloud.sequence.push_back(cell->sequence);
    cloud.voxel.push_back(*key);
    if (!fusion_.fuse || cell->weight <= 0.0f) {
      cloud.points.push_back(cell->point);
      cloud.intensity.push_back(cell->intensity);
      cloud.observations.push_back(1);
      cloud.spread.push_back({});
      const double angular_sigma = cell->range * angle_sigma;
      cloud.model_sigma.push_back(static_cast<float>(
          std::sqrt(fusion_.range_sigma * fusion_.range_sigma +
                    angular_sigma * angular_sigma)));
      continue;
    }
    const double weight = cell->weight;
    const Vec3 mean(cell->sum[0] / weight, cell->sum[1] / weight,
                    cell->sum[2] / weight);
    cloud.points.push_back(VoxelCentre(*key, voxel_) + mean);
    cloud.intensity.push_back(
        static_cast<float>(cell->weighted_intensity / weight));
    cloud.observations.push_back(cell->count);
    std::array<float, kSymmetricTerms> spread{};
    int term = 0;
    for (int row = 0; row < kSpaceDim; ++row) {
      for (int column = row; column < kSpaceDim; ++column) {
        double value = cell->outer[term] / weight - mean[row] * mean[column];
        if (row == column) value = std::max(value, 0.0);
        spread[term++] = static_cast<float>(value);
      }
    }
    cloud.spread.push_back(spread);
    cloud.model_sigma.push_back(
        static_cast<float>(fusion_.range_sigma / std::sqrt(weight)));
  }
  return cloud;
}

std::vector<float> PointSigmas(const MapAccumulator::Cloud& cloud,
                               const Points& normals, double sigma_floor) {
  GR_CHECK(normals.size() == cloud.points.size());
  std::vector<float> out(cloud.points.size());
  for (size_t i = 0; i < out.size(); ++i) {
    const std::array<float, kSymmetricTerms>& spread = cloud.spread[i];
    const Mat3 covariance =
        (Mat3() << spread[0], spread[1], spread[2],  //
         spread[1], spread[3], spread[4],            //
         spread[2], spread[4], spread[5])
            .finished();
    const Vec3& normal = normals[i];
    const double along_normal =
        normal.isZero() ? covariance.trace() / kSpaceDim
                        : normal.dot(covariance * normal) /
                              normal.squaredNorm();
    const double model = cloud.model_sigma[i];
    const double variance =
        std::max(along_normal / std::max(cloud.observations[i], 1),
                 model * model) +
        sigma_floor * sigma_floor;
    out[i] = static_cast<float>(std::sqrt(variance));
  }
  return out;
}

NormalsResult EstimateOrientedNormals(const Points& points,
                                      const std::vector<int>& keyframe,
                                      const std::vector<Vec3>& keyframe_origins,
                                      double radius, int min_neighbors,
                                      int max_neighbors) {
  GR_CHECK(keyframe.size() == points.size());
  NormalsResult result =
      EstimateNormals(points, radius, min_neighbors, max_neighbors);
  for (size_t i = 0; i < points.size(); ++i) {
    const size_t observer = static_cast<size_t>(keyframe[i]);
    GR_CHECK(observer < keyframe_origins.size());
    if (result.normals[i].dot(keyframe_origins[observer] - points[i]) < 0.0) {
      result.normals[i] = -result.normals[i];
    }
  }
  return result;
}

double EstimateBeamSpacing(const std::vector<Points>& sweeps) {
  std::vector<double> elevations;
  for (const Points& sweep : sweeps) {
    for (const Vec3& point : sweep) {
      const double range = std::max(point.norm(), kMinRange);
      const double degrees =
          RadToDeg(std::asin(std::clamp(point.z() / range, -1.0, 1.0)));
      elevations.push_back(std::nearbyint(degrees * kElevationStepsPerDegree) /
                           kElevationStepsPerDegree);
    }
  }
  std::sort(elevations.begin(), elevations.end());
  elevations.erase(std::unique(elevations.begin(), elevations.end()),
                   elevations.end());
  // Beams are clusters of elevations separated by wider gaps.
  std::vector<double> centers;
  size_t start = 0;
  for (size_t end = 1; end <= elevations.size(); ++end) {
    if (end < elevations.size() &&
        elevations[end] - elevations[end - 1] <= kBeamGap) {
      continue;
    }
    double sum = 0.0;
    for (size_t member = start; member < end; ++member) {
      sum += elevations[member];
    }
    centers.push_back(sum / static_cast<double>(end - start));
    start = end;
  }
  double spacing = kDefaultBeamSpacing;
  if (centers.size() >= kMinBeamsForSpacing) {
    std::vector<double> gaps;
    for (size_t beam = 1; beam < centers.size(); ++beam) {
      gaps.push_back(centers[beam] - centers[beam - 1]);
    }
    spacing = Median(gaps);
  }
  return std::clamp(spacing, kMinBeamSpacing, kMaxBeamSpacing);
}

DynamicDetector::DynamicDetector(const Points& map, const Points& normals,
                                 const std::vector<int>& trip,
                                 const DynamicConfig& config)
    : config_(config),
      map_(map),
      trip_(trip),
      voter_(map, normals, config.bucket),
      elevation_resolution_(config.el_res) {
  GR_CHECK(normals.size() == map.size() && trip.size() == map.size());
}

void DynamicDetector::SetHeights(const std::vector<double>& height) {
  GR_CHECK(height.size() == map_.size());
  height_ = height;
}

void DynamicDetector::AddSweep(int trip, const Pose& world_from_lidar,
                               const Points& sweep) {
  if (elevation_resolution_ <= 0.0) {
    elevation_resolution_ = EstimateBeamSpacing({sweep});
  }
  VisibilityVoteOptions options;
  options.image.az_res_deg = config_.az_res;
  options.image.el_res_deg = elevation_resolution_;
  options.image.el_min_deg = config_.el_min;
  options.image.el_max_deg = config_.el_max;
  options.max_range = config_.max_range;
  options.margin_abs = config_.margin_abs;
  options.margin_rel = config_.margin_rel;
  options.min_cos = config_.min_cos;
  voter_.AddSweep(world_from_lidar, VoxelDownsample(sweep, config_.ray_voxel),
                  options, trip);
}

void DynamicDetector::AddEgoPath(int trip, const Points& vehicle_positions) {
  Points& path = ego_paths_[trip];
  path.insert(path.end(), vehicle_positions.begin(), vehicle_positions.end());
}

std::vector<bool> DynamicDetector::EgoSwept(int trip) const {
  std::vector<bool> swept(map_.size(), false);
  const auto trip_path = ego_paths_.find(trip);
  if (trip_path == ego_paths_.end() || trip_path->second.empty() ||
      config_.ego_radius <= 0.0) {
    return swept;
  }
  const Points& path = trip_path->second;
  const double radius = config_.ego_radius;
  // Path samples per cell of the ego radius.
  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> samples_by_cell;
  for (size_t sample = 0; sample < path.size(); ++sample) {
    samples_by_cell[CellOf(path[sample], radius)].push_back(
        static_cast<int>(sample));
  }
  // vector<bool> is not safe for parallel writes; collect into bytes first.
  std::vector<uint8_t> hit(map_.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t index = 0; index < static_cast<int64_t>(map_.size());
       ++index) {
    const Vec3& map_point = map_[static_cast<size_t>(index)];
    const CellKey cell = CellOf(map_point, radius);
    double best_squared = radius * radius;
    int nearest = kNoSample;
    for (int dx = -kCellNeighbourhood; dx <= kCellNeighbourhood; ++dx) {
      for (int dy = -kCellNeighbourhood; dy <= kCellNeighbourhood; ++dy) {
        const auto neighbors = samples_by_cell.find({cell.x + dx, cell.y + dy});
        if (neighbors == samples_by_cell.end()) continue;
        for (int sample : neighbors->second) {
          const double squared =
              (path[static_cast<size_t>(sample)].head<2>() -
               map_point.head<2>())
                  .squaredNorm();
          if (squared < best_squared) {
            best_squared = squared;
            nearest = sample;
          }
        }
      }
    }
    if (nearest == kNoSample) continue;
    // Height above the ego vehicle's floor.
    const double height =
        map_point.z() - path[static_cast<size_t>(nearest)].z();
    if (height > config_.min_height && height < config_.ego_height) {
      hit[static_cast<size_t>(index)] = 1;
    }
  }
  for (size_t i = 0; i < hit.size(); ++i) swept[i] = hit[i] != 0;
  return swept;
}

std::vector<DynamicKind> DynamicDetector::Classify() const {
  const size_t count = map_.size();
  std::vector<uint8_t> moved(count, 0);
  std::vector<uint8_t> changed(count, 0);
  for (int session : voter_.Sessions()) {
    const std::vector<int>& seen = voter_.SeenCounts(session);
    const std::vector<int>& through = voter_.ThroughCounts(session);
    for (size_t i = 0; i < count; ++i) {
      const int seen_through = through[i];
      if (seen_through < config_.min_through) continue;
      if (static_cast<double>(seen_through) /
              std::max(seen[i] + seen_through, 1) <
          config_.min_ratio) {
        continue;
      }
      (trip_[i] == session ? moved : changed)[i] = 1;
    }
  }
  for (const auto& [trip, path] : ego_paths_) {
    const std::vector<bool> swept = EgoSwept(trip);
    for (size_t i = 0; i < count; ++i) {
      if (swept[i]) (trip_[i] == trip ? moved : changed)[i] = 1;
    }
  }
  std::vector<DynamicKind> kind(count, DynamicKind::kStatic);
  for (size_t i = 0; i < count; ++i) {
    // The floor itself is grazed by many rays.
    if (!height_.empty() && !(height_[i] > config_.min_height)) continue;
    if (moved[i]) {
      kind[i] = DynamicKind::kMoved;
    } else if (changed[i]) {
      kind[i] = DynamicKind::kChanged;
    }
  }
  return kind;
}

}  // namespace gr
