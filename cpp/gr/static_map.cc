#include "gr/static_map.h"

#include <algorithm>
#include <cmath>

#include "gr/check.h"
#include "gr/numeric.h"

namespace gr {

MapAccumulator::MapAccumulator(double voxel, double max_range,
                               const MapFusionConfig& fusion)
    : voxel_(voxel),
      inv_voxel_(1.0 / voxel),
      max_range_(max_range),
      fusion_(fusion) {
  GR_CHECK(voxel > 0.0);
}

void MapAccumulator::AddRanges(const Pose& world_from_lidar,
                               const Points& sweep) {
  for (const Vec3& p : sweep) {
    const double range = p.norm();
    if (range > max_range_) continue;
    const auto [it, inserted] = nearest_.try_emplace(
        VoxelOf(world_from_lidar * p, inv_voxel_), static_cast<float>(range));
    if (!inserted) it->second = std::min(it->second, static_cast<float>(range));
  }
}

void MapAccumulator::Add(int keyframe, const Pose& world_from_lidar,
                         const Points& sweep,
                         const std::vector<float>& intensity) {
  const double angle_sigma = DegToRad(fusion_.angle_sigma_deg);
  const double reference = fusion_.range_sigma * fusion_.range_sigma;
  for (size_t i = 0; i < sweep.size(); ++i) {
    const double range = sweep[i].norm();
    if (range > max_range_) continue;
    const Vec3 point = world_from_lidar * sweep[i];
    const VoxelKey key = VoxelOf(point, inv_voxel_);
    const float value = i < intensity.size() ? intensity[i] : 0.0f;
    Cell& cell = cells_[key];
    if (cell.keyframe < 0 || range < cell.range) {
      cell.range = range;
      cell.sequence = sequence_;
      cell.point = point;
      cell.intensity = value;
      cell.keyframe = keyframe;
    }
    ++sequence_;
    if (!fusion_.fuse) continue;
    const auto nearest = nearest_.find(key);
    const double best = nearest == nearest_.end() ? range : nearest->second;
    if (range > fusion_.range_ratio * best + fusion_.range_margin) continue;
    const double angular = range * angle_sigma;
    const float weight =
        static_cast<float>(reference / (reference + angular * angular));
    const Vec3 center =
        (Vec3(key.x, key.y, key.z) + Vec3::Constant(0.5)) * voxel_;
    const Vec3 d = point - center;
    cell.weight += weight;
    for (int a = 0; a < 3; ++a)
      cell.sum[a] += weight * static_cast<float>(d[a]);
    int k = 0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        cell.outer[k++] += weight * static_cast<float>(d[a] * d[b]);
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
  const double angle_sigma = DegToRad(fusion_.angle_sigma_deg);
  for (const auto& [key, cell] : cells) {
    cloud.keyframe.push_back(cell->keyframe);
    if (!fusion_.fuse || cell->weight <= 0.0f) {
      cloud.points.push_back(cell->point);
      cloud.intensity.push_back(cell->intensity);
      cloud.observations.push_back(1);
      cloud.spread.push_back({});
      const double angular = cell->range * angle_sigma;
      cloud.model_sigma.push_back(static_cast<float>(std::sqrt(
          fusion_.range_sigma * fusion_.range_sigma + angular * angular)));
      continue;
    }
    const double w = cell->weight;
    const Vec3 mean(cell->sum[0] / w, cell->sum[1] / w, cell->sum[2] / w);
    const Vec3 center =
        (Vec3(key->x, key->y, key->z) + Vec3::Constant(0.5)) * voxel_;
    cloud.points.push_back(center + mean);
    cloud.intensity.push_back(static_cast<float>(cell->weighted_intensity / w));
    cloud.observations.push_back(cell->count);
    std::array<float, 6> spread{};
    int k = 0;
    for (int a = 0; a < 3; ++a) {
      for (int b = a; b < 3; ++b) {
        double value = cell->outer[k] / w - mean[a] * mean[b];
        if (a == b) value = std::max(value, 0.0);
        spread[k++] = static_cast<float>(value);
      }
    }
    cloud.spread.push_back(spread);
    cloud.model_sigma.push_back(
        static_cast<float>(fusion_.range_sigma / std::sqrt(w)));
  }
  return cloud;
}

std::vector<float> PointSigmas(const MapAccumulator::Cloud& cloud,
                               const Points& normals, double sigma_floor) {
  GR_CHECK(normals.size() == cloud.points.size());
  std::vector<float> out(cloud.points.size());
  for (size_t i = 0; i < out.size(); ++i) {
    const std::array<float, 6>& c = cloud.spread[i];
    const Mat3 covariance = (Mat3() << c[0], c[1], c[2],  //
                             c[1], c[3], c[4],            //
                             c[2], c[4], c[5])
                                .finished();
    const Vec3& n = normals[i];
    const double along = n.isZero() ? covariance.trace() / 3.0
                                    : n.dot(covariance * n) / n.squaredNorm();
    const double model = cloud.model_sigma[i];
    const double variance =
        std::max(along / std::max(cloud.observations[i], 1), model * model) +
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
    const size_t k = static_cast<size_t>(keyframe[i]);
    GR_CHECK(k < keyframe_origins.size());
    if (result.normals[i].dot(keyframe_origins[k] - points[i]) < 0.0) {
      result.normals[i] = -result.normals[i];
    }
  }
  return result;
}

double EstimateBeamSpacing(const std::vector<Points>& sweeps) {
  std::vector<double> elevations;
  for (const Points& sweep : sweeps) {
    for (const Vec3& p : sweep) {
      const double r = std::max(p.norm(), 1e-6);
      const double degrees =
          RadToDeg(std::asin(std::clamp(p.z() / r, -1.0, 1.0)));
      elevations.push_back(std::nearbyint(degrees * 10.0) / 10.0);
    }
  }
  std::sort(elevations.begin(), elevations.end());
  elevations.erase(std::unique(elevations.begin(), elevations.end()),
                   elevations.end());
  // Beams are clusters of elevations separated by gaps wider than 0.35 deg.
  std::vector<double> centers;
  size_t start = 0;
  for (size_t i = 1; i <= elevations.size(); ++i) {
    if (i < elevations.size() && elevations[i] - elevations[i - 1] <= 0.35) {
      continue;
    }
    double sum = 0.0;
    for (size_t a = start; a < i; ++a) sum += elevations[a];
    centers.push_back(sum / static_cast<double>(i - start));
    start = i;
  }
  double spacing = 1.0;
  if (centers.size() > 2) {
    std::vector<double> gaps;
    for (size_t a = 1; a < centers.size(); ++a) {
      gaps.push_back(centers[a] - centers[a - 1]);
    }
    spacing = Median(gaps);
  }
  return std::clamp(spacing, 0.3, 4.0);
}

DynamicDetector::DynamicDetector(const Points& map, const Points& normals,
                                 const std::vector<int>& trip,
                                 const DynamicConfig& config)
    : config_(config),
      map_(map),
      trip_(trip),
      voter_(map, normals, config.bucket),
      el_res_(config.el_res) {
  GR_CHECK(normals.size() == map.size() && trip.size() == map.size());
}

void DynamicDetector::SetHeights(const std::vector<double>& height) {
  GR_CHECK(height.size() == map_.size());
  height_ = height;
}

void DynamicDetector::AddSweep(int trip, const Pose& world_from_lidar,
                               const Points& sweep) {
  if (el_res_ <= 0.0) el_res_ = EstimateBeamSpacing({sweep});
  VisibilityVoteOptions options;
  options.image.az_res_deg = config_.az_res;
  options.image.el_res_deg = el_res_;
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
  const auto it = ego_paths_.find(trip);
  if (it == ego_paths_.end() || it->second.empty() ||
      config_.ego_radius <= 0.0) {
    return swept;
  }
  const Points& path = it->second;
  const double radius = config_.ego_radius;
  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> grid;
  for (size_t j = 0; j < path.size(); ++j) {
    grid[CellOf(path[j], radius)].push_back(static_cast<int>(j));
  }
  // vector<bool> is not safe for parallel writes; collect into bytes first.
  std::vector<uint8_t> hit(map_.size(), 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(map_.size()); ++ii) {
    const Vec3& q = map_[static_cast<size_t>(ii)];
    const CellKey cell = CellOf(q, radius);
    double best = radius * radius;
    int nearest = -1;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        const auto g = grid.find({cell.x + dx, cell.y + dy});
        if (g == grid.end()) continue;
        for (int j : g->second) {
          const double d2 =
              (path[static_cast<size_t>(j)].head<2>() - q.head<2>())
                  .squaredNorm();
          if (d2 < best) {
            best = d2;
            nearest = j;
          }
        }
      }
    }
    if (nearest < 0) continue;
    // Height above the ego vehicle's floor.
    const double dz = q.z() - path[static_cast<size_t>(nearest)].z();
    if (dz > config_.min_height && dz < config_.ego_height) {
      hit[static_cast<size_t>(ii)] = 1;
    }
  }
  for (size_t i = 0; i < hit.size(); ++i) swept[i] = hit[i] != 0;
  return swept;
}

std::vector<DynamicKind> DynamicDetector::Classify() const {
  const size_t n = map_.size();
  std::vector<uint8_t> moved(n, 0);
  std::vector<uint8_t> changed(n, 0);
  for (int session : voter_.Sessions()) {
    const std::vector<int>& seen = voter_.SeenCounts(session);
    const std::vector<int>& through = voter_.ThroughCounts(session);
    for (size_t i = 0; i < n; ++i) {
      const int t = through[i];
      if (t < config_.min_through) continue;
      if (static_cast<double>(t) / std::max(seen[i] + t, 1) <
          config_.min_ratio) {
        continue;
      }
      (trip_[i] == session ? moved : changed)[i] = 1;
    }
  }
  for (const auto& [trip, path] : ego_paths_) {
    const std::vector<bool> swept = EgoSwept(trip);
    for (size_t i = 0; i < n; ++i) {
      if (swept[i]) (trip_[i] == trip ? moved : changed)[i] = 1;
    }
  }
  std::vector<DynamicKind> kind(n, DynamicKind::kStatic);
  for (size_t i = 0; i < n; ++i) {
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
