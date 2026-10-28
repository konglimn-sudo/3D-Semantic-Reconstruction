#include "gr/mapping/tiled_map.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Horizontal distance from `p` to the box of a voxel region (m).
double DistanceToRegion(const Vec3& p, const VoxelRegion& region,
                        double voxel) {
  const double dx = std::max(
      {region.x0 * voxel - p.x(), 0.0, p.x() - region.x1 * voxel});
  const double dy = std::max(
      {region.y0 * voxel - p.y(), 0.0, p.y() - region.y1 * voxel});
  return std::hypot(dx, dy);
}

VoxelRegion Grow(const VoxelRegion& region, int margin) {
  return {region.x0 - margin, region.x1 + margin, region.y0 - margin,
          region.y1 + margin};
}

int FloorDiv(int a, int b) {
  return a / b - (a % b != 0 && (a < 0) != (b < 0));
}

struct Builder {
  const std::vector<Pose>& poses;
  const std::vector<int>& trips;
  const std::vector<Points>& paths;
  const SweepLoader& load;
  const TiledMapConfig& config;
  std::vector<Vec3> origins;

  // Keyframes in order whose sensor is within `reach` of `region` (all for
  // a whole map).
  std::vector<int> Reaching(const MapTile& tile, const VoxelRegion& region,
                            double reach) const {
    std::vector<int> out;
    for (size_t k = 0; k < poses.size(); ++k) {
      if (tile.whole || DistanceToRegion(poses[k].translation, region,
                                         config.voxel) <= reach) {
        out.push_back(static_cast<int>(k));
      }
    }
    return out;
  }

  // The tile's points with normals and sigmas: its voxels plus a halo of
  // the normal radius are accumulated, the halo only serves the normals.
  MapChunk Fuse(const MapTile& tile) const {
    const MapFusionConfig& fusion = config.fusion;
    MapAccumulator accumulator(config.voxel, config.max_range, fusion);
    const int halo =
        static_cast<int>(std::ceil(config.normal_radius / config.voxel)) + 1;
    const VoxelRegion region = Grow(tile.region, halo);
    if (!tile.whole) accumulator.SetRegion(region);
    const std::vector<int> sweeps = Reaching(tile, region, config.max_range);
    Points points;
    std::vector<float> intensity;
    if (fusion.fuse) {
      for (int k : sweeps) {
        load(k, &points, &intensity);
        accumulator.AddRanges(poses[static_cast<size_t>(k)], points);
      }
    }
    for (int k : sweeps) {
      load(k, &points, &intensity);
      accumulator.Add(k, poses[static_cast<size_t>(k)], points, intensity);
    }
    const MapAccumulator::Cloud cloud = accumulator.Collect();
    const NormalsResult normals = EstimateOrientedNormals(
        cloud.points, cloud.keyframe, origins, config.normal_radius, 6, 48);
    std::vector<float> sigma;
    if (fusion.fuse) {
      sigma = PointSigmas(cloud, normals.normals, fusion.sigma_floor);
    }
    MapChunk chunk;
    for (size_t i = 0; i < cloud.points.size(); ++i) {
      if (!tile.whole && !tile.region.Contains(cloud.voxel[i])) continue;
      const int k = cloud.keyframe[i];
      chunk.xyz.push_back(cloud.points[i]);
      chunk.intensity.push_back(cloud.intensity[i]);
      chunk.keyframe.push_back(k);
      chunk.trip.push_back(
          static_cast<int16_t>(trips[static_cast<size_t>(k)]));
      chunk.normals.push_back(normals.normals[i]);
      chunk.planarity.push_back(static_cast<float>(normals.planarity[i]));
      if (!sigma.empty()) chunk.sigma.push_back(sigma[i]);
      chunk.observations.push_back(
          static_cast<uint16_t>(std::min(cloud.observations[i], 65535)));
      chunk.range.push_back(cloud.range[i]);
      chunk.sequence.push_back(cloud.sequence[i]);
    }
    return chunk;
  }

  // Levels (the later steps see the stored float32 normals) and dynamic
  // points of a tile.
  void Finish(const MapTile& tile, const LevelModel& levels, double el_res,
              MapChunk* chunk) const {
    const LevelAssignment assignment = AssignLevels(levels, chunk->xyz);
    chunk->level = assignment.level;
    chunk->height = assignment.height;
    chunk->on_ramp = assignment.on_ramp;
    const size_t n = chunk->size();
    chunk->dynamic_kind.assign(n, 0);
    if (!config.dynamic || n == 0) return;
    Points normals = chunk->normals;
    for (Vec3& v : normals) v = v.cast<float>().cast<double>();
    DynamicDetector detector(
        chunk->xyz, normals,
        std::vector<int>(chunk->trip.begin(), chunk->trip.end()),
        config.dynamic_config);
    detector.SetElevationResolution(el_res);
    Points points;
    std::vector<float> intensity;
    for (int k : Reaching(tile, tile.region, config.dynamic_config.max_range)) {
      load(k, &points, &intensity);
      detector.AddSweep(trips[static_cast<size_t>(k)],
                        poses[static_cast<size_t>(k)], points);
    }
    for (size_t t = 0; t < paths.size(); ++t) {
      detector.AddEgoPath(static_cast<int>(t), paths[t]);
    }
    detector.SetHeights(
        std::vector<double>(chunk->height.begin(), chunk->height.end()));
    const std::vector<DynamicKind> kinds = detector.Classify();
    for (size_t i = 0; i < n; ++i) {
      chunk->dynamic_kind[i] = static_cast<uint8_t>(kinds[i]);
    }
  }

  // The tiles covering every return within max_range of a sensor position.
  std::vector<MapTile> Tiles() const {
    if (config.tile_size <= 0.0 || poses.empty()) return {MapTile{}};
    const int n = std::max(
        1, static_cast<int>(std::lround(config.tile_size / config.voxel)));
    Vec2 lo = Vec2::Constant(kInf);
    Vec2 hi = Vec2::Constant(-kInf);
    for (const Pose& pose : poses) {
      lo = lo.cwiseMin(pose.translation.head<2>());
      hi = hi.cwiseMax(pose.translation.head<2>());
    }
    const auto tile_of = [&](double x) {
      return FloorDiv(static_cast<int>(std::floor(x / config.voxel)), n);
    };
    std::vector<MapTile> tiles;
    for (int iy = tile_of(lo.y() - config.max_range);
         iy <= tile_of(hi.y() + config.max_range); ++iy) {
      for (int ix = tile_of(lo.x() - config.max_range);
           ix <= tile_of(hi.x() + config.max_range); ++ix) {
        MapTile tile;
        tile.ix = ix;
        tile.iy = iy;
        tile.whole = false;
        tile.region = {ix * n, (ix + 1) * n, iy * n, (iy + 1) * n};
        tiles.push_back(tile);
      }
    }
    return tiles;
  }

  // Sweep elevation spacing of the range images (DynamicDetector).
  double ElevationResolution() const {
    if (config.dynamic_config.el_res > 0.0) return config.dynamic_config.el_res;
    // np.linspace(0, K - 1, min(3, K)).astype(int).
    const size_t count = std::min<size_t>(3, poses.size());
    std::vector<Points> sample;
    std::vector<float> intensity;
    for (size_t m = 0; m < count; ++m) {
      const double step = count > 1 ? static_cast<double>(poses.size() - 1) /
                                          static_cast<double>(count - 1)
                                    : 0.0;
      const size_t k = m + 1 == count && count > 1
                           ? poses.size() - 1
                           : static_cast<size_t>(static_cast<double>(m) * step);
      sample.emplace_back();
      load(static_cast<int>(k), &sample.back(), &intensity);
    }
    return EstimateBeamSpacing(sample);
  }
};

}  // namespace

TiledMapSummary BuildTiledMap(const std::vector<Pose>& world_from_lidar,
                              const std::vector<int>& keyframe_trip,
                              const std::vector<Points>& vehicle_paths,
                              const SweepLoader& load,
                              const TiledMapConfig& config,
                              const ChunkSink& sink, const Logger& log) {
  GR_CHECK(config.voxel > 0.0);
  GR_CHECK(keyframe_trip.size() == world_from_lidar.size());
  Builder builder{world_from_lidar, keyframe_trip, vehicle_paths, load,
                  config, {}};
  for (const Pose& pose : world_from_lidar) {
    builder.origins.push_back(pose.translation);
  }
  const std::vector<MapTile> tiles = builder.Tiles();

  // Pass 1: the floors and ceilings of the whole map for the levels.
  Points level_xyz;
  std::vector<double> level_nz;
  std::vector<std::pair<double, int64_t>> level_order;
  MapBounds bounds{Vec2::Constant(kInf), Vec2::Constant(-kInf)};
  std::vector<uint8_t> occupied(tiles.size(), 0);
  std::optional<MapChunk> cached;  // A single tile is not fused twice.
  for (size_t t = 0; t < tiles.size(); ++t) {
    MapChunk chunk = builder.Fuse(tiles[t]);
    occupied[t] = chunk.size() > 0;
    for (size_t i = 0; i < chunk.size(); ++i) {
      bounds.lo = bounds.lo.cwiseMin(chunk.xyz[i].head<2>());
      bounds.hi = bounds.hi.cwiseMax(chunk.xyz[i].head<2>());
      const double nz = static_cast<double>(static_cast<float>(
          chunk.normals[i].z()));
      if (nz > 0.8 || nz < -0.85) {
        level_xyz.push_back(chunk.xyz[i]);
        level_nz.push_back(nz);
        level_order.emplace_back(chunk.range[i], chunk.sequence[i]);
      }
    }
    if (tiles.size() == 1) cached = std::move(chunk);
  }
  std::vector<size_t> order(level_xyz.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return level_order[a] < level_order[b];
  });
  Points xyz(order.size());
  Points normals(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    xyz[i] = level_xyz[order[i]];
    normals[i] = Vec3(0.0, 0.0, level_nz[order[i]]);
  }
  level_xyz = Points();
  level_nz = std::vector<double>();

  TiledMapSummary summary;
  const bool any = std::any_of(occupied.begin(), occupied.end(),
                               [](uint8_t o) { return o != 0; });
  summary.levels =
      DetectLevels(vehicle_paths, xyz, normals, config.levels, log,
                   any ? std::optional<MapBounds>(bounds) : std::nullopt);
  summary.elevation_resolution =
      config.dynamic ? builder.ElevationResolution() : 0.0;

  // Pass 2: every tile again, with levels and dynamic points.
  for (size_t t = 0; t < tiles.size(); ++t) {
    if (!occupied[t]) continue;
    MapChunk chunk = cached ? std::move(*cached) : builder.Fuse(tiles[t]);
    builder.Finish(tiles[t], summary.levels, summary.elevation_resolution,
                   &chunk);
    ++summary.tiles;
    summary.points += chunk.size();
    for (uint8_t kind : chunk.dynamic_kind) {
      summary.moved += kind == static_cast<uint8_t>(DynamicKind::kMoved);
      summary.changed += kind == static_cast<uint8_t>(DynamicKind::kChanged);
    }
    if (tiles.size() > 1) {
      LogLine(log, StringPrintf("[map] tile (%d, %d): %zu points",
                                tiles[t].ix, tiles[t].iy, chunk.size()));
    }
    sink(tiles[t], std::move(chunk));
  }
  return summary;
}

}  // namespace gr
