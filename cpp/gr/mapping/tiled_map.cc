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
// The halo of a tile is the normal radius in voxels plus this many: a point
// of a halo voxel may lie anywhere in it.
constexpr int kHaloExtraVoxels = 1;
// uint16 observation counts saturate.
constexpr int kMaxObservations = 65535;
// Sweeps that measure the LiDAR beam spacing (first, middle, last).
constexpr size_t kBeamSpacingSweeps = 3;

// Horizontal distance from `point` to the box of a voxel region (m).
double DistanceToRegion(const Vec3& point, const VoxelRegion& region,
                        double voxel) {
  const double dx = std::max(
      {region.x0 * voxel - point.x(), 0.0, point.x() - region.x1 * voxel});
  const double dy = std::max(
      {region.y0 * voxel - point.y(), 0.0, point.y() - region.y1 * voxel});
  return std::hypot(dx, dy);
}

VoxelRegion Grow(const VoxelRegion& region, int margin) {
  return {region.x0 - margin, region.x1 + margin, region.y0 - margin,
          region.y1 + margin};
}

// Integer division rounding down (Python's //).
int FloorDiv(int dividend, int divisor) {
  return dividend / divisor -
         (dividend % divisor != 0 && (dividend < 0) != (divisor < 0));
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
    for (size_t keyframe = 0; keyframe < poses.size(); ++keyframe) {
      if (tile.whole || DistanceToRegion(poses[keyframe].translation, region,
                                         config.voxel) <= reach) {
        out.push_back(static_cast<int>(keyframe));
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
        static_cast<int>(std::ceil(config.normal_radius / config.voxel)) +
        kHaloExtraVoxels;
    const VoxelRegion region = Grow(tile.region, halo);
    if (!tile.whole) accumulator.SetRegion(region);
    const std::vector<int> sweeps = Reaching(tile, region, config.max_range);
    Points points;
    std::vector<float> intensity;
    if (fusion.fuse) {
      for (int keyframe : sweeps) {
        load(keyframe, &points, &intensity);
        accumulator.AddRanges(poses[static_cast<size_t>(keyframe)], points);
      }
    }
    for (int keyframe : sweeps) {
      load(keyframe, &points, &intensity);
      accumulator.Add(keyframe, poses[static_cast<size_t>(keyframe)], points,
                      intensity);
    }
    const MapAccumulator::Cloud cloud = accumulator.Collect();
    const NormalsResult normals = EstimateOrientedNormals(
        cloud.points, cloud.keyframe, origins, config.normal_radius,
        kMapNormalMinNeighbors, kMapNormalMaxNeighbors);
    std::vector<float> sigma;
    if (fusion.fuse) {
      sigma = PointSigmas(cloud, normals.normals, fusion.sigma_floor);
    }
    MapChunk chunk;
    for (size_t i = 0; i < cloud.points.size(); ++i) {
      if (!tile.whole && !tile.region.Contains(cloud.voxel[i])) continue;
      const int keyframe = cloud.keyframe[i];
      chunk.xyz.push_back(cloud.points[i]);
      chunk.intensity.push_back(cloud.intensity[i]);
      chunk.keyframe.push_back(keyframe);
      chunk.trip.push_back(
          static_cast<int16_t>(trips[static_cast<size_t>(keyframe)]));
      chunk.normals.push_back(normals.normals[i]);
      chunk.planarity.push_back(static_cast<float>(normals.planarity[i]));
      if (!sigma.empty()) chunk.sigma.push_back(sigma[i]);
      chunk.observations.push_back(static_cast<uint16_t>(
          std::min(cloud.observations[i], kMaxObservations)));
      chunk.range.push_back(cloud.range[i]);
      chunk.sequence.push_back(cloud.sequence[i]);
    }
    return chunk;
  }

  // Levels (the later steps see the stored float32 normals) and dynamic
  // points of a tile.
  void Finish(const MapTile& tile, const LevelModel& levels,
              double elevation_resolution, MapChunk* chunk) const {
    const LevelAssignment assignment = AssignLevels(levels, chunk->xyz);
    chunk->level = assignment.level;
    chunk->height = assignment.height;
    chunk->on_ramp = assignment.on_ramp;
    const size_t count = chunk->size();
    chunk->dynamic_kind.assign(count, 0);
    if (!config.dynamic || count == 0) return;
    Points normals = chunk->normals;
    for (Vec3& normal : normals) normal = normal.cast<float>().cast<double>();
    DynamicDetector detector(
        chunk->xyz, normals,
        std::vector<int>(chunk->trip.begin(), chunk->trip.end()),
        config.dynamic_config);
    detector.SetElevationResolution(elevation_resolution);
    Points points;
    std::vector<float> intensity;
    for (int keyframe :
         Reaching(tile, tile.region, config.dynamic_config.max_range)) {
      load(keyframe, &points, &intensity);
      detector.AddSweep(trips[static_cast<size_t>(keyframe)],
                        poses[static_cast<size_t>(keyframe)], points);
    }
    for (size_t trip = 0; trip < paths.size(); ++trip) {
      detector.AddEgoPath(static_cast<int>(trip), paths[trip]);
    }
    detector.SetHeights(
        std::vector<double>(chunk->height.begin(), chunk->height.end()));
    const std::vector<DynamicKind> kinds = detector.Classify();
    for (size_t i = 0; i < count; ++i) {
      chunk->dynamic_kind[i] = static_cast<uint8_t>(kinds[i]);
    }
  }

  // The tiles covering every return within max_range of a sensor position.
  std::vector<MapTile> Tiles() const {
    if (config.tile_size <= 0.0 || poses.empty()) return {MapTile{}};
    const int voxels_per_tile = std::max(
        1, static_cast<int>(std::lround(config.tile_size / config.voxel)));
    Vec2 lo = Vec2::Constant(kInf);
    Vec2 hi = Vec2::Constant(-kInf);
    for (const Pose& pose : poses) {
      lo = lo.cwiseMin(pose.translation.head<2>());
      hi = hi.cwiseMax(pose.translation.head<2>());
    }
    const auto tile_of = [&](double coordinate) {
      return FloorDiv(static_cast<int>(std::floor(coordinate / config.voxel)),
                      voxels_per_tile);
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
        tile.region = {ix * voxels_per_tile, (ix + 1) * voxels_per_tile,
                       iy * voxels_per_tile, (iy + 1) * voxels_per_tile};
        tiles.push_back(tile);
      }
    }
    return tiles;
  }

  // Sweep elevation spacing of the range images (DynamicDetector).
  double ElevationResolution() const {
    if (config.dynamic_config.el_res > 0.0) {
      return config.dynamic_config.el_res;
    }
    // np.linspace(0, K - 1, min(kBeamSpacingSweeps, K)).astype(int).
    const size_t count = std::min(kBeamSpacingSweeps, poses.size());
    std::vector<Points> sample;
    std::vector<float> intensity;
    for (size_t m = 0; m < count; ++m) {
      const double step = count > 1 ? static_cast<double>(poses.size() - 1) /
                                          static_cast<double>(count - 1)
                                    : 0.0;
      const size_t keyframe =
          m + 1 == count && count > 1
              ? poses.size() - 1
              : static_cast<size_t>(static_cast<double>(m) * step);
      sample.emplace_back();
      load(static_cast<int>(keyframe), &sample.back(), &intensity);
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
  std::vector<double> level_normal_z;
  std::vector<std::pair<double, int64_t>> level_order;
  MapBounds bounds{Vec2::Constant(kInf), Vec2::Constant(-kInf)};
  std::vector<uint8_t> occupied(tiles.size(), 0);
  std::optional<MapChunk> cached;  // A single tile is not fused twice.
  for (size_t tile = 0; tile < tiles.size(); ++tile) {
    MapChunk chunk = builder.Fuse(tiles[tile]);
    occupied[tile] = chunk.size() > 0;
    for (size_t i = 0; i < chunk.size(); ++i) {
      bounds.lo = bounds.lo.cwiseMin(chunk.xyz[i].head<2>());
      bounds.hi = bounds.hi.cwiseMax(chunk.xyz[i].head<2>());
      // DetectLevels sees the stored float32 normals.
      const double normal_z =
          static_cast<double>(static_cast<float>(chunk.normals[i].z()));
      if (normal_z > kRampMinNormalZ || normal_z < kCeilingMaxNormalZ) {
        level_xyz.push_back(chunk.xyz[i]);
        level_normal_z.push_back(normal_z);
        level_order.emplace_back(chunk.range[i], chunk.sequence[i]);
      }
    }
    if (tiles.size() == 1) cached = std::move(chunk);
  }
  std::vector<size_t> order(level_xyz.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t left, size_t right) {
    return level_order[left] < level_order[right];
  });
  Points xyz(order.size());
  Points normals(order.size());
  for (size_t i = 0; i < order.size(); ++i) {
    xyz[i] = level_xyz[order[i]];
    normals[i] = Vec3(0.0, 0.0, level_normal_z[order[i]]);
  }
  level_xyz = Points();
  level_normal_z = std::vector<double>();

  TiledMapSummary summary;
  const bool any = std::any_of(occupied.begin(), occupied.end(),
                               [](uint8_t flag) { return flag != 0; });
  summary.levels =
      DetectLevels(vehicle_paths, xyz, normals, config.levels, log,
                   any ? std::optional<MapBounds>(bounds) : std::nullopt);
  summary.elevation_resolution =
      config.dynamic ? builder.ElevationResolution() : 0.0;

  // Pass 2: every tile again, with levels and dynamic points.
  for (size_t tile = 0; tile < tiles.size(); ++tile) {
    if (!occupied[tile]) continue;
    MapChunk chunk = cached ? std::move(*cached) : builder.Fuse(tiles[tile]);
    builder.Finish(tiles[tile], summary.levels, summary.elevation_resolution,
                   &chunk);
    ++summary.tiles;
    summary.points += chunk.size();
    for (uint8_t kind : chunk.dynamic_kind) {
      summary.moved += kind == static_cast<uint8_t>(DynamicKind::kMoved);
      summary.changed += kind == static_cast<uint8_t>(DynamicKind::kChanged);
    }
    if (tiles.size() > 1) {
      LogLine(log, StringPrintf("[map] tile (%d, %d): %zu points",
                                tiles[tile].ix, tiles[tile].iy,
                                chunk.size()));
    }
    sink(tiles[tile], std::move(chunk));
  }
  return summary;
}

}  // namespace gr
