// The map stage (static_map: accumulation, normals and sigmas, levels,
// dynamic points) built tile by tile, so that its memory follows the tile
// size instead of the garage.
//
// Every step is local: a voxel only depends on the returns that fall into
// it, a normal on the points within the normal radius, the dynamic votes of
// a point on the sweeps that see it. A tile therefore accumulates its own
// voxels plus a halo of the normal radius, from the sweeps that can reach
// it, and its points come out identical to the same points of the map built
// in one piece; sorted on (range, sequence) the tiles give that map's point
// order too. The levels need the floors and ceilings of the whole garage: a
// first pass over the tiles keeps only those points, a second pass rebuilds
// every tile and adds the levels and the dynamic points.

#ifndef GARAGE_RECON_GR_MAPPING_TILED_MAP_H_
#define GARAGE_RECON_GR_MAPPING_TILED_MAP_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "gr/core/logging.h"
#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/mapping/levels.h"
#include "gr/mapping/static_map.h"

namespace gr {

struct TiledMapConfig {
  double voxel = 0.05;
  double max_range = 20.0;
  double normal_radius = 0.2;
  // Tile side (m), aligned to the voxel grid; <= 0: the map in one piece.
  double tile_size = 0.0;
  MapFusionConfig fusion;
  LevelConfig levels;
  bool dynamic = true;
  // el_res <= 0: measured on three sweeps (the first, middle and last).
  DynamicConfig dynamic_config;
};

// The points of one tile with the attributes of map/point_map.npz, in map
// order (range, then sequence).
struct MapChunk {
  Points xyz;
  std::vector<float> intensity;
  std::vector<int32_t> keyframe;
  std::vector<int16_t> trip;
  Points normals;  // Oriented towards the sensor; stored as float32.
  std::vector<float> planarity;
  std::vector<float> sigma;  // Empty without fusion.
  std::vector<uint16_t> observations;
  std::vector<int16_t> level;
  std::vector<float> height;
  std::vector<uint8_t> on_ramp;
  std::vector<uint8_t> dynamic_kind;  // DynamicKind.
  std::vector<double> range;
  std::vector<int64_t> sequence;

  size_t size() const { return xyz.size(); }
};

// Tile (ix, iy) holds the voxels with x index in [ix * n, (ix + 1) * n) and
// y likewise, n = round(tile_size / voxel); a single tile holds all.
struct MapTile {
  int ix = 0;
  int iy = 0;
  bool whole = true;  // The map in one piece (no bounds).
  VoxelRegion region;
};

// Loads keyframe `k`'s sweep (LiDAR frame) and intensities.
using SweepLoader =
    std::function<void(int k, Points* points, std::vector<float>* intensity)>;
// Receives every non-empty tile once, after the second pass.
using ChunkSink = std::function<void(const MapTile& tile, MapChunk&& chunk)>;

struct TiledMapSummary {
  LevelModel levels;
  int tiles = 0;  // Non-empty tiles.
  size_t points = 0;
  size_t moved = 0;
  size_t changed = 0;
  double elevation_resolution = 0.0;  // Of the dynamic range images (deg).
};

// Builds the map of the keyframes (world_from_lidar and trip index per
// keyframe, added in this order) and hands it to `sink` tile by tile.
// `vehicle_paths` (vehicle origins on the floor, one polyline per trip)
// give the levels and the ego vehicle's swept volume.
TiledMapSummary BuildTiledMap(const std::vector<Pose>& world_from_lidar,
                              const std::vector<int>& keyframe_trip,
                              const std::vector<Points>& vehicle_paths,
                              const SweepLoader& load,
                              const TiledMapConfig& config,
                              const ChunkSink& sink,
                              const Logger& log = nullptr);

}  // namespace gr

#endif  // GARAGE_RECON_GR_MAPPING_TILED_MAP_H_
