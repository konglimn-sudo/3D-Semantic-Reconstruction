#include "gr/mapping/tiled_map.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "gr/sim/sim.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::GarageStructure;
using ::gr::testing::LoopRoute;
using ::gr::testing::MakeScene;
using ::gr::testing::ParkedCars;
using ::gr::testing::ScanFromPose;
using ::gr::testing::SimBox;

// The test maps: voxel, returns used and normal radius (m).
constexpr double kVoxel = 0.1;
constexpr double kMaxRange = 20.0;
constexpr double kNormalRadius = 0.3;
// Tiles of this side (m) cut the test garage into a dozen or more.
constexpr double kTileSize = 8.0;

// Static scans along the test garage's loop: trip 0 without, trip 1 with an
// extra parked car (points that change between sessions).
struct Scans {
  std::vector<Pose> poses;
  std::vector<int> trip;
  std::vector<Points> sweeps;
  std::vector<std::vector<float>> intensity;
  std::vector<Points> paths;
};

Scans MakeScans() {
  constexpr int kTrips = 2;
  constexpr int kScansPerTrip = 24;      // Around the loop.
  constexpr double kTripOffset = 0.3;    // Scans of trip 1 lie in between.
  constexpr double kLidarHeight = 1.8;   // m
  constexpr double kScanRange = 30.0;    // m
  constexpr int kIntensityLevels = 17;   // A made-up intensity pattern.
  Scans scans;
  const Pose vehicle_from_lidar(Mat3::Identity(),
                                Vec3(0.0, 0.0, kLidarHeight));
  const LoopRoute route;
  for (int trip = 0; trip < kTrips; ++trip) {
    std::vector<SimBox> boxes = GarageStructure();
    const std::vector<SimBox> cars = ParkedCars(trip == 1);
    boxes.insert(boxes.end(), cars.begin(), cars.end());
    const BoxScene scene = MakeScene(boxes);
    Points path;
    for (int scan = 0; scan < kScansPerTrip; ++scan) {
      const double time = (scan + kTripOffset * trip) * route.length() /
                          kScansPerTrip / route.speed();
      const Pose vehicle = route.PoseAt(time);
      path.push_back(vehicle.translation);
      const Pose pose = vehicle * vehicle_from_lidar;
      scans.poses.push_back(pose);
      scans.trip.push_back(trip);
      scans.sweeps.push_back(
          ScanFromPose(scene, pose, kScanRange).points_sensor);
      std::vector<float> intensity(scans.sweeps.back().size());
      for (size_t i = 0; i < intensity.size(); ++i) {
        intensity[i] = static_cast<float>(i % kIntensityLevels) /
                       static_cast<float>(kIntensityLevels);
      }
      scans.intensity.push_back(std::move(intensity));
    }
    scans.paths.push_back(std::move(path));
  }
  return scans;
}

// The chunks of a build merged into map order (range, then sequence).
MapChunk Build(const Scans& scans, double tile_size, int* tiles) {
  TiledMapConfig config;
  config.voxel = kVoxel;
  config.max_range = kMaxRange;
  config.normal_radius = kNormalRadius;
  config.tile_size = tile_size;
  const SweepLoader load = [&](int keyframe, Points* points,
                               std::vector<float>* intensity) {
    *points = scans.sweeps[static_cast<size_t>(keyframe)];
    *intensity = scans.intensity[static_cast<size_t>(keyframe)];
  };
  std::vector<MapChunk> chunks;
  const TiledMapSummary summary = BuildTiledMap(
      scans.poses, scans.trip, scans.paths, load, config,
      [&](const MapTile&, MapChunk&& chunk) {
        chunks.push_back(std::move(chunk));
      });
  *tiles = summary.tiles;
  // (chunk, point) pairs in map order.
  std::vector<std::pair<size_t, size_t>> index;
  for (size_t chunk = 0; chunk < chunks.size(); ++chunk) {
    for (size_t i = 0; i < chunks[chunk].size(); ++i) {
      index.emplace_back(chunk, i);
    }
  }
  std::sort(index.begin(), index.end(),
            [&](const auto& left, const auto& right) {
              const MapChunk& a = chunks[left.first];
              const MapChunk& b = chunks[right.first];
              return std::make_pair(a.range[left.second],
                                    a.sequence[left.second]) <
                     std::make_pair(b.range[right.second],
                                    b.sequence[right.second]);
            });
  MapChunk out;
  for (const auto& [chunk, i] : index) {
    const MapChunk& x = chunks[chunk];
    out.xyz.push_back(x.xyz[i]);
    out.intensity.push_back(x.intensity[i]);
    out.keyframe.push_back(x.keyframe[i]);
    out.trip.push_back(x.trip[i]);
    out.normals.push_back(x.normals[i]);
    out.planarity.push_back(x.planarity[i]);
    out.sigma.push_back(x.sigma[i]);
    out.observations.push_back(x.observations[i]);
    out.level.push_back(x.level[i]);
    out.height.push_back(x.height[i]);
    out.on_ramp.push_back(x.on_ramp[i]);
    out.dynamic_kind.push_back(x.dynamic_kind[i]);
    out.range.push_back(x.range[i]);
    out.sequence.push_back(x.sequence[i]);
  }
  return out;
}

GR_TEST(TiledMapTest, TilesGiveTheMapBuiltInOnePiece) {
  constexpr int kMinTiles = 12;
  constexpr size_t kMinPoints = 50000;
  constexpr size_t kMinChanged = 20;  // Points of the extra car.
  const Scans scans = MakeScans();
  int whole_tiles = 0;
  int tiled_tiles = 0;
  const MapChunk whole = Build(scans, 0.0, &whole_tiles);
  const MapChunk tiled = Build(scans, kTileSize, &tiled_tiles);
  GR_EXPECT_EQ(whole_tiles, 1);
  GR_EXPECT_GE(tiled_tiles, kMinTiles);
  GR_ASSERT_TRUE(whole.size() > kMinPoints);
  GR_ASSERT_TRUE(tiled.size() == whole.size());
  // Bit for bit: every point, normal, sigma, level and dynamic flag.
  GR_EXPECT_TRUE(tiled.xyz == whole.xyz);
  GR_EXPECT_TRUE(tiled.normals == whole.normals);
  GR_EXPECT_TRUE(tiled.intensity == whole.intensity);
  GR_EXPECT_TRUE(tiled.keyframe == whole.keyframe);
  GR_EXPECT_TRUE(tiled.trip == whole.trip);
  GR_EXPECT_TRUE(tiled.planarity == whole.planarity);
  GR_EXPECT_TRUE(tiled.sigma == whole.sigma);
  GR_EXPECT_TRUE(tiled.observations == whole.observations);
  GR_EXPECT_TRUE(tiled.level == whole.level);
  GR_EXPECT_TRUE(tiled.on_ramp == whole.on_ramp);
  GR_EXPECT_TRUE(tiled.dynamic_kind == whole.dynamic_kind);
  bool heights_equal = true;  // NaN below every floor.
  for (size_t i = 0; i < whole.size(); ++i) {
    heights_equal &= whole.height[i] == tiled.height[i] ||
                     (whole.height[i] != whole.height[i] &&
                      tiled.height[i] != tiled.height[i]);
  }
  GR_EXPECT_TRUE(heights_equal);
  // The extra car of trip 1 is a change between sessions.
  const size_t changed = static_cast<size_t>(
      std::count(whole.dynamic_kind.begin(), whole.dynamic_kind.end(),
                 static_cast<uint8_t>(DynamicKind::kChanged)));
  GR_EXPECT_GT(changed, kMinChanged);
}

GR_TEST(TiledMapTest, TilesKeepEachVoxelOnce) {
  const Scans scans = MakeScans();
  TiledMapConfig config;
  config.voxel = kVoxel;
  config.normal_radius = kNormalRadius;
  config.tile_size = kTileSize;
  config.dynamic = false;
  size_t outside = 0;
  BuildTiledMap(
      scans.poses, scans.trip, scans.paths,
      [&](int keyframe, Points* points, std::vector<float>* intensity) {
        *points = scans.sweeps[static_cast<size_t>(keyframe)];
        *intensity = scans.intensity[static_cast<size_t>(keyframe)];
      },
      config, [&](const MapTile& tile, MapChunk&& chunk) {
        GR_EXPECT_FALSE(tile.whole);
        for (const Vec3& point : chunk.xyz) {
          outside += !tile.region.Contains(VoxelOf(point, 1.0 / config.voxel));
        }
      });
  GR_EXPECT_EQ(outside, size_t{0});
}

}  // namespace
}  // namespace gr
