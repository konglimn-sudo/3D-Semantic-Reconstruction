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
  Scans scans;
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(0.0, 0.0, 1.8));
  const LoopRoute route;
  for (int trip = 0; trip < 2; ++trip) {
    std::vector<SimBox> boxes = GarageStructure();
    const std::vector<SimBox> cars = ParkedCars(trip == 1);
    boxes.insert(boxes.end(), cars.begin(), cars.end());
    const BoxScene scene = MakeScene(boxes);
    Points path;
    for (int k = 0; k < 24; ++k) {
      const double t = (k + 0.3 * trip) * route.length() / 24.0 / route.speed();
      const Pose vehicle = route.PoseAt(t);
      path.push_back(vehicle.translation);
      const Pose pose = vehicle * vehicle_from_lidar;
      scans.poses.push_back(pose);
      scans.trip.push_back(trip);
      scans.sweeps.push_back(ScanFromPose(scene, pose, 30.0).points_sensor);
      std::vector<float> intensity(scans.sweeps.back().size());
      for (size_t i = 0; i < intensity.size(); ++i) {
        intensity[i] = static_cast<float>(i % 17) / 17.0f;
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
  config.voxel = 0.1;
  config.max_range = 20.0;
  config.normal_radius = 0.3;
  config.tile_size = tile_size;
  const SweepLoader load = [&](int k, Points* points,
                               std::vector<float>* intensity) {
    *points = scans.sweeps[static_cast<size_t>(k)];
    *intensity = scans.intensity[static_cast<size_t>(k)];
  };
  std::vector<MapChunk> chunks;
  const TiledMapSummary summary = BuildTiledMap(
      scans.poses, scans.trip, scans.paths, load, config,
      [&](const MapTile&, MapChunk&& chunk) {
        chunks.push_back(std::move(chunk));
      });
  *tiles = summary.tiles;
  std::vector<std::pair<size_t, size_t>> index;
  for (size_t c = 0; c < chunks.size(); ++c) {
    for (size_t i = 0; i < chunks[c].size(); ++i) index.emplace_back(c, i);
  }
  std::sort(index.begin(), index.end(), [&](const auto& a, const auto& b) {
    const MapChunk& x = chunks[a.first];
    const MapChunk& y = chunks[b.first];
    return std::make_pair(x.range[a.second], x.sequence[a.second]) <
           std::make_pair(y.range[b.second], y.sequence[b.second]);
  });
  MapChunk out;
  for (const auto& [c, i] : index) {
    const MapChunk& x = chunks[c];
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
  const Scans scans = MakeScans();
  int whole_tiles = 0;
  int tiled_tiles = 0;
  const MapChunk whole = Build(scans, 0.0, &whole_tiles);
  const MapChunk tiled = Build(scans, 8.0, &tiled_tiles);
  GR_EXPECT_EQ(whole_tiles, 1);
  GR_EXPECT_GE(tiled_tiles, 12);
  GR_ASSERT_TRUE(whole.size() > 50000);
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
  GR_EXPECT_GT(changed, size_t{20});
}

GR_TEST(TiledMapTest, TilesKeepEachVoxelOnce) {
  const Scans scans = MakeScans();
  TiledMapConfig config;
  config.voxel = 0.1;
  config.normal_radius = 0.3;
  config.tile_size = 8.0;
  config.dynamic = false;
  size_t outside = 0;
  BuildTiledMap(
      scans.poses, scans.trip, scans.paths,
      [&](int k, Points* points, std::vector<float>* intensity) {
        *points = scans.sweeps[static_cast<size_t>(k)];
        *intensity = scans.intensity[static_cast<size_t>(k)];
      },
      config, [&](const MapTile& tile, MapChunk&& chunk) {
        GR_EXPECT_FALSE(tile.whole);
        for (const Vec3& p : chunk.xyz) {
          outside += !tile.region.Contains(VoxelOf(p, 1.0 / config.voxel));
        }
      });
  GR_EXPECT_EQ(outside, size_t{0});
}

}  // namespace
}  // namespace gr
