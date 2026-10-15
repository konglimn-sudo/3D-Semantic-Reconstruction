#include "gr/static_map.h"

#include <cmath>
#include <utility>
#include <vector>

#include "gr/sim.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::ScanFromPose;
using ::gr::testing::StaticScan;

GR_TEST(MapAccumulatorTest, KeepsTheNearestObservationPerVoxel) {
  MapFusionConfig nearest_only;
  nearest_only.fuse = false;
  MapAccumulator accumulator(0.1, 20.0, nearest_only);
  // The same world point seen from 10 m (keyframe 0) and from 2 m (keyframe 1).
  accumulator.Add(0, Pose(Mat3::Identity(), Vec3(-10, 0, 0)),
                  {Vec3(10.01, 0.02, 0.0), Vec3(25, 0, 0)}, {0.1f, 0.2f});
  accumulator.Add(1, Pose(Mat3::Identity(), Vec3(-2, 0, 0)),
                  {Vec3(2.03, 0.01, 0.0)}, {0.9f});
  const MapAccumulator::Cloud cloud = accumulator.Collect();
  GR_ASSERT_TRUE(cloud.points.size() == 1);  // 25 m is beyond max_range.
  GR_EXPECT_EQ(cloud.keyframe[0], 1);
  GR_EXPECT_NEAR(cloud.intensity[0], 0.9f, 1e-7);
  GR_EXPECT_NEAR(cloud.points[0].x(), 0.03, 1e-12);
}

GR_TEST(MapAccumulatorTest, FusesObservationsFromComparableRanges) {
  MapAccumulator accumulator(0.1, 20.0);
  // One voxel seen from 10 m, 2 m and 3 m. The 10 m observation is outside
  // the range window (2 * 2.02 + 3 m) of the nearest one.
  const std::vector<std::pair<Pose, Points>> sweeps = {
      {Pose(Mat3::Identity(), Vec3(-10, 0, 0)), {Vec3(10.01, 0.02, 0.0)}},
      {Pose(Mat3::Identity(), Vec3(-2, 0, 0)), {Vec3(2.02, 0.02, 0.0)}},
      {Pose(Mat3::Identity(), Vec3(-3, 0, 0)), {Vec3(3.08, 0.02, 0.0)}}};
  for (const auto& [pose, sweep] : sweeps) accumulator.AddRanges(pose, sweep);
  for (size_t k = 0; k < sweeps.size(); ++k) {
    accumulator.Add(static_cast<int>(k), sweeps[k].first, sweeps[k].second,
                    {0.5f});
  }
  const MapAccumulator::Cloud cloud = accumulator.Collect();
  GR_ASSERT_TRUE(cloud.points.size() == 1);
  GR_EXPECT_EQ(cloud.keyframe[0], 1);  // The nearest observation.
  GR_EXPECT_EQ(cloud.observations[0], 2);
  GR_EXPECT_NEAR(cloud.points[0].x(), 0.05, 1e-4);  // Mean of 0.02, 0.08.
  GR_EXPECT_NEAR(cloud.spread[0][0], 9e-4, 1e-5);   // Variance along x.
  GR_EXPECT_NEAR(cloud.model_sigma[0], 0.01 / std::sqrt(2.0), 1e-4);
  // Along x the observations disagree by 6 cm: the spread over sqrt(2)
  // (2.1 cm) dominates the formal sigma (7 mm).
  GR_EXPECT_NEAR(PointSigmas(cloud, {Vec3::UnitX()}, 0.003)[0], 0.0214, 3e-4);
  // Along y they agree: formal sigma and floor, 7.7 mm.
  GR_EXPECT_NEAR(PointSigmas(cloud, {Vec3::UnitY()}, 0.003)[0], 0.0077, 2e-4);
}

GR_TEST(StaticMapTest, BeamSpacingIsMeasuredFromSweeps) {
  BoxScene scene;
  scene.AddBox(Vec3(0, 0, 0), Vec3(40, 40, 0.2), Mat3::Identity());
  scene.AddBox(Vec3(0, 0, 6), Vec3(40, 40, 0.2), Mat3::Identity());
  scene.AddBox(Vec3(12, 0, 3), Vec3(0.2, 40, 6), Mat3::Identity());
  const StaticScan scan =
      ScanFromPose(scene, Pose(Mat3::Identity(), Vec3(0, 0, 1.8)), 30.0);
  GR_EXPECT_NEAR(EstimateBeamSpacing({scan.points_sensor}), 2.0, 0.15);
}

GR_TEST(DynamicDetectorTest, SeparatesMovedChangedAndEgoSweptPoints) {
  const std::vector<std::pair<Vec3, Vec3>> structure = {
      {Vec3(5, 0, -0.1), Vec3(30, 20, 0.2)},    // Floor.
      {Vec3(10.15, 0, 1.5), Vec3(0.3, 20, 3)},  // Wall behind the objects.
      {Vec3(3, 4, 0.75), Vec3(0.6, 0.6, 1.5)},  // Post on the ego path.
  };
  // A is parked during session 0 only; B leaves during session 1.
  const std::pair<Vec3, Vec3> object_a{Vec3(5, -1.5, 0.75), Vec3(1, 1.2, 1.5)};
  const std::pair<Vec3, Vec3> object_b{Vec3(5, 1.5, 0.75), Vec3(1, 1.2, 1.5)};
  const auto scene_with = [&](bool with_a, bool with_b) {
    BoxScene scene;
    for (const auto& [center, size] : structure) {
      scene.AddBox(center, size, Mat3::Identity());
    }
    if (with_a) scene.AddBox(object_a.first, object_a.second, Mat3::Identity());
    if (with_b) scene.AddBox(object_b.first, object_b.second, Mat3::Identity());
    return scene;
  };
  const Pose sensor(Mat3::Identity(), Vec3(0, 0, 1.0));
  const StaticScan s0 = ScanFromPose(scene_with(true, false), sensor, 30.0);
  const StaticScan s1a = ScanFromPose(scene_with(false, true), sensor, 30.0);
  const StaticScan s1b = ScanFromPose(scene_with(false, false), sensor, 30.0);

  // Map: everything session 0 saw (trip 0) plus B as seen in session 1.
  enum Part { kFloor = 0, kWall = 1, kPost = 2, kObjectA = 3, kObjectB = 4 };
  Points map;
  Points normals;
  std::vector<int> trip;
  std::vector<int> part;
  for (size_t i = 0; i < s0.points_world.size(); ++i) {
    map.push_back(s0.points_world[i]);
    normals.push_back(s0.normals[i]);
    trip.push_back(0);
    part.push_back(s0.box_ids[i] < 3 ? s0.box_ids[i] : kObjectA);
  }
  for (size_t i = 0; i < s1a.points_world.size(); ++i) {
    if (s1a.box_ids[i] != 3) continue;
    map.push_back(s1a.points_world[i]);
    normals.push_back(s1a.normals[i]);
    trip.push_back(1);
    part.push_back(kObjectB);
  }
  DynamicConfig config;
  config.ray_voxel = 0.02;
  DynamicDetector detector(map, normals, trip, config);
  detector.SetElevationResolution(2.0);
  std::vector<double> height;
  for (const Vec3& p : map) height.push_back(p.z());
  detector.SetHeights(height);
  detector.AddSweep(0, sensor, s0.points_sensor);
  detector.AddSweep(1, sensor, s1a.points_sensor);
  detector.AddSweep(1, sensor, s1b.points_sensor);
  detector.AddSweep(1, sensor, s1b.points_sensor);
  Points path;
  for (int i = 0; i <= 60; ++i) path.emplace_back(0.1 * i, 4.0, 0.0);
  detector.AddEgoPath(0, path);
  const std::vector<DynamicKind> kind = detector.Classify();

  int count[5] = {0, 0, 0, 0, 0};
  int moved[5] = {0, 0, 0, 0, 0};
  int changed[5] = {0, 0, 0, 0, 0};
  for (size_t i = 0; i < map.size(); ++i) {
    const size_t p = static_cast<size_t>(part[i]);
    ++count[p];
    moved[p] += kind[i] == DynamicKind::kMoved;
    changed[p] += kind[i] == DynamicKind::kChanged;
  }
  GR_ASSERT_TRUE(count[kObjectA] > 100 && count[kObjectB] > 100 &&
                 count[kPost] > 20);
  GR_EXPECT_GT(changed[kObjectA], 0.6 * count[kObjectA]);
  GR_EXPECT_GT(moved[kObjectB], 0.6 * count[kObjectB]);
  // The post stands in the ego vehicle's swept volume.
  GR_EXPECT_GT(moved[kPost], 0.8 * count[kPost]);
  GR_EXPECT_EQ(moved[kFloor] + changed[kFloor], 0);  // Below the height gate.
  GR_EXPECT_LT(moved[kWall] + changed[kWall], 0.01 * count[kWall]);
}

}  // namespace
}  // namespace gr
