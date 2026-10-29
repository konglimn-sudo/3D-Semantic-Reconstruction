#include "gr/mapping/static_map.h"

#include <cmath>
#include <utility>
#include <vector>

#include "gr/sim/sim.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::ScanFromPose;
using ::gr::testing::StaticScan;

// Map voxel (m) and range (m) of the accumulator tests.
constexpr double kVoxel = 0.1;
constexpr double kMaxRange = 20.0;

GR_TEST(MapAccumulatorTest, KeepsTheNearestObservationPerVoxel) {
  constexpr double kTolerance = 1e-12;
  constexpr float kIntensityTolerance = 1e-7f;
  MapFusionConfig nearest_only;
  nearest_only.fuse = false;
  MapAccumulator accumulator(kVoxel, kMaxRange, nearest_only);
  // The same world point seen from 10 m (keyframe 0) and from 2 m (keyframe 1).
  accumulator.Add(0, Pose(Mat3::Identity(), Vec3(-10, 0, 0)),
                  {Vec3(10.01, 0.02, 0.0), Vec3(25, 0, 0)}, {0.1f, 0.2f});
  accumulator.Add(1, Pose(Mat3::Identity(), Vec3(-2, 0, 0)),
                  {Vec3(2.03, 0.01, 0.0)}, {0.9f});
  const MapAccumulator::Cloud cloud = accumulator.Collect();
  GR_ASSERT_TRUE(cloud.points.size() == 1);  // 25 m is beyond max_range.
  GR_EXPECT_EQ(cloud.keyframe[0], 1);
  GR_EXPECT_NEAR(cloud.intensity[0], 0.9f, kIntensityTolerance);
  GR_EXPECT_NEAR(cloud.points[0].x(), 0.03, kTolerance);
}

GR_TEST(MapAccumulatorTest, FusesObservationsFromComparableRanges) {
  constexpr float kIntensity = 0.5f;
  constexpr double kSigmaFloor = 0.003;  // m
  constexpr double kRangeSigma = 0.01;   // m, MapFusionConfig default.
  constexpr double kMeanTolerance = 1e-4;
  constexpr double kSpreadTolerance = 1e-5;
  MapAccumulator accumulator(kVoxel, kMaxRange);
  // One voxel seen from 10 m, 2 m and 3 m. The 10 m observation is outside
  // the range window (2 * 2.02 + 3 m) of the nearest one.
  const std::vector<std::pair<Pose, Points>> sweeps = {
      {Pose(Mat3::Identity(), Vec3(-10, 0, 0)), {Vec3(10.01, 0.02, 0.0)}},
      {Pose(Mat3::Identity(), Vec3(-2, 0, 0)), {Vec3(2.02, 0.02, 0.0)}},
      {Pose(Mat3::Identity(), Vec3(-3, 0, 0)), {Vec3(3.08, 0.02, 0.0)}}};
  for (const auto& [pose, sweep] : sweeps) accumulator.AddRanges(pose, sweep);
  for (size_t k = 0; k < sweeps.size(); ++k) {
    accumulator.Add(static_cast<int>(k), sweeps[k].first, sweeps[k].second,
                    {kIntensity});
  }
  const MapAccumulator::Cloud cloud = accumulator.Collect();
  GR_ASSERT_TRUE(cloud.points.size() == 1);
  GR_EXPECT_EQ(cloud.keyframe[0], 1);  // The nearest observation.
  GR_EXPECT_EQ(cloud.observations[0], 2);
  // Mean of 0.02 and 0.08, and the variance along x.
  GR_EXPECT_NEAR(cloud.points[0].x(), 0.05, kMeanTolerance);
  GR_EXPECT_NEAR(cloud.spread[0][0], 9e-4, kSpreadTolerance);
  GR_EXPECT_NEAR(cloud.model_sigma[0], kRangeSigma / std::sqrt(2.0),
                 kMeanTolerance);
  // Along x the observations disagree by 6 cm: the spread over sqrt(2)
  // (2.1 cm) dominates the formal sigma (7 mm).
  GR_EXPECT_NEAR(PointSigmas(cloud, {Vec3::UnitX()}, kSigmaFloor)[0], 0.0214,
                 3e-4);
  // Along y they agree: formal sigma and floor, 7.7 mm.
  GR_EXPECT_NEAR(PointSigmas(cloud, {Vec3::UnitY()}, kSigmaFloor)[0], 0.0077,
                 2e-4);
}

GR_TEST(StaticMapTest, BeamSpacingIsMeasuredFromSweeps) {
  // ScanFromPose: 16 beams over [-15, 15] deg, 2 deg apart.
  constexpr double kBeamSpacing = 2.0;    // deg
  constexpr double kTolerance = 0.15;     // deg
  constexpr double kSensorHeight = 1.8;   // m
  constexpr double kScanRange = 30.0;     // m
  BoxScene scene;
  scene.AddBox(Vec3(0, 0, 0), Vec3(40, 40, 0.2), Mat3::Identity());
  scene.AddBox(Vec3(0, 0, 6), Vec3(40, 40, 0.2), Mat3::Identity());
  scene.AddBox(Vec3(12, 0, 3), Vec3(0.2, 40, 6), Mat3::Identity());
  const StaticScan scan = ScanFromPose(
      scene, Pose(Mat3::Identity(), Vec3(0, 0, kSensorHeight)), kScanRange);
  GR_EXPECT_NEAR(EstimateBeamSpacing({scan.points_sensor}), kBeamSpacing,
                 kTolerance);
}

GR_TEST(DynamicDetectorTest, SeparatesMovedChangedAndEgoSweptPoints) {
  constexpr double kSensorHeight = 1.0;    // m
  constexpr double kScanRange = 30.0;      // m
  constexpr double kRayVoxel = 0.02;       // m
  constexpr double kBeamSpacing = 2.0;     // deg (ScanFromPose)
  constexpr int kPathSamples = 60;         // 6 m of ego path at y = 4 m.
  constexpr double kPathStep = 0.1;        // m
  constexpr double kPathY = 4.0;           // m
  constexpr int kMinObjectPoints = 100;
  constexpr int kMinPostPoints = 20;
  // Shares of a part's points expected to be flagged (or at most flagged).
  constexpr double kObjectShare = 0.6;
  constexpr double kPostShare = 0.8;
  constexpr double kWallShare = 0.01;
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
  const Pose sensor(Mat3::Identity(), Vec3(0, 0, kSensorHeight));
  const StaticScan session0 =
      ScanFromPose(scene_with(true, false), sensor, kScanRange);
  const StaticScan session1_with_b =
      ScanFromPose(scene_with(false, true), sensor, kScanRange);
  const StaticScan session1_empty =
      ScanFromPose(scene_with(false, false), sensor, kScanRange);

  // Map: everything session 0 saw (trip 0) plus B as seen in session 1.
  // Parts by box id: the structure boxes, then the object.
  enum Part {
    kFloor = 0,
    kWall = 1,
    kPost = 2,
    kObjectA = 3,
    kObjectB = 4,
    kParts = 5
  };
  const int kObjectBox = static_cast<int>(structure.size());
  Points map;
  Points normals;
  std::vector<int> trip;
  std::vector<int> part;
  for (size_t i = 0; i < session0.points_world.size(); ++i) {
    map.push_back(session0.points_world[i]);
    normals.push_back(session0.normals[i]);
    trip.push_back(0);
    part.push_back(session0.box_ids[i] < kObjectBox ? session0.box_ids[i]
                                                    : kObjectA);
  }
  for (size_t i = 0; i < session1_with_b.points_world.size(); ++i) {
    if (session1_with_b.box_ids[i] != kObjectBox) continue;
    map.push_back(session1_with_b.points_world[i]);
    normals.push_back(session1_with_b.normals[i]);
    trip.push_back(1);
    part.push_back(kObjectB);
  }
  DynamicConfig config;
  config.ray_voxel = kRayVoxel;
  DynamicDetector detector(map, normals, trip, config);
  detector.SetElevationResolution(kBeamSpacing);
  std::vector<double> height;
  for (const Vec3& point : map) height.push_back(point.z());
  detector.SetHeights(height);
  detector.AddSweep(0, sensor, session0.points_sensor);
  detector.AddSweep(1, sensor, session1_with_b.points_sensor);
  detector.AddSweep(1, sensor, session1_empty.points_sensor);
  detector.AddSweep(1, sensor, session1_empty.points_sensor);
  Points path;
  for (int sample = 0; sample <= kPathSamples; ++sample) {
    path.emplace_back(kPathStep * sample, kPathY, 0.0);
  }
  detector.AddEgoPath(0, path);
  const std::vector<DynamicKind> kind = detector.Classify();

  int count[kParts] = {};
  int moved[kParts] = {};
  int changed[kParts] = {};
  for (size_t i = 0; i < map.size(); ++i) {
    const size_t index = static_cast<size_t>(part[i]);
    ++count[index];
    moved[index] += kind[i] == DynamicKind::kMoved;
    changed[index] += kind[i] == DynamicKind::kChanged;
  }
  GR_ASSERT_TRUE(count[kObjectA] > kMinObjectPoints &&
                 count[kObjectB] > kMinObjectPoints &&
                 count[kPost] > kMinPostPoints);
  GR_EXPECT_GT(changed[kObjectA], kObjectShare * count[kObjectA]);
  GR_EXPECT_GT(moved[kObjectB], kObjectShare * count[kObjectB]);
  // The post stands in the ego vehicle's swept volume.
  GR_EXPECT_GT(moved[kPost], kPostShare * count[kPost]);
  GR_EXPECT_EQ(moved[kFloor] + changed[kFloor], 0);  // Below the height gate.
  GR_EXPECT_LT(moved[kWall] + changed[kWall], kWallShare * count[kWall]);
}

}  // namespace
}  // namespace gr
