#include "gr/mapping/levels.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Two levels kLevelDrop apart joined by a ramp along x (level 0 at z = 0 for
// x < kRampStart, the ramp down to z = -kLevelDrop at kRampEnd, level 1
// beyond), floor points on a grid over y in [-kHalfWidth, kHalfWidth], a
// ceiling over the flat parts, and a vehicle path along y = 0.
constexpr double kLevelDrop = 3.6;       // m
constexpr double kRampStart = 10.0;      // m
constexpr double kRampEnd = 34.0;        // m
constexpr double kSceneStart = -20.0;    // m
constexpr double kSceneEnd = 64.0;       // m
constexpr double kHalfWidth = 3.0;       // m
constexpr double kPathStep = 0.2;        // m
constexpr double kGridStep = 0.25;       // m
constexpr double kCeilingHeight = 3.3;   // m above the floor.
// The ceiling stops this far (m) before the ramp on either side.
constexpr double kCeilingSetback = 2.0;

struct Scene {
  Points path;
  Points xyz;
  Points normals;
};

double FloorHeight(double x) {
  if (x < kRampStart) return 0.0;
  if (x < kRampEnd) {
    return -kLevelDrop * (x - kRampStart) / (kRampEnd - kRampStart);
  }
  return -kLevelDrop;
}

Scene MakeScene() {
  Scene scene;
  for (double x = kSceneStart; x <= kSceneEnd; x += kPathStep) {
    scene.path.emplace_back(x, 0.0, FloorHeight(x));
  }
  for (double x = kSceneStart; x <= kSceneEnd; x += kGridStep) {
    for (double y = -kHalfWidth; y <= kHalfWidth; y += kGridStep) {
      const double z = FloorHeight(x);
      const double slope = (x >= kRampStart && x < kRampEnd)
                               ? kLevelDrop / (kRampEnd - kRampStart)
                               : 0.0;
      scene.xyz.emplace_back(x, y, z);
      scene.normals.push_back(Vec3(slope, 0.0, 1.0).normalized());
      if (x < kRampStart - kCeilingSetback || x > kRampEnd + kCeilingSetback) {
        scene.xyz.emplace_back(x, y, z + kCeilingHeight);
        scene.normals.emplace_back(0.0, 0.0, -1.0);
      }
    }
  }
  return scene;
}

GR_TEST(LevelsTest, DetectsTwoLevelsAndTheRampBetweenThem) {
  constexpr double kTolerance = 1e-6;
  constexpr double kSlopeTolerance = 1e-3;
  constexpr double kHeightTolerance = 1e-5;  // float32 heights.
  constexpr double kRampTolerance = 0.01;    // m
  constexpr double kRampSlope = kLevelDrop / (kRampEnd - kRampStart);
  constexpr double kOnLevel0X = -5.0;
  constexpr double kOnLevel1X = 50.0;
  constexpr double kOnRampX = 22.0;
  const Scene scene = MakeScene();
  const LevelModel model =
      DetectLevels({scene.path}, scene.xyz, scene.normals, LevelConfig());
  GR_ASSERT_TRUE(model.num_levels() == 2);
  GR_EXPECT_NEAR(model.z[0], 0.0, kTolerance);
  GR_EXPECT_NEAR(model.z[1], -kLevelDrop, kTolerance);
  GR_EXPECT_NEAR(model.ceiling[0], kCeilingHeight, kTolerance);
  GR_EXPECT_NEAR(model.FloorZ(Vec2(kOnLevel0X, 1.0), 0), 0.0, kTolerance);
  GR_EXPECT_NEAR(model.FloorZ(Vec2(kOnLevel1X, 1.0), 1), -kLevelDrop,
                 kTolerance);
  GR_ASSERT_TRUE(model.ramps.size() == 1);
  const Ramp& ramp = model.ramps[0];
  GR_EXPECT_EQ(ramp.level_from, 0);
  GR_EXPECT_EQ(ramp.level_to, 1);
  GR_EXPECT_NEAR(ramp.slope, kRampSlope, kSlopeTolerance);
  GR_EXPECT_NEAR(ramp.ZAt(Vec2(kOnRampX, 0.0)), FloorHeight(kOnRampX),
                 kRampTolerance);
  GR_EXPECT_TRUE(ramp.Contains(Vec2(kOnRampX, 0.0)));
  GR_EXPECT_FALSE(ramp.Contains(Vec2(0.0, 0.0)));
  // Points: on level 0, on level 1, on the ramp, under nothing.
  constexpr double kAboveLevel0 = 1.0;
  constexpr double kAboveLevel1 = 0.6;
  constexpr double kRampPointZ = -1.5;
  constexpr double kUnderEverything = -10.0;
  const LevelAssignment assignment = AssignLevels(
      model, {Vec3(kOnLevel0X, 0, kAboveLevel0),
              Vec3(kOnLevel1X, 0, -kLevelDrop + kAboveLevel1),
              Vec3(kOnRampX, 0, kRampPointZ),
              Vec3(kOnLevel0X, 0, kUnderEverything)});
  GR_EXPECT_EQ(assignment.level[0], 0);
  GR_EXPECT_NEAR(assignment.height[0], kAboveLevel0, kHeightTolerance);
  GR_EXPECT_EQ(assignment.level[1], 1);
  GR_EXPECT_NEAR(assignment.height[1], kAboveLevel1, kHeightTolerance);
  GR_EXPECT_EQ(assignment.level[2], 0);
  GR_EXPECT_EQ(assignment.on_ramp[2], 1);
  GR_EXPECT_NEAR(assignment.height[2], kRampPointZ - FloorHeight(kOnRampX),
                 kRampTolerance);
  GR_EXPECT_EQ(assignment.level[3], kNoLevel);
  GR_EXPECT_TRUE(std::isnan(assignment.height[3]));
}

}  // namespace
}  // namespace gr
