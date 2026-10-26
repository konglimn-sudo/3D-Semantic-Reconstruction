#include "gr/mapping/levels.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Two levels 3.6 m apart joined by a 24 m ramp along x (level 0 at z = 0 for
// x < 10, the ramp down to z = -3.6 at x = 34, level 1 beyond), floor points
// on a 0.25 m grid over y in [-3, 3], and a vehicle path along y = 0.
struct Scene {
  Points path;
  Points xyz;
  Points normals;
};

double FloorHeight(double x) {
  if (x < 10.0) return 0.0;
  if (x < 34.0) return -3.6 * (x - 10.0) / 24.0;
  return -3.6;
}

Scene MakeScene() {
  Scene scene;
  for (double x = -20.0; x <= 64.0; x += 0.2) {
    scene.path.emplace_back(x, 0.0, FloorHeight(x));
  }
  for (double x = -20.0; x <= 64.0; x += 0.25) {
    for (double y = -3.0; y <= 3.0; y += 0.25) {
      const double z = FloorHeight(x);
      const double slope = (x >= 10.0 && x < 34.0) ? 3.6 / 24.0 : 0.0;
      scene.xyz.emplace_back(x, y, z);
      scene.normals.push_back(Vec3(slope, 0.0, 1.0).normalized());
      // A ceiling 3.3 m above the flat parts.
      if (x < 8.0 || x > 36.0) {
        scene.xyz.emplace_back(x, y, z + 3.3);
        scene.normals.emplace_back(0.0, 0.0, -1.0);
      }
    }
  }
  return scene;
}

GR_TEST(LevelsTest, DetectsTwoLevelsAndTheRampBetweenThem) {
  const Scene scene = MakeScene();
  const LevelModel model =
      DetectLevels({scene.path}, scene.xyz, scene.normals, LevelConfig());
  GR_ASSERT_TRUE(model.num_levels() == 2);
  GR_EXPECT_NEAR(model.z[0], 0.0, 1e-6);
  GR_EXPECT_NEAR(model.z[1], -3.6, 1e-6);
  GR_EXPECT_NEAR(model.ceiling[0], 3.3, 1e-6);
  GR_EXPECT_NEAR(model.FloorZ(Vec2(-5.0, 1.0), 0), 0.0, 1e-6);
  GR_EXPECT_NEAR(model.FloorZ(Vec2(50.0, 1.0), 1), -3.6, 1e-6);
  GR_ASSERT_TRUE(model.ramps.size() == 1);
  const Ramp& ramp = model.ramps[0];
  GR_EXPECT_EQ(ramp.level_from, 0);
  GR_EXPECT_EQ(ramp.level_to, 1);
  GR_EXPECT_NEAR(ramp.slope, 0.15, 1e-3);
  GR_EXPECT_NEAR(ramp.ZAt(Vec2(22.0, 0.0)), FloorHeight(22.0), 0.01);
  GR_EXPECT_TRUE(ramp.Contains(Vec2(22.0, 0.0)));
  GR_EXPECT_FALSE(ramp.Contains(Vec2(0.0, 0.0)));
  // Points: on level 0, on level 1, on the ramp, under nothing.
  const LevelAssignment a = AssignLevels(
      model, {Vec3(-5, 0, 1.0), Vec3(50, 0, -3.0), Vec3(22, 0, -1.5),
              Vec3(-5, 0, -10.0)});
  GR_EXPECT_EQ(a.level[0], 0);
  GR_EXPECT_NEAR(a.height[0], 1.0, 1e-5);
  GR_EXPECT_EQ(a.level[1], 1);
  GR_EXPECT_NEAR(a.height[1], 0.6, 1e-5);
  GR_EXPECT_EQ(a.level[2], 0);
  GR_EXPECT_EQ(a.on_ramp[2], 1);
  GR_EXPECT_NEAR(a.height[2], -1.5 - FloorHeight(22.0), 0.01);
  GR_EXPECT_EQ(a.level[3], -1);
  GR_EXPECT_TRUE(std::isnan(a.height[3]));
}

}  // namespace
}  // namespace gr
