#include "gr/semantics/structure.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// One level (floor at z = 0, ceiling at 3 m) with four 0.6 m columns at
// (5, 5), (13, 5), (5, 13), (13, 13) and a wall along y = 18, x in [0, 18].
StructureInput MakeLevel() {
  StructureInput in;
  auto add = [&](const Vec3& p, const Vec3& n) {
    in.xyz.push_back(p);
    in.normals.push_back(n);
    in.height.push_back(static_cast<float>(p.z()));
    in.level.push_back(0);
    in.on_ramp.push_back(0);
  };
  const double step = 0.05;
  for (double x = -1.0; x <= 19.0; x += 0.1) {
    for (double y = -1.0; y <= 19.0; y += 0.1) {
      add(Vec3(x, y, 0.0), Vec3::UnitZ());
      add(Vec3(x, y, 3.0), -Vec3::UnitZ());
    }
  }
  const double half = 0.3;
  for (double cx : {5.0, 13.0}) {
    for (double cy : {5.0, 13.0}) {
      for (double z = 0.05; z < 2.95; z += step) {
        for (double t = -half; t <= half + 1e-9; t += step) {
          add(Vec3(cx - half, cy + t, z), -Vec3::UnitX());
          add(Vec3(cx + half, cy + t, z), Vec3::UnitX());
          add(Vec3(cx + t, cy - half, z), -Vec3::UnitY());
          add(Vec3(cx + t, cy + half, z), Vec3::UnitY());
        }
      }
    }
  }
  for (double x = 0.0; x <= 18.0; x += step) {
    for (double z = 0.05; z < 2.95; z += step) {
      add(Vec3(x, 18.0, z), -Vec3::UnitY());
      add(Vec3(x, 18.2, z), Vec3::UnitY());
    }
  }
  in.level_z = {0.0};
  in.ceiling = {3.0};
  for (double x = 0.0; x <= 18.0; x += 0.5) in.path_xy.emplace_back(x, 9.0);
  return in;
}

GR_TEST(StructureTest, FindsColumnsWallsAndTheirLabels) {
  const StructureInput in = MakeLevel();
  const StructureResult result = ExtractStructure(in);
  GR_EXPECT_NEAR(result.yaw, 0.0, 1e-9);
  GR_ASSERT_TRUE(result.columns.size() == 4);
  for (const StructureColumn& column : result.columns) {
    const Vec2 c = column.center;
    const double dx = std::min(std::abs(c.x() - 5.0), std::abs(c.x() - 13.0));
    const double dy = std::min(std::abs(c.y() - 5.0), std::abs(c.y() - 13.0));
    GR_EXPECT_LT(dx, 0.03);
    GR_EXPECT_LT(dy, 0.03);
    GR_EXPECT_NEAR(column.size.x(), 0.6, 0.05);
    GR_EXPECT_NEAR(column.size.y(), 0.6, 0.05);
    GR_EXPECT_FALSE(column.completed);
  }
  // Axis labels: rows A / B along y, numbers along x.
  GR_EXPECT_TRUE(result.columns[0].axis_label == "A1");
  GR_EXPECT_TRUE(result.columns[0].id == "L0-A1");
  GR_EXPECT_TRUE(result.columns[3].id == "L0-B2");
  GR_ASSERT_TRUE(result.walls.size() >= 1);
  bool wall_found = false;
  for (const StructureWall& w : result.walls) {
    wall_found |= std::abs(w.p0.y() - 18.1) < 0.15 &&
                  std::abs(w.p1.x() - w.p0.x()) > 15.0;
  }
  GR_EXPECT_TRUE(wall_found);
  // Labels: floor, ceiling, a column face, the wall.
  const StructureConfig config;
  GR_EXPECT_EQ(result.labels[0], config.floor_id);
  GR_EXPECT_EQ(result.labels[1], config.ceiling_id);
  int column_points = 0;
  int wall_points = 0;
  for (size_t i = 0; i < in.xyz.size(); ++i) {
    column_points += result.labels[i] == config.column_id;
    wall_points += result.labels[i] == config.wall_id;
  }
  GR_EXPECT_GT(column_points, 4 * 4 * 50 * 10);
  GR_EXPECT_GT(wall_points, 2 * 300 * 50);
}

}  // namespace
}  // namespace gr
