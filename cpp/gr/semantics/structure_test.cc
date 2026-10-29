#include "gr/semantics/structure.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// One level (floor at z = 0, ceiling at kCeiling) with four kColumnSide
// columns at (kColumnLow | kColumnHigh)^2 and a wall along y = kWallY (its
// back face kWallThickness behind), x in [0, kWallLength].
constexpr double kCeiling = 3.0;          // m
constexpr double kColumnSide = 0.6;       // m
constexpr double kColumnLow = 5.0;        // m
constexpr double kColumnHigh = 13.0;      // m
constexpr double kWallY = 18.0;           // m
constexpr double kWallThickness = 0.2;    // m
constexpr double kWallLength = 18.0;      // m
constexpr double kSceneMin = -1.0;        // m: the floor and ceiling extent.
constexpr double kSceneMax = 19.0;        // m
constexpr double kSlabStep = 0.1;         // m between floor / ceiling points.
constexpr double kSurfaceStep = 0.05;     // m between column / wall points.
// Column and wall points stay this far (m) off the floor and the ceiling.
constexpr double kSurfaceMargin = 0.05;
constexpr double kStepSlack = 1e-9;       // m: rounding of the side loops.
constexpr double kPathY = 9.0;            // m
constexpr double kPathStep = 0.5;         // m

StructureInput MakeLevel() {
  StructureInput in;
  const auto add = [&](const Vec3& point, const Vec3& normal) {
    in.xyz.push_back(point);
    in.normals.push_back(normal);
    in.height.push_back(static_cast<float>(point.z()));
    in.level.push_back(0);
    in.on_ramp.push_back(0);
  };
  for (double x = kSceneMin; x <= kSceneMax; x += kSlabStep) {
    for (double y = kSceneMin; y <= kSceneMax; y += kSlabStep) {
      add(Vec3(x, y, 0.0), Vec3::UnitZ());
      add(Vec3(x, y, kCeiling), -Vec3::UnitZ());
    }
  }
  const double half = kColumnSide / 2.0;
  for (double center_x : {kColumnLow, kColumnHigh}) {
    for (double center_y : {kColumnLow, kColumnHigh}) {
      for (double z = kSurfaceMargin; z < kCeiling - kSurfaceMargin;
           z += kSurfaceStep) {
        for (double offset = -half; offset <= half + kStepSlack;
             offset += kSurfaceStep) {
          add(Vec3(center_x - half, center_y + offset, z), -Vec3::UnitX());
          add(Vec3(center_x + half, center_y + offset, z), Vec3::UnitX());
          add(Vec3(center_x + offset, center_y - half, z), -Vec3::UnitY());
          add(Vec3(center_x + offset, center_y + half, z), Vec3::UnitY());
        }
      }
    }
  }
  for (double x = 0.0; x <= kWallLength; x += kSurfaceStep) {
    for (double z = kSurfaceMargin; z < kCeiling - kSurfaceMargin;
         z += kSurfaceStep) {
      add(Vec3(x, kWallY, z), -Vec3::UnitY());
      add(Vec3(x, kWallY + kWallThickness, z), Vec3::UnitY());
    }
  }
  in.level_z = {0.0};
  in.ceiling = {kCeiling};
  for (double x = 0.0; x <= kWallLength; x += kPathStep) {
    in.path_xy.emplace_back(x, kPathY);
  }
  return in;
}

GR_TEST(StructureTest, FindsColumnsWallsAndTheirLabels) {
  constexpr double kYawTolerance = 1e-9;
  constexpr double kCentreTolerance = 0.03;  // m
  constexpr double kSizeTolerance = 0.05;    // m
  constexpr double kWallTolerance = 0.15;    // m off the wall's middle.
  constexpr double kMinWallLength = 15.0;    // m
  // At least 4 columns x 4 faces x 50 heights x 10 of the 13 offsets, and
  // 2 faces x 300 of the 361 stations x 50 heights.
  constexpr int kMinColumnPoints = 4 * 4 * 50 * 10;
  constexpr int kMinWallPoints = 2 * 300 * 50;
  const StructureInput in = MakeLevel();
  const StructureResult result = ExtractStructure(in);
  GR_EXPECT_NEAR(result.yaw, 0.0, kYawTolerance);
  GR_ASSERT_TRUE(result.columns.size() == 4);
  for (const StructureColumn& column : result.columns) {
    const Vec2 center = column.center;
    const double off_x = std::min(std::abs(center.x() - kColumnLow),
                                  std::abs(center.x() - kColumnHigh));
    const double off_y = std::min(std::abs(center.y() - kColumnLow),
                                  std::abs(center.y() - kColumnHigh));
    GR_EXPECT_LT(off_x, kCentreTolerance);
    GR_EXPECT_LT(off_y, kCentreTolerance);
    GR_EXPECT_NEAR(column.size.x(), kColumnSide, kSizeTolerance);
    GR_EXPECT_NEAR(column.size.y(), kColumnSide, kSizeTolerance);
    GR_EXPECT_FALSE(column.completed);
  }
  // Axis labels: rows A / B along y, numbers along x.
  GR_EXPECT_TRUE(result.columns[0].axis_label == "A1");
  GR_EXPECT_TRUE(result.columns[0].id == "L0-A1");
  GR_EXPECT_TRUE(result.columns[3].id == "L0-B2");
  GR_ASSERT_TRUE(result.walls.size() >= 1);
  bool wall_found = false;
  for (const StructureWall& wall : result.walls) {
    wall_found |=
        std::abs(wall.p0.y() - (kWallY + kWallThickness / 2.0)) <
            kWallTolerance &&
        std::abs(wall.p1.x() - wall.p0.x()) > kMinWallLength;
  }
  GR_EXPECT_TRUE(wall_found);
  // Labels: floor, ceiling, a column face, the wall.
  const StructureConfig config;
  GR_EXPECT_EQ(result.labels[0], config.floor_id);
  GR_EXPECT_EQ(result.labels[1], config.ceiling_id);
  int column_points = 0;
  int wall_points = 0;
  for (size_t point = 0; point < in.xyz.size(); ++point) {
    column_points += result.labels[point] == config.column_id;
    wall_points += result.labels[point] == config.wall_id;
  }
  GR_EXPECT_GT(column_points, kMinColumnPoints);
  GR_EXPECT_GT(wall_points, kMinWallPoints);
}

}  // namespace
}  // namespace gr
