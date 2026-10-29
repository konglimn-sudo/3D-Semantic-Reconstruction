#include "gr/semantics/occupancy.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr int kFloorLabel = 1;
constexpr int kCarLabel = 13;
constexpr double kVoxel = 0.1;  // m, OccupancyConfig's default.
// Rounding slack of the floor grid's sample count.
constexpr double kCountSlack = 1e-6;

// Floor grid (kFloorLabel) on z = 0 over [x_min, x_max] x [y_min, y_max].
void AddFloor(double x_min, double x_max, double y_min, double y_max,
              double step, Points& points, std::vector<int>& labels) {
  const int steps_x = static_cast<int>((x_max - x_min) / step + kCountSlack);
  const int steps_y = static_cast<int>((y_max - y_min) / step + kCountSlack);
  for (int i = 0; i <= steps_x; ++i) {
    for (int j = 0; j <= steps_y; ++j) {
      points.emplace_back(x_min + step * i, y_min + step * j, 0.0);
      labels.push_back(kFloorLabel);
    }
  }
}

GR_TEST(OccupancyTest, StaticLabelsBoxesAndMasks) {
  // A 10 x 10 x 3 m grid; a floor point in every column 2 cm below z = 0,
  // a car at x = 3 and the LiDAR 1.9 m up at the origin.
  constexpr int kHalfFloorCells = 50;
  constexpr double kFloorZ = -0.02;  // m
  constexpr int kGridXY = 100;       // Voxels.
  constexpr int kGridZ = 30;         // Voxels.
  constexpr size_t kMinFloorVoxels = 5000;
  constexpr size_t kMinCarVoxels = 2000;
  OccupancyConfig config;
  config.min_bound = Vec3(-5, -5, -0.5);
  config.max_bound = Vec3(5, 5, 2.5);
  Points floor;
  std::vector<int> labels;
  for (int i = -kHalfFloorCells; i < kHalfFloorCells; ++i) {
    for (int j = -kHalfFloorCells; j < kHalfFloorCells; ++j) {
      floor.emplace_back(kVoxel * i + 0.5 * kVoxel, kVoxel * j + 0.5 * kVoxel,
                         kFloorZ);
      labels.push_back(kFloorLabel);
    }
  }
  OrientedBox car;
  car.center = Vec3(3, 0, 0.75);
  car.size = Vec3(4.0, 1.8, 1.5);
  car.label = kCarLabel;
  const OccupancyResult result = BuildOccupancy(
      config, Pose(), floor, labels, {car}, Vec3(0, 0, 1.9), floor, {});
  GR_ASSERT_TRUE(result.nx == kGridXY && result.ny == kGridXY &&
                 result.nz == kGridZ);
  size_t num_floor = 0;
  size_t num_car = 0;
  size_t num_observed = 0;
  for (size_t voxel = 0; voxel < result.semantics.size(); ++voxel) {
    num_floor += result.semantics[voxel] == kFloorLabel;
    num_car += result.semantics[voxel] == kCarLabel;
    num_observed += result.mask_lidar[voxel];
  }
  GR_EXPECT_GT(num_floor, kMinFloorVoxels);
  GR_EXPECT_GT(num_car, kMinCarVoxels);
  GR_EXPECT_GT(num_observed, num_floor);
}

GR_TEST(OccupancyGeneratorTest, PlacesWorldBoxesInTheVehicleFrame) {
  // A 12 x 12 x 3 m grid around a vehicle at (100, 50) facing world +y; the
  // LiDAR is 1 m ahead of and 1.8 m above the vehicle origin.
  constexpr double kHalfGrid = 6.0;  // m
  constexpr double kGridBottom = -0.4;  // m
  constexpr int kGridXY = 120;       // Voxels.
  constexpr int kGridZ = 30;         // Voxels.
  constexpr double kFloorStep = 0.05;  // m
  constexpr double kCarHalfHeight = 0.75;  // m
  Points floor;
  std::vector<int> labels;
  AddFloor(85, 115, 35, 65, kFloorStep, floor, labels);
  OccupancyConfig config;
  config.min_bound = Vec3(-kHalfGrid, -kHalfGrid, kGridBottom);
  config.max_bound = Vec3(kHalfGrid, kHalfGrid, 2.6);
  const OccupancyGenerator generator(
      floor, labels, config, Pose(Mat3::Identity(), Vec3(1, 0, 1.8)), {});
  const Pose world_from_vehicle(
      Eigen::AngleAxisd(kPi / 2, Vec3::UnitZ()).toRotationMatrix(),
      Vec3(100, 50, 0));
  // A car 4 m ahead of the vehicle (world +y), aligned with it.
  ObjectBox car;
  car.label = kCarLabel;
  car.center = Vec3(100, 54, kCarHalfHeight);
  car.size = Vec3(4.6, 1.9, 2.0 * kCarHalfHeight);
  car.yaw = kPi / 2;
  const OccupancyResult result =
      generator.Generate(world_from_vehicle, {car}, {Vec3(3, 0, -1.8)});
  GR_ASSERT_TRUE(result.nx == kGridXY && result.ny == kGridXY &&
                 result.nz == kGridZ);
  const auto at = [&](double x, double y, double z) {
    const size_t i = static_cast<size_t>((x + kHalfGrid) / kVoxel);
    const size_t j = static_cast<size_t>((y + kHalfGrid) / kVoxel);
    const size_t k = static_cast<size_t>((z - kGridBottom) / kVoxel);
    return result.semantics[(i * kGridXY + j) * kGridZ + k];
  };
  constexpr double kNearFloor = 0.05;  // m
  GR_EXPECT_EQ(at(4.0, 0.0, kCarHalfHeight), kCarLabel);  // Ahead.
  GR_EXPECT_EQ(at(0.0, 4.0, kCarHalfHeight), config.free_label);  // Not left.
  GR_EXPECT_EQ(at(-3.0, 2.0, kNearFloor), kFloorLabel);  // Floor around.
  size_t observed = 0;
  for (uint8_t seen : result.mask_lidar) observed += seen;
  GR_EXPECT_GT(observed, 0u);
}

}  // namespace
}  // namespace gr
