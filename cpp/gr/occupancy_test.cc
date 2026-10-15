#include "gr/occupancy.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Floor grid (label 1) on z = 0 over [x0, x1] x [y0, y1].
void AddFloor(double x0, double x1, double y0, double y1, double step,
              Points& points, std::vector<int>& labels) {
  const int nx = static_cast<int>((x1 - x0) / step + 1e-6);
  const int ny = static_cast<int>((y1 - y0) / step + 1e-6);
  for (int i = 0; i <= nx; ++i) {
    for (int j = 0; j <= ny; ++j) {
      points.emplace_back(x0 + step * i, y0 + step * j, 0.0);
      labels.push_back(1);
    }
  }
}

GR_TEST(OccupancyTest, StaticLabelsBoxesAndMasks) {
  OccupancyConfig config;
  config.min_bound = Vec3(-5, -5, -0.5);
  config.max_bound = Vec3(5, 5, 2.5);
  Points floor;
  std::vector<int> labels;
  for (int i = -50; i < 50; ++i) {
    for (int j = -50; j < 50; ++j) {
      floor.emplace_back(0.1 * i + 0.05, 0.1 * j + 0.05, -0.02);
      labels.push_back(1);
    }
  }
  OrientedBox car;
  car.center = Vec3(3, 0, 0.75);
  car.size = Vec3(4.0, 1.8, 1.5);
  car.label = 13;
  const OccupancyResult result = BuildOccupancy(
      config, Pose(), floor, labels, {car}, Vec3(0, 0, 1.9), floor, {});
  GR_ASSERT_TRUE(result.nx == 100 && result.ny == 100 && result.nz == 30);
  size_t num_floor = 0;
  size_t num_car = 0;
  size_t num_observed = 0;
  for (size_t i = 0; i < result.semantics.size(); ++i) {
    num_floor += result.semantics[i] == 1;
    num_car += result.semantics[i] == 13;
    num_observed += result.mask_lidar[i];
  }
  GR_EXPECT_GT(num_floor, 5000u);
  GR_EXPECT_GT(num_car, 2000u);
  GR_EXPECT_GT(num_observed, num_floor);
}

GR_TEST(OccupancyGeneratorTest, PlacesWorldBoxesInTheVehicleFrame) {
  Points floor;
  std::vector<int> labels;
  AddFloor(85, 115, 35, 65, 0.05, floor, labels);
  OccupancyConfig config;
  config.min_bound = Vec3(-6, -6, -0.4);
  config.max_bound = Vec3(6, 6, 2.6);
  const OccupancyGenerator generator(
      floor, labels, config, Pose(Mat3::Identity(), Vec3(1, 0, 1.8)), {});
  const Pose world_from_vehicle(
      Eigen::AngleAxisd(kPi / 2, Vec3::UnitZ()).toRotationMatrix(),
      Vec3(100, 50, 0));
  // A car 4 m ahead of the vehicle (world +y), aligned with it.
  ObjectBox car;
  car.label = 13;
  car.center = Vec3(100, 54, 0.75);
  car.size = Vec3(4.6, 1.9, 1.5);
  car.yaw = kPi / 2;
  const OccupancyResult result =
      generator.Generate(world_from_vehicle, {car}, {Vec3(3, 0, -1.8)});
  GR_ASSERT_TRUE(result.nx == 120 && result.ny == 120 && result.nz == 30);
  const auto at = [&](double x, double y, double z) {
    const size_t i = static_cast<size_t>((x + 6) / 0.1);
    const size_t j = static_cast<size_t>((y + 6) / 0.1);
    const size_t k = static_cast<size_t>((z + 0.4) / 0.1);
    return result.semantics[(i * 120 + j) * 30 + k];
  };
  GR_EXPECT_EQ(at(4.0, 0.0, 0.75), 13);  // Ahead in the vehicle frame.
  GR_EXPECT_EQ(at(0.0, 4.0, 0.75), 0);   // Not to the left.
  GR_EXPECT_EQ(at(-3.0, 2.0, 0.05), 1);  // Floor around the vehicle.
  size_t observed = 0;
  for (uint8_t m : result.mask_lidar) observed += m;
  GR_EXPECT_GT(observed, 0u);
}

}  // namespace
}  // namespace gr
