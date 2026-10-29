#include "gr/mapping/raycast.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(RaycastTest, GridTraversalVisitsExpectedVoxels) {
  constexpr int kGridSize = 10;
  constexpr double kRayLength = 4.2;  // Voxels: ends in voxel 4.
  std::vector<int> visited_x;
  bool off_axis = false;
  TraverseGrid(Vec3(0.5, 0.5, 0.5), Vec3::UnitX(), kRayLength,
               Eigen::Vector3i::Constant(kGridSize), [&](int x, int y, int z) {
                 off_axis = off_axis || y != 0 || z != 0;
                 visited_x.push_back(x);
                 return true;
               });
  GR_EXPECT_FALSE(off_axis);
  GR_ASSERT_TRUE(visited_x.size() == 5);
  GR_EXPECT_EQ(visited_x.front(), 0);
  GR_EXPECT_EQ(visited_x.back(), 4);
}

GR_TEST(RaycastTest, FreeSpaceCarvingFlagsARemovedObject) {
  constexpr double kVoxel = 0.2;            // m
  constexpr double kEndpointMargin = 0.3;   // m
  constexpr double kObjectX = 5.0;          // m
  constexpr double kWallFactor = 4.0;       // The wall is 4x farther (x = 20).
  constexpr int kHalfColumns = 10;          // Points across the object.
  constexpr int kRows = 8;                  // Points up the object.
  constexpr double kColumnStep = 0.05;      // m
  constexpr double kRowStep = 0.1;          // m
  constexpr double kLowestRow = 0.5;        // m
  constexpr int kWallSweeps = 3;
  // Share of the object's points flagged free (in tenths).
  constexpr int kFlaggedTenths = 8;
  constexpr int kMinMisses = 2;
  constexpr int kMaxHits = 0;
  FreeSpaceCarver carver(Vec3(-1, -6, -2), Vec3(21, 6, 4), kVoxel,
                         kEndpointMargin);
  // Sweep 1 sees an object; later sweeps (object gone) see the wall behind it
  // along the same rays.
  const Vec3 origin(0, 0, 1);
  Points object;
  Points wall;
  for (int column = -kHalfColumns; column <= kHalfColumns; ++column) {
    for (int row = 0; row < kRows; ++row) {
      const Vec3 point(kObjectX, kColumnStep * column,
                       kLowestRow + kRowStep * row);
      object.push_back(point);
      wall.push_back(origin + (point - origin) * kWallFactor);
    }
  }
  carver.Integrate(origin, object);
  for (int sweep = 0; sweep < kWallSweeps; ++sweep) {
    carver.Integrate(origin, wall);
  }
  const FreeSpaceCarver::Counts on_object = carver.Query(object);
  int flagged = 0;
  for (size_t i = 0; i < object.size(); ++i) {
    if (on_object.misses[i] > 0 && on_object.misses[i] >= on_object.hits[i]) {
      ++flagged;
    }
  }
  GR_EXPECT_GT(flagged, static_cast<int>(object.size()) * kFlaggedTenths / 10);
  const FreeSpaceCarver::Counts on_wall = carver.Query(wall);
  for (size_t i = 0; i < wall.size(); ++i) GR_EXPECT_GT(on_wall.hits[i], 0);
  // Dense export: free in front of the wall, not on it.
  const std::vector<uint8_t> free = carver.FreeMask(kMinMisses, kMaxHits);
  const Eigen::Vector3i dims = carver.dims();
  GR_ASSERT_TRUE(free.size() ==
                 static_cast<size_t>(dims.x()) * dims.y() * dims.z());
  const auto free_at = [&](const Vec3& point) {
    const Eigen::Vector3i cell = ((point - carver.min_bound()) / carver.voxel())
                                     .array()
                                     .floor()
                                     .cast<int>();
    return free[(static_cast<size_t>(cell.x()) * dims.y() + cell.y()) *
                    dims.z() +
                cell.z()];
  };
  GR_EXPECT_EQ(free_at(Vec3(10.0, 0.0, 1.0)), 1);  // Between object and wall.
  GR_EXPECT_EQ(free_at(wall[wall.size() / 2]), 0);
}

}  // namespace
}  // namespace gr
