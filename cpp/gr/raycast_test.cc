#include "gr/raycast.h"

#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(RaycastTest, GridTraversalVisitsExpectedVoxels) {
  std::vector<int> xs;
  bool off_axis = false;
  TraverseGrid(Vec3(0.5, 0.5, 0.5), Vec3(1, 0, 0), 4.2, 10, 10, 10,
               [&](int i, int j, int k) {
                 off_axis = off_axis || j != 0 || k != 0;
                 xs.push_back(i);
                 return true;
               });
  GR_EXPECT_FALSE(off_axis);
  GR_ASSERT_TRUE(xs.size() == 5);
  GR_EXPECT_EQ(xs.front(), 0);
  GR_EXPECT_EQ(xs.back(), 4);
}

GR_TEST(RaycastTest, FreeSpaceCarvingFlagsARemovedObject) {
  FreeSpaceCarver carver(Vec3(-1, -6, -2), Vec3(21, 6, 4), 0.2, 0.3);
  // Sweep 1 sees an object at x = 5; later sweeps (object gone) see the wall
  // at x = 20 along the same rays.
  const Vec3 origin(0, 0, 1);
  Points object;
  Points wall;
  for (int j = -10; j <= 10; ++j) {
    for (int k = 0; k < 8; ++k) {
      const Vec3 p(5.0, 0.05 * j, 0.5 + 0.1 * k);
      object.push_back(p);
      wall.push_back(origin + (p - origin) * 4.0);
    }
  }
  carver.Integrate(origin, object);
  for (int r = 0; r < 3; ++r) carver.Integrate(origin, wall);
  const FreeSpaceCarver::Counts on_object = carver.Query(object);
  int flagged = 0;
  for (size_t i = 0; i < object.size(); ++i) {
    if (on_object.misses[i] > 0 && on_object.misses[i] >= on_object.hits[i]) {
      ++flagged;
    }
  }
  GR_EXPECT_GT(flagged, static_cast<int>(object.size()) * 8 / 10);
  const FreeSpaceCarver::Counts on_wall = carver.Query(wall);
  for (size_t i = 0; i < wall.size(); ++i) GR_EXPECT_GT(on_wall.hits[i], 0);
  // Dense export: free in front of the wall, not on it.
  const std::vector<uint8_t> free = carver.FreeMask(2, 0);
  const Eigen::Vector3i dims = carver.dims();
  GR_ASSERT_TRUE(free.size() ==
                 static_cast<size_t>(dims.x()) * dims.y() * dims.z());
  auto at = [&](const Vec3& p) {
    const Eigen::Vector3i g =
        ((p - carver.min_bound()) / carver.voxel()).array().floor().cast<int>();
    return free[(static_cast<size_t>(g.x()) * dims.y() + g.y()) * dims.z() +
                g.z()];
  };
  GR_EXPECT_EQ(at(Vec3(10.0, 0.0, 1.0 + 9.0 * 0.0)), 1);
  GR_EXPECT_EQ(at(wall[wall.size() / 2]), 0);
}

}  // namespace
}  // namespace gr
