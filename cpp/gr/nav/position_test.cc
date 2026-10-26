#include "gr/nav/position.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

FloorHeights Floor(int level, double z) {
  FloorHeights floor;
  floor.level = level;
  floor.x0 = -10.0;
  floor.y0 = -10.0;
  floor.res = 1.0;
  floor.width = 40;
  floor.height = 40;
  floor.z.assign(40 * 40, z);
  return floor;
}

GR_TEST(PositionTest, FindsLevelEdgeAndSlot) {
  const std::vector<FloorHeights> floors = {Floor(0, 0.0), Floor(1, 3.0)};
  const std::vector<NavEdge> edges = {
      {0, "aisle", {Vec2(0, 0), Vec2(10, 0), Vec2(10, 10)}},
      {0, "connector", {Vec2(4, 0), Vec2(4, 1)}},
      {1, "aisle", {Vec2(0, 1), Vec2(10, 1)}}};
  const std::vector<NavSlot> slots = {
      {0, "L0-001", {Vec2(3, 0.5), Vec2(5, 0.5), Vec2(5, 3), Vec2(3, 3)}}};
  // Heading west on level 0, 0.8 m left of the aisle's first leg (+y).
  const Pose vehicle(
      Eigen::AngleAxisd(kPi, Vec3::UnitZ()).toRotationMatrix(),
      Vec3(4.0, 0.8, 0.1));
  const NavPosition p = Where(vehicle, edges, slots, floors);
  GR_ASSERT_TRUE(p.level.has_value() && p.edge.has_value());
  GR_EXPECT_EQ(*p.level, 0);
  GR_EXPECT_EQ(p.edge->edge, 0);  // The connector does not count.
  GR_EXPECT_NEAR(p.edge->distance, 0.8, 1e-12);
  GR_EXPECT_NEAR(p.edge->along, 4.0, 1e-12);
  GR_EXPECT_NEAR(p.edge->lateral, 0.8, 1e-12);
  GR_EXPECT_NEAR(std::abs(p.edge->heading_rel_deg), 180.0, 1e-9);
  GR_ASSERT_TRUE(p.slot.has_value());
  GR_EXPECT_TRUE(*p.slot == "L0-001");
  // On the upper level: its own aisle; outside every floor: no level.
  const NavPosition up =
      Where(Pose(Mat3::Identity(), Vec3(2.0, 1.2, 3.1)), edges, slots, floors);
  GR_EXPECT_EQ(up.edge->edge, 2);
  GR_EXPECT_FALSE(up.slot.has_value());
  GR_EXPECT_FALSE(Where(Pose(Mat3::Identity(), Vec3(100, 0, 0)), edges,
                        slots, floors)
                      .level.has_value());
}

}  // namespace
}  // namespace gr
