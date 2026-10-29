#include "gr/nav/position.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// A flat floor grid of kFloorCells x kFloorCells cells of 1 m from
// (-10, -10).
constexpr int kFloorCells = 40;
constexpr double kFloorOrigin = -10.0;
constexpr double kExact = 1e-12;

FloorHeights Floor(int level, double z) {
  FloorHeights floor;
  floor.level = level;
  floor.x0 = kFloorOrigin;
  floor.y0 = kFloorOrigin;
  floor.res = 1.0;
  floor.width = kFloorCells;
  floor.height = kFloorCells;
  floor.z.assign(kFloorCells * kFloorCells, z);
  return floor;
}

GR_TEST(PositionTest, FindsLevelEdgeAndSlot) {
  constexpr double kUpperFloor = 3.0;  // m
  constexpr double kOffAisle = 0.8;    // m left of the first leg (+y).
  constexpr double kAlongAisle = 4.0;  // m
  constexpr double kHeadingTolerance = 1e-9;  // deg
  const std::vector<FloorHeights> floors = {Floor(0, 0.0),
                                            Floor(1, kUpperFloor)};
  const std::vector<NavEdge> edges = {
      {0, "aisle", {Vec2(0, 0), Vec2(10, 0), Vec2(10, 10)}},
      {0, "connector", {Vec2(4, 0), Vec2(4, 1)}},
      {1, "aisle", {Vec2(0, 1), Vec2(10, 1)}}};
  const std::vector<NavSlot> slots = {
      {0, "L0-001", {Vec2(3, 0.5), Vec2(5, 0.5), Vec2(5, 3), Vec2(3, 3)}}};
  // Heading west on level 0, beside the aisle's first leg.
  const Pose vehicle(
      Eigen::AngleAxisd(kPi, Vec3::UnitZ()).toRotationMatrix(),
      Vec3(kAlongAisle, kOffAisle, 0.1));
  const NavPosition position = Where(vehicle, edges, slots, floors);
  GR_ASSERT_TRUE(position.level.has_value() && position.edge.has_value());
  GR_EXPECT_EQ(*position.level, 0);
  GR_EXPECT_EQ(position.edge->edge, 0);  // The connector does not count.
  GR_EXPECT_NEAR(position.edge->distance, kOffAisle, kExact);
  GR_EXPECT_NEAR(position.edge->along, kAlongAisle, kExact);
  GR_EXPECT_NEAR(position.edge->lateral, kOffAisle, kExact);
  GR_EXPECT_NEAR(std::abs(position.edge->heading_rel_deg),
                 kDegreesPerHalfTurn, kHeadingTolerance);
  GR_ASSERT_TRUE(position.slot.has_value());
  GR_EXPECT_TRUE(*position.slot == "L0-001");
  // On the upper level: its own aisle; outside every floor: no level.
  const NavPosition up = Where(Pose(Mat3::Identity(), Vec3(2.0, 1.2, 3.1)),
                               edges, slots, floors);
  GR_EXPECT_EQ(up.edge->edge, 2);
  GR_EXPECT_FALSE(up.slot.has_value());
  GR_EXPECT_FALSE(Where(Pose(Mat3::Identity(), Vec3(100, 0, 0)), edges,
                        slots, floors)
                      .level.has_value());
}

}  // namespace
}  // namespace gr
