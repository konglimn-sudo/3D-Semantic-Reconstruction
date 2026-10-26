#include "gr/semantics/slots.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// A level raster 20 m x 15 m from (-5, -5) at 0.1 m: empty floor, observed
// everywhere unless `hidden` says otherwise.
SlotRasters Rasters() {
  SlotRasters r;
  r.frame = RasterFrame{-5.0, -5.0, 0.1, 200, 150};
  const size_t n = 200 * 150;
  r.height_max.assign(n, 0.0f);
  r.floor_z.assign(n, 0.0f);
  r.floor_count.assign(n, 3.0f);
  return r;
}

// Hides the floor of x in [x0, x1] (a parked car).
void ParkCar(SlotRasters& r, double x0, double x1) {
  for (int j = 0; j < r.frame.height; ++j) {
    for (int i = 0; i < r.frame.width; ++i) {
      const double x = r.frame.x0 + (i + 0.5) * r.frame.res;
      const double y = r.frame.y0 + (j + 0.5) * r.frame.res;
      if (x >= x0 && x <= x1 && y >= 0.3 && y <= 5.0) {
        r.height_max[static_cast<size_t>(j * r.frame.width + i)] = 1.4f;
        r.floor_count[static_cast<size_t>(j * r.frame.width + i)] = 0.0f;
      }
    }
  }
}

MarkingSegment Separator(double x, double y0, double y1, bool ok0, bool ok1) {
  MarkingSegment s;
  s.p0 = Vec2(x, y0);
  s.p1 = Vec2(x, y1);
  s.width = 0.15;
  s.end_ok[0] = ok0;
  s.end_ok[1] = ok1;
  return s;
}

// The aisle (driven path) runs along y = -3, below the row.
std::vector<Vec2> Path() {
  std::vector<Vec2> path;
  for (double x = -4.0; x <= 14.0; x += 0.5) path.emplace_back(x, -3.0);
  return path;
}

GR_TEST(SlotsTest, RowEndsComeFromTheMeasuredSeparators) {
  // Back ends measured at 5.30 / 5.29 / 5.31, one line stops short; the
  // entrance ends are not measured.
  const std::vector<MarkingSegment> segs = {
      Separator(0.0, 0.04, 5.30, false, true),
      Separator(2.5, 0.10, 5.29, false, true),
      Separator(5.0, 0.20, 5.31, false, true),
      Separator(7.5, 0.30, 4.90, false, false)};
  const std::vector<ParkingSlot> slots =
      AssembleSlots(segs, 0, 0.0, Path(), {}, Rasters());
  GR_ASSERT_TRUE(slots.size() == 3);
  for (const ParkingSlot& s : slots) {
    GR_EXPECT_NEAR(s.width, 2.5, 1e-9);
    // Back line: the median of the measured ends; the hidden entrance one
    // slot depth (5.3 m) before it.
    GR_EXPECT_NEAR(s.depth, 5.3, 1e-9);
    GR_EXPECT_NEAR(s.corners[2].y(), 5.30, 1e-9);
    // The entrance faces the aisle (low y): corners 0 and 1 there.
    GR_EXPECT_NEAR(s.corners[0].y(), 0.0, 1e-9);
    GR_EXPECT_FALSE(s.inferred);
    GR_EXPECT_EQ(s.occupied, 0);
  }
  GR_EXPECT_TRUE(slots[0].slot_id == "L0-R01-001");
}

GR_TEST(SlotsTest, InfersHiddenSeparatorsOnlyWhereTheFloorIsHidden) {
  // A 5 m gap between x = 2.5 and 7.5 in a row of 2.5 m slots.
  const std::vector<MarkingSegment> segs = {
      Separator(0.0, 0.0, 5.3, true, true),
      Separator(2.5, 0.0, 5.3, true, true),
      Separator(7.5, 0.0, 5.3, true, true),
      Separator(10.0, 0.0, 5.3, true, true)};
  // Empty floor seen in the gap: the missing line would have been seen.
  std::vector<ParkingSlot> slots =
      AssembleSlots(segs, 0, 0.0, Path(), {}, Rasters());
  int inferred = 0;
  for (const ParkingSlot& s : slots) inferred += s.inferred;
  GR_EXPECT_EQ(static_cast<int>(slots.size()), 2);
  GR_EXPECT_EQ(inferred, 0);
  // A car over the hidden line: two inferred slots, one of them occupied.
  SlotRasters rasters = Rasters();
  ParkCar(rasters, 4.0, 6.0);
  slots = AssembleSlots(segs, 0, 0.0, Path(), {}, rasters);
  inferred = 0;
  int occupied = 0;
  for (const ParkingSlot& s : slots) {
    inferred += s.inferred;
    occupied += s.occupied == 1;
  }
  GR_EXPECT_EQ(static_cast<int>(slots.size()), 4);
  GR_EXPECT_EQ(inferred, 2);
  GR_EXPECT_GE(occupied, 1);
}

}  // namespace
}  // namespace gr
