#include "gr/semantics/slots.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// A level raster 20 m x 15 m from (-5, -5) at 0.1 m: empty floor, observed
// everywhere unless a car hides it.
constexpr double kRasterOrigin = -5.0;  // m, both axes.
constexpr double kRes = 0.1;            // m
constexpr int kWidth = 200;             // Cells.
constexpr int kHeight = 150;            // Cells.
constexpr float kFloorObservations = 3.0f;
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;
// Parked cars stand over y in [kCarStartY, kCarEndY] and are kCarHeight high.
constexpr double kCarStartY = 0.3;   // m
constexpr double kCarEndY = 5.0;     // m
constexpr float kCarHeight = 1.4f;   // m
constexpr double kLineWidth = 0.15;  // m
constexpr double kSlotWidth = 2.5;   // m
constexpr double kSlotDepth = 5.3;   // m
constexpr double kTolerance = 1e-9;
// The aisle (driven path) runs along y = kPathY, below the row.
constexpr double kPathY = -3.0;
constexpr double kPathStart = -4.0;
constexpr double kPathEnd = 14.0;
constexpr double kPathStep = 0.5;

SlotRasters Rasters() {
  SlotRasters rasters;
  rasters.frame =
      RasterFrame{kRasterOrigin, kRasterOrigin, kRes, kWidth, kHeight};
  const size_t cells = kWidth * kHeight;
  rasters.height_max.assign(cells, 0.0f);
  rasters.floor_z.assign(cells, 0.0f);
  rasters.floor_count.assign(cells, kFloorObservations);
  return rasters;
}

// Hides the floor of x in [x_min, x_max] (a parked car).
void ParkCar(SlotRasters& rasters, double x_min, double x_max) {
  const RasterFrame& frame = rasters.frame;
  for (int row = 0; row < frame.height; ++row) {
    for (int column = 0; column < frame.width; ++column) {
      const double x = frame.x0 + (column + kCellCentre) * frame.res;
      const double y = frame.y0 + (row + kCellCentre) * frame.res;
      if (x >= x_min && x <= x_max && y >= kCarStartY && y <= kCarEndY) {
        const size_t cell = static_cast<size_t>(row * frame.width + column);
        rasters.height_max[cell] = kCarHeight;
        rasters.floor_count[cell] = 0.0f;
      }
    }
  }
}

MarkingSegment Separator(double x, double start_y, double end_y,
                         bool start_measured, bool end_measured) {
  MarkingSegment segment;
  segment.p0 = Vec2(x, start_y);
  segment.p1 = Vec2(x, end_y);
  segment.width = kLineWidth;
  segment.end_ok[0] = start_measured;
  segment.end_ok[1] = end_measured;
  return segment;
}

std::vector<Vec2> Path() {
  std::vector<Vec2> path;
  for (double x = kPathStart; x <= kPathEnd; x += kPathStep) {
    path.emplace_back(x, kPathY);
  }
  return path;
}

GR_TEST(SlotsTest, RowEndsComeFromTheMeasuredSeparators) {
  // Back ends measured at 5.30 / 5.29 / 5.31, one line stops short; the
  // entrance ends are not measured.
  const std::vector<MarkingSegment> segments = {
      Separator(0.0 * kSlotWidth, 0.04, 5.30, false, true),
      Separator(1.0 * kSlotWidth, 0.10, 5.29, false, true),
      Separator(2.0 * kSlotWidth, 0.20, 5.31, false, true),
      Separator(3.0 * kSlotWidth, 0.30, 4.90, false, false)};
  const std::vector<ParkingSlot> slots =
      AssembleSlots(segments, 0, 0.0, Path(), {}, Rasters());
  GR_ASSERT_TRUE(slots.size() == 3);
  for (const ParkingSlot& slot : slots) {
    GR_EXPECT_NEAR(slot.width, kSlotWidth, kTolerance);
    // Back line: the median of the measured ends; the hidden entrance one
    // slot depth before it.
    GR_EXPECT_NEAR(slot.depth, kSlotDepth, kTolerance);
    GR_EXPECT_NEAR(slot.corners[2].y(), kSlotDepth, kTolerance);
    // The entrance faces the aisle (low y): corners 0 and 1 there.
    GR_EXPECT_NEAR(slot.corners[0].y(), 0.0, kTolerance);
    GR_EXPECT_FALSE(slot.inferred);
    GR_EXPECT_EQ(slot.occupied, 0);
  }
  GR_EXPECT_TRUE(slots[0].slot_id == "L0-R01-001");
}

GR_TEST(SlotsTest, InfersHiddenSeparatorsOnlyWhereTheFloorIsHidden) {
  // A two-slot gap between x = 2.5 and 7.5 in a row of 2.5 m slots.
  const std::vector<MarkingSegment> segments = {
      Separator(0.0 * kSlotWidth, 0.0, kSlotDepth, true, true),
      Separator(1.0 * kSlotWidth, 0.0, kSlotDepth, true, true),
      Separator(3.0 * kSlotWidth, 0.0, kSlotDepth, true, true),
      Separator(4.0 * kSlotWidth, 0.0, kSlotDepth, true, true)};
  // Empty floor seen in the gap: the missing line would have been seen.
  std::vector<ParkingSlot> slots =
      AssembleSlots(segments, 0, 0.0, Path(), {}, Rasters());
  int inferred = 0;
  for (const ParkingSlot& slot : slots) inferred += slot.inferred;
  GR_EXPECT_EQ(static_cast<int>(slots.size()), 2);
  GR_EXPECT_EQ(inferred, 0);
  // A car over the hidden line (x = 5): two inferred slots, one of them
  // occupied.
  constexpr double kCarStartX = 4.0;
  constexpr double kCarEndX = 6.0;
  SlotRasters rasters = Rasters();
  ParkCar(rasters, kCarStartX, kCarEndX);
  slots = AssembleSlots(segments, 0, 0.0, Path(), {}, rasters);
  inferred = 0;
  int occupied = 0;
  for (const ParkingSlot& slot : slots) {
    inferred += slot.inferred;
    occupied += slot.occupied == 1;
  }
  GR_EXPECT_EQ(static_cast<int>(slots.size()), 4);
  GR_EXPECT_EQ(inferred, 2);
  GR_EXPECT_GE(occupied, 1);
}

}  // namespace
}  // namespace gr
