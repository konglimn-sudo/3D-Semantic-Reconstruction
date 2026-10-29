#include "gr/semantics/markings.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kRes = 0.05;  // m
constexpr int kSize = 200;     // Cells: 10 m x 10 m.
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;
constexpr double kLineWidth = 0.15;  // m

GR_TEST(MarkingsTest, MergesCollinearFragments) {
  constexpr double kMergedLength = 5.0;  // m
  constexpr double kTolerance = 1e-6;
  MarkingConfig config;
  std::vector<MarkingSegment> segments(3);
  segments[0].p0 = Vec2(0, 0);
  segments[0].p1 = Vec2(2, 0);
  segments[1].p0 = Vec2(3.5, 0.05);  // 1.5 m gap, 5 cm off the line.
  segments[1].p1 = Vec2(kMergedLength, 0.05);
  segments[2].p0 = Vec2(0, 1);  // A parallel line 1 m away stays apart.
  segments[2].p1 = Vec2(kMergedLength, 1);
  for (MarkingSegment& segment : segments) segment.width = kLineWidth;
  const std::vector<MarkingSegment> out = MergeCollinear(segments, config);
  GR_ASSERT_TRUE(out.size() == 2);
  GR_EXPECT_NEAR(out[0].length(), kMergedLength, kTolerance);
  GR_EXPECT_NEAR(out[1].length(), kMergedLength, kTolerance);
}

GR_TEST(MarkingsTest, VectorizesPaintedLines) {
  constexpr int kLevel = 1;
  constexpr double kMinLength = 4.5;     // m of the 5 m separators.
  constexpr double kWidthTolerance = 0.06;  // m
  constexpr double kMaxAxisAngle = 5.0;  // deg
  constexpr double kOffGridAngle = 30.0;  // deg
  // Floor is observed in the first kObservedColumns columns (x < 3.5 m).
  constexpr int kObservedColumns = 70;
  std::vector<uint8_t> mask(kSize * kSize, 0);
  const auto paint = [&](double x_min, double x_max, double y_min,
                         double y_max) {
    for (int row = 0; row < kSize; ++row) {
      for (int column = 0; column < kSize; ++column) {
        const double x = (column + kCellCentre) * kRes;
        const double y = (row + kCellCentre) * kRes;
        if (x >= x_min && x <= x_max && y >= y_min && y <= y_max) {
          mask[row * kSize + column] = 1;
        }
      }
    }
  };
  paint(2.0, 2.0 + kLineWidth, 1.0, 6.0);  // Two separators, 2.5 m apart.
  paint(4.5, 4.5 + kLineWidth, 1.0, 6.0);
  paint(7.0, 7.4, 3.0, 3.1);  // Too short.
  RasterFrame frame{0.0, 0.0, kRes, kSize, kSize};
  const std::vector<MarkingSegment> segments =
      VectorizeMarkings(mask, frame, MarkingConfig(), kLevel);
  GR_ASSERT_TRUE(segments.size() == 2);
  for (const MarkingSegment& segment : segments) {
    GR_EXPECT_EQ(segment.level, kLevel);
    GR_EXPECT_GT(segment.length(), kMinLength);
    GR_EXPECT_NEAR(segment.width, kLineWidth, kWidthTolerance);
    GR_EXPECT_TRUE(GridAligned(segment, 0.0, kMaxAxisAngle));
    GR_EXPECT_FALSE(
        GridAligned(segment, DegToRad(kOffGridAngle), kMaxAxisAngle));
  }
  // Observed floor on both sides of the first line only.
  std::vector<uint8_t> floor(kSize * kSize, 0);
  for (int row = 0; row < kSize; ++row) {
    for (int column = 0; column < kObservedColumns; ++column) {
      floor[row * kSize + column] = 1;
    }
  }
  const MarkingSegment& left =
      segments[0].p0.x() < segments[1].p0.x() ? segments[0] : segments[1];
  const MarkingSegment& right =
      segments[0].p0.x() < segments[1].p0.x() ? segments[1] : segments[0];
  GR_EXPECT_TRUE(BothSidesObserved(left, floor, frame));
  GR_EXPECT_FALSE(BothSidesObserved(right, floor, frame));
}

// A bright stripe two cells (0.1 m) wide along x = 2, from y = 1 to y = 4,
// on a dark floor, observed everywhere except beyond y = 4.2 (a car hides
// the far end).
constexpr double kStripeX = 2.0;          // m
constexpr double kStripeStart = 1.0;      // m (y)
constexpr double kStripeEnd = 4.0;        // m (y)
constexpr double kStripeWidth = 0.1;      // m
constexpr double kStripeHalfSpan = 0.07;  // m: cell centres within it.
constexpr double kHiddenBeyond = 4.2;     // m (y)
constexpr double kFloorBrightness = 0.2;
constexpr double kPaintBrightness = 0.9;

FloorImage StripeImage() {
  FloorImage image;
  image.name = "lidar";
  image.width = kSize;
  image.height = kSize;
  image.res = kRes;
  image.brightness.assign(kSize * kSize, kFloorBrightness);
  image.valid.assign(kSize * kSize, 1.0);
  for (int row = 0; row < kSize; ++row) {
    for (int column = 0; column < kSize; ++column) {
      const double x = (column + kCellCentre) * kRes;
      const double y = (row + kCellCentre) * kRes;
      const size_t cell = static_cast<size_t>(row * kSize + column);
      if (std::abs(x - kStripeX) < kStripeHalfSpan && y >= kStripeStart &&
          y <= kStripeEnd) {
        image.brightness[cell] = kPaintBrightness;
      }
      if (y > kHiddenBeyond) image.valid[cell] = 0.0;
    }
  }
  return image;
}

GR_TEST(VectorFitTest, MeasuresTheCentreLineAndTheEnds) {
  // The skeleton's guess: 3 cm off the centre, ends eroded by 0.1 m.
  const Vec2 skeleton_start(2.03, 1.1);
  const Vec2 skeleton_end(2.03, 3.9);
  constexpr double kCentreTolerance = 0.01;  // m
  constexpr double kWidthTolerance = 0.03;   // m
  constexpr double kEndTolerance = 0.03;     // m
  const FloorImage image = StripeImage();
  const std::optional<LineFit> fit =
      FitLine(skeleton_start, skeleton_end, image);
  GR_ASSERT_TRUE(fit.has_value());
  GR_EXPECT_NEAR(fit->p0.x(), kStripeX, kCentreTolerance);
  GR_EXPECT_NEAR(fit->p1.x(), kStripeX, kCentreTolerance);
  GR_EXPECT_NEAR(fit->width, kStripeWidth, kWidthTolerance);
  // The near end is measured (floor seen beyond it), the far one is not.
  GR_EXPECT_TRUE(fit->end_ok[0]);
  GR_EXPECT_NEAR(fit->p0.y(), kStripeStart, kEndTolerance);
  GR_EXPECT_FALSE(fit->end_ok[1]);
  std::vector<MarkingSegment> segments(1);
  segments[0].p0 = skeleton_start;
  segments[0].p1 = skeleton_end;
  GR_EXPECT_EQ(RefineSegments(segments, {&image}), 1);
  GR_EXPECT_NEAR(segments[0].p0.x(), kStripeX, kCentreTolerance);
}

}  // namespace
}  // namespace gr
