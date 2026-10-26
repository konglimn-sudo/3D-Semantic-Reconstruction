#include "gr/semantics/markings.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kRes = 0.05;
constexpr int kSize = 200;  // 10 m x 10 m.

GR_TEST(MarkingsTest, MergesCollinearFragments) {
  MarkingConfig config;
  std::vector<MarkingSegment> segs(3);
  segs[0].p0 = Vec2(0, 0);
  segs[0].p1 = Vec2(2, 0);
  segs[1].p0 = Vec2(3.5, 0.05);  // 1.5 m gap, 5 cm off the line.
  segs[1].p1 = Vec2(5, 0.05);
  segs[2].p0 = Vec2(0, 1);  // A parallel line 1 m away stays apart.
  segs[2].p1 = Vec2(5, 1);
  for (MarkingSegment& s : segs) s.width = 0.15;
  const std::vector<MarkingSegment> out = MergeCollinear(segs, config);
  GR_ASSERT_TRUE(out.size() == 2);
  GR_EXPECT_NEAR(out[0].length(), 5.0, 1e-6);
  GR_EXPECT_NEAR(out[1].length(), 5.0, 1e-6);
}

GR_TEST(MarkingsTest, VectorizesPaintedLines) {
  std::vector<uint8_t> mask(kSize * kSize, 0);
  auto paint = [&](double x0, double x1, double y0, double y1) {
    for (int j = 0; j < kSize; ++j) {
      for (int i = 0; i < kSize; ++i) {
        const double x = (i + 0.5) * kRes;
        const double y = (j + 0.5) * kRes;
        if (x >= x0 && x <= x1 && y >= y0 && y <= y1) mask[j * kSize + i] = 1;
      }
    }
  };
  paint(2.0, 2.15, 1.0, 6.0);  // Two separators, 2.5 m apart.
  paint(4.5, 4.65, 1.0, 6.0);
  paint(7.0, 7.4, 3.0, 3.1);   // Too short.
  RasterFrame frame{0.0, 0.0, kRes, kSize, kSize};
  const std::vector<MarkingSegment> segs =
      VectorizeMarkings(mask, frame, MarkingConfig(), 1);
  GR_ASSERT_TRUE(segs.size() == 2);
  for (const MarkingSegment& s : segs) {
    GR_EXPECT_EQ(s.level, 1);
    GR_EXPECT_GT(s.length(), 4.5);
    GR_EXPECT_NEAR(s.width, 0.15, 0.06);
    GR_EXPECT_TRUE(GridAligned(s, 0.0, 5.0));
    GR_EXPECT_FALSE(GridAligned(s, DegToRad(30.0), 5.0));
  }
  // Observed floor on both sides of the first line only.
  std::vector<uint8_t> floor(kSize * kSize, 0);
  for (int j = 0; j < kSize; ++j) {
    for (int i = 0; i < 70; ++i) floor[j * kSize + i] = 1;
  }
  const MarkingSegment& left =
      segs[0].p0.x() < segs[1].p0.x() ? segs[0] : segs[1];
  const MarkingSegment& right =
      segs[0].p0.x() < segs[1].p0.x() ? segs[1] : segs[0];
  GR_EXPECT_TRUE(BothSidesObserved(left, floor, frame));
  GR_EXPECT_FALSE(BothSidesObserved(right, floor, frame));
}

// A bright stripe two cells (0.1 m) wide along x = 2, from y = 1 to y = 4,
// on a dark floor, observed everywhere except beyond y = 4.2 (a car hides
// the far end).
FloorImage StripeImage() {
  FloorImage image;
  image.name = "lidar";
  image.width = kSize;
  image.height = kSize;
  image.res = kRes;
  image.brightness.assign(kSize * kSize, 0.2);
  image.valid.assign(kSize * kSize, 1.0);
  for (int j = 0; j < kSize; ++j) {
    for (int i = 0; i < kSize; ++i) {
      const double x = (i + 0.5) * kRes;
      const double y = (j + 0.5) * kRes;
      const size_t c = static_cast<size_t>(j * kSize + i);
      if (std::abs(x - 2.0) < 0.07 && y >= 1.0 && y <= 4.0) {
        image.brightness[c] = 0.9;
      }
      if (y > 4.2) image.valid[c] = 0.0;
    }
  }
  return image;
}

GR_TEST(VectorFitTest, MeasuresTheCentreLineAndTheEnds) {
  const FloorImage image = StripeImage();
  const std::optional<LineFit> fit =
      FitLine(Vec2(2.03, 1.1), Vec2(2.03, 3.9), image);
  GR_ASSERT_TRUE(fit.has_value());
  GR_EXPECT_NEAR(fit->p0.x(), 2.0, 0.01);
  GR_EXPECT_NEAR(fit->p1.x(), 2.0, 0.01);
  GR_EXPECT_NEAR(fit->width, 0.1, 0.03);
  // The near end is measured (floor seen beyond it), the far one is not.
  GR_EXPECT_TRUE(fit->end_ok[0]);
  GR_EXPECT_NEAR(fit->p0.y(), 1.0, 0.03);
  GR_EXPECT_FALSE(fit->end_ok[1]);
  std::vector<MarkingSegment> segs(1);
  segs[0].p0 = Vec2(2.03, 1.1);
  segs[0].p1 = Vec2(2.03, 3.9);
  GR_EXPECT_EQ(RefineSegments(segs, {&image}), 1);
  GR_EXPECT_NEAR(segs[0].p0.x(), 2.0, 0.01);
}

}  // namespace
}  // namespace gr
