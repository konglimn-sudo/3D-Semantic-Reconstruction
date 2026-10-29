#include "gr/semantics/slot_paint.h"

#include <cmath>
#include <vector>

#include "gr/core/image.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(SlotPaintTest, GreenSlotsStandOutFromConcrete) {
  // 10 x 6 m at 0.1 m: grey concrete, a green slot at x 1 .. 3.5, white
  // separators at its sides, an unobserved patch (a parked car) in a grey
  // slot at x 5 .. 7.5.
  constexpr int kWidth = 100;   // Cells.
  constexpr int kHeight = 60;   // Cells.
  constexpr double kRes = 0.1;  // m
  constexpr double kGreenStart = 1.0;   // m
  constexpr double kGreenEnd = 3.5;     // m
  constexpr double kGreyStart = 5.0;    // m
  constexpr double kGreyEnd = 7.5;      // m
  constexpr double kSlotNear = 0.5;     // m (y)
  constexpr double kSlotFar = 5.5;      // m (y)
  constexpr double kLineHalfWidth = 0.08;  // m
  // The car hides x 5.5 .. 7 over rows 11 .. 44.
  constexpr double kCarStart = 5.5;
  constexpr double kCarEnd = 7.0;
  constexpr int kCarFirstRow = 10;  // Exclusive.
  constexpr int kCarLastRow = 45;   // Exclusive.
  constexpr uint8_t kConcrete[kRgbChannels] = {108, 108, 112};
  constexpr uint8_t kGreen[kRgbChannels] = {70, 150, 90};
  constexpr uint8_t kWhite = 232;
  constexpr double kMinGreenShare = 0.95;
  constexpr double kTolerance = 1e-9;
  const RasterFrame frame{0.0, 0.0, kRes, kWidth, kHeight};
  std::vector<uint8_t> rgb(kWidth * kHeight * kRgbChannels);
  std::vector<uint8_t> observed(kWidth * kHeight, 1);
  for (int row = 0; row < kHeight; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      const double x = (column + 0.5) * kRes;
      const size_t cell = static_cast<size_t>(row) * kWidth + column;
      uint8_t colour[kRgbChannels] = {kConcrete[0], kConcrete[1],
                                      kConcrete[2]};
      if (x > kGreenStart && x < kGreenEnd) {
        for (int channel = 0; channel < kRgbChannels; ++channel) {
          colour[channel] = kGreen[channel];
        }
      }
      if (std::abs(x - kGreenStart) < kLineHalfWidth ||
          std::abs(x - kGreenEnd) < kLineHalfWidth) {
        colour[0] = colour[1] = colour[2] = kWhite;
      }
      for (int channel = 0; channel < kRgbChannels; ++channel) {
        rgb[kRgbChannels * cell + channel] = colour[channel];
      }
      if (x > kCarStart && x < kCarEnd && row > kCarFirstRow &&
          row < kCarLastRow) {
        observed[cell] = 0;
      }
    }
  }
  const auto slot = [&](double start, double end) {
    return std::array<Vec2, kParkingSlotCorners>{
        Vec2(start, kSlotNear), Vec2(end, kSlotNear), Vec2(end, kSlotFar),
        Vec2(start, kSlotFar)};
  };
  const std::vector<SlotPaint> paint = MeasureSlotPaint(
      {slot(kGreenStart, kGreenEnd), slot(kGreyStart, kGreyEnd)}, rgb,
      observed, frame);
  GR_ASSERT_TRUE(paint.size() == 2);
  GR_EXPECT_GT(paint[0].in_band, kMinGreenShare);
  GR_EXPECT_NEAR(paint[0].mean_rgb.y(), kGreen[1], kTolerance);
  GR_EXPECT_EQ(paint[1].in_band, 0.0);
  GR_EXPECT_LT(paint[1].observed, paint[1].cells / 2);  // Mostly hidden.
}

}  // namespace
}  // namespace gr
