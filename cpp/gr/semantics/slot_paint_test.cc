#include "gr/semantics/slot_paint.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(SlotPaintTest, GreenSlotsStandOutFromConcrete) {
  // 10 x 6 m at 0.1 m: grey concrete, a green slot at x 1 .. 3.5, white
  // separators at its sides, an unobserved patch (a parked car) in a grey
  // slot at x 5 .. 7.5.
  const RasterFrame frame{0.0, 0.0, 0.1, 100, 60};
  std::vector<uint8_t> rgb(100 * 60 * 3);
  std::vector<uint8_t> observed(100 * 60, 1);
  for (int r = 0; r < 60; ++r) {
    for (int c = 0; c < 100; ++c) {
      const double x = (c + 0.5) * 0.1;
      const size_t k = static_cast<size_t>(r) * 100 + c;
      uint8_t colour[3] = {108, 108, 112};
      if (x > 1.0 && x < 3.5) {
        colour[0] = 70;
        colour[1] = 150;
        colour[2] = 90;
      }
      if (std::abs(x - 1.0) < 0.08 || std::abs(x - 3.5) < 0.08) {
        colour[0] = colour[1] = colour[2] = 232;
      }
      for (int i = 0; i < 3; ++i) rgb[3 * k + i] = colour[i];
      if (x > 5.5 && x < 7.0 && r > 10 && r < 45) observed[k] = 0;
    }
  }
  auto slot = [](double x0, double x1) {
    return std::array<Vec2, 4>{Vec2(x0, 0.5), Vec2(x1, 0.5), Vec2(x1, 5.5),
                               Vec2(x0, 5.5)};
  };
  const std::vector<SlotPaint> paint = MeasureSlotPaint(
      {slot(1.0, 3.5), slot(5.0, 7.5)}, rgb, observed, frame);
  GR_ASSERT_TRUE(paint.size() == 2);
  GR_EXPECT_GT(paint[0].in_band, 0.95);
  GR_EXPECT_NEAR(paint[0].mean_rgb.y(), 150.0, 1e-9);
  GR_EXPECT_EQ(paint[1].in_band, 0.0);
  GR_EXPECT_LT(paint[1].observed, paint[1].cells / 2);
}

}  // namespace
}  // namespace gr
