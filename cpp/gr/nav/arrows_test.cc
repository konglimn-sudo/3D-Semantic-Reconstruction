#include "gr/nav/arrows.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kRes = 0.05;  // m per cell.
constexpr int kSize = 400;     // Cells: 20 m x 20 m.
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;

// The arrows the synthetic garage paints: a kShaftLength x kShaftWidth
// shaft, then a head of kHeadSteps steps of kHeadStep narrowing from
// kHeadWidth.
constexpr double kShaftLength = 2.3;  // m
constexpr double kShaftWidth = 0.2;   // m
constexpr double kHeadStep = 0.2;     // m
constexpr int kHeadSteps = 6;
constexpr double kHeadWidth = 0.7;    // m
constexpr double kArrowLength = 3.5;  // m

struct Canvas {
  std::vector<uint8_t> mask = std::vector<uint8_t>(kSize * kSize, 0);

  // Paints the cells whose centre is inside the rectangle centred at
  // `centre`, `length` along `axis` and `width` across it.
  void Rect(const Vec2& centre, const Vec2& axis, double length,
            double width) {
    const Vec2 across(-axis.y(), axis.x());
    for (int row = 0; row < kSize; ++row) {
      for (int column = 0; column < kSize; ++column) {
        const Vec2 cell((column + kCellCentre) * kRes,
                        (row + kCellCentre) * kRes);
        const Vec2 offset = cell - centre;
        if (std::abs(offset.dot(axis)) <= length / 2 &&
            std::abs(offset.dot(across)) <= width / 2) {
          mask[static_cast<size_t>(row * kSize + column)] = 1;
        }
      }
    }
  }

  void Arrow(const Vec2& tail, const Vec2& direction) {
    Rect(tail + direction * (kShaftLength / 2), direction, kShaftLength,
         kShaftWidth);
    for (int k = 0; k < kHeadSteps; ++k) {
      Rect(tail + direction * (kShaftLength + kHeadStep * k + kHeadStep / 2),
           direction, kHeadStep,
           kHeadWidth * (1.0 - static_cast<double>(k) / kHeadSteps));
    }
  }
};

GR_TEST(ArrowsTest, FindsArrowsAndTheirDirection) {
  constexpr double kSlantAngle = 0.5;           // rad
  constexpr double kLineWidth = 0.15;           // m, parking lines.
  constexpr double kMinDirectionCos = 0.995;
  constexpr double kLengthTolerance = 0.15;     // m
  constexpr double kShaftTolerance = 0.05;      // m
  constexpr double kMinHeadWidth = 0.5;         // m
  constexpr double kTipTolerance = 0.15;        // m
  // The first arrow lies left of this x (m), the second right of it.
  constexpr double kSplitX = 9.0;
  Canvas canvas;
  const Vec2 east(1.0, 0.0);
  const Vec2 north(0.0, 1.0);
  const Vec2 slanted = Vec2(std::cos(kSlantAngle), std::sin(kSlantAngle));
  const Vec2 first_tail(2.0, 3.0);
  const Vec2 second_tail(16.0, 8.0);
  canvas.Arrow(first_tail, east);
  canvas.Arrow(second_tail, -slanted);
  // Not arrows: a parking line, a T junction of lines, a blob.
  canvas.Rect(Vec2(10.0, 15.0), north, 5.3, kLineWidth);
  canvas.Rect(Vec2(4.0, 12.0), east, 3.0, kLineWidth);
  canvas.Rect(Vec2(5.5, 12.0), north, 2.5, kLineWidth);
  canvas.Rect(Vec2(15.0, 16.0), east, 1.2, 1.0);
  const std::vector<FloorArrow> arrows =
      DetectArrows(canvas.mask, kSize, kSize, kRes, 0.0, 0.0);
  GR_ASSERT_TRUE(arrows.size() == 2);
  for (const FloorArrow& arrow : arrows) {
    const bool first = arrow.center.x() < kSplitX;
    const Vec2 expected = first ? east : Vec2(-slanted);
    GR_EXPECT_GT(arrow.direction.dot(expected), kMinDirectionCos);
    GR_EXPECT_NEAR(arrow.length, kArrowLength, kLengthTolerance);
    GR_EXPECT_NEAR(arrow.shaft_width, kShaftWidth, kShaftTolerance);
    GR_EXPECT_GT(arrow.head_width, kMinHeadWidth);
    const Vec2 tip = first ? Vec2(first_tail + kArrowLength * east)
                           : Vec2(second_tail - kArrowLength * slanted);
    GR_EXPECT_LT((arrow.tip - tip).norm(), kTipTolerance);
  }
}

}  // namespace
}  // namespace gr
