#include "gr/nav/arrows.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kRes = 0.05;
constexpr int kSize = 400;  // 20 m x 20 m.

struct Canvas {
  std::vector<uint8_t> mask = std::vector<uint8_t>(kSize * kSize, 0);

  // Paints the cells whose centre is inside the rectangle centred at
  // `centre`, `length` along `axis` and `width` across it.
  void Rect(const Vec2& centre, const Vec2& axis, double length,
            double width) {
    const Vec2 across(-axis.y(), axis.x());
    for (int row = 0; row < kSize; ++row) {
      for (int col = 0; col < kSize; ++col) {
        const Vec2 p((col + 0.5) * kRes, (row + 0.5) * kRes);
        const Vec2 d = p - centre;
        if (std::abs(d.dot(axis)) <= length / 2 &&
            std::abs(d.dot(across)) <= width / 2) {
          mask[static_cast<size_t>(row * kSize + col)] = 1;
        }
      }
    }
  }

  // Shaft 2.3 m x 0.2 m, then a 1.2 m head narrowing from 0.7 m, as the
  // synthetic garage paints them.
  void Arrow(const Vec2& tail, const Vec2& direction) {
    Rect(tail + direction * 1.15, direction, 2.3, 0.2);
    for (int k = 0; k < 6; ++k) {
      Rect(tail + direction * (2.3 + 0.2 * k + 0.1), direction, 0.2,
           0.7 * (1.0 - k / 6.0));
    }
  }
};

GR_TEST(ArrowsTest, FindsArrowsAndTheirDirection) {
  Canvas canvas;
  const Vec2 east(1.0, 0.0);
  const Vec2 slanted = Vec2(std::cos(0.5), std::sin(0.5));
  canvas.Arrow(Vec2(2.0, 3.0), east);
  canvas.Arrow(Vec2(16.0, 8.0), -slanted);
  // Not arrows: a parking line, a T junction of lines, a blob.
  canvas.Rect(Vec2(10.0, 15.0), Vec2(0.0, 1.0), 5.3, 0.15);
  canvas.Rect(Vec2(4.0, 12.0), east, 3.0, 0.15);
  canvas.Rect(Vec2(5.5, 12.0), Vec2(0.0, 1.0), 2.5, 0.15);
  canvas.Rect(Vec2(15.0, 16.0), east, 1.2, 1.0);
  const std::vector<FloorArrow> arrows =
      DetectArrows(canvas.mask, kSize, kSize, kRes, 0.0, 0.0);
  GR_ASSERT_TRUE(arrows.size() == 2);
  for (const FloorArrow& arrow : arrows) {
    const bool first = arrow.center.x() < 9.0;
    const Vec2 expected = first ? east : Vec2(-slanted);
    GR_EXPECT_GT(arrow.direction.dot(expected), 0.995);
    GR_EXPECT_NEAR(arrow.length, 3.5, 0.15);
    GR_EXPECT_NEAR(arrow.shaft_width, 0.2, 0.05);
    GR_EXPECT_GT(arrow.head_width, 0.5);
    const Vec2 tip = first ? Vec2(5.5, 3.0) : Vec2(16.0, 8.0) - 3.5 * slanted;
    GR_EXPECT_LT((arrow.tip - tip).norm(), 0.15);
  }
}

}  // namespace
}  // namespace gr
