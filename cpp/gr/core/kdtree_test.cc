#include "gr/core/kdtree.h"

#include <random>
#include <vector>

#include "gr/core/geometry.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(KdTreeTest, NearestMatchesBruteForce) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> u(-50.0, 50.0);
  std::vector<Vec2> points(3000);
  for (Vec2& p : points) p = Vec2(u(rng), u(rng));
  const KdTree2 tree(points);
  for (int q = 0; q < 500; ++q) {
    const Vec2 query(u(rng), u(rng));
    int best = -1;
    double best_d = 1e18;
    for (size_t i = 0; i < points.size(); ++i) {
      const double d = (points[i] - query).norm();
      if (d < best_d) {
        best_d = d;
        best = static_cast<int>(i);
      }
    }
    const KdTree2::Hit hit = tree.Nearest(query);
    GR_EXPECT_EQ(hit.index, best);
    GR_EXPECT_NEAR(hit.distance, best_d, 1e-9);
    // With a bound below the nearest distance there is no hit.
    GR_EXPECT_EQ(tree.Nearest(query, 0.99 * best_d).index, -1);
  }
}

GR_TEST(GeometryTest, ConvexHullAndPointInPolygon) {
  std::vector<Vec2> points = {{0, 0}, {4, 0}, {4, 3}, {0, 3}, {2, 1},
                              {1, 2}, {2, 0}, {4, 3}};
  const std::vector<Vec2> hull = ConvexHull2D(points);
  GR_ASSERT_TRUE(hull.size() == 4);
  GR_EXPECT_TRUE(hull[0] == Vec2(0, 0));
  GR_EXPECT_TRUE(hull[1] == Vec2(4, 0));  // Counter-clockwise.
  GR_EXPECT_TRUE(hull[2] == Vec2(4, 3));
  GR_EXPECT_TRUE(PointInPolygon(hull, Vec2(1.0, 1.0)));
  GR_EXPECT_FALSE(PointInPolygon(hull, Vec2(5.0, 1.0)));
  GR_EXPECT_FALSE(PointInPolygon(hull, Vec2(2.0, -0.5)));
}

}  // namespace
}  // namespace gr
