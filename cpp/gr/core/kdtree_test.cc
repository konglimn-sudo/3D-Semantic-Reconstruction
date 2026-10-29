#include "gr/core/kdtree.h"

#include <limits>
#include <random>
#include <vector>

#include "gr/core/geometry.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(KdTreeTest, NearestMatchesBruteForce) {
  // Points and queries uniform over a 100 x 100 square.
  constexpr int kSeed = 5;
  constexpr double kHalfExtent = 50.0;
  constexpr int kPointCount = 3000;
  constexpr int kQueryCount = 500;
  constexpr double kDistanceTolerance = 1e-9;
  // Bounds this fraction of the nearest distance exclude it.
  constexpr double kBoundFraction = 0.99;

  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(-kHalfExtent, kHalfExtent);
  std::vector<Vec2> points(kPointCount);
  for (Vec2& point : points) point = Vec2(uniform(rng), uniform(rng));
  const KdTree2 tree(points);
  for (int query_index = 0; query_index < kQueryCount; ++query_index) {
    const Vec2 query(uniform(rng), uniform(rng));
    int best = KdTree2::kNoPoint;
    double best_distance = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < points.size(); ++i) {
      const double distance = (points[i] - query).norm();
      if (distance < best_distance) {
        best_distance = distance;
        best = static_cast<int>(i);
      }
    }
    const KdTree2::Hit hit = tree.Nearest(query);
    GR_EXPECT_EQ(hit.index, best);
    GR_EXPECT_NEAR(hit.distance, best_distance, kDistanceTolerance);
    GR_EXPECT_EQ(tree.Nearest(query, kBoundFraction * best_distance).index,
                 KdTree2::kNoPoint);
  }
}

GR_TEST(GeometryTest, ConvexHullAndPointInPolygon) {
  // A 4 x 3 rectangle with points inside, on an edge and a repeated corner,
  // none of which are hull vertices.
  constexpr double kWidth = 4.0;
  constexpr double kHeight = 3.0;
  const std::vector<Vec2> corners = {
      {0.0, 0.0}, {kWidth, 0.0}, {kWidth, kHeight}, {0.0, kHeight}};
  std::vector<Vec2> points = corners;
  for (const Vec2& other : std::vector<Vec2>{
           {2.0, 1.0}, {1.0, 2.0}, {2.0, 0.0}, {kWidth, kHeight}}) {
    points.push_back(other);
  }
  const std::vector<Vec2> hull = ConvexHull2D(points);
  GR_ASSERT_TRUE(hull.size() == corners.size());
  GR_EXPECT_TRUE(hull[0] == corners[0]);
  GR_EXPECT_TRUE(hull[1] == corners[1]);  // Counter-clockwise.
  GR_EXPECT_TRUE(hull[2] == corners[2]);
  GR_EXPECT_TRUE(PointInPolygon(hull, Vec2(1.0, 1.0)));
  GR_EXPECT_FALSE(PointInPolygon(hull, Vec2(kWidth + 1.0, 1.0)));
  GR_EXPECT_FALSE(PointInPolygon(hull, Vec2(2.0, -0.5)));
}

}  // namespace
}  // namespace gr
