#include "gr/nav_map.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

std::vector<uint8_t> Rectangles(int width, int height,
                                const std::vector<Eigen::Vector4i>& boxes) {
  std::vector<uint8_t> mask(static_cast<size_t>(width) * height, 0);
  for (const Eigen::Vector4i& b : boxes) {  // u0, v0, u1, v1 (inclusive).
    for (int v = b[1]; v <= b[3]; ++v) {
      for (int u = b[0]; u <= b[2]; ++u) {
        mask[static_cast<size_t>(v * width + u)] = 1;
      }
    }
  }
  return mask;
}

GR_TEST(NavMapTest, DistanceTransformIsEuclidean) {
  std::vector<uint8_t> sources(10 * 8, 0);
  sources[3 * 10 + 2] = 1;  // (u 2, v 3).
  const std::vector<float> d = DistanceTransform(sources, 10, 8);
  GR_EXPECT_NEAR(d[7 * 10 + 5], 5.0, 1e-5);
  GR_EXPECT_NEAR(d[3 * 10 + 2], 0.0, 1e-9);
  GR_EXPECT_NEAR(d[0], std::sqrt(4.0 + 9.0), 1e-5);
  const std::vector<float> none =
      DistanceTransform(std::vector<uint8_t>(10 * 8, 0), 10, 8);
  GR_EXPECT_TRUE(std::isinf(none[5]));
}

GR_TEST(NavMapTest, SkeletonGraphOfATJunction) {
  // A 7-wide horizontal corridor (u 5..74, v 10..16) and a branch down from
  // its middle (u 37..43, v 17..49).
  const int width = 80;
  const int height = 55;
  const std::vector<uint8_t> mask = Rectangles(
      width, height,
      {Eigen::Vector4i(5, 10, 74, 16), Eigen::Vector4i(37, 17, 43, 49)});
  const std::vector<uint8_t> skeleton = ThinMask(mask, width, height);
  for (size_t i = 0; i < mask.size(); ++i) {
    if (skeleton[i]) GR_EXPECT_EQ(mask[i], 1);  // Inside the corridors.
  }
  const SkeletonGraph graph =
      ExtractSkeletonGraph(skeleton, width, height, 6.0);
  GR_ASSERT_TRUE(graph.edges.size() == 3u);
  GR_ASSERT_TRUE(graph.nodes.size() == 4u);
  std::vector<int> degree(graph.nodes.size(), 0);
  double total = 0.0;
  for (const SkeletonEdge& e : graph.edges) {
    ++degree[static_cast<size_t>(e.a)];
    ++degree[static_cast<size_t>(e.b)];
    total += e.length;
    const Vec2& a = graph.nodes[static_cast<size_t>(e.a)];
    const Vec2& b = graph.nodes[static_cast<size_t>(e.b)];
    GR_EXPECT_NEAR((e.cells.front() - a).norm(), 0.0, 2.0);
    GR_EXPECT_NEAR((e.cells.back() - b).norm(), 0.0, 2.0);
  }
  int junction = -1;
  for (size_t n = 0; n < degree.size(); ++n) {
    if (degree[n] == 3) junction = static_cast<int>(n);
  }
  GR_ASSERT_TRUE(junction >= 0);
  GR_EXPECT_NEAR(graph.nodes[static_cast<size_t>(junction)].x(), 40.0, 2.0);
  GR_EXPECT_NEAR(graph.nodes[static_cast<size_t>(junction)].y(), 13.0, 2.0);
  // Centre lines: about 64 + 33 cells (thinning shortens the ends).
  GR_EXPECT_GT(total, 85.0);
  GR_EXPECT_LT(total, 105.0);
}

GR_TEST(NavMapTest, ARingBecomesOneLoop) {
  const int width = 60;
  const int height = 40;
  std::vector<uint8_t> mask =
      Rectangles(width, height, {Eigen::Vector4i(5, 5, 54, 34)});
  for (int v = 12; v <= 27; ++v) {
    for (int u = 12; u <= 47; ++u) mask[static_cast<size_t>(v * width + u)] = 0;
  }
  const SkeletonGraph graph =
      ExtractSkeletonGraph(ThinMask(mask, width, height), width, height, 6.0);
  GR_ASSERT_TRUE(graph.edges.size() == 1u);
  GR_EXPECT_EQ(graph.edges[0].a, graph.edges[0].b);
  // Centre line of the ring (cells 8..50 x 8..30): about 2 * (42 + 22).
  GR_EXPECT_NEAR(graph.edges[0].length, 128.0, 10.0);
}

GR_TEST(NavMapTest, SimplifyKeepsCorners) {
  std::vector<Vec2> line;
  for (int i = 0; i <= 20; ++i) line.emplace_back(i, (i % 2) * 0.1);
  GR_EXPECT_EQ(SimplifyPolyline(line, 0.2).size(), 2u);
  std::vector<Vec2> corner;
  for (int i = 0; i <= 10; ++i) corner.emplace_back(i, 0);
  for (int i = 1; i <= 10; ++i) corner.emplace_back(10, i);
  const std::vector<Vec2> simple = SimplifyPolyline(corner, 0.1);
  GR_ASSERT_TRUE(simple.size() == 3u);
  GR_EXPECT_NEAR((simple[1] - Vec2(10, 0)).norm(), 0.0, 1e-9);
}

GR_TEST(NavMapTest, RouteGraphRespectsOneWayEdges) {
  RouteGraph graph(4);
  graph.AddEdge(0, 1, 1.0, false);
  const int shortcut = graph.AddEdge(1, 3, 1.0, true);  // 1 -> 3 only.
  graph.AddEdge(1, 2, 2.0, false);
  graph.AddEdge(2, 3, 2.0, false);
  const GraphPath forward = graph.ShortestPath(0, 3);
  GR_ASSERT_TRUE(forward.found);
  GR_EXPECT_NEAR(forward.cost, 2.0, 1e-12);
  GR_EXPECT_TRUE((forward.nodes == std::vector<int>{0, 1, 3}));
  GR_EXPECT_EQ(forward.edges.back(), shortcut);
  const GraphPath back = graph.ShortestPath(3, 0);
  GR_ASSERT_TRUE(back.found);
  GR_EXPECT_NEAR(back.cost, 5.0, 1e-12);  // 3 -> 2 -> 1 -> 0.
  RouteGraph split(3);
  split.AddEdge(0, 1, 1.0, false);
  GR_EXPECT_FALSE(split.ShortestPath(0, 2).found);
  GR_EXPECT_TRUE(std::isinf(split.Distances(0)[2]));
}

GR_TEST(NavMapTest, GridAStarFindsTheGapInAWall) {
  const int width = 30;
  const int height = 20;
  std::vector<float> cost(static_cast<size_t>(width) * height, 0.0f);
  for (int v = 0; v < height; ++v) {
    if (v != 16) cost[static_cast<size_t>(v * width + 15)] = -1.0f;  // Wall.
  }
  const auto path = GridAStar(cost, width, height, {2, 2}, {27, 2});
  GR_ASSERT_TRUE(!path.empty());
  GR_EXPECT_TRUE((path.front() == Eigen::Vector2i(2, 2)));
  GR_EXPECT_TRUE((path.back() == Eigen::Vector2i(27, 2)));
  bool through_gap = false;
  for (const Eigen::Vector2i& c : path) {
    GR_EXPECT_GE(cost[static_cast<size_t>(c.y() * width + c.x())], 0.0f);
    if (c.x() == 15) through_gap = c.y() == 16;
  }
  GR_EXPECT_TRUE(through_gap);
  cost[static_cast<size_t>(16 * width + 15)] = -1.0f;
  GR_EXPECT_TRUE(GridAStar(cost, width, height, {2, 2}, {27, 2}).empty());
}

}  // namespace
}  // namespace gr
