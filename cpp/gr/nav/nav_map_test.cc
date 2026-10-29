#include "gr/nav/nav_map.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kExact = 1e-12;
// Spurs shorter than this many cells are pruned.
constexpr double kMinSpur = 6.0;

// Rectangles (x0, y0, x1, y1), inclusive, set in a width x height mask.
std::vector<uint8_t> Rectangles(int width, int height,
                                const std::vector<Eigen::Vector4i>& boxes) {
  std::vector<uint8_t> mask(static_cast<size_t>(width) * height, 0);
  for (const Eigen::Vector4i& box : boxes) {
    for (int y = box[1]; y <= box[3]; ++y) {
      for (int x = box[0]; x <= box[2]; ++x) {
        mask[static_cast<size_t>(y * width + x)] = 1;
      }
    }
  }
  return mask;
}

GR_TEST(NavMapTest, DistanceTransformIsEuclidean) {
  constexpr int kWidth = 10;
  constexpr int kHeight = 8;
  constexpr int kSourceX = 2;
  constexpr int kSourceY = 3;
  constexpr double kTolerance = 1e-5;  // float32 distances.
  std::vector<uint8_t> sources(kWidth * kHeight, 0);
  sources[kSourceY * kWidth + kSourceX] = 1;
  const std::vector<float> distance =
      DistanceTransform(sources, kWidth, kHeight);
  // (5, 7) is (3, 4) away; the corner (0, 0) is (2, 3) away.
  GR_EXPECT_NEAR(distance[7 * kWidth + 5], 5.0, kTolerance);
  GR_EXPECT_NEAR(distance[kSourceY * kWidth + kSourceX], 0.0, kExact);
  GR_EXPECT_NEAR(distance[0], std::sqrt(4.0 + 9.0), kTolerance);
  const std::vector<float> none =
      DistanceTransform(std::vector<uint8_t>(kWidth * kHeight, 0), kWidth,
                        kHeight);
  GR_EXPECT_TRUE(std::isinf(none[5]));
}

GR_TEST(NavMapTest, SkeletonGraphOfATJunction) {
  // A 7-wide horizontal corridor (x 5..74, y 10..16) and a branch down from
  // its middle (x 37..43, y 17..49).
  constexpr int kWidth = 80;
  constexpr int kHeight = 55;
  constexpr double kEndTolerance = 2.0;       // Cells.
  constexpr double kJunctionX = 40.0;         // Cells.
  constexpr double kJunctionY = 13.0;
  constexpr double kJunctionTolerance = 2.0;  // Cells.
  constexpr int kJunctionDegree = 3;
  // Centre lines: about 64 + 33 cells (thinning shortens the ends).
  constexpr double kMinTotal = 85.0;
  constexpr double kMaxTotal = 105.0;
  const std::vector<uint8_t> mask = Rectangles(
      kWidth, kHeight,
      {Eigen::Vector4i(5, 10, 74, 16), Eigen::Vector4i(37, 17, 43, 49)});
  const std::vector<uint8_t> skeleton = ThinMask(mask, kWidth, kHeight);
  for (size_t i = 0; i < mask.size(); ++i) {
    if (skeleton[i]) GR_EXPECT_EQ(mask[i], 1);  // Inside the corridors.
  }
  const SkeletonGraph graph =
      ExtractSkeletonGraph(skeleton, kWidth, kHeight, kMinSpur);
  GR_ASSERT_TRUE(graph.edges.size() == 3u);
  GR_ASSERT_TRUE(graph.nodes.size() == 4u);
  std::vector<int> degree(graph.nodes.size(), 0);
  double total = 0.0;
  for (const SkeletonEdge& edge : graph.edges) {
    ++degree[static_cast<size_t>(edge.start_node)];
    ++degree[static_cast<size_t>(edge.end_node)];
    total += edge.length;
    const Vec2& start = graph.nodes[static_cast<size_t>(edge.start_node)];
    const Vec2& end = graph.nodes[static_cast<size_t>(edge.end_node)];
    GR_EXPECT_NEAR((edge.cells.front() - start).norm(), 0.0, kEndTolerance);
    GR_EXPECT_NEAR((edge.cells.back() - end).norm(), 0.0, kEndTolerance);
  }
  int junction = SkeletonEdge::kNoNode;
  for (size_t node = 0; node < degree.size(); ++node) {
    if (degree[node] == kJunctionDegree) junction = static_cast<int>(node);
  }
  GR_ASSERT_TRUE(junction != SkeletonEdge::kNoNode);
  GR_EXPECT_NEAR(graph.nodes[static_cast<size_t>(junction)].x(), kJunctionX,
                 kJunctionTolerance);
  GR_EXPECT_NEAR(graph.nodes[static_cast<size_t>(junction)].y(), kJunctionY,
                 kJunctionTolerance);
  GR_EXPECT_GT(total, kMinTotal);
  GR_EXPECT_LT(total, kMaxTotal);
}

GR_TEST(NavMapTest, ARingBecomesOneLoop) {
  constexpr int kWidth = 60;
  constexpr int kHeight = 40;
  // Centre line of the ring (cells 8..50 x 8..30): about 2 * (42 + 22).
  constexpr double kLoopLength = 128.0;
  constexpr double kLoopTolerance = 10.0;
  std::vector<uint8_t> mask =
      Rectangles(kWidth, kHeight, {Eigen::Vector4i(5, 5, 54, 34)});
  for (int y = 12; y <= 27; ++y) {  // The hole.
    for (int x = 12; x <= 47; ++x) {
      mask[static_cast<size_t>(y * kWidth + x)] = 0;
    }
  }
  const SkeletonGraph graph = ExtractSkeletonGraph(
      ThinMask(mask, kWidth, kHeight), kWidth, kHeight, kMinSpur);
  GR_ASSERT_TRUE(graph.edges.size() == 1u);
  GR_EXPECT_EQ(graph.edges[0].start_node, graph.edges[0].end_node);
  GR_EXPECT_NEAR(graph.edges[0].length, kLoopLength, kLoopTolerance);
}

GR_TEST(NavMapTest, SimplifyKeepsCorners) {
  constexpr int kSamples = 20;
  constexpr double kZigzag = 0.1;          // m, below the tolerance.
  constexpr double kZigzagTolerance = 0.2;  // m
  constexpr int kLeg = 10;                 // m along each side of the corner.
  constexpr double kCornerTolerance = 0.1;  // m
  constexpr double kTolerance = 1e-9;
  std::vector<Vec2> line;
  for (int i = 0; i <= kSamples; ++i) line.emplace_back(i, (i % 2) * kZigzag);
  GR_EXPECT_EQ(SimplifyPolyline(line, kZigzagTolerance).size(), 2u);
  std::vector<Vec2> corner;
  for (int i = 0; i <= kLeg; ++i) corner.emplace_back(i, 0);
  for (int i = 1; i <= kLeg; ++i) corner.emplace_back(kLeg, i);
  const std::vector<Vec2> simple = SimplifyPolyline(corner, kCornerTolerance);
  GR_ASSERT_TRUE(simple.size() == 3u);
  GR_EXPECT_NEAR((simple[1] - Vec2(kLeg, 0)).norm(), 0.0, kTolerance);
}

GR_TEST(NavMapTest, RouteGraphRespectsOneWayEdges) {
  constexpr int kNodes = 4;
  constexpr double kShortCost = 1.0;
  constexpr double kLongCost = 2.0;
  RouteGraph graph(kNodes);
  graph.AddEdge(0, 1, kShortCost, false);
  const int shortcut = graph.AddEdge(1, 3, kShortCost, true);  // 1 -> 3 only.
  graph.AddEdge(1, 2, kLongCost, false);
  graph.AddEdge(2, 3, kLongCost, false);
  const GraphPath forward = graph.ShortestPath(0, 3);
  GR_ASSERT_TRUE(forward.found);
  GR_EXPECT_NEAR(forward.cost, 2 * kShortCost, kExact);
  GR_EXPECT_TRUE((forward.nodes == std::vector<int>{0, 1, 3}));
  GR_EXPECT_EQ(forward.edges.back(), shortcut);
  const GraphPath back = graph.ShortestPath(3, 0);
  GR_ASSERT_TRUE(back.found);
  // 3 -> 2 -> 1 -> 0.
  GR_EXPECT_NEAR(back.cost, 2 * kLongCost + kShortCost, kExact);
  RouteGraph split(3);
  split.AddEdge(0, 1, kShortCost, false);
  GR_EXPECT_FALSE(split.ShortestPath(0, 2).found);
  GR_EXPECT_TRUE(std::isinf(split.Distances(0)[2]));
}

GR_TEST(NavMapTest, GridAStarFindsTheGapInAWall) {
  constexpr int kWidth = 30;
  constexpr int kHeight = 20;
  constexpr int kWallX = 15;
  constexpr int kGapY = 16;
  constexpr float kBlocked = -1.0f;
  const Eigen::Vector2i start(2, 2);
  const Eigen::Vector2i goal(27, 2);
  std::vector<float> cost(static_cast<size_t>(kWidth) * kHeight, 0.0f);
  for (int y = 0; y < kHeight; ++y) {
    if (y != kGapY) cost[static_cast<size_t>(y * kWidth + kWallX)] = kBlocked;
  }
  const auto path = GridAStar(cost, kWidth, kHeight, start, goal);
  GR_ASSERT_TRUE(!path.empty());
  GR_EXPECT_TRUE((path.front() == start));
  GR_EXPECT_TRUE((path.back() == goal));
  bool through_gap = false;
  for (const Eigen::Vector2i& cell : path) {
    GR_EXPECT_GE(cost[static_cast<size_t>(cell.y() * kWidth + cell.x())],
                 0.0f);
    if (cell.x() == kWallX) through_gap = cell.y() == kGapY;
  }
  GR_EXPECT_TRUE(through_gap);
  cost[static_cast<size_t>(kGapY * kWidth + kWallX)] = kBlocked;
  GR_EXPECT_TRUE(GridAStar(cost, kWidth, kHeight, start, goal).empty());
}

}  // namespace
}  // namespace gr
