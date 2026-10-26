#include "gr/splat/gaussians.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

// Points of the plane x = `x` facing the origin, `half` metres around the x
// axis, 5 cm apart.
Points Wall(double x, double half) {
  Points points;
  const int n = static_cast<int>(std::lround(half / 0.05));
  for (int i = -n; i <= n; ++i) {
    for (int j = -n; j <= n; ++j) points.emplace_back(x, 0.05 * i, 0.05 * j);
  }
  return points;
}

GR_TEST(InitGaussiansTest, FlatDiscsAlongTheNormal) {
  Points points = Wall(2.0, 0.5);
  Points normals(points.size(), Vec3(-1, 0, 0));
  const std::vector<double> sigma(points.size(), 0.002);
  points.emplace_back(5.0, 5.0, 5.0);  // Alone and without a normal.
  normals.push_back(Vec3::Zero());
  std::vector<double> sigma_all = sigma;
  sigma_all.push_back(0.002);
  const GaussianSet set =
      InitGaussians(points, normals, sigma_all, GaussianInitConfig());
  GR_ASSERT_TRUE(set.size() == points.size());
  // An interior point: 4 neighbours at 5 cm and 2 at 7.07 cm.
  const size_t center = (points.size() - 1) / 2;
  const double spacing = (4 * 0.05 + 2 * 0.05 * std::sqrt(2.0)) / 6.0;
  GR_EXPECT_NEAR(set.scales[center].x(), 0.8 * spacing, 1e-6);
  GR_EXPECT_NEAR(set.scales[center].y(), 0.8 * spacing, 1e-6);
  GR_EXPECT_NEAR(set.scales[center].z(), 0.002, 1e-9);
  const Vec3 axis = set.rotations[center].toRotationMatrix().col(2);
  GR_EXPECT_NEAR(axis.x(), -1.0, 1e-9);
  const size_t alone = points.size() - 1;
  GR_EXPECT_NEAR(set.scales[alone].x(), 0.8 * 0.05, 1e-9);
  GR_EXPECT_NEAR(set.scales[alone].z(), 0.8 * 0.05, 1e-9);
  GR_EXPECT_NEAR(set.opacity[center], 0.9, 1e-6);
}

GaussianSet WallGaussians(double x, double half, int trip) {
  const Points points = Wall(x, half);
  const Points normals(points.size(), Vec3(-1, 0, 0));
  GaussianSet set = InitGaussians(points, normals,
                                  std::vector<double>(points.size(), 0.002),
                                  GaussianInitConfig());
  for (size_t i = 0; i < set.size(); ++i) {
    set.intensity[i] = points[i].y() < 0.0 ? 0.3f : 0.8f;
    set.trip[i] = trip;
  }
  return set;
}

GR_TEST(GaussianRayRendererTest, DepthAndIntensityOfAWall) {
  const GaussianRayRenderer renderer(WallGaussians(5.0, 1.5, -1));
  Points origins, directions;
  std::vector<double> expected;
  for (double angle : {-12.0, -6.0, 3.0, 9.0}) {
    const double a = DegToRad(angle);
    origins.push_back(Vec3::Zero());
    directions.emplace_back(std::cos(a), std::sin(a), 0.07);
    expected.push_back(5.0 / std::cos(a) *
                       directions.back().norm() / std::hypot(
                           std::cos(a), std::sin(a)));
  }
  const RayRender render = renderer.Render(origins, directions, -1);
  for (size_t r = 0; r < origins.size(); ++r) {
    GR_EXPECT_NEAR(render.depth[r], expected[r], 0.005);
    GR_EXPECT_NEAR(render.mean_depth[r], expected[r], 0.005);
    GR_EXPECT_GT(render.opacity[r], 0.99f);
    GR_EXPECT_TRUE(render.dominant[r] >= 0);
  }
  GR_EXPECT_NEAR(render.intensity[0], 0.3, 0.01);
  GR_EXPECT_NEAR(render.intensity[3], 0.8, 0.01);
}

GR_TEST(GaussianRayRendererTest, TripSpecificGaussiansOnlyInTheirTrip) {
  GaussianSet set = WallGaussians(5.0, 1.0, -1);
  const GaussianSet near = WallGaussians(3.0, 1.0, 1);  // A changed object.
  for (size_t i = 0; i < near.size(); ++i) {
    set.means.push_back(near.means[i]);
    set.rotations.push_back(near.rotations[i]);
    set.scales.push_back(near.scales[i]);
    set.opacity.push_back(near.opacity[i]);
    set.intensity.push_back(near.intensity[i]);
    set.trip.push_back(near.trip[i]);
  }
  const GaussianRayRenderer renderer(std::move(set));
  const Points origins = {Vec3::Zero()};
  const Points directions = {Vec3(1, 0.01, 0.02).normalized()};
  const double scale = 1.0 / directions[0].x();
  GR_EXPECT_NEAR(renderer.Render(origins, directions, 1).depth[0], 3 * scale,
                 0.005);
  GR_EXPECT_NEAR(renderer.Render(origins, directions, 0).depth[0], 5 * scale,
                 0.005);
  GR_EXPECT_NEAR(renderer.Render(origins, directions, -1).depth[0], 5 * scale,
                 0.005);
}

GR_TEST(GaussianRayRendererTest, MissesGiveNaN) {
  const GaussianRayRenderer renderer(WallGaussians(5.0, 1.0, -1));
  const RayRender render =
      renderer.Render({Vec3::Zero()}, {Vec3(-1, 0, 0)}, -1);
  GR_EXPECT_TRUE(std::isnan(render.depth[0]));
  GR_EXPECT_EQ(render.dominant[0], -1);
  GR_EXPECT_NEAR(render.opacity[0], 0.0, 1e-9);
  const GaussianRayRenderer empty(GaussianSet{});
  GR_EXPECT_TRUE(
      std::isnan(empty.Render({Vec3::Zero()}, {Vec3::UnitX()}, -1).depth[0]));
}

}  // namespace
}  // namespace gr
