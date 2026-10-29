#include "gr/splat/gaussians.h"

#include <cmath>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr double kWallStep = 0.05;     // m between wall points.
constexpr double kNormalSigma = 0.002;  // m
constexpr double kDepthTolerance = 0.005;  // m

// Points of the plane x = `x` facing the origin, `half` metres around the x
// axis, kWallStep apart.
Points Wall(double x, double half) {
  Points points;
  const int steps = static_cast<int>(std::lround(half / kWallStep));
  for (int i = -steps; i <= steps; ++i) {
    for (int j = -steps; j <= steps; ++j) {
      points.emplace_back(x, kWallStep * i, kWallStep * j);
    }
  }
  return points;
}

GR_TEST(InitGaussiansTest, FlatDiscsAlongTheNormal) {
  constexpr double kTolerance = 1e-9;
  constexpr double kScaleTolerance = 1e-6;
  const GaussianInitConfig config;
  Points points = Wall(2.0, 0.5);
  Points normals(points.size(), Vec3(-1, 0, 0));
  const std::vector<double> sigma(points.size(), kNormalSigma);
  points.emplace_back(5.0, 5.0, 5.0);  // Alone and without a normal.
  normals.push_back(Vec3::Zero());
  std::vector<double> sigma_all = sigma;
  sigma_all.push_back(kNormalSigma);
  const GaussianSet set = InitGaussians(points, normals, sigma_all, config);
  GR_ASSERT_TRUE(set.size() == points.size());
  // An interior point: its 6 nearest neighbours are 4 at kWallStep and 2 on
  // the diagonal.
  const size_t center = (points.size() - 1) / 2;
  const double spacing =
      (4 * kWallStep + 2 * kWallStep * std::sqrt(2.0)) / config.neighbors;
  GR_EXPECT_NEAR(set.scales[center].x(), config.tangent_factor * spacing,
                 kScaleTolerance);
  GR_EXPECT_NEAR(set.scales[center].y(), config.tangent_factor * spacing,
                 kScaleTolerance);
  GR_EXPECT_NEAR(set.scales[center].z(), kNormalSigma, kTolerance);
  const Vec3 axis = set.rotations[center].toRotationMatrix().col(2);
  GR_EXPECT_NEAR(axis.x(), -1.0, kTolerance);
  // The lonely point: isotropic, from the default spacing.
  const size_t alone = points.size() - 1;
  const double default_sigma = config.tangent_factor * config.default_spacing;
  GR_EXPECT_NEAR(set.scales[alone].x(), default_sigma, kTolerance);
  GR_EXPECT_NEAR(set.scales[alone].z(), default_sigma, kTolerance);
  GR_EXPECT_NEAR(set.opacity[center], config.opacity, kScaleTolerance);
}

// The left (y < 0) and right halves of the walls have these intensities.
constexpr float kLeftIntensity = 0.3f;
constexpr float kRightIntensity = 0.8f;

GaussianSet WallGaussians(double x, double half, int trip) {
  const Points points = Wall(x, half);
  const Points normals(points.size(), Vec3(-1, 0, 0));
  GaussianSet set = InitGaussians(
      points, normals, std::vector<double>(points.size(), kNormalSigma),
      GaussianInitConfig());
  for (size_t i = 0; i < set.size(); ++i) {
    set.intensity[i] = points[i].y() < 0.0 ? kLeftIntensity : kRightIntensity;
    set.trip[i] = trip;
  }
  return set;
}

GR_TEST(GaussianRayRendererTest, DepthAndIntensityOfAWall) {
  constexpr double kWallX = 5.0;   // m
  constexpr double kRise = 0.07;   // z component of the ray directions.
  constexpr float kMinOpacity = 0.99f;
  constexpr double kIntensityTolerance = 0.01;
  const GaussianRayRenderer renderer(
      WallGaussians(kWallX, 1.5, GaussianSet::kSharedTrip));
  Points origins;
  Points directions;
  std::vector<double> expected;
  for (double angle_deg : {-12.0, -6.0, 3.0, 9.0}) {
    const double angle = DegToRad(angle_deg);
    origins.push_back(Vec3::Zero());
    directions.emplace_back(std::cos(angle), std::sin(angle), kRise);
    expected.push_back(kWallX / std::cos(angle) * directions.back().norm() /
                       std::hypot(std::cos(angle), std::sin(angle)));
  }
  const RayRender render =
      renderer.Render(origins, directions, GaussianSet::kSharedTrip);
  for (size_t ray = 0; ray < origins.size(); ++ray) {
    GR_EXPECT_NEAR(render.depth[ray], expected[ray], kDepthTolerance);
    GR_EXPECT_NEAR(render.mean_depth[ray], expected[ray], kDepthTolerance);
    GR_EXPECT_GT(render.opacity[ray], kMinOpacity);
    GR_EXPECT_TRUE(render.dominant[ray] != RayRender::kNoGaussian);
  }
  GR_EXPECT_NEAR(render.intensity[0], kLeftIntensity, kIntensityTolerance);
  GR_EXPECT_NEAR(render.intensity[3], kRightIntensity, kIntensityTolerance);
}

GR_TEST(GaussianRayRendererTest, TripSpecificGaussiansOnlyInTheirTrip) {
  constexpr double kFarX = 5.0;   // m: the shared wall.
  constexpr double kNearX = 3.0;  // m: a changed object of kChangedTrip.
  constexpr int kChangedTrip = 1;
  constexpr int kOtherTrip = 0;
  GaussianSet set = WallGaussians(kFarX, 1.0, GaussianSet::kSharedTrip);
  const GaussianSet near = WallGaussians(kNearX, 1.0, kChangedTrip);
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
  const double scale = 1.0 / directions[0].x();  // Range per metre of x.
  GR_EXPECT_NEAR(renderer.Render(origins, directions, kChangedTrip).depth[0],
                 kNearX * scale, kDepthTolerance);
  GR_EXPECT_NEAR(renderer.Render(origins, directions, kOtherTrip).depth[0],
                 kFarX * scale, kDepthTolerance);
  GR_EXPECT_NEAR(
      renderer.Render(origins, directions, GaussianSet::kSharedTrip).depth[0],
      kFarX * scale, kDepthTolerance);
}

GR_TEST(GaussianRayRendererTest, MissesGiveNaN) {
  constexpr double kTolerance = 1e-9;
  const GaussianRayRenderer renderer(
      WallGaussians(5.0, 1.0, GaussianSet::kSharedTrip));
  const RayRender render = renderer.Render({Vec3::Zero()}, {Vec3(-1, 0, 0)},
                                           GaussianSet::kSharedTrip);
  GR_EXPECT_TRUE(std::isnan(render.depth[0]));
  GR_EXPECT_EQ(render.dominant[0], RayRender::kNoGaussian);
  GR_EXPECT_NEAR(render.opacity[0], 0.0, kTolerance);
  const GaussianRayRenderer empty(GaussianSet{});
  GR_EXPECT_TRUE(std::isnan(
      empty.Render({Vec3::Zero()}, {Vec3::UnitX()}, GaussianSet::kSharedTrip)
          .depth[0]));
}

}  // namespace
}  // namespace gr
