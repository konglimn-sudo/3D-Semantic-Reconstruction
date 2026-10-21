#include "gr/geometry.h"

#include <cmath>
#include <random>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(GeometryTest, NormalsAndThicknessOnAPlane) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> uniform(0.0, 2.0);
  std::normal_distribution<double> noise(0.0, 0.002);
  Points points;
  for (int i = 0; i < 4000; ++i) {
    points.emplace_back(uniform(rng), uniform(rng), 1.0 + noise(rng));
  }
  const NormalsResult normals = EstimateNormals(points, 0.15, 5);
  int interior = 0;
  int planar = 0;
  for (size_t i = 0; i < points.size(); ++i) {
    GR_EXPECT_LT(std::abs(std::abs(normals.normals[i].z()) - 1.0), 0.02);
    // Planarity is only isotropic away from the patch border.
    if (points[i].x() < 0.2 || points[i].x() > 1.8 || points[i].y() < 0.2 ||
        points[i].y() > 1.8) {
      continue;
    }
    ++interior;
    if (normals.planarity[i] > 0.5) ++planar;
  }
  GR_EXPECT_GT(planar, interior * 95 / 100);
  const ThicknessStats stats = MeasurePlaneThickness(points, 0.5, 20, 0.7);
  GR_EXPECT_GE(stats.num_voxels, 10);
  GR_EXPECT_NEAR(stats.median, 0.002, 0.001);
}

GR_TEST(GeometryTest, RansacPlaneIgnoresOutliers) {
  std::mt19937 rng(2);
  std::uniform_real_distribution<double> uniform(-3.0, 3.0);
  Points points;
  for (int i = 0; i < 2000; ++i) {
    const double x = uniform(rng);
    const double y = uniform(rng);
    points.emplace_back(x, y, 0.1 * x - 0.2 * y + 0.5);
  }
  for (int i = 0; i < 500; ++i) {
    points.emplace_back(uniform(rng), uniform(rng), uniform(rng));
  }
  const PlaneFit fit = FitPlaneRansac(points, 0.01, 300, 7);
  const Vec3 expected = Vec3(0.1, -0.2, -1.0).normalized();
  GR_EXPECT_NEAR(std::abs(fit.normal.dot(expected)), 1.0, 1e-4);
  GR_EXPECT_GE(fit.inliers.size(), 2000u);
}

GR_TEST(GeometryTest, ProjectionOntoLocalPlanesRemovesTheNoiseOfAWall) {
  // A 4 x 3 m wall (x = 0) sampled every 5 cm with 1.5 cm noise along its
  // normal, and a corner edge where a second wall (y = 0) meets it.
  std::mt19937 rng(11);
  std::normal_distribution<double> noise(0.0, 0.015);
  Points points;
  for (double y = 0.05; y < 4.0; y += 0.05) {
    for (double z = 0.0; z < 3.0; z += 0.05) {
      points.emplace_back(noise(rng), y, z);
    }
  }
  std::vector<double> shift;
  const Points projected =
      ProjectToLocalPlanes(points, 0.15, 8, 64, 0.3, 0.1, &shift);
  double before = 0.0, after = 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    before += points[i].x() * points[i].x();
    after += projected[i].x() * projected[i].x();
    // Only along the normal: the wall coordinates stay.
    GR_EXPECT_NEAR(projected[i].y(), points[i].y(), 0.02);
    GR_EXPECT_LE(std::abs(shift[i]), 0.1);
  }
  before = std::sqrt(before / points.size());
  after = std::sqrt(after / points.size());
  GR_EXPECT_NEAR(before, 0.015, 0.002);
  GR_EXPECT_LT(after, 0.006);
}

}  // namespace
}  // namespace gr
