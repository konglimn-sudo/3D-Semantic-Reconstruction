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

}  // namespace
}  // namespace gr
