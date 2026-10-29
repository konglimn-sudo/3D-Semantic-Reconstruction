#include "gr/core/geometry.h"

#include <cmath>
#include <random>

#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(GeometryTest, NormalsAndThicknessOnAPlane) {
  // A 2 x 2 m patch of the plane z = 1 with 2 mm of noise along its normal.
  constexpr int kSeed = 1;
  constexpr int kPointCount = 4000;
  constexpr double kPatchSize = 2.0;
  constexpr double kPlaneHeight = 1.0;
  constexpr double kNoiseSigma = 0.002;
  constexpr double kNormalRadius = 0.15;
  constexpr int kMinNeighbors = 5;
  constexpr double kNormalTolerance = 0.02;
  // Planarity is only isotropic away from the patch border.
  constexpr double kInteriorMin = 0.2;
  constexpr double kInteriorMax = 1.8;
  constexpr double kPlanarPlanarity = 0.5;
  constexpr int kPercent = 100;
  constexpr int kMinPlanarPercent = 95;
  constexpr double kVoxelSize = 0.5;
  constexpr int kMinVoxelPoints = 20;
  constexpr double kMinVoxelPlanarity = 0.7;
  constexpr int kMinVoxels = 10;
  // The thickness of the patch is its noise.
  constexpr double kThicknessTolerance = 0.001;

  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(0.0, kPatchSize);
  std::normal_distribution<double> noise(0.0, kNoiseSigma);
  Points points;
  for (int i = 0; i < kPointCount; ++i) {
    points.emplace_back(uniform(rng), uniform(rng), kPlaneHeight + noise(rng));
  }
  const NormalsResult normals =
      EstimateNormals(points, kNormalRadius, kMinNeighbors);
  int interior = 0;
  int planar = 0;
  for (size_t i = 0; i < points.size(); ++i) {
    GR_EXPECT_LT(std::abs(std::abs(normals.normals[i].z()) - 1.0),
                 kNormalTolerance);
    if (points[i].x() < kInteriorMin || points[i].x() > kInteriorMax ||
        points[i].y() < kInteriorMin || points[i].y() > kInteriorMax) {
      continue;
    }
    ++interior;
    if (normals.planarity[i] > kPlanarPlanarity) ++planar;
  }
  GR_EXPECT_GT(planar, interior * kMinPlanarPercent / kPercent);
  const ThicknessStats stats = MeasurePlaneThickness(
      points, kVoxelSize, kMinVoxelPoints, kMinVoxelPlanarity);
  GR_EXPECT_GE(stats.num_voxels, kMinVoxels);
  GR_EXPECT_NEAR(stats.median, kNoiseSigma, kThicknessTolerance);
}

GR_TEST(GeometryTest, RansacPlaneIgnoresOutliers) {
  // Points of the plane z = kSlopeX x + kSlopeY y + kHeight over a 6 x 6 m
  // square, and outliers anywhere in the 6 m cube around it.
  constexpr int kSeed = 2;
  constexpr double kHalfExtent = 3.0;
  constexpr int kPlanePoints = 2000;
  constexpr int kOutliers = 500;
  constexpr double kSlopeX = 0.1;
  constexpr double kSlopeY = -0.2;
  constexpr double kHeight = 0.5;
  constexpr double kInlierThreshold = 0.01;
  constexpr int kIterations = 300;
  constexpr int kRansacSeed = 7;
  constexpr double kNormalTolerance = 1e-4;

  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(-kHalfExtent, kHalfExtent);
  Points points;
  for (int i = 0; i < kPlanePoints; ++i) {
    const double x = uniform(rng);
    const double y = uniform(rng);
    points.emplace_back(x, y, kSlopeX * x + kSlopeY * y + kHeight);
  }
  for (int i = 0; i < kOutliers; ++i) {
    points.emplace_back(uniform(rng), uniform(rng), uniform(rng));
  }
  const PlaneFit fit =
      FitPlaneRansac(points, kInlierThreshold, kIterations, kRansacSeed);
  const Vec3 expected = Vec3(kSlopeX, kSlopeY, -1.0).normalized();
  GR_EXPECT_NEAR(std::abs(fit.normal.dot(expected)), 1.0, kNormalTolerance);
  GR_EXPECT_GE(fit.inliers.size(), static_cast<size_t>(kPlanePoints));
}

GR_TEST(GeometryTest, ProjectionOntoLocalPlanesRemovesTheNoiseOfAWall) {
  // A 4 x 3 m wall (x = 0) sampled every 5 cm with 1.5 cm noise along its
  // normal, and a corner edge where a second wall (y = 0) meets it.
  constexpr int kSeed = 11;
  constexpr double kWallLength = 4.0;
  constexpr double kWallHeight = 3.0;
  constexpr double kSpacing = 0.05;
  constexpr double kNoiseSigma = 0.015;
  constexpr double kRadius = 0.15;
  constexpr int kMinNeighbors = 8;
  constexpr int kMaxNeighbors = 64;
  constexpr double kMinPlanarity = 0.3;
  constexpr double kMaxShift = 0.1;
  constexpr double kInPlaneTolerance = 0.02;
  constexpr double kNoiseTolerance = 0.002;
  constexpr double kMaxResidualNoise = 0.006;

  std::mt19937 rng(kSeed);
  std::normal_distribution<double> noise(0.0, kNoiseSigma);
  Points points;
  for (double y = kSpacing; y < kWallLength; y += kSpacing) {
    for (double z = 0.0; z < kWallHeight; z += kSpacing) {
      points.emplace_back(noise(rng), y, z);
    }
  }
  std::vector<double> shift;
  const Points projected =
      ProjectToLocalPlanes(points, kRadius, kMinNeighbors, kMaxNeighbors,
                           kMinPlanarity, kMaxShift, &shift);
  double squared_before = 0.0;
  double squared_after = 0.0;
  for (size_t i = 0; i < points.size(); ++i) {
    squared_before += points[i].x() * points[i].x();
    squared_after += projected[i].x() * projected[i].x();
    // Only along the normal: the wall coordinates stay.
    GR_EXPECT_NEAR(projected[i].y(), points[i].y(), kInPlaneTolerance);
    GR_EXPECT_LE(std::abs(shift[i]), kMaxShift);
  }
  const double rms_before = std::sqrt(squared_before / points.size());
  const double rms_after = std::sqrt(squared_after / points.size());
  GR_EXPECT_NEAR(rms_before, kNoiseSigma, kNoiseTolerance);
  GR_EXPECT_LT(rms_after, kMaxResidualNoise);
}

}  // namespace
}  // namespace gr
