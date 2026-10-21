#include "gr/depth_prior.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr int kHeight = 96;
constexpr int kWidth = 128;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

// A floor seen from a camera: range grows towards the top of the image.
std::vector<float> TrueRange() {
  std::vector<float> range(kHeight * kWidth);
  for (int r = 0; r < kHeight; ++r) {
    for (int c = 0; c < kWidth; ++c) {
      range[r * kWidth + c] = static_cast<float>(
          2.0 + 20.0 * (kHeight - r) / kHeight + 0.01 * c);
    }
  }
  return range;
}

// LiDAR on every other row of the lower two thirds only.
std::vector<float> Lidar(const std::vector<float>& truth) {
  std::vector<float> lidar(truth.size(), kNaN);
  for (int r = kHeight / 3; r < kHeight; r += 2) {
    for (int c = 0; c < kWidth; ++c) lidar[r * kWidth + c] = truth[r * kWidth + c];
  }
  return lidar;
}

GR_TEST(DepthPriorTest, RecoversScaleAndShiftAndFillsOnlyTheHoles) {
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  // Prior = (truth - 0.3) / 1.07 with 1 % noise and 5 % gross outliers.
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 0.01);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<float> prior(truth.size());
  for (size_t i = 0; i < truth.size(); ++i) {
    double p = (truth[i] - 0.3) / 1.07 * (1.0 + noise(rng));
    if (uniform(rng) < 0.05) p *= 0.5;
    prior[i] = static_cast<float>(p);
  }
  const FusedDepth fused = FuseDepthPrior(prior, {}, lidar, {}, kHeight, kWidth);
  GR_ASSERT_TRUE(fused.alignment.valid);
  GR_EXPECT_NEAR(fused.alignment.scale, 1.07, 0.01);
  GR_EXPECT_NEAR(fused.alignment.shift, 0.3, 0.05);
  GR_EXPECT_LT(fused.alignment.residual_median, 0.02);
  int prior_pixels = 0, lidar_kept = 0;
  std::vector<double> errors;
  for (size_t i = 0; i < truth.size(); ++i) {
    if (std::isfinite(lidar[i])) {
      GR_EXPECT_EQ(fused.source[i], kDepthLidar);
      lidar_kept += fused.range[i] == lidar[i];
    } else if (fused.source[i] == kDepthPrior) {
      ++prior_pixels;
      errors.push_back(std::abs(fused.range[i] - truth[i]) / truth[i]);
    }
  }
  GR_EXPECT_EQ(prior_pixels, fused.prior_pixels);
  // The upper third without LiDAR is filled, up to hole_radius rows from it;
  // the gaps between the LiDAR rows are not holes.
  const int hole_rows = kHeight / 3 - DepthAlignConfig().hole_radius;
  GR_EXPECT_EQ(prior_pixels, hole_rows * kWidth);
  int good = 0;
  for (double e : errors) good += e < 0.03;
  GR_EXPECT_GT(good, static_cast<int>(0.9 * errors.size()));
  GR_EXPECT_EQ(lidar_kept,
               static_cast<int>((kHeight - kHeight / 3 + 1) / 2) * kWidth);
}

GR_TEST(DepthPriorTest, DropsTheCellsWhereThePriorDisagrees) {
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  std::vector<float> prior = truth;
  // A 30 x 30 px block of wrong depth (a thin pole drawn behind the wall).
  for (int r = 50; r < 80; ++r) {
    for (int c = 40; c < 70; ++c) prior[r * kWidth + c] *= 1.6f;
  }
  DepthAlignConfig config;
  config.cell = 16;
  config.hole_radius = 0;  // Every pixel without LiDAR is a hole.
  const FusedDepth fused = FuseDepthPrior(prior, {}, lidar, {}, kHeight,
                                          kWidth, config);
  GR_ASSERT_TRUE(fused.alignment.valid);
  GR_EXPECT_NEAR(fused.alignment.scale, 1.0, 0.01);
  GR_EXPECT_GT(fused.alignment.cells_rejected, 0);
  for (int r = 50; r < 80; ++r) {
    for (int c = 40; c < 70; ++c) {
      GR_EXPECT_TRUE(fused.source[r * kWidth + c] != kDepthPrior);
    }
  }
  // Far from the block the gaps between LiDAR rows are filled.
  GR_EXPECT_EQ(fused.source[(kHeight - 3) * kWidth + 5], kDepthPrior);
}

GR_TEST(DepthPriorTest, RejectsAnImageThatDoesNotFit) {
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> uniform(1.0, 30.0);
  std::vector<float> prior(truth.size());
  for (float& p : prior) p = static_cast<float>(uniform(rng));
  const FusedDepth fused = FuseDepthPrior(prior, {}, lidar, {}, kHeight, kWidth);
  GR_EXPECT_FALSE(fused.alignment.valid);
  GR_EXPECT_EQ(fused.prior_pixels, 0);
  for (size_t i = 0; i < truth.size(); ++i) {
    GR_EXPECT_TRUE(fused.source[i] != kDepthPrior);
  }
}

GR_TEST(DepthPriorTest, ConfidenceFollowsThePriorAndTheResidual) {
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  std::vector<float> confidence(truth.size(), 1.0f);
  for (int r = 0; r < kHeight / 3; ++r) {
    for (int c = 0; c < kWidth / 2; ++c) confidence[r * kWidth + c] = 0.02f;
  }
  const FusedDepth fused =
      FuseDepthPrior(truth, confidence, lidar, {}, kHeight, kWidth);
  GR_ASSERT_TRUE(fused.alignment.valid);
  // Low own confidence: dropped (min_confidence 0.05); the upper third is a
  // hole.
  GR_EXPECT_EQ(fused.source[5 * kWidth + 5], kDepthNone);
  GR_EXPECT_EQ(fused.source[5 * kWidth + kWidth - 5], kDepthPrior);
  GR_EXPECT_GT(fused.confidence[5 * kWidth + kWidth - 5], 0.95f);
}

// A floor at a grazing angle: the z-buffer of the map points keeps the
// nearest point in each pixel and reads short; the points themselves, at
// their sub-pixel position, fit the prior exactly.
GR_TEST(DepthPriorTest, FitsAtTheLidarPointsNotTheZBuffer) {
  auto range = [](double u, double v) {
    return 40.0 / (1.0 + 0.2 * v) + 0.01 * u;  // Steep in v (grazing).
  };
  std::vector<float> prior(kHeight * kWidth), lidar(kHeight * kWidth, kNaN);
  std::vector<DepthSample> samples;
  for (int r = 0; r < kHeight; ++r) {
    for (int c = 0; c < kWidth; ++c) {
      prior[r * kWidth + c] = static_cast<float>(range(c, r));
      // Four points per pixel at +-0.4 px: the z-buffer keeps the nearest.
      double nearest = 1e9;
      for (double dv : {-0.4, 0.4}) {
        for (double du : {-0.4, 0.4}) {
          nearest = std::min(nearest, range(c + du, r + dv));
          if (r % 3 == 0 && c % 3 == 0) {
            samples.push_back({c + du, r + dv, range(c + du, r + dv)});
          }
        }
      }
      lidar[r * kWidth + c] = static_cast<float>(nearest);
    }
  }
  DepthAlignConfig config;
  config.fit_shift = false;
  const FusedDepth zbuffer =
      FuseDepthPrior(prior, {}, lidar, {}, kHeight, kWidth, config);
  const FusedDepth points =
      FuseDepthPrior(prior, {}, lidar, samples, kHeight, kWidth, config);
  GR_ASSERT_TRUE(zbuffer.alignment.valid && points.alignment.valid);
  GR_EXPECT_LT(zbuffer.alignment.scale, 0.99);
  GR_EXPECT_NEAR(points.alignment.scale, 1.0, 2e-3);
}

}  // namespace
}  // namespace gr
