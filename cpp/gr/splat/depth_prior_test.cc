#include "gr/splat/depth_prior.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

constexpr int kHeight = 96;   // Pixels.
constexpr int kWidth = 128;   // Pixels.
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
// The upper kHeight / kUnmeasuredShare rows have no LiDAR.
constexpr int kUnmeasuredShare = 3;
// LiDAR is on every kLidarRowStep-th row below them.
constexpr int kLidarRowStep = 2;

// A floor seen from a camera: range grows towards the top of the image, from
// kNearRange (m) by kRangeSpan over the height and kRangePerColumn per
// column.
std::vector<float> TrueRange() {
  constexpr double kNearRange = 2.0;
  constexpr double kRangeSpan = 20.0;
  constexpr double kRangePerColumn = 0.01;
  std::vector<float> range(kHeight * kWidth);
  for (int row = 0; row < kHeight; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      range[row * kWidth + column] = static_cast<float>(
          kNearRange + kRangeSpan * (kHeight - row) / kHeight +
          kRangePerColumn * column);
    }
  }
  return range;
}

// LiDAR on every other row of the lower two thirds only.
std::vector<float> Lidar(const std::vector<float>& truth) {
  std::vector<float> lidar(truth.size(), kNaN);
  for (int row = kHeight / kUnmeasuredShare; row < kHeight;
       row += kLidarRowStep) {
    for (int column = 0; column < kWidth; ++column) {
      lidar[row * kWidth + column] = truth[row * kWidth + column];
    }
  }
  return lidar;
}

GR_TEST(DepthPriorTest, RecoversScaleAndShiftAndFillsOnlyTheHoles) {
  // Prior = (truth - kShift) / kScale with kNoise relative noise and a
  // kOutlierShare of gross outliers at kOutlierFactor.
  constexpr double kScale = 1.07;
  constexpr double kShift = 0.3;  // m
  constexpr double kNoise = 0.01;
  constexpr double kOutlierShare = 0.05;
  constexpr double kOutlierFactor = 0.5;
  constexpr unsigned kSeed = 3;
  constexpr double kScaleTolerance = 0.01;
  constexpr double kShiftTolerance = 0.05;  // m
  constexpr double kMaxResidual = 0.02;
  constexpr double kGoodError = 0.03;  // Relative.
  constexpr double kMinGoodShare = 0.9;
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  std::mt19937 rng(kSeed);
  std::normal_distribution<double> noise(0.0, kNoise);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<float> prior(truth.size());
  for (size_t pixel = 0; pixel < truth.size(); ++pixel) {
    double value = (truth[pixel] - kShift) / kScale * (1.0 + noise(rng));
    if (uniform(rng) < kOutlierShare) value *= kOutlierFactor;
    prior[pixel] = static_cast<float>(value);
  }
  const FusedDepth fused =
      FuseDepthPrior(prior, {}, lidar, {}, kHeight, kWidth);
  GR_ASSERT_TRUE(fused.alignment.valid);
  GR_EXPECT_NEAR(fused.alignment.scale, kScale, kScaleTolerance);
  GR_EXPECT_NEAR(fused.alignment.shift, kShift, kShiftTolerance);
  GR_EXPECT_LT(fused.alignment.residual_median, kMaxResidual);
  int prior_pixels = 0;
  int lidar_kept = 0;
  std::vector<double> errors;
  for (size_t pixel = 0; pixel < truth.size(); ++pixel) {
    if (std::isfinite(lidar[pixel])) {
      GR_EXPECT_EQ(fused.source[pixel], kDepthLidar);
      lidar_kept += fused.range[pixel] == lidar[pixel];
    } else if (fused.source[pixel] == kDepthPrior) {
      ++prior_pixels;
      errors.push_back(std::abs(fused.range[pixel] - truth[pixel]) /
                       truth[pixel]);
    }
  }
  GR_EXPECT_EQ(prior_pixels, fused.prior_pixels);
  // The upper third without LiDAR is filled, up to hole_radius rows from it;
  // the gaps between the LiDAR rows are not holes.
  const int hole_rows =
      kHeight / kUnmeasuredShare - DepthAlignConfig().hole_radius;
  GR_EXPECT_EQ(prior_pixels, hole_rows * kWidth);
  int good = 0;
  for (double error : errors) good += error < kGoodError;
  GR_EXPECT_GT(good, static_cast<int>(kMinGoodShare * errors.size()));
  const int lidar_rows =
      (kHeight - kHeight / kUnmeasuredShare + 1) / kLidarRowStep;
  GR_EXPECT_EQ(lidar_kept, lidar_rows * kWidth);
}

GR_TEST(DepthPriorTest, DropsTheCellsWhereThePriorDisagrees) {
  // A 30 x 30 px block of wrong depth (a thin pole drawn behind the wall).
  constexpr int kBlockTop = 50;
  constexpr int kBlockBottom = 80;  // Exclusive.
  constexpr int kBlockLeft = 40;
  constexpr int kBlockRight = 70;   // Exclusive.
  constexpr float kBlockFactor = 1.6f;
  constexpr int kCell = 16;         // Pixels.
  constexpr double kScaleTolerance = 0.01;
  // A pixel between two LiDAR rows far from the block.
  constexpr int kGapRow = kHeight - 3;
  constexpr int kGapColumn = 5;
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  std::vector<float> prior = truth;
  for (int row = kBlockTop; row < kBlockBottom; ++row) {
    for (int column = kBlockLeft; column < kBlockRight; ++column) {
      prior[row * kWidth + column] *= kBlockFactor;
    }
  }
  DepthAlignConfig config;
  config.cell = kCell;
  config.hole_radius = 0;  // Every pixel without LiDAR is a hole.
  const FusedDepth fused = FuseDepthPrior(prior, {}, lidar, {}, kHeight,
                                          kWidth, config);
  GR_ASSERT_TRUE(fused.alignment.valid);
  GR_EXPECT_NEAR(fused.alignment.scale, 1.0, kScaleTolerance);
  GR_EXPECT_GT(fused.alignment.cells_rejected, 0);
  for (int row = kBlockTop; row < kBlockBottom; ++row) {
    for (int column = kBlockLeft; column < kBlockRight; ++column) {
      GR_EXPECT_TRUE(fused.source[row * kWidth + column] != kDepthPrior);
    }
  }
  // Far from the block the gaps between LiDAR rows are filled.
  GR_EXPECT_EQ(fused.source[kGapRow * kWidth + kGapColumn], kDepthPrior);
}

GR_TEST(DepthPriorTest, RejectsAnImageThatDoesNotFit) {
  // Random ranges in [kMinRange, kMaxRange] m.
  constexpr unsigned kSeed = 5;
  constexpr double kMinRange = 1.0;
  constexpr double kMaxRange = 30.0;
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(kMinRange, kMaxRange);
  std::vector<float> prior(truth.size());
  for (float& value : prior) value = static_cast<float>(uniform(rng));
  const FusedDepth fused =
      FuseDepthPrior(prior, {}, lidar, {}, kHeight, kWidth);
  GR_EXPECT_FALSE(fused.alignment.valid);
  GR_EXPECT_EQ(fused.prior_pixels, 0);
  for (size_t pixel = 0; pixel < truth.size(); ++pixel) {
    GR_EXPECT_TRUE(fused.source[pixel] != kDepthPrior);
  }
}

GR_TEST(DepthPriorTest, ConfidenceFollowsThePriorAndTheResidual) {
  // The left half of the upper third has a low own confidence.
  constexpr float kLowConfidence = 0.02f;
  constexpr int kProbeRow = 5;
  constexpr int kProbeMargin = 5;  // Columns from the image's sides.
  constexpr float kMinHighConfidence = 0.95f;
  const std::vector<float> truth = TrueRange();
  const std::vector<float> lidar = Lidar(truth);
  std::vector<float> confidence(truth.size(), 1.0f);
  for (int row = 0; row < kHeight / kUnmeasuredShare; ++row) {
    for (int column = 0; column < kWidth / 2; ++column) {
      confidence[row * kWidth + column] = kLowConfidence;
    }
  }
  const FusedDepth fused =
      FuseDepthPrior(truth, confidence, lidar, {}, kHeight, kWidth);
  GR_ASSERT_TRUE(fused.alignment.valid);
  // Low own confidence: dropped (below min_confidence); the upper third is a
  // hole.
  const int left = kProbeRow * kWidth + kProbeMargin;
  const int right = kProbeRow * kWidth + kWidth - kProbeMargin;
  GR_EXPECT_EQ(fused.source[left], kDepthNone);
  GR_EXPECT_EQ(fused.source[right], kDepthPrior);
  GR_EXPECT_GT(fused.confidence[right], kMinHighConfidence);
}

// A floor at a grazing angle: the z-buffer of the map points keeps the
// nearest point in each pixel and reads short; the points themselves, at
// their sub-pixel position, fit the prior exactly.
GR_TEST(DepthPriorTest, FitsAtTheLidarPointsNotTheZBuffer) {
  // Range kFarRange / (1 + kRangeFalloff v) + kRangePerColumn u: steep in v.
  constexpr double kFarRange = 40.0;
  constexpr double kRangeFalloff = 0.2;
  constexpr double kRangePerColumn = 0.01;
  // Four points per pixel at +-kPointOffset px; every kSampleStride-th row
  // and column gives samples.
  constexpr double kPointOffset = 0.4;
  constexpr int kSampleStride = 3;
  constexpr double kMaxZBufferScale = 0.99;
  constexpr double kScaleTolerance = 2e-3;
  const auto range = [&](double u, double v) {
    return kFarRange / (1.0 + kRangeFalloff * v) + kRangePerColumn * u;
  };
  std::vector<float> prior(kHeight * kWidth);
  std::vector<float> lidar(kHeight * kWidth, kNaN);
  std::vector<DepthSample> samples;
  for (int row = 0; row < kHeight; ++row) {
    for (int column = 0; column < kWidth; ++column) {
      prior[row * kWidth + column] = static_cast<float>(range(column, row));
      // The z-buffer keeps the nearest of the pixel's points.
      double nearest = std::numeric_limits<double>::infinity();
      for (double offset_v : {-kPointOffset, kPointOffset}) {
        for (double offset_u : {-kPointOffset, kPointOffset}) {
          const double u = column + offset_u;
          const double v = row + offset_v;
          nearest = std::min(nearest, range(u, v));
          if (row % kSampleStride == 0 && column % kSampleStride == 0) {
            samples.push_back({u, v, range(u, v)});
          }
        }
      }
      lidar[row * kWidth + column] = static_cast<float>(nearest);
    }
  }
  DepthAlignConfig config;
  config.fit_shift = false;
  const FusedDepth zbuffer =
      FuseDepthPrior(prior, {}, lidar, {}, kHeight, kWidth, config);
  const FusedDepth points =
      FuseDepthPrior(prior, {}, lidar, samples, kHeight, kWidth, config);
  GR_ASSERT_TRUE(zbuffer.alignment.valid && points.alignment.valid);
  GR_EXPECT_LT(zbuffer.alignment.scale, kMaxZBufferScale);
  GR_EXPECT_NEAR(points.alignment.scale, 1.0, kScaleTolerance);
}

}  // namespace
}  // namespace gr
