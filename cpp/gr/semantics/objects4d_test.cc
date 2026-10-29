#include "gr/semantics/objects4d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "gr/core/geometry.h"
#include "gr/core/morphology.h"
#include "gr/core/numeric.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

GR_TEST(LinearSumAssignmentTest, MatchesBruteForceOnRectangularCosts) {
  // Random costs in [0, kMaxCost), a kGatedShare of them gated (kGated).
  constexpr unsigned kSeed = 7;
  constexpr double kMaxCost = 10.0;
  constexpr double kGatedBelow = 3.0;  // Of kMaxCost: 30 % gated.
  constexpr double kGated = 1e6;
  constexpr int kTrials = 40;
  constexpr int kMaxRows = 4;
  constexpr int kMaxColumns = 5;
  constexpr double kTolerance = 1e-9;
  std::mt19937 rng(kSeed);
  std::uniform_real_distribution<double> uniform(0.0, kMaxCost);
  for (int trial = 0; trial < kTrials; ++trial) {
    const int rows = 1 + trial % kMaxRows;
    const int columns = 1 + (trial / kMaxRows) % kMaxColumns;
    std::vector<double> cost(static_cast<size_t>(rows) * columns);
    for (double& entry : cost) {
      entry = uniform(rng) < kGatedBelow ? kGated : uniform(rng);
    }
    const auto pairs = LinearSumAssignment(cost, rows, columns);
    GR_ASSERT_TRUE(static_cast<int>(pairs.size()) == std::min(rows, columns));
    double total = 0.0;
    for (const auto& [row, column] : pairs)
      total += cost[row * columns + column];
    // Brute force over the column permutations (rows <= columns) or the row
    // permutations.
    const int small = std::min(rows, columns);
    const int large = std::max(rows, columns);
    std::vector<int> permutation(large);
    std::iota(permutation.begin(), permutation.end(), 0);
    double best = std::numeric_limits<double>::infinity();
    do {
      double sum = 0.0;
      for (int k = 0; k < small; ++k) {
        sum += rows <= columns ? cost[k * columns + permutation[k]]
                               : cost[permutation[k] * columns + k];
      }
      best = std::min(best, sum);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    GR_EXPECT_NEAR(total, best, kTolerance);
    for (size_t k = 1; k < pairs.size(); ++k) {
      GR_EXPECT_LT(pairs[k - 1].first, pairs[k].first);
    }
  }
}

GR_TEST(MinAreaRectTest, FindsARotatedRectangle) {
  // A kLength x kWidth grid of points, kStep apart, rotated by kYaw.
  constexpr double kYaw = 0.4;
  constexpr double kStep = 0.1;
  constexpr int kHalfLengthSteps = 22;
  constexpr int kHalfWidthSteps = 9;
  constexpr double kLength = 2 * kHalfLengthSteps * kStep;
  constexpr double kWidth = 2 * kHalfWidthSteps * kStep;
  constexpr double kSizeTolerance = 1e-6;
  constexpr double kTolerance = 1e-9;
  std::vector<Vec2> points;
  const Vec2 center(3.0, -1.0);
  for (int along = -kHalfLengthSteps; along <= kHalfLengthSteps; ++along) {
    for (int across = -kHalfWidthSteps; across <= kHalfWidthSteps; ++across) {
      const double x = kStep * along;
      const double y = kStep * across;
      points.push_back(center + Vec2(std::cos(kYaw) * x - std::sin(kYaw) * y,
                                     std::sin(kYaw) * x + std::cos(kYaw) * y));
    }
  }
  const RotatedRect rect = MinAreaRect(points);
  GR_EXPECT_NEAR(rect.length, kLength, kSizeTolerance);
  GR_EXPECT_NEAR(rect.width, kWidth, kSizeTolerance);
  GR_EXPECT_NEAR(rect.yaw, kYaw, kTolerance);
  GR_EXPECT_NEAR((rect.center - center).norm(), 0.0, kTolerance);
}

GR_TEST(FillHolesCrossTest, FillsEnclosedBackgroundOnly) {
  // A ring (rows 1..4, columns 1..5) with a hole at (3, 2), and a notch at
  // column 2 open to the bottom border.
  constexpr int kWidth = 7;
  constexpr int kHeight = 6;
  constexpr int kHole = 2 * kWidth + 3;
  constexpr int kNotch = 4 * kWidth + 2;
  constexpr int kNotchBorder = 5 * kWidth + 2;
  Mask mask(kWidth * kHeight, 0);
  for (int row = 1; row <= 4; ++row) {
    for (int column = 1; column <= 5; ++column) mask[row * kWidth + column] = 1;
  }
  mask[kHole] = 0;
  mask[kNotch] = 0;  // On the ring's edge row, next to the border.
  mask[kNotchBorder] = 0;
  const Mask out = FillHolesCross(mask, kWidth, kHeight);
  GR_EXPECT_EQ(static_cast<int>(out[kHole]), 1);
  GR_EXPECT_EQ(static_cast<int>(out[kNotch]), 0);
  GR_EXPECT_EQ(static_cast<int>(out[0]), 0);
}

// Car-shaped point clusters: a roof of kRoofHalfLengthSteps x
// kRoofHalfWidthSteps points kCarStep apart at kRoofHeight, and a side line
// kSideOffset to the right at kSideHeight.
constexpr double kCarStep = 0.05;      // m
constexpr int kRoofHalfLengthSteps = 44;
constexpr int kRoofHalfWidthSteps = 18;
constexpr double kRoofHeight = 1.4;    // m
constexpr double kSideOffset = 0.9;    // m
constexpr double kSideHeight = 0.6;    // m

// A box-shaped cluster of points (world frame) at `center`.
Points CarPoints(const Vec2& center, double floor_z) {
  Points points;
  for (int along = -kRoofHalfLengthSteps; along <= kRoofHalfLengthSteps;
       ++along) {
    const double x = kCarStep * along;
    for (int across = -kRoofHalfWidthSteps; across <= kRoofHalfWidthSteps;
         ++across) {
      points.emplace_back(center.x() + x, center.y() + kCarStep * across,
                          floor_z + kRoofHeight);
    }
    points.emplace_back(center.x() + x, center.y() - kSideOffset,
                        floor_z + kSideHeight);
  }
  return points;
}

GR_TEST(ObjectsTest, TracksAMovingCarAcrossKeyframes) {
  // The car drives 1 m along x per keyframe, one keyframe per 0.5 s, the
  // sensor 1.8 m up beside it.
  constexpr int kKeyframes = 6;
  constexpr double kStartX = 2.0;      // m
  constexpr double kStepX = 1.0;       // m per keyframe
  constexpr double kCarY = 5.0;        // m
  constexpr double kPeriod = 0.5;      // s per keyframe
  constexpr double kSensorHeight = 1.8;  // m
  constexpr double kCeiling = 3.0;     // m
  constexpr double kStructureVoxel = 0.1;  // m
  constexpr int kMiddleKeyframe = 3;
  constexpr double kSpeed = kStepX / kPeriod;  // m/s
  constexpr double kSpeedTolerance = 0.3;
  constexpr double kYawTolerance = 1e-6;
  constexpr double kLength = 2 * kRoofHalfLengthSteps * kCarStep;
  constexpr double kLengthTolerance = 0.4;
  constexpr double kWidth = 2 * kSideOffset;
  constexpr double kWidthTolerance = 0.1;
  LevelModel levels;
  levels.z = {0.0};
  levels.ceiling = {kCeiling};
  // Structure far away.
  const StructureIndex structure({Vec3(100.0, 100.0, 0.0)}, kStructureVoxel);
  ObjectsConfig config;
  std::vector<std::vector<Detection>> frames;
  for (int k = 0; k < kKeyframes; ++k) {
    const Vec2 center(kStartX + kStepX * k, kCarY);
    frames.push_back(DetectObjects(CarPoints(center, 0.0), k, kPeriod * k,
                                   Vec3(center.x(), 0.0, kSensorHeight),
                                   levels, structure, config));
    GR_ASSERT_TRUE(frames.back().size() == 1);
    GR_EXPECT_EQ(frames.back()[0].label, config.vehicle_id);
  }
  const std::vector<Track> tracks = TrackDetections(frames, config, 0);
  GR_ASSERT_TRUE(tracks.size() == 1);
  GR_EXPECT_EQ(tracks[0].detections.size(), size_t{kKeyframes});
  const std::map<int, ObjectBox> boxes = MovingTrackBoxes(tracks[0], config);
  GR_ASSERT_TRUE(boxes.size() == kKeyframes);
  const ObjectBox& box = boxes.at(kMiddleKeyframe);
  GR_EXPECT_EQ(box.label, config.vehicle_moving_id);
  // Edge-padded smoothing slows the ends of a track down a little.
  GR_EXPECT_NEAR(box.velocity.x(), kSpeed, kSpeedTolerance);
  GR_EXPECT_NEAR(box.yaw, 0.0, kYawTolerance);
  GR_EXPECT_NEAR(box.size.x(), kLength, kLengthTolerance);
  GR_EXPECT_NEAR(box.size.y(), kWidth, kWidthTolerance);
}

// A detection with `count` scattered points around (x, y) (the size of a
// car), seeded by its keyframe.
Detection MakeDetection(int keyframe, double stamp, double x, double y,
                        int label, int count = 60) {
  constexpr double kAlongSigma = 0.5;   // m
  constexpr double kAcrossSigma = 0.3;  // m
  constexpr double kPointHeight = 0.8;  // m
  std::mt19937 rng(static_cast<unsigned>(keyframe));
  std::normal_distribution<double> along(x, kAlongSigma);
  std::normal_distribution<double> across(y, kAcrossSigma);
  Detection detection;
  detection.keyframe = keyframe;
  detection.stamp = stamp;
  detection.center = Vec3(x, y, 0.75);
  detection.size = Vec3(4.5, 1.8, 1.5);
  detection.label = label;
  detection.score = 1.0;
  for (int i = 0; i < count; ++i) {
    detection.points.emplace_back(along(rng), across(rng), kPointHeight);
  }
  return detection;
}

GR_TEST(ObjectsTest, StitchesFragmentsAndFlagsMotion) {
  // 1.5 m per keyframe, one keyframe per 0.5 s; a two-wheeler for the first
  // kTwoWheelerFrames keyframes, then a vehicle; keyframe kMissedFrame lost.
  constexpr int kKeyframes = 10;
  constexpr double kPeriod = 0.5;  // s
  constexpr double kStepX = 1.5;   // m
  constexpr int kTwoWheelerFrames = 3;
  constexpr int kMissedFrame = 3;
  constexpr double kSpeed = kStepX / kPeriod;
  constexpr double kSpeedTolerance = 0.5;
  const ObjectsConfig config;
  std::vector<std::vector<Detection>> frames;
  for (int k = 0; k < kKeyframes; ++k) {
    const int label =
        k < kTwoWheelerFrames ? config.two_wheeler_id : config.vehicle_id;
    frames.push_back({MakeDetection(k, kPeriod * k, kStepX * k, 0.0, label)});
  }
  frames[kMissedFrame].clear();
  const std::vector<Track> tracks = TrackDetections(frames, config, 0);
  GR_ASSERT_TRUE(tracks.size() == 1);
  const std::map<int, ObjectBox> boxes = MovingTrackBoxes(tracks[0], config);
  GR_ASSERT_TRUE(!boxes.empty());  // Moving.
  std::vector<double> velocity_x;
  for (const auto& [keyframe, box] : boxes) {
    velocity_x.push_back(box.velocity.x());
    GR_EXPECT_EQ(box.label, config.vehicle_moving_id);
  }
  GR_EXPECT_NEAR(Median(velocity_x), kSpeed, kSpeedTolerance);
}

}  // namespace
}  // namespace gr
