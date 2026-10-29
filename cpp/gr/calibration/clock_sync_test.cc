#include "gr/calibration/clock_sync.h"

#include <cmath>
#include <random>
#include <vector>

#include "Eigen/Geometry"

#include "gr/core/pose.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

// Angular rate of a rig that turns and sways (rad/s), at `time` (s).
Vec3 Rate(double time) {
  return Vec3(0.3 * std::sin(1.7 * time) + 0.1 * std::sin(5.3 * time + 1.0),
              0.2 * std::sin(2.3 * time + 0.4) + 0.1 * std::cos(7.1 * time),
              0.5 * std::sin(0.9 * time + 2.0) + 0.2 * std::sin(3.7 * time));
}

GR_TEST(ClockSyncTest, RecoversOffsetDriftAndRotation) {
  // The reference gyro at 200 Hz over 0..200 s, the other at 100 Hz over
  // 5..195 s of its own clock.
  constexpr int kReferenceSamples = 40000;
  constexpr double kReferencePeriod = 0.005;  // s
  constexpr int kOtherSamples = 19000;
  constexpr double kOtherStart = 5.0;    // s
  constexpr double kOtherPeriod = 0.01;  // s
  constexpr double kOffsetTime = 100.0;  // s: the offset holds there.
  constexpr double kOffset = 0.042;      // s
  constexpr double kDrift = 30e-6;       // s/s: 3 ms over 100 s.
  constexpr int kSeed = 3;
  constexpr double kNoise = 0.005;  // rad/s
  constexpr double kWindow = 20.0;  // s
  constexpr size_t kMinWindows = 8;
  constexpr double kWindowLagTolerance = 0.001;  // s
  constexpr double kOffsetTolerance = 3e-4;      // s
  constexpr double kDriftTolerance = 5e-6;       // s/s
  constexpr double kRotationTolerance = 2e-3;    // rad
  const Mat3 reference_from_other = (Eigen::AngleAxisd(0.02, Vec3::UnitZ()) *
                                     Eigen::AngleAxisd(-0.01, Vec3::UnitX()))
                                        .toRotationMatrix();
  std::mt19937 rng(kSeed);
  std::normal_distribution<double> noise(0.0, kNoise);
  std::vector<double> reference_stamps;
  std::vector<Vec3> reference_rates;
  for (int i = 0; i <= kReferenceSamples; ++i) {
    const double time = kReferencePeriod * i;
    reference_stamps.push_back(time);
    reference_rates.push_back(Rate(time) +
                              Vec3(noise(rng), noise(rng), noise(rng)));
  }
  std::vector<double> other_stamps;
  std::vector<Vec3> other_rates;
  for (int i = 0; i <= kOtherSamples; ++i) {
    const double time = kOtherStart + kOtherPeriod * i;
    const double lag = kOffset + kDrift * (time - kOffsetTime);
    other_stamps.push_back(time);
    other_rates.push_back(reference_from_other.transpose() * Rate(time + lag) +
                          Vec3(noise(rng), noise(rng), noise(rng)));
  }
  ClockSyncConfig config;
  config.window = kWindow;
  const ClockSyncResult result = EstimateClockOffset(
      reference_stamps, reference_rates, other_stamps, other_rates, config);
  GR_ASSERT_TRUE(result.valid);
  GR_EXPECT_GE(result.windows.size(), kMinWindows);
  for (const ClockSyncWindow& window : result.windows) {
    GR_EXPECT_NEAR(window.lag, kOffset + kDrift * (window.center - kOffsetTime),
                   kWindowLagTolerance);
  }
  GR_EXPECT_NEAR(
      result.offset + result.drift * (kOffsetTime - result.reference_time),
      kOffset, kOffsetTolerance);
  GR_EXPECT_NEAR(result.drift, kDrift, kDriftTolerance);
  GR_EXPECT_LT(
      RotationAngle(result.rotation.transpose() * reference_from_other),
      kRotationTolerance);
}

GR_TEST(ClockSyncTest, StillRigGivesNoEstimate) {
  // 60 s at 100 Hz of a rig standing still (rates far below min_rate).
  constexpr int kSamples = 6000;
  constexpr double kPeriod = 0.01;  // s
  std::vector<double> stamps;
  std::vector<Vec3> rates;
  for (int i = 0; i < kSamples; ++i) {
    stamps.push_back(kPeriod * i);
    rates.push_back(Vec3(1e-4, -2e-4, 1e-4));
  }
  const ClockSyncResult result =
      EstimateClockOffset(stamps, rates, stamps, rates, ClockSyncConfig());
  GR_EXPECT_FALSE(result.valid);
  GR_EXPECT_TRUE(result.windows.empty());
}

}  // namespace
}  // namespace gr
