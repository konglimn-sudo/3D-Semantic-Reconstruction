#include "gr/calibration/clock_sync.h"

#include <cmath>
#include <random>
#include <vector>

#include "Eigen/Geometry"

#include "gr/core/pose.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

// Angular rate of a rig that turns and sways (rad/s).
Vec3 Rate(double t) {
  return Vec3(0.3 * std::sin(1.7 * t) + 0.1 * std::sin(5.3 * t + 1.0),
              0.2 * std::sin(2.3 * t + 0.4) + 0.1 * std::cos(7.1 * t),
              0.5 * std::sin(0.9 * t + 2.0) + 0.2 * std::sin(3.7 * t));
}

GR_TEST(ClockSyncTest, RecoversOffsetDriftAndRotation) {
  const Mat3 reference_from_other = (Eigen::AngleAxisd(0.02, Vec3::UnitZ()) *
                                     Eigen::AngleAxisd(-0.01, Vec3::UnitX()))
                                        .toRotationMatrix();
  const double offset = 0.042;  // At t = 100 s.
  const double drift = 30e-6;   // 3 ms over 100 s.
  std::mt19937 rng(3);
  std::normal_distribution<double> noise(0.0, 0.005);
  std::vector<double> reference_stamps;
  std::vector<Vec3> reference_rates;
  for (int i = 0; i <= 40000; ++i) {  // 200 Hz, 0..200 s.
    const double t = 0.005 * i;
    reference_stamps.push_back(t);
    reference_rates.push_back(Rate(t) +
                              Vec3(noise(rng), noise(rng), noise(rng)));
  }
  std::vector<double> other_stamps;
  std::vector<Vec3> other_rates;
  for (int i = 0; i <= 19000; ++i) {  // 100 Hz, other clock 5..195 s.
    const double t = 5.0 + 0.01 * i;
    const double lag = offset + drift * (t - 100.0);
    other_stamps.push_back(t);
    other_rates.push_back(reference_from_other.transpose() * Rate(t + lag) +
                          Vec3(noise(rng), noise(rng), noise(rng)));
  }
  ClockSyncConfig config;
  config.window = 20.0;
  const ClockSyncResult result = EstimateClockOffset(
      reference_stamps, reference_rates, other_stamps, other_rates, config);
  GR_ASSERT_TRUE(result.valid);
  GR_EXPECT_GE(result.windows.size(), 8u);
  for (const ClockSyncWindow& w : result.windows) {
    GR_EXPECT_NEAR(w.lag, offset + drift * (w.center - 100.0), 0.001);
  }
  GR_EXPECT_NEAR(result.offset + result.drift * (100.0 - result.reference_time),
                 offset, 3e-4);
  GR_EXPECT_NEAR(result.drift, drift, 5e-6);
  GR_EXPECT_LT(
      RotationAngle(result.rotation.transpose() * reference_from_other), 2e-3);
}

GR_TEST(ClockSyncTest, StillRigGivesNoEstimate) {
  std::vector<double> stamps;
  std::vector<Vec3> rates;
  for (int i = 0; i < 6000; ++i) {
    stamps.push_back(0.01 * i);
    rates.push_back(Vec3(1e-4, -2e-4, 1e-4));
  }
  const ClockSyncResult result =
      EstimateClockOffset(stamps, rates, stamps, rates, ClockSyncConfig());
  GR_EXPECT_FALSE(result.valid);
  GR_EXPECT_TRUE(result.windows.empty());
}

}  // namespace
}  // namespace gr
