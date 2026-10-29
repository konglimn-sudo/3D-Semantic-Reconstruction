#include "gr/backend/scan_context.h"

#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::CorridorScene;
using ::gr::testing::CorridorSensorPose;
using ::gr::testing::ScanFromPose;

GR_TEST(ScanContextTest, RecoversYawAndSeparatesPlaces) {
  constexpr double kLidarHeight = 2.0;  // m
  constexpr double kPlace = 12.0;       // m along the corridor
  constexpr double kFarPlace = 40.0;    // m
  constexpr double kYaw = 0.5;          // rad
  constexpr double kMaxDistance = 0.1;  // Scan Context distance.
  const BoxScene scene = CorridorScene();
  ScanContextConfig config;
  config.lidar_height = kLidarHeight;
  const auto descriptor = [&](const Pose& pose) {
    return ComputeScanContext(ScanFromPose(scene, pose).points_sensor, config);
  };
  const Eigen::MatrixXd reference = descriptor(CorridorSensorPose(kPlace, 0.0));
  // Same place, rotated.
  const Eigen::MatrixXd rotated = descriptor(CorridorSensorPose(kPlace, kYaw));
  const ScanContextMatch match = MatchScanContexts(reference, rotated);
  GR_EXPECT_LT(match.distance, kMaxDistance);
  // Within one sector.
  GR_EXPECT_NEAR(match.yaw, kYaw, 2.0 * kPi / config.sectors);
  const Eigen::MatrixXd far = descriptor(CorridorSensorPose(kFarPlace, 0.0));
  GR_EXPECT_GT(MatchScanContexts(reference, far).distance, match.distance);
}

}  // namespace
}  // namespace gr
