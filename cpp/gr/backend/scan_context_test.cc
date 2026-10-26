#include "gr/backend/scan_context.h"

#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::CorridorScene;
using ::gr::testing::CorridorSensorPose;
using ::gr::testing::ScanFromPose;

GR_TEST(ScanContextTest, RecoversYawAndSeparatesPlaces) {
  const BoxScene scene = CorridorScene();
  ScanContextConfig config;
  config.lidar_height = 2.0;
  const auto descriptor = [&](const Pose& pose) {
    return ComputeScanContext(ScanFromPose(scene, pose).points_sensor, config);
  };
  const Eigen::MatrixXd a = descriptor(CorridorSensorPose(12.0, 0.0));
  // Same place, rotated by 0.5 rad.
  const Eigen::MatrixXd b = descriptor(CorridorSensorPose(12.0, 0.5));
  const ScanContextMatch match = MatchScanContexts(a, b);
  GR_EXPECT_LT(match.distance, 0.1);
  GR_EXPECT_NEAR(match.yaw, 0.5, 2.0 * kPi / config.sectors);
  const Eigen::MatrixXd far = descriptor(CorridorSensorPose(40.0, 0.0));
  GR_EXPECT_GT(MatchScanContexts(a, far).distance, match.distance);
}

}  // namespace
}  // namespace gr
