#include "gr/registration/registration.h"

#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::CorridorScene;
using ::gr::testing::CorridorSensorPose;
using ::gr::testing::ScanFromPose;

GR_TEST(RegistrationTest, IcpRecoversKnownOffset) {
  const BoxScene scene = CorridorScene();
  const Pose start = CorridorSensorPose(10.0, 0.0);
  VoxelHashMap map(0.3, 1e9, 20);
  map.AddPoints(ScanFromPose(scene, start).points_world);
  const Pose truth = CorridorSensorPose(10.4, 0.06);
  IcpOptions options;
  options.max_corr_dist = 1.0;
  options.kernel = 0.2;
  const IcpResult result =
      AlignToMap(VoxelDownsample(ScanFromPose(scene, truth).points_sensor, 0.3),
                 map, start, options);
  GR_EXPECT_NEAR((result.pose.translation - truth.translation).norm(), 0.0,
                 0.03);
  GR_EXPECT_NEAR(
      So3Log(result.pose.rotation.transpose() * truth.rotation).norm(), 0.0,
      0.003);
  GR_EXPECT_GT(result.fitness, 0.8);
}

GR_TEST(RegistrationTest, MultiScaleIcpConvergesFromAFarStart) {
  const BoxScene scene = CorridorScene();
  const Pose target_pose = CorridorSensorPose(20.0, 0.0);
  const Pose source_pose = CorridorSensorPose(21.0, 0.1);
  const Points target = ScanFromPose(scene, target_pose).points_sensor;
  const Points source = ScanFromPose(scene, source_pose).points_sensor;
  const Pose truth = target_pose.Inverse() * source_pose;  // target <- source
  const IcpResult result = AlignMultiScale(VoxelDownsample(source, 0.2),
                                           VoxelDownsample(target, 0.2), Pose(),
                                           {2.0, 1.0, 0.5, 0.3}, 0.2);
  GR_EXPECT_NEAR((result.pose.translation - truth.translation).norm(), 0.0,
                 0.05);
  GR_EXPECT_GT(result.fitness, 0.6);
}

GR_TEST(RegistrationTest, NearestHonoursTheGateAndExcludedFrames) {
  VoxelHashMap map(0.5);
  map.AddPoints({Vec3(0, 0, 0)}, 1);
  map.AddPoints({Vec3(0.3, 0, 0)}, 2);
  const auto nearest = map.Nearest(Vec3(0.05, 0, 0), 1.0);
  GR_ASSERT_TRUE(nearest.has_value());
  GR_EXPECT_NEAR(nearest->point.x(), 0.0, 1e-12);
  const auto other = map.Nearest(Vec3(0.05, 0, 0), 1.0, /*exclude_frame=*/1);
  GR_ASSERT_TRUE(other.has_value());
  GR_EXPECT_NEAR(other->point.x(), 0.3, 1e-12);
  GR_EXPECT_FALSE(map.Nearest(Vec3(5, 0, 0), 1.0).has_value());
  GR_EXPECT_EQ(map.CountPoints(), 2u);
}

}  // namespace
}  // namespace gr
