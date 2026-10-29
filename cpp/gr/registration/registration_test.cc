#include "gr/registration/registration.h"

#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::CorridorScene;
using ::gr::testing::CorridorSensorPose;
using ::gr::testing::ScanFromPose;

GR_TEST(RegistrationTest, IcpRecoversKnownOffset) {
  // The scan is taken 0.4 m further and turned by 0.06 rad.
  constexpr double kStartX = 10.0;  // m along the corridor.
  constexpr double kTruthX = 10.4;
  constexpr double kTruthYaw = 0.06;             // rad
  constexpr double kVoxel = 0.3;                 // m
  constexpr double kMaxTranslationError = 0.03;  // m
  constexpr double kMaxRotationError = 0.003;    // rad
  constexpr double kMinFitness = 0.8;
  const BoxScene scene = CorridorScene();
  const Pose start = CorridorSensorPose(kStartX, 0.0);
  VoxelHashMap map(kVoxel);
  map.AddPoints(ScanFromPose(scene, start).points_world);
  const Pose truth = CorridorSensorPose(kTruthX, kTruthYaw);
  IcpOptions options;
  options.max_corr_dist = 1.0;
  options.kernel = 0.2;
  const IcpResult result = AlignToMap(
      VoxelDownsample(ScanFromPose(scene, truth).points_sensor, kVoxel), map,
      start, options);
  GR_EXPECT_NEAR((result.pose.translation - truth.translation).norm(), 0.0,
                 kMaxTranslationError);
  GR_EXPECT_NEAR(
      So3Log(result.pose.rotation.transpose() * truth.rotation).norm(), 0.0,
      kMaxRotationError);
  GR_EXPECT_GT(result.fitness, kMinFitness);
}

GR_TEST(RegistrationTest, MultiScaleIcpConvergesFromAFarStart) {
  // 1 m and 0.1 rad apart, aligned from the identity.
  constexpr double kTargetX = 20.0;  // m along the corridor.
  constexpr double kSourceX = 21.0;
  constexpr double kSourceYaw = 0.1;             // rad
  constexpr double kVoxel = 0.2;                 // m
  constexpr double kMaxTranslationError = 0.05;  // m
  constexpr double kMinFitness = 0.6;
  const BoxScene scene = CorridorScene();
  const Pose target_pose = CorridorSensorPose(kTargetX, 0.0);
  const Pose source_pose = CorridorSensorPose(kSourceX, kSourceYaw);
  const Points target = ScanFromPose(scene, target_pose).points_sensor;
  const Points source = ScanFromPose(scene, source_pose).points_sensor;
  const Pose truth = target_pose.Inverse() * source_pose;  // target <- source
  const IcpResult result = AlignMultiScale(
      VoxelDownsample(source, kVoxel), VoxelDownsample(target, kVoxel), Pose(),
      {2.0, 1.0, 0.5, 0.3}, kVoxel);
  GR_EXPECT_NEAR((result.pose.translation - truth.translation).norm(), 0.0,
                 kMaxTranslationError);
  GR_EXPECT_GT(result.fitness, kMinFitness);
}

GR_TEST(RegistrationTest, NearestHonoursTheGateAndExcludedFrames) {
  constexpr double kVoxel = 0.5;  // m
  constexpr double kGate = 1.0;   // m
  constexpr int kFirstFrame = 1;
  constexpr int kSecondFrame = 2;
  constexpr double kTolerance = 1e-12;
  // A point of each frame, 0.3 m apart; the query lies next to the first.
  const Vec3 first(0, 0, 0);
  const Vec3 second(0.3, 0, 0);
  const Vec3 query(0.05, 0, 0);
  VoxelHashMap map(kVoxel);
  map.AddPoints({first}, kFirstFrame);
  map.AddPoints({second}, kSecondFrame);
  const auto nearest = map.Nearest(query, kGate);
  GR_ASSERT_TRUE(nearest.has_value());
  GR_EXPECT_NEAR(nearest->point.x(), first.x(), kTolerance);
  const auto other = map.Nearest(query, kGate, kFirstFrame);
  GR_ASSERT_TRUE(other.has_value());
  GR_EXPECT_NEAR(other->point.x(), second.x(), kTolerance);
  GR_EXPECT_FALSE(map.Nearest(Vec3(5, 0, 0), kGate).has_value());
  GR_EXPECT_EQ(map.CountPoints(), 2u);
}

}  // namespace
}  // namespace gr
