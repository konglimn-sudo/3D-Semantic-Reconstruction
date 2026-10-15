#include "gr/backend.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "gr/testing/test.h"

namespace gr {
namespace {

Pose PoseXyzYaw(double x, double y, double z, double yaw) {
  return Pose(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
              Vec3(x, y, z));
}

double TiltDegrees(const Pose& pose) {
  return RadToDeg(std::acos(std::clamp(pose.rotation(2, 2), -1.0, 1.0)));
}

GR_TEST(BackendTest, GravityFactorsKeepADriftingChainLevel) {
  // Odometry that pitches by 0.1 deg per keyframe (as on a ramp exit) tilts
  // the whole chain; level-gravity measurements keep it horizontal.
  KeyframeSet keyframes({0.2});
  Pose pose;
  const Pose step(
      Eigen::AngleAxisd(DegToRad(0.1), Vec3::UnitY()).toRotationMatrix(),
      Vec3(1, 0, 0));
  std::vector<GravityMeasurement> gravity;
  for (int k = 0; k < 60; ++k) {
    Keyframe keyframe;
    keyframe.odom = pose;
    keyframes.Add(keyframe, {});
    GravityMeasurement measurement;
    measurement.index = k;
    measurement.up = Vec3::UnitZ();  // The true vehicle stays level.
    measurement.sigma_rad = DegToRad(0.2);
    gravity.push_back(measurement);
    pose = pose * step;
  }
  Anchor anchor;
  anchor.keyframe = 0;
  anchor.sigma_t = 0.01;
  anchor.sigma_r = 0.01;
  const BackendConfig config;
  const std::vector<Pose> without =
      OptimizeGraph(keyframes, {}, {anchor}, {}, config);
  GraphSummary summary;
  const std::vector<Pose> with = OptimizeGraph(
      keyframes, {}, {anchor}, gravity, config, nullptr, nullptr, &summary);
  GR_EXPECT_LT(summary.final_cost, summary.initial_cost);
  GR_EXPECT_GT(TiltDegrees(without.back()), 5.0);
  GR_EXPECT_LT(TiltDegrees(with.back()), 0.5);
  GR_EXPECT_LT(std::abs(with.back().translation.z()),
               0.25 * std::abs(without.back().translation.z()));
}

GR_TEST(BackendTest, AnchorsUseInsPoseGnssLeverArmOrAWeakPrior) {
  KeyframeSet keyframes({0.2});
  for (int t = 0; t < 3; ++t) {
    for (int k = 0; k < 3; ++k) {
      Keyframe keyframe;
      keyframe.trip = t;
      keyframe.stamp = 10.0 + k;
      keyframe.odom = PoseXyzYaw(k, 0, 0, kPi / 2);
      keyframes.Add(keyframe, {});
    }
  }
  std::vector<TripInputs> trips(3);
  for (TripInputs& trip : trips) {
    trip.vehicle_from_lidar = Pose(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
    trip.frame_stamps = {10.0, 11.0, 12.0};
    trip.frame_odometry = {PoseXyzYaw(0, 0, 0, kPi / 2),
                           PoseXyzYaw(1, 0, 0, kPi / 2),
                           PoseXyzYaw(2, 0, 0, kPi / 2)};
  }
  trips[0].initial_pose = PoseXyzYaw(5, 6, 0, 0);
  trips[1].gnss.resize(2, 5);
  trips[1].gnss << 11.2, 100, 200, 0, 0.03,  //
      30.0, 0, 0, 0, 0.03;                   // Too far from any keyframe.
  std::vector<std::string> lines;
  const std::vector<Anchor> anchors =
      MakeAnchors(keyframes, trips, BackendConfig(),
                  [&](const std::string& line) { lines.push_back(line); });
  GR_ASSERT_TRUE(anchors.size() == 3);
  GR_EXPECT_TRUE(anchors[0].type == Anchor::Type::kPose);
  GR_EXPECT_EQ(anchors[0].keyframe, 0);
  // INS pose * vehicle_from_lidar.
  GR_EXPECT_LT((anchors[0].pose.translation - Vec3(6, 6, 1.8)).norm(), 1e-12);
  GR_EXPECT_TRUE(anchors[1].type == Anchor::Type::kPosition);
  GR_EXPECT_EQ(anchors[1].keyframe, 4);
  GR_EXPECT_LT((anchors[1].lever - Vec3(-1.0, 0, -1.8)).norm(), 1e-12);
  // The fix at t = 11.2 moves to the keyframe at t = 11 with the odometry
  // increment (-0.2 m in x).
  GR_EXPECT_LT((anchors[1].position - Vec3(99.8, 200, 0)).norm(), 1e-9);
  GR_EXPECT_NEAR(anchors[1].sigma, 0.03, 1e-12);
  GR_EXPECT_TRUE(anchors[2].type == Anchor::Type::kPose);
  GR_EXPECT_EQ(anchors[2].keyframe, 6);
  GR_EXPECT_NEAR(anchors[2].sigma_t, 50.0, 1e-12);
  GR_EXPECT_EQ(lines.size(), 1u);
}

GR_TEST(BackendTest, FramePosesFollowInterpolatedKeyframeCorrections) {
  const std::vector<double> kf_stamps{0.0, 1.0};
  const std::vector<Pose> kf_odom{PoseXyzYaw(0, 0, 0, 0),
                                  PoseXyzYaw(10, 0, 0, 0)};
  const std::vector<Pose> kf_poses{PoseXyzYaw(0, 1, 0, 0),
                                   PoseXyzYaw(10, 3, 0, 0)};
  const std::vector<double> frame_stamps{0.0, 0.5, 1.0, 2.0};
  const std::vector<Pose> frame_odom{
      PoseXyzYaw(0, 0, 0, 0), PoseXyzYaw(5, 0, 0, 0), PoseXyzYaw(10, 0, 0, 0),
      PoseXyzYaw(20, 0, 0, 0)};
  const std::vector<Pose> out =
      CorrectFramePoses(kf_stamps, kf_odom, kf_poses, frame_stamps, frame_odom);
  GR_EXPECT_LT((out[0].translation - Vec3(0, 1, 0)).norm(), 1e-12);
  GR_EXPECT_LT((out[1].translation - Vec3(5, 2, 0)).norm(), 1e-12);
  GR_EXPECT_LT((out[2].translation - Vec3(10, 3, 0)).norm(), 1e-12);
  // Clamped to the last correction.
  GR_EXPECT_LT((out[3].translation - Vec3(20, 3, 0)).norm(), 1e-12);
}

}  // namespace
}  // namespace gr
