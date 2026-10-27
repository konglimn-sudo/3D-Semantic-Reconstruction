#include "gr/backend/backend.h"
#include "gr/backend/loop_closure.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "gr/testing/sim_garage.h"
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
  GR_EXPECT_TRUE(!anchors[0].weak && !anchors[1].weak && anchors[2].weak);
  GR_EXPECT_EQ(lines.size(), 1u);

  // A control point: the target 2 m ahead of the LiDAR (its x axis, yaw
  // 90 deg: world +y) at t = 11.6, surveyed at (1.6, 2, 0): anchored at the
  // keyframe of t = 12 with the target in that keyframe's frame.
  trips[2].control_points.resize(1, 8);
  trips[2].control_points << 11.6, 2.0, 0.0, 0.0, 1.6, 2.0, 0.0, 0.004;
  const std::vector<Anchor> with_control =
      MakeAnchors(keyframes, trips, BackendConfig(), {});
  GR_ASSERT_TRUE(with_control.size() == 3);
  const Anchor& control = with_control[2];
  GR_EXPECT_TRUE(control.type == Anchor::Type::kPosition && !control.weak);
  GR_EXPECT_EQ(control.keyframe, 8);
  GR_EXPECT_NEAR(control.sigma, 0.004, 1e-12);
  // In the keyframe frame (at x = 2) the target is 2 m ahead, 0.4 m back.
  GR_EXPECT_LT((keyframes[8].odom * control.lever - Vec3(1.6, 2.0, 0.0))
                   .norm(),
               1e-9);
}

// Two sessions around the test garage loop without any anchor; the second
// starts 30 m further along the loop and its odometry is in the frame of its
// own start (as a handheld session's is).
struct TwoSessions {
  KeyframeSet keyframes{std::vector<double>{0.2}};
  std::vector<TripInputs> trips{2};
  std::vector<Pose> truth;  // world_from_lidar per keyframe.
  Pose world_from_start1;   // The second session's frame.
};

TwoSessions SimulateTwoSessions() {
  TwoSessions out;
  const BoxScene scene = testing::MakeScene([] {
    std::vector<testing::SimBox> boxes = testing::GarageStructure();
    const std::vector<testing::SimBox> cars = testing::ParkedCars(false);
    boxes.insert(boxes.end(), cars.begin(), cars.end());
    return boxes;
  }());
  const Pose vehicle_from_lidar(Mat3::Identity(), Vec3(0.0, 0.0, 1.8));
  const testing::LoopRoute routes[2] = {testing::LoopRoute(0.0),
                                        testing::LoopRoute(30.0)};
  out.world_from_start1 = routes[1].PoseAt(0.0);
  for (int t = 0; t < 2; ++t) {
    const Pose frame = t == 0 ? Pose() : out.world_from_start1.Inverse();
    TripInputs& inputs = out.trips[static_cast<size_t>(t)];
    inputs.vehicle_from_lidar = vehicle_from_lidar;
    // One keyframe per metre around the loop.
    for (int k = 0; k * 0.25 * routes[t].speed() < routes[t].length(); ++k) {
      const double stamp = 0.25 * k;
      const Pose world_from_lidar = routes[t].PoseAt(stamp) * vehicle_from_lidar;
      Keyframe keyframe;
      keyframe.trip = t;
      keyframe.frame = k;
      keyframe.stamp = stamp;
      keyframe.odom = frame * world_from_lidar;
      out.keyframes.Add(keyframe,
                        testing::ScanFromPose(scene, world_from_lidar)
                            .points_sensor);
      out.truth.push_back(world_from_lidar);
      inputs.frame_stamps.push_back(stamp);
      inputs.frame_odometry.push_back(keyframe.odom);
    }
  }
  return out;
}

double AngleDegrees(const Pose& a, const Pose& b) {
  return RadToDeg(
      Eigen::AngleAxisd(a.rotation.transpose() * b.rotation).angle());
}

GR_TEST(BackendTest, AlignTripsPlacesASessionStartedInItsOwnFrame) {
  const TwoSessions sim = SimulateTwoSessions();
  std::vector<Anchor> anchors =
      MakeAnchors(sim.keyframes, sim.trips, BackendConfig());
  GR_ASSERT_TRUE(anchors.size() == 2 && anchors[0].weak && anchors[1].weak);
  std::vector<Pose> poses = sim.keyframes.OdomPoses();
  std::vector<std::string> lines;
  const std::vector<TripAlignment> alignments =
      AlignTrips(sim.keyframes, TripAlignConfig(), poses, anchors,
                 [&](const std::string& line) { lines.push_back(line); });
  // Trip 0 is the reference; trip 1 is placed.
  GR_ASSERT_TRUE(alignments.size() == 1);
  GR_EXPECT_EQ(alignments[0].trip, 1);
  GR_ASSERT_TRUE(alignments[0].aligned);
  GR_EXPECT_LT((alignments[0].correction.translation -
                sim.world_from_start1.translation)
                   .norm(),
               0.05);
  GR_EXPECT_LT(AngleDegrees(alignments[0].correction, sim.world_from_start1),
               0.3);
  for (int k : sim.keyframes.TripIndices(1)) {
    const size_t kk = static_cast<size_t>(k);
    GR_EXPECT_LT((poses[kk].translation - sim.truth[kk].translation).norm(),
                 0.1);
  }
  // The weak anchor moved with the trip and was tightened.
  GR_EXPECT_LT((anchors[1].pose.translation -
                sim.truth[static_cast<size_t>(anchors[1].keyframe)].translation)
                   .norm(),
               0.1);
  GR_EXPECT_NEAR(anchors[1].sigma_t, TripAlignConfig().sigma_t, 1e-12);
  GR_EXPECT_EQ(lines.size(), 2u);  // Reference + placement.
}

GR_TEST(BackendTest, AppearanceLoopsRegisterPairsWithoutAPoseGuess) {
  // Trip 1 is in the frame of its own start (no prior between the trips).
  // Appearance candidates: each of its keyframes paired with the trip 0
  // keyframe at the same place (true) or across the garage (false).
  const TwoSessions sim = SimulateTwoSessions();
  const std::vector<Pose> poses = sim.keyframes.OdomPoses();
  const std::vector<int> trip0 = sim.keyframes.TripIndices(0);
  std::vector<std::pair<int, int>> pairs;
  std::vector<bool> is_true;
  for (int i : sim.keyframes.TripIndices(1)) {
    if (i % 6 != 0) continue;
    int near = trip0.front(), far = trip0.front();
    double best = 1e9, farthest = 0.0;
    for (int j : trip0) {
      const double d = (sim.truth[static_cast<size_t>(i)].translation -
                        sim.truth[static_cast<size_t>(j)].translation)
                           .norm();
      if (d < best) best = d, near = j;
      if (d > farthest) farthest = d, far = j;
    }
    pairs.push_back({i, near});
    is_true.push_back(true);
    pairs.push_back({i, far});
    is_true.push_back(false);
  }
  int tested = 0;
  const std::vector<LoopConstraint> loops = VerifyAppearanceLoops(
      sim.keyframes, poses, pairs, LoopConfig(), AppearanceLoopConfig(),
      &tested);
  GR_EXPECT_GT(tested, static_cast<int>(pairs.size()));
  // In this symmetric test garage a pair from across the garage can fit
  // the mirrored pose; PCM (consistency between the loops of the two trips
  // through their odometry) removes those, as in the graph stage.
  const std::vector<LoopConstraint> kept =
      SelectConsistentLoops(loops, sim.keyframes, poses, LoopConfig());
  int true_found = 0, num_true = 0;
  for (bool t : is_true) num_true += t;
  for (const LoopConstraint& loop : kept) {
    const Pose truth = sim.truth[static_cast<size_t>(loop.j)].Inverse() *
                       sim.truth[static_cast<size_t>(loop.i)];
    // Every accepted loop is right, whichever pair it came from.
    GR_EXPECT_LT((loop.j_from_i.translation - truth.translation).norm(),
                 0.05);
    GR_EXPECT_LT(AngleDegrees(loop.j_from_i, truth), 0.3);
    for (size_t p = 0; p < pairs.size(); ++p) {
      if (is_true[p] && pairs[p] == std::make_pair(loop.i, loop.j)) {
        ++true_found;
      }
    }
  }
  std::printf("       appearance loops: %zu of %zu pairs verified, %zu kept "
              "by PCM (%d of %d true pairs), %d ICP runs\n",
              loops.size(), pairs.size(), kept.size(), true_found, num_true,
              tested);
  GR_EXPECT_GE(true_found, (8 * num_true) / 10);
}

GR_TEST(BackendTest, AlignTripsRefusesATripSeenOnOneOfTwoIdenticalLevels) {
  // Two copies of the test garage, 3.6 m apart. Trip 0 drives the loop on
  // both levels; trip 1 only on the upper one, in the frame of its start:
  // placing it on either level fits equally well.
  std::vector<testing::SimBox> boxes = testing::GarageStructure();
  const std::vector<testing::SimBox> cars = testing::ParkedCars(false);
  boxes.insert(boxes.end(), cars.begin(), cars.end());
  const size_t one_level = boxes.size();
  for (size_t b = 0; b < one_level; ++b) {
    testing::SimBox upper = boxes[b];
    upper.center.z() += 3.6;
    boxes.push_back(upper);
  }
  const BoxScene scene = testing::MakeScene(boxes);
  const Pose lift(Mat3::Identity(), Vec3(0.0, 0.0, 1.8));
  const Pose up(Mat3::Identity(), Vec3(0.0, 0.0, 3.6));
  KeyframeSet keyframes({0.2});
  const auto drive = [&](int trip, const testing::LoopRoute& route,
                         const Pose& world_from_route, const Pose& frame) {
    for (int k = 0; k * 0.25 * route.speed() < route.length(); ++k) {
      const Pose world_from_lidar =
          world_from_route * route.PoseAt(0.25 * k) * lift;
      Keyframe keyframe;
      keyframe.trip = trip;
      keyframe.odom = frame * world_from_lidar;
      keyframes.Add(keyframe,
                    testing::ScanFromPose(scene, world_from_lidar)
                        .points_sensor);
    }
  };
  drive(0, testing::LoopRoute(0.0), Pose(), Pose());
  drive(0, testing::LoopRoute(0.0), up, Pose());
  const testing::LoopRoute later(30.0);
  drive(1, later, up, (up * later.PoseAt(0.0)).Inverse());
  std::vector<Anchor> anchors(2);
  for (int t = 0; t < 2; ++t) {
    anchors[static_cast<size_t>(t)].keyframe = keyframes.TripIndices(t)[0];
    anchors[static_cast<size_t>(t)].weak = true;
  }
  std::vector<Pose> poses = keyframes.OdomPoses();
  const std::vector<Pose> before = poses;
  const std::vector<TripAlignment> alignments =
      AlignTrips(keyframes, TripAlignConfig(), poses, anchors);
  GR_ASSERT_TRUE(alignments.size() == 1);
  GR_EXPECT_TRUE(!alignments[0].aligned);
  GR_EXPECT_GT(alignments[0].runner_up, alignments[0].fitness - 0.03);
  // Nothing moved.
  for (size_t k = 0; k < poses.size(); ++k) {
    GR_EXPECT_LT((poses[k].translation - before[k].translation).norm(),
                 1e-12);
  }
}

GR_TEST(BackendTest, InterTripLoopsNeedTheAlignmentOfAnchorlessTrips) {
  const TwoSessions sim = SimulateTwoSessions();
  GraphStageConfig config;
  config.ba.enabled = false;
  const auto inter_trip = [&](const GraphStageResult& result) {
    return std::count_if(
        result.loops.begin(), result.loops.end(),
        [&](const LoopConstraint& loop) {
          return sim.keyframes[static_cast<size_t>(loop.i)].trip !=
                 sim.keyframes[static_cast<size_t>(loop.j)].trip;
        });
  };
  config.trip_align.enabled = false;
  const GraphStageResult without = RunGraphStage(sim.keyframes, sim.trips,
                                                 config);
  GR_EXPECT_EQ(inter_trip(without), 0);
  config.trip_align.enabled = true;
  const GraphStageResult with = RunGraphStage(sim.keyframes, sim.trips,
                                              config);
  GR_EXPECT_GT(inter_trip(with), 10);
  double worst = 0.0;
  for (size_t k = 0; k < sim.truth.size(); ++k) {
    worst = std::max(
        worst, (with.poses[k].translation - sim.truth[k].translation).norm());
  }
  GR_EXPECT_LT(worst, 0.05);
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
