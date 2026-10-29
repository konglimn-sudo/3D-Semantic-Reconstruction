#include "gr/backend/backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "gr/backend/loop_closure.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::GarageStructure;
using ::gr::testing::LoopRoute;
using ::gr::testing::MakeScene;
using ::gr::testing::ParkedCars;
using ::gr::testing::ScanFromPose;
using ::gr::testing::SimBox;

constexpr double kExactTolerance = 1e-12;
constexpr double kNumericTolerance = 1e-9;
// Sweep resolution the keyframe sets keep (m).
constexpr double kSweepVoxel = 0.2;
// The simulated vehicles carry the LiDAR this high above their origin (m).
constexpr double kLidarHeight = 1.8;
// The simulated drives take a keyframe every quarter second (s).
constexpr double kKeyframePeriod = 0.25;
// The second session starts this far along the loop (m).
constexpr double kSecondSessionStart = 30.0;
// Accuracy of placed trips and verified loops.
constexpr double kMaxPlacementError = 0.05;  // m
constexpr double kMaxAngleError = 0.3;       // deg
constexpr double kMaxKeyframeError = 0.1;    // m

Pose PoseXyzYaw(double x, double y, double z, double yaw) {
  return Pose(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
              Vec3(x, y, z));
}

// Angle between the pose's z axis and the world's.
double TiltDegrees(const Pose& pose) {
  return RadToDeg(std::acos(std::clamp(pose.rotation(2, 2), -1.0, 1.0)));
}

Pose LiftToLidar() {
  return Pose(Mat3::Identity(), Vec3(0.0, 0.0, kLidarHeight));
}

GR_TEST(BackendTest, GravityFactorsKeepADriftingChainLevel) {
  // Odometry that pitches by 0.1 deg per keyframe (as on a ramp exit) tilts
  // the whole chain; level-gravity measurements keep it horizontal.
  constexpr int kKeyframes = 60;
  constexpr double kPitchPerKeyframe = 0.1;  // deg
  constexpr double kStep = 1.0;              // m per keyframe
  constexpr double kGravitySigma = 0.2;      // deg
  constexpr double kAnchorSigma = 0.01;      // m, deg
  constexpr double kMinDriftedTilt = 5.0;    // deg
  constexpr double kMaxLevelTilt = 0.5;      // deg
  constexpr double kMaxHeightRatio = 0.25;
  KeyframeSet keyframes({kSweepVoxel});
  Pose pose;
  const Pose step(Eigen::AngleAxisd(DegToRad(kPitchPerKeyframe), Vec3::UnitY())
                      .toRotationMatrix(),
                  Vec3(kStep, 0, 0));
  std::vector<GravityMeasurement> gravity;
  for (int index = 0; index < kKeyframes; ++index) {
    Keyframe keyframe;
    keyframe.odom = pose;
    keyframes.Add(keyframe, {});
    GravityMeasurement measurement;
    measurement.index = index;
    measurement.up = Vec3::UnitZ();  // The true vehicle stays level.
    measurement.sigma_rad = DegToRad(kGravitySigma);
    gravity.push_back(measurement);
    pose = pose * step;
  }
  Anchor anchor;
  anchor.keyframe = 0;
  anchor.sigma_t = kAnchorSigma;
  anchor.sigma_r = kAnchorSigma;
  const BackendConfig config;
  const std::vector<Pose> without =
      OptimizeGraph(keyframes, {}, {anchor}, {}, config);
  GraphSummary summary;
  const std::vector<Pose> with = OptimizeGraph(
      keyframes, {}, {anchor}, gravity, config, nullptr, nullptr, &summary);
  GR_EXPECT_LT(summary.final_cost, summary.initial_cost);
  GR_EXPECT_GT(TiltDegrees(without.back()), kMinDriftedTilt);
  GR_EXPECT_LT(TiltDegrees(with.back()), kMaxLevelTilt);
  GR_EXPECT_LT(std::abs(with.back().translation.z()),
               kMaxHeightRatio * std::abs(without.back().translation.z()));
}

GR_TEST(BackendTest, AnchorsUseInsPoseGnssLeverArmOrAWeakPrior) {
  // Three trips of three keyframes, one second apart, heading along +y.
  constexpr int kTrips = 3;
  constexpr int kKeyframesPerTrip = 3;
  constexpr double kFirstStamp = 10.0;  // s
  constexpr double kHeading = kPi / 2;
  KeyframeSet keyframes({kSweepVoxel});
  for (int trip = 0; trip < kTrips; ++trip) {
    for (int index = 0; index < kKeyframesPerTrip; ++index) {
      Keyframe keyframe;
      keyframe.trip = trip;
      keyframe.stamp = kFirstStamp + index;
      keyframe.odom = PoseXyzYaw(index, 0, 0, kHeading);
      keyframes.Add(keyframe, {});
    }
  }
  std::vector<TripInputs> trips(kTrips);
  for (TripInputs& trip : trips) {
    trip.vehicle_from_lidar = Pose(Mat3::Identity(), Vec3(1.0, 0.0, 1.8));
    trip.frame_stamps = {10.0, 11.0, 12.0};
    trip.frame_odometry = {PoseXyzYaw(0, 0, 0, kHeading),
                           PoseXyzYaw(1, 0, 0, kHeading),
                           PoseXyzYaw(2, 0, 0, kHeading)};
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
  GR_EXPECT_LT((anchors[0].pose.translation - Vec3(6, 6, 1.8)).norm(),
               kExactTolerance);
  GR_EXPECT_TRUE(anchors[1].type == Anchor::Type::kPosition);
  GR_EXPECT_EQ(anchors[1].keyframe, 4);
  GR_EXPECT_LT((anchors[1].lever - Vec3(-1.0, 0, -1.8)).norm(),
               kExactTolerance);
  // The fix at t = 11.2 moves to the keyframe at t = 11 with the odometry
  // increment (-0.2 m in x).
  GR_EXPECT_LT((anchors[1].position - Vec3(99.8, 200, 0)).norm(),
               kNumericTolerance);
  GR_EXPECT_NEAR(anchors[1].sigma, 0.03, kExactTolerance);
  GR_EXPECT_TRUE(anchors[2].type == Anchor::Type::kPose);
  GR_EXPECT_EQ(anchors[2].keyframe, 6);
  GR_EXPECT_NEAR(anchors[2].sigma_t, 50.0, kExactTolerance);
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
  GR_EXPECT_NEAR(control.sigma, 0.004, kExactTolerance);
  // In the keyframe frame (at x = 2) the target is 2 m ahead, 0.4 m back.
  GR_EXPECT_LT((keyframes[8].odom * control.lever - Vec3(1.6, 2.0, 0.0)).norm(),
               kNumericTolerance);
}

// Two sessions around the test garage loop without any anchor; the second
// starts kSecondSessionStart further along the loop and its odometry is in
// the frame of its own start (as a handheld session's is).
struct TwoSessions {
  KeyframeSet keyframes{std::vector<double>{kSweepVoxel}};
  std::vector<TripInputs> trips{2};
  std::vector<Pose> truth;  // world_from_lidar per keyframe.
  Pose world_from_start1;   // The second session's frame.
};

TwoSessions SimulateTwoSessions() {
  TwoSessions sim;
  const BoxScene scene = MakeScene([] {
    std::vector<SimBox> boxes = GarageStructure();
    const std::vector<SimBox> cars = ParkedCars(false);
    boxes.insert(boxes.end(), cars.begin(), cars.end());
    return boxes;
  }());
  const Pose vehicle_from_lidar = LiftToLidar();
  const LoopRoute routes[2] = {LoopRoute(0.0), LoopRoute(kSecondSessionStart)};
  sim.world_from_start1 = routes[1].PoseAt(0.0);
  for (int trip = 0; trip < 2; ++trip) {
    const Pose frame = trip == 0 ? Pose() : sim.world_from_start1.Inverse();
    const LoopRoute& route = routes[trip];
    TripInputs& inputs = sim.trips[static_cast<size_t>(trip)];
    inputs.vehicle_from_lidar = vehicle_from_lidar;
    // One keyframe per metre around the loop.
    for (int index = 0;
         index * kKeyframePeriod * route.speed() < route.length(); ++index) {
      const double stamp = kKeyframePeriod * index;
      const Pose world_from_lidar = route.PoseAt(stamp) * vehicle_from_lidar;
      Keyframe keyframe;
      keyframe.trip = trip;
      keyframe.frame = index;
      keyframe.stamp = stamp;
      keyframe.odom = frame * world_from_lidar;
      sim.keyframes.Add(keyframe,
                        ScanFromPose(scene, world_from_lidar).points_sensor);
      sim.truth.push_back(world_from_lidar);
      inputs.frame_stamps.push_back(stamp);
      inputs.frame_odometry.push_back(keyframe.odom);
    }
  }
  return sim;
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
      AlignTrips(sim.keyframes, TripAlignConfig(), &poses, &anchors,
                 [&](const std::string& line) { lines.push_back(line); });
  // Trip 0 is the reference; trip 1 is placed.
  GR_ASSERT_TRUE(alignments.size() == 1);
  GR_EXPECT_EQ(alignments[0].trip, 1);
  GR_ASSERT_TRUE(alignments[0].aligned);
  GR_EXPECT_LT(
      (alignments[0].correction.translation - sim.world_from_start1.translation)
          .norm(),
      kMaxPlacementError);
  GR_EXPECT_LT(AngleDegrees(alignments[0].correction, sim.world_from_start1),
               kMaxAngleError);
  for (int index : sim.keyframes.TripIndices(1)) {
    const size_t keyframe = static_cast<size_t>(index);
    GR_EXPECT_LT(
        (poses[keyframe].translation - sim.truth[keyframe].translation).norm(),
        kMaxKeyframeError);
  }
  // The weak anchor moved with the trip and was tightened.
  GR_EXPECT_LT((anchors[1].pose.translation -
                sim.truth[static_cast<size_t>(anchors[1].keyframe)].translation)
                   .norm(),
               kMaxKeyframeError);
  GR_EXPECT_NEAR(anchors[1].sigma_t, TripAlignConfig().sigma_t,
                 kExactTolerance);
  GR_EXPECT_EQ(lines.size(), 2u);  // Reference + placement.
}

GR_TEST(BackendTest, AppearanceLoopsRegisterPairsWithoutAPoseGuess) {
  // Trip 1 is in the frame of its own start (no prior between the trips).
  // Appearance candidates: every sixth of its keyframes paired with the trip
  // 0 keyframe at the same place (true) or across the garage (false).
  constexpr int kQueryEvery = 6;
  constexpr double kMinTrueFound = 0.8;  // Of the true pairs.
  const TwoSessions sim = SimulateTwoSessions();
  const std::vector<Pose> poses = sim.keyframes.OdomPoses();
  const std::vector<int> trip0 = sim.keyframes.TripIndices(0);
  const auto position = [&](int keyframe) -> const Vec3& {
    return sim.truth[static_cast<size_t>(keyframe)].translation;
  };
  std::vector<std::pair<int, int>> pairs;
  std::vector<bool> is_true;
  for (int query : sim.keyframes.TripIndices(1)) {
    if (query % kQueryEvery != 0) continue;
    int nearest = trip0.front();
    int farthest = trip0.front();
    double nearest_distance = std::numeric_limits<double>::infinity();
    double farthest_distance = 0.0;
    for (int candidate : trip0) {
      const double distance = (position(query) - position(candidate)).norm();
      if (distance < nearest_distance) {
        nearest_distance = distance;
        nearest = candidate;
      }
      if (distance > farthest_distance) {
        farthest_distance = distance;
        farthest = candidate;
      }
    }
    pairs.push_back({query, nearest});
    is_true.push_back(true);
    pairs.push_back({query, farthest});
    is_true.push_back(false);
  }
  int tested = 0;
  const std::vector<LoopConstraint> loops =
      VerifyAppearanceLoops(sim.keyframes, poses, pairs, LoopConfig(),
                            AppearanceLoopConfig(), &tested);
  GR_EXPECT_GT(tested, static_cast<int>(pairs.size()));
  // In this symmetric test garage a pair from across the garage can fit
  // the mirrored pose; PCM (consistency between the loops of the two trips
  // through their odometry) removes those, as in the graph stage.
  const std::vector<LoopConstraint> kept =
      SelectConsistentLoops(loops, sim.keyframes, poses, LoopConfig());
  int true_found = 0;
  const int num_true =
      static_cast<int>(std::count(is_true.begin(), is_true.end(), true));
  for (const LoopConstraint& loop : kept) {
    const Pose truth = sim.truth[static_cast<size_t>(loop.j)].Inverse() *
                       sim.truth[static_cast<size_t>(loop.i)];
    // Every accepted loop is right, whichever pair it came from.
    GR_EXPECT_LT((loop.j_from_i.translation - truth.translation).norm(),
                 kMaxPlacementError);
    GR_EXPECT_LT(AngleDegrees(loop.j_from_i, truth), kMaxAngleError);
    for (size_t pair = 0; pair < pairs.size(); ++pair) {
      if (is_true[pair] && pairs[pair] == std::make_pair(loop.i, loop.j)) {
        ++true_found;
      }
    }
  }
  std::printf(
      "       appearance loops: %zu of %zu pairs verified, %zu kept "
      "by PCM (%d of %d true pairs), %d ICP runs\n",
      loops.size(), pairs.size(), kept.size(), true_found, num_true, tested);
  GR_EXPECT_GE(true_found, static_cast<int>(kMinTrueFound * num_true));
}

GR_TEST(BackendTest, AlignTripsRefusesATripSeenOnOneOfTwoIdenticalLevels) {
  // Two copies of the test garage, one level apart. Trip 0 drives the loop
  // on both levels; trip 1 only on the upper one, in the frame of its start:
  // placing it on either level fits equally well.
  constexpr double kLevelHeight = 3.6;  // m
  std::vector<SimBox> boxes = GarageStructure();
  const std::vector<SimBox> cars = ParkedCars(false);
  boxes.insert(boxes.end(), cars.begin(), cars.end());
  const size_t one_level = boxes.size();
  for (size_t box = 0; box < one_level; ++box) {
    SimBox upper = boxes[box];
    upper.center.z() += kLevelHeight;
    boxes.push_back(upper);
  }
  const BoxScene scene = MakeScene(boxes);
  const Pose lift = LiftToLidar();
  const Pose up(Mat3::Identity(), Vec3(0.0, 0.0, kLevelHeight));
  KeyframeSet keyframes({kSweepVoxel});
  const auto drive = [&](int trip, const LoopRoute& route,
                         const Pose& world_from_route, const Pose& frame) {
    for (int index = 0;
         index * kKeyframePeriod * route.speed() < route.length(); ++index) {
      const Pose world_from_lidar =
          world_from_route * route.PoseAt(kKeyframePeriod * index) * lift;
      Keyframe keyframe;
      keyframe.trip = trip;
      keyframe.odom = frame * world_from_lidar;
      keyframes.Add(keyframe,
                    ScanFromPose(scene, world_from_lidar).points_sensor);
    }
  };
  drive(0, LoopRoute(0.0), Pose(), Pose());
  drive(0, LoopRoute(0.0), up, Pose());
  const LoopRoute later(kSecondSessionStart);
  drive(1, later, up, (up * later.PoseAt(0.0)).Inverse());
  std::vector<Anchor> anchors(2);
  for (int trip = 0; trip < 2; ++trip) {
    anchors[static_cast<size_t>(trip)].keyframe =
        keyframes.TripIndices(trip)[0];
    anchors[static_cast<size_t>(trip)].weak = true;
  }
  std::vector<Pose> poses = keyframes.OdomPoses();
  const std::vector<Pose> before = poses;
  const TripAlignConfig config;
  const std::vector<TripAlignment> alignments =
      AlignTrips(keyframes, config, &poses, &anchors);
  GR_ASSERT_TRUE(alignments.size() == 1);
  GR_EXPECT_TRUE(!alignments[0].aligned);
  GR_EXPECT_GT(alignments[0].runner_up,
               alignments[0].fitness - config.min_fitness_gap);
  // Nothing moved.
  for (size_t keyframe = 0; keyframe < poses.size(); ++keyframe) {
    GR_EXPECT_LT(
        (poses[keyframe].translation - before[keyframe].translation).norm(),
        kExactTolerance);
  }
}

GR_TEST(BackendTest, InterTripLoopsNeedTheAlignmentOfAnchorlessTrips) {
  constexpr int kMinInterTripLoops = 10;
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
  const GraphStageResult without =
      RunGraphStage(sim.keyframes, sim.trips, config);
  GR_EXPECT_EQ(inter_trip(without), 0);
  config.trip_align.enabled = true;
  const GraphStageResult with = RunGraphStage(sim.keyframes, sim.trips, config);
  GR_EXPECT_GT(inter_trip(with), kMinInterTripLoops);
  double worst = 0.0;
  for (size_t keyframe = 0; keyframe < sim.truth.size(); ++keyframe) {
    worst = std::max(worst, (with.poses[keyframe].translation -
                             sim.truth[keyframe].translation)
                                .norm());
  }
  GR_EXPECT_LT(worst, kMaxPlacementError);
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
  const std::vector<Pose> frame_poses =
      CorrectFramePoses(kf_stamps, kf_odom, kf_poses, frame_stamps, frame_odom);
  GR_EXPECT_LT((frame_poses[0].translation - Vec3(0, 1, 0)).norm(),
               kExactTolerance);
  GR_EXPECT_LT((frame_poses[1].translation - Vec3(5, 2, 0)).norm(),
               kExactTolerance);
  GR_EXPECT_LT((frame_poses[2].translation - Vec3(10, 3, 0)).norm(),
               kExactTolerance);
  // Clamped to the last correction.
  GR_EXPECT_LT((frame_poses[3].translation - Vec3(20, 3, 0)).norm(),
               kExactTolerance);
}

}  // namespace
}  // namespace gr
