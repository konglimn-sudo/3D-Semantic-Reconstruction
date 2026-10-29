#include "gr/nav/localization.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "gr/core/numeric.h"
#include "gr/core/voxel.h"
#include "gr/testing/sim_garage.h"
#include "gr/testing/test.h"

namespace gr {
namespace {

using ::gr::testing::GarageStructure;
using ::gr::testing::LidarModel;
using ::gr::testing::LoopRoute;
using ::gr::testing::MakeScene;
using ::gr::testing::ParkedCars;
using ::gr::testing::ScanFromPose;
using ::gr::testing::SimBox;
using ::gr::testing::SimulateSweep;
using ::gr::testing::Sweep;

// The LiDAR is mounted kLidarForward ahead of and kLidarHeight above the
// vehicle origin.
constexpr double kLidarForward = 1.0;  // m
constexpr double kLidarHeight = 1.8;   // m
// The sweeps come every kFrameInterval (s).
constexpr double kFrameInterval = 0.1;
// Maps are voxelized at kMapVoxel (m) from static scans every kMapSpacing.
constexpr double kMapVoxel = 0.05;
constexpr double kMapSpacing = 1.0;  // m
constexpr double kSecondsPerMinute = 60.0;
// Localization quality of the tracked sweeps.
constexpr double kMinRegisteredShare = 0.95;
constexpr double kMaxMedianPositionError = 0.02;  // m
constexpr double kMaxP95PositionError = 0.05;     // m
constexpr double kErrorPercentile = 95.0;

Pose VehicleFromLidar() {
  return Pose(Mat3::Identity(), Vec3(kLidarForward, 0, kLidarHeight));
}

Pose YawPose(double yaw_deg, const Vec3& translation) {
  return Pose(
      Eigen::AngleAxisd(DegToRad(yaw_deg), Vec3::UnitZ()).toRotationMatrix(),
      translation);
}

// Map of the permanent structure: static scans every kMapSpacing around the
// loop.
Points StructureMap(const BoxScene& structure, const LoopRoute& route) {
  Points map;
  for (double along = 0.0; along < route.length(); along += kMapSpacing) {
    const Pose sensor =
        route.PoseAt(along / route.speed()) * VehicleFromLidar();
    const Points world = ScanFromPose(structure, sensor).points_world;
    map.insert(map.end(), world.begin(), world.end());
  }
  return VoxelDownsample(map, kMapVoxel);
}

struct Errors {
  static constexpr int kNeverTracked = -1;
  int registered = 0;
  int tracked = 0;  // Sweeps after the first initialization.
  int first_tracked = kNeverTracked;
  std::vector<double> position;  // Of the tracked sweeps (m).
  std::vector<double> yaw;       // deg
};

// Drives `laps` loops through `scene` (parked cars differ from the map) and
// feeds the localizer an odometry that drifts and lives in `odom_from_world`.
Errors Drive(Localizer& localizer, const BoxScene& scene,
             const LoopRoute& route, const Pose& odom_from_world,
             double laps) {
  constexpr unsigned kSeed = 7;
  // Odometry drift: a share of the distance driven (along x) and a heading
  // rate.
  constexpr double kDistanceDrift = 0.01;
  constexpr double kHeadingDrift = 0.5;  // deg / min
  std::mt19937 rng(kSeed);
  const LidarModel lidar;
  Errors errors;
  const double duration = laps * route.length() / route.speed();
  int frame = 0;
  for (double time = kFrameInterval; time <= duration;
       time += kFrameInterval, ++frame) {
    const Pose truth = route.PoseAt(time) * VehicleFromLidar();
    // The odometry drifts on top of the unknown frame.
    const Pose drift =
        YawPose(kHeadingDrift * time / kSecondsPerMinute,
                Vec3(kDistanceDrift * route.speed() * time, 0.0, 0.0));
    const Pose odom = odom_from_world * drift * truth;
    const Sweep sweep =
        SimulateSweep(scene, route, VehicleFromLidar(), time, lidar, rng);
    const LocalizationResult result =
        localizer.Process(sweep.points, sweep.times, time, odom);
    if (result.status != LocalizationStatus::kTracking) continue;
    if (errors.first_tracked == Errors::kNeverTracked) {
      errors.first_tracked = frame;
    }
    ++errors.tracked;
    if (result.registered) ++errors.registered;
    errors.position.push_back(
        (result.pose.translation - truth.translation).norm());
    errors.yaw.push_back(std::abs(
        RadToDeg(WrapAngle(YawOf(result.pose.rotation) -
                           YawOf(truth.rotation)))));
  }
  return errors;
}

std::vector<SimBox> Structure() { return GarageStructure(); }

std::vector<SimBox> StructureAndCars() {
  std::vector<SimBox> boxes = GarageStructure();
  const std::vector<SimBox> cars = ParkedCars(true);
  boxes.insert(boxes.end(), cars.begin(), cars.end());
  return boxes;
}

GR_TEST(LocalizerTest, TracksFromAPriorAndRemovesTheOdometryDrift) {
  // The prior is 0.6 m and 4 deg off (e.g. the entrance pose of the map).
  constexpr double kPriorYaw = 4.0;  // deg
  constexpr double kMaxP95YawError = 0.3;  // deg
  constexpr double kLaps = 1.0;
  const LoopRoute route;
  const Points map = StructureMap(MakeScene(Structure()), route);
  Localizer localizer(map, LocalizerConfig());
  const Pose first_truth = route.PoseAt(kFrameInterval) * VehicleFromLidar();
  localizer.SetPrior(YawPose(kPriorYaw, Vec3(0.4, -0.45, 0.0)) * first_truth);
  const Errors errors =
      Drive(localizer, MakeScene(StructureAndCars()), route, Pose(), kLaps);
  GR_EXPECT_EQ(errors.first_tracked, 0);
  GR_EXPECT_GE(errors.registered, kMinRegisteredShare * errors.tracked);
  GR_EXPECT_LT(Median(errors.position), kMaxMedianPositionError);
  GR_EXPECT_LT(Percentile(errors.position, kErrorPercentile),
               kMaxP95PositionError);
  GR_EXPECT_LT(Percentile(errors.yaw, kErrorPercentile), kMaxP95YawError);
}

GR_TEST(LocalizerTest, PlacesItselfWithoutAPrior) {
  // The odometry starts in its own frame, 70 deg and 120 m away.
  constexpr double kOdomYaw = 70.0;  // deg
  constexpr double kLaps = 1.5;
  // It places itself within the first lap: tracked for at least this share
  // of the sweeps.
  constexpr double kMinTrackedShare = 0.3;
  const LoopRoute route;
  const Points map = StructureMap(MakeScene(Structure()), route);
  LocalizerConfig config;
  Localizer localizer(map, config);
  const Pose odom_from_world = YawPose(kOdomYaw, Vec3(120.0, -40.0, 0.0));
  const Errors errors = Drive(localizer, MakeScene(StructureAndCars()), route,
                              odom_from_world, kLaps);
  GR_EXPECT_GT(errors.first_tracked, 0);
  GR_EXPECT_GE(localizer.placement_attempts(), 1);
  GR_EXPECT_GT(errors.tracked, kMinTrackedShare * kLaps * route.length() /
                                   route.speed() / kFrameInterval);
  GR_EXPECT_GE(errors.registered, kMinRegisteredShare * errors.tracked);
  GR_EXPECT_LT(Median(errors.position), kMaxMedianPositionError);
  GR_EXPECT_LT(Percentile(errors.position, kErrorPercentile),
               kMaxP95PositionError);
}

GR_TEST(LocalizerTest, KeepsThePredictionAlongAFeaturelessCorridor) {
  // Floor, ceiling and two walls only: nothing fixes the position along x.
  constexpr double kSpeed = 2.0;          // m/s
  constexpr double kDuration = 10.0;      // s
  constexpr double kMapHalfLength = 40.0;  // m
  constexpr unsigned kSeed = 3;
  // The odometry is 3 % too long along the corridor and 2 cm off across it
  // (which ICP fixes).
  constexpr double kOdomScaleError = 0.03;
  constexpr double kOdomLateral = 0.02;   // m
  constexpr double kMinAlongVariance = 0.5;     // m^2
  constexpr double kMaxLateralVariance = 0.01;  // m^2
  constexpr int kMinRegistered = 90;
  constexpr double kAlongTolerance = 0.05;  // m
  constexpr double kMaxLateralError = 0.02;  // m
  BoxScene scene;
  const Mat3 identity = Mat3::Identity();
  scene.AddBox(Vec3(0, 0, -0.15), Vec3(400, 8, 0.3), identity);
  scene.AddBox(Vec3(0, 0, 3.15), Vec3(400, 8, 0.3), identity);
  scene.AddBox(Vec3(0, -4.15, 1.5), Vec3(400, 0.3, 3), identity);
  scene.AddBox(Vec3(0, 4.15, 1.5), Vec3(400, 0.3, 3), identity);
  Points map;
  for (double x = -kMapHalfLength; x <= kMapHalfLength; x += kMapSpacing) {
    const Points world =
        ScanFromPose(scene, YawPose(0.0, Vec3(x, 0.0, kLidarHeight)))
            .points_world;
    map.insert(map.end(), world.begin(), world.end());
  }
  Localizer localizer(VoxelDownsample(map, kMapVoxel), LocalizerConfig());
  const auto truth_at = [](double time) {
    return YawPose(0.0, Vec3(kSpeed * time, 0, 0));
  };
  localizer.SetPrior(truth_at(kFrameInterval) * VehicleFromLidar());
  std::mt19937 rng(kSeed);
  int degenerate = 0;
  int registered = 0;
  double along = 0.0;
  double lateral = 0.0;
  for (double time = kFrameInterval; time <= kDuration;
       time += kFrameInterval) {
    const Pose truth = truth_at(time) * VehicleFromLidar();
    const Pose odom(Mat3::Identity(),
                    truth.translation + Vec3(kOdomScaleError * kSpeed * time,
                                             kOdomLateral, 0.0));
    const Sweep sweep = SimulateSweep(scene, truth_at, VehicleFromLidar(),
                                      time, LidarModel(), rng);
    const LocalizationResult result =
        localizer.Process(sweep.points, sweep.times, time, odom);
    if (!result.registered) continue;
    ++registered;
    if (result.degenerate_t > 0) ++degenerate;
    along = result.pose.translation.x() - truth.translation.x();
    lateral = std::max(lateral, std::abs(result.pose.translation.y() -
                                         truth.translation.y()));
    // The reported uncertainty is large along the corridor only.
    GR_EXPECT_GT(result.covariance(0, 0), kMinAlongVariance);
    GR_EXPECT_LT(result.covariance(1, 1), kMaxLateralVariance);
  }
  GR_EXPECT_GT(registered, kMinRegistered);
  GR_EXPECT_EQ(degenerate, registered);
  // Along x the odometry error stays (no snapping to arbitrary positions);
  // across the corridor it is corrected.
  GR_EXPECT_NEAR(along, kOdomScaleError * kSpeed * kDuration,
                 kAlongTolerance);
  GR_EXPECT_LT(lateral, kMaxLateralError);
}

}  // namespace
}  // namespace gr
