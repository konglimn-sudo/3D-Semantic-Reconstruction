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

Pose VehicleFromLidar() { return Pose(Mat3::Identity(), Vec3(1.0, 0, 1.8)); }

Pose YawPose(double yaw_deg, const Vec3& translation) {
  return Pose(
      Eigen::AngleAxisd(DegToRad(yaw_deg), Vec3::UnitZ()).toRotationMatrix(),
      translation);
}

// Map of the permanent structure: static scans every metre around the loop.
Points StructureMap(const BoxScene& structure, const LoopRoute& route) {
  Points map;
  for (double s = 0.0; s < route.length(); s += 1.0) {
    const Pose sensor = route.PoseAt(s / route.speed()) * VehicleFromLidar();
    const Points world = ScanFromPose(structure, sensor).points_world;
    map.insert(map.end(), world.begin(), world.end());
  }
  return VoxelDownsample(map, 0.05);
}

struct Errors {
  int registered = 0;
  int tracked = 0;  // Sweeps after the first initialization.
  int first_tracked = -1;
  std::vector<double> position;  // Of the tracked sweeps (m).
  std::vector<double> yaw;       // deg
};

// Drives `laps` loops through `scene` (parked cars differ from the map) and
// feeds the localizer an odometry that drifts and lives in `odom_from_world`.
Errors Drive(Localizer& localizer, const BoxScene& scene,
             const LoopRoute& route, const Pose& odom_from_world,
             double laps) {
  std::mt19937 rng(7);
  const LidarModel lidar;
  Errors errors;
  const double duration = laps * route.length() / route.speed();
  int frame = 0;
  for (double t = 0.1; t <= duration; t += 0.1, ++frame) {
    const Pose truth = route.PoseAt(t) * VehicleFromLidar();
    // Odometry: drifts 1 % of the distance driven (along x) and 0.5 deg/min
    // in heading, on top of the unknown frame.
    const Pose drift = YawPose(0.5 * t / 60.0, Vec3(0.01 * route.speed() * t,
                                                    0.0, 0.0));
    const Pose odom = odom_from_world * drift * truth;
    const Sweep sweep =
        SimulateSweep(scene, route, VehicleFromLidar(), t, lidar, rng);
    const LocalizationResult result =
        localizer.Process(sweep.points, sweep.times, t, odom);
    if (result.status != LocalizationStatus::kTracking) continue;
    if (errors.first_tracked < 0) errors.first_tracked = frame;
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
  const LoopRoute route;
  const Points map = StructureMap(MakeScene(Structure()), route);
  Localizer localizer(map, LocalizerConfig());
  // Prior 0.6 m and 4 deg off (e.g. the entrance pose of the map).
  const Pose truth0 = route.PoseAt(0.1) * VehicleFromLidar();
  localizer.SetPrior(YawPose(4.0, Vec3(0.4, -0.45, 0.0)) * truth0);
  const Errors errors =
      Drive(localizer, MakeScene(StructureAndCars()), route, Pose(), 1.0);
  GR_EXPECT_EQ(errors.first_tracked, 0);
  GR_EXPECT_GE(errors.registered, 0.95 * errors.tracked);
  GR_EXPECT_LT(Median(errors.position), 0.02);
  GR_EXPECT_LT(Percentile(errors.position, 95.0), 0.05);
  GR_EXPECT_LT(Percentile(errors.yaw, 95.0), 0.3);
}

GR_TEST(LocalizerTest, PlacesItselfWithoutAPrior) {
  const LoopRoute route;
  const Points map = StructureMap(MakeScene(Structure()), route);
  LocalizerConfig config;
  Localizer localizer(map, config);
  // The odometry starts in its own frame, 70 deg and 120 m away.
  const Pose odom_from_world = YawPose(70.0, Vec3(120.0, -40.0, 0.0));
  const Errors errors = Drive(localizer, MakeScene(StructureAndCars()), route,
                              odom_from_world, 1.5);
  GR_EXPECT_GT(errors.first_tracked, 0);
  GR_EXPECT_GE(localizer.placement_attempts(), 1);
  // Placed within the loop.
  GR_EXPECT_GT(errors.tracked,
               0.3 * 1.5 * route.length() / route.speed() / 0.1);
  GR_EXPECT_GE(errors.registered, 0.95 * errors.tracked);
  GR_EXPECT_LT(Median(errors.position), 0.02);
  GR_EXPECT_LT(Percentile(errors.position, 95.0), 0.05);
}

GR_TEST(LocalizerTest, KeepsThePredictionAlongAFeaturelessCorridor) {
  // Floor, ceiling and two walls only: nothing fixes the position along x.
  BoxScene scene;
  const Mat3 identity = Mat3::Identity();
  scene.AddBox(Vec3(0, 0, -0.15), Vec3(400, 8, 0.3), identity);
  scene.AddBox(Vec3(0, 0, 3.15), Vec3(400, 8, 0.3), identity);
  scene.AddBox(Vec3(0, -4.15, 1.5), Vec3(400, 0.3, 3), identity);
  scene.AddBox(Vec3(0, 4.15, 1.5), Vec3(400, 0.3, 3), identity);
  Points map;
  for (double x = -40.0; x <= 40.0; x += 1.0) {
    const Points world =
        ScanFromPose(scene, YawPose(0.0, Vec3(x, 0.0, 1.8))).points_world;
    map.insert(map.end(), world.begin(), world.end());
  }
  Localizer localizer(VoxelDownsample(map, 0.05), LocalizerConfig());
  auto truth_at = [](double t) { return YawPose(0.0, Vec3(2.0 * t, 0, 0)); };
  localizer.SetPrior(truth_at(0.1) * VehicleFromLidar());
  std::mt19937 rng(3);
  int degenerate = 0;
  int registered = 0;
  double along = 0.0;
  double lateral = 0.0;
  for (double t = 0.1; t <= 10.0; t += 0.1) {
    const Pose truth = truth_at(t) * VehicleFromLidar();
    // 3 % too long along the corridor, 2 cm lateral offset that ICP fixes.
    const Pose odom(Mat3::Identity(),
                    truth.translation + Vec3(0.03 * 2.0 * t, 0.02, 0.0));
    const Sweep sweep = SimulateSweep(scene, truth_at, VehicleFromLidar(), t,
                                      LidarModel(), rng);
    const LocalizationResult result =
        localizer.Process(sweep.points, sweep.times, t, odom);
    if (!result.registered) continue;
    ++registered;
    if (result.degenerate_t > 0) ++degenerate;
    along = result.pose.translation.x() - truth.translation.x();
    lateral = std::max(lateral, std::abs(result.pose.translation.y() -
                                         truth.translation.y()));
    // The reported uncertainty is large along the corridor only.
    GR_EXPECT_GT(result.covariance(0, 0), 0.5);
    GR_EXPECT_LT(result.covariance(1, 1), 0.01);
  }
  GR_EXPECT_GT(registered, 90);
  GR_EXPECT_EQ(degenerate, registered);
  // Along x the odometry error stays (no snapping to arbitrary positions);
  // across the corridor it is corrected.
  GR_EXPECT_NEAR(along, 0.03 * 2.0 * 10.0, 0.05);
  GR_EXPECT_LT(lateral, 0.02);
}

}  // namespace
}  // namespace gr
