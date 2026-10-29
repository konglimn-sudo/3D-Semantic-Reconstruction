#include "gr/testing/sim_garage.h"

#include <cmath>

#include "gr/sensors/imu_preintegration.h"

namespace gr::testing {
namespace {

// Static scans: kScanBeams beams from kLowestElevation over kElevationSpan
// (deg), kScanAzimuths azimuth steps.
constexpr int kScanBeams = 16;
constexpr int kScanAzimuths = 720;
constexpr double kLowestElevation = -15.0;
constexpr double kElevationSpan = 30.0;
// Swept LiDAR returns closer than this (m) are dropped.
constexpr double kMinReturnRange = 0.5;
constexpr double kDegreesPerTurn = 360.0;
// Proprioception: IMU at 200 Hz, wheel speed at 100 Hz; sample counts are
// rounded with kCountSlack.
constexpr double kImuPeriod = 0.005;   // s
constexpr double kWheelPeriod = 0.01;  // s
constexpr double kCountSlack = 1e-6;
// The loop route: the low straight along y = kLowStraightY from x =
// kRouteStartX, the half-turns around (kRouteTurnX | kRouteStartX,
// kTurnCentreY), the high straight along y = kHighStraightY.
constexpr double kRouteStartX = 2.0;     // m
constexpr double kRouteTurnX = 28.0;     // m
constexpr double kLowStraightY = 5.0;    // m
constexpr double kTurnCentreY = 10.0;    // m
constexpr double kHighStraightY = 15.0;  // m

}  // namespace

StaticScan ScanFromPose(const BoxScene& scene, const Pose& world_from_sensor,
                        double max_range) {
  RowMatX3d directions(kScanBeams * kScanAzimuths, kSpaceDim);
  RowMatX3d local(kScanBeams * kScanAzimuths, kSpaceDim);
  for (int beam = 0; beam < kScanBeams; ++beam) {
    const double elevation = DegToRad(
        kLowestElevation + kElevationSpan * beam / (kScanBeams - 1));
    for (int step = 0; step < kScanAzimuths; ++step) {
      const double azimuth = 2.0 * kPi * step / kScanAzimuths;
      const Vec3 direction(std::cos(elevation) * std::cos(azimuth),
                           std::cos(elevation) * std::sin(azimuth),
                           std::sin(elevation));
      local.row(beam * kScanAzimuths + step) = direction.transpose();
      directions.row(beam * kScanAzimuths + step) =
          (world_from_sensor.rotation * direction).transpose();
    }
  }
  RowMatX3d origin(1, kSpaceDim);
  origin.row(0) = world_from_sensor.translation.transpose();
  const RaycastHits hits = scene.Raycast(origin, directions, max_range);
  StaticScan scan;
  for (Eigen::Index ray = 0; ray < hits.distance.size(); ++ray) {
    if (hits.box_ids[ray] == RaycastHits::kMiss) continue;
    const Vec3 point = local.row(ray).transpose() * hits.distance[ray];
    scan.points_sensor.push_back(point);
    scan.points_world.push_back(world_from_sensor * point);
    scan.normals.push_back(hits.normals.row(ray).transpose());
    scan.box_ids.push_back(hits.box_ids[ray]);
  }
  return scan;
}

BoxScene CorridorScene() {
  // Pairs of columns every kColumnSpacing m from x = kFirstColumnX, at y =
  // -kColumnY and (shifted by kColumnStagger along x) at y = kColumnY.
  constexpr int kColumnPairs = 9;
  constexpr double kFirstColumnX = -4.0;
  constexpr double kColumnSpacing = 8.0;
  constexpr double kColumnStagger = 2.0;
  constexpr double kColumnY = 3.5;
  BoxScene scene;
  const Mat3 identity = Mat3::Identity();
  scene.AddBox(Vec3(30, 0, -0.15), Vec3(80, 12, 0.3), identity);   // Floor.
  scene.AddBox(Vec3(30, 0, 3.15), Vec3(80, 12, 0.3), identity);    // Ceiling.
  scene.AddBox(Vec3(30, -6.15, 1.5), Vec3(80, 0.3, 3), identity);  // Walls.
  scene.AddBox(Vec3(30, 6.15, 1.5), Vec3(80, 0.3, 3), identity);
  scene.AddBox(Vec3(-10.15, 0, 1.5), Vec3(0.3, 12, 3), identity);
  for (int pair = 0; pair < kColumnPairs; ++pair) {
    const double x = kFirstColumnX + kColumnSpacing * pair;
    scene.AddBox(Vec3(x, -kColumnY, 1.5), Vec3(0.5, 0.5, 3), identity);
    scene.AddBox(Vec3(x + kColumnStagger, kColumnY, 1.5), Vec3(0.5, 0.5, 3),
                 identity);
  }
  // A few small boxes break the symmetry.
  scene.AddBox(Vec3(5, 4.5, 0.4), Vec3(1.0, 1.0, 0.8), identity);
  scene.AddBox(Vec3(17, -4.8, 0.6), Vec3(2.0, 1.0, 1.2), identity);
  scene.AddBox(Vec3(29, 4.0, 0.3), Vec3(0.6, 2.0, 0.6), identity);
  return scene;
}

Pose CorridorSensorPose(double x, double yaw) {
  // The sensor weaves kWeaveAmplitude (m) sideways with kWeavePerMetre
  // (rad / m) along x, kSensorHeight (m) above the floor.
  constexpr double kWeaveAmplitude = 0.3;
  constexpr double kWeavePerMetre = 0.2;
  constexpr double kSensorHeight = 1.9;
  return Pose(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
              Vec3(x, kWeaveAmplitude * std::sin(kWeavePerMetre * x),
                   kSensorHeight));
}

std::vector<SimBox> GarageStructure() {
  std::vector<SimBox> boxes;
  boxes.push_back({Vec3(15, 10, -0.1), Vec3(46, 32, 0.2)});  // Floor slab.
  boxes.push_back({Vec3(15, 10, 3.1), Vec3(46, 32, 0.2)});   // Ceiling slab.
  boxes.push_back({Vec3(-8, 10, 1.5), Vec3(0.3, 32, 3)});    // Walls.
  boxes.push_back({Vec3(38, 10, 1.5), Vec3(0.3, 32, 3)});
  boxes.push_back({Vec3(15, -6, 1.5), Vec3(46, 0.3, 3)});
  boxes.push_back({Vec3(15, 26, 1.5), Vec3(46, 0.3, 3)});
  // The column grid.
  for (double x : {0.0, 8.0, 16.0, 24.0}) {
    for (double y : {0.0, 10.0, 20.0}) {
      boxes.push_back({Vec3(x, y, 1.5), Vec3(0.6, 0.6, 3.0)});
    }
  }
  // Irregular clutter so that the aisles are not perfectly self-similar.
  boxes.push_back({Vec3(-6.5, 3.0, 0.6), Vec3(1.2, 2.0, 1.2)});
  boxes.push_back({Vec3(36.5, 17.0, 1.0), Vec3(1.0, 3.0, 2.0)});
  boxes.push_back({Vec3(20.0, 25.2, 1.2), Vec3(5.0, 1.0, 2.4)});
  return boxes;
}

std::vector<SimBox> ParkedCars(bool with_extra) {
  // Cars of one size, standing on the floor.
  const Vec3 car_size(4.6, 1.9, 1.5);
  const double center_z = car_size.z() / 2.0;
  std::vector<SimBox> cars = {{Vec3(4.0, -2.6, center_z), car_size},
                              {Vec3(20.0, -2.6, center_z), car_size},
                              {Vec3(28.0, 22.6, center_z), car_size}};
  if (with_extra) cars.push_back({Vec3(12.0, 8.2, center_z), car_size});
  return cars;
}

BoxScene MakeScene(const std::vector<SimBox>& boxes) {
  BoxScene scene;
  for (const SimBox& box : boxes) {
    scene.AddBox(box.center, box.size, Mat3::Identity());
  }
  return scene;
}

LoopRoute::State LoopRoute::StateAt(double t) const {
  double distance = std::fmod(start_ + speed_ * t, length());
  if (distance < 0) distance += length();
  const double arc = kPi * kRadius;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double yaw_rate = 0.0;
  if (distance < kStraight) {
    x = kRouteStartX + distance;
    y = kLowStraightY;
  } else if (distance < kStraight + arc) {
    const double turned = (distance - kStraight) / kRadius;
    x = kRouteTurnX + kRadius * std::sin(turned);
    y = kTurnCentreY - kRadius * std::cos(turned);
    yaw = turned;
    yaw_rate = speed_ / kRadius;
  } else if (distance < 2.0 * kStraight + arc) {
    x = kRouteTurnX - (distance - kStraight - arc);
    y = kHighStraightY;
    yaw = kPi;
  } else {
    const double turned = (distance - 2.0 * kStraight - arc) / kRadius;
    x = kRouteStartX - kRadius * std::sin(turned);
    y = kTurnCentreY + kRadius * std::cos(turned);
    yaw = kPi + turned;
    yaw_rate = speed_ / kRadius;
  }
  return {Pose(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
               Vec3(x, y, 0.0)),
          yaw_rate};
}

Pose LoopRoute::PoseAt(double t) const { return StateAt(t).pose; }

double LoopRoute::YawRateAt(double t) const { return StateAt(t).yaw_rate; }

Sweep SimulateSweep(const BoxScene& scene, const LoopRoute& route,
                    const Pose& vehicle_from_lidar, double stamp,
                    const LidarModel& lidar, std::mt19937& rng) {
  return SimulateSweep(
      scene, [&route](double t) { return route.PoseAt(t); }, vehicle_from_lidar,
      stamp, lidar, rng);
}

Sweep SimulateSweep(const BoxScene& scene,
                    const std::function<Pose(double)>& world_from_vehicle,
                    const Pose& vehicle_from_lidar, double stamp,
                    const LidarModel& lidar, std::mt19937& rng) {
  const int num_azimuths =
      static_cast<int>(std::lround(kDegreesPerTurn / lidar.az_res));
  const int num_rays = num_azimuths * lidar.beams;
  RowMatX3d origins(num_rays, kSpaceDim);
  RowMatX3d directions(num_rays, kSpaceDim);
  std::vector<Vec3> local(static_cast<size_t>(num_rays));
  std::vector<double> ray_time(static_cast<size_t>(num_azimuths));
  for (int step = 0; step < num_azimuths; ++step) {
    const double time_offset =
        -lidar.sweep + lidar.sweep * (step + 1) / num_azimuths;
    ray_time[static_cast<size_t>(step)] = time_offset;
    const Pose world_from_lidar =
        world_from_vehicle(stamp + time_offset) * vehicle_from_lidar;
    const double azimuth = DegToRad(step * lidar.az_res);
    for (int beam = 0; beam < lidar.beams; ++beam) {
      const double elevation =
          DegToRad(lidar.el_min +
                   (lidar.el_max - lidar.el_min) * beam / (lidar.beams - 1));
      const Vec3 direction(std::cos(elevation) * std::cos(azimuth),
                           std::cos(elevation) * std::sin(azimuth),
                           std::sin(elevation));
      const int ray = step * lidar.beams + beam;
      local[static_cast<size_t>(ray)] = direction;
      origins.row(ray) = world_from_lidar.translation.transpose();
      directions.row(ray) = (world_from_lidar.rotation * direction).transpose();
    }
  }
  const RaycastHits hits = scene.Raycast(origins, directions, lidar.max_range);
  std::normal_distribution<double> noise(0.0, lidar.noise);
  Sweep sweep;
  for (int step = 0; step < num_azimuths; ++step) {
    for (int beam = 0; beam < lidar.beams; ++beam) {
      const int ray = step * lidar.beams + beam;
      if (hits.box_ids[ray] == RaycastHits::kMiss ||
          hits.distance[ray] < kMinReturnRange) {
        continue;
      }
      sweep.points.push_back(local[static_cast<size_t>(ray)] *
                             (hits.distance[ray] + noise(rng)));
      sweep.times.push_back(ray_time[static_cast<size_t>(step)]);
    }
  }
  return sweep;
}

Proprioception SimulateProprioception(const LoopRoute& route, double t0,
                                      double t1, const Vec6& imu_bias) {
  Proprioception out;
  const int num_imu =
      static_cast<int>(std::floor((t1 - t0) / kImuPeriod + kCountSlack));
  for (int sample = 0; sample <= num_imu; ++sample) {
    const double t = t0 + kImuPeriod * sample;
    const double yaw_rate = route.YawRateAt(t);
    out.imu.stamps.push_back(t);
    out.imu.gyro.push_back(Vec3(0.0, 0.0, yaw_rate) +
                           imu_bias.segment<kSpaceDim>(kGyroBiasOffset));
    // Centripetal acceleration plus the reaction to gravity.
    out.imu.accel.push_back(Vec3(0.0, route.speed() * yaw_rate, kGravity) +
                            imu_bias.segment<kSpaceDim>(kAccelBiasOffset));
  }
  const int num_wheel =
      static_cast<int>(std::floor((t1 - t0) / kWheelPeriod + kCountSlack));
  for (int sample = 0; sample <= num_wheel; ++sample) {
    out.wheel.stamps.push_back(t0 + kWheelPeriod * sample);
    out.wheel.speed.push_back(route.speed());
  }
  return out;
}

}  // namespace gr::testing
