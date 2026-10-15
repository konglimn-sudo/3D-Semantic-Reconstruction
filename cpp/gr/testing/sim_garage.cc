#include "gr/testing/sim_garage.h"

#include <cmath>

namespace gr::testing {

StaticScan ScanFromPose(const BoxScene& scene, const Pose& world_from_sensor,
                        double max_range) {
  constexpr int kBeams = 16;
  constexpr int kAzimuths = 720;
  RowMatX3d directions(kBeams * kAzimuths, 3);
  RowMatX3d local(kBeams * kAzimuths, 3);
  for (int b = 0; b < kBeams; ++b) {
    const double elevation = DegToRad(-15.0 + 30.0 * b / (kBeams - 1));
    for (int a = 0; a < kAzimuths; ++a) {
      const double azimuth = 2.0 * kPi * a / kAzimuths;
      const Vec3 d(std::cos(elevation) * std::cos(azimuth),
                   std::cos(elevation) * std::sin(azimuth),
                   std::sin(elevation));
      local.row(b * kAzimuths + a) = d.transpose();
      directions.row(b * kAzimuths + a) =
          (world_from_sensor.rotation * d).transpose();
    }
  }
  RowMatX3d origin(1, 3);
  origin.row(0) = world_from_sensor.translation.transpose();
  const RaycastHits hits = scene.Raycast(origin, directions, max_range);
  StaticScan scan;
  for (Eigen::Index i = 0; i < hits.distance.size(); ++i) {
    if (hits.box_ids[i] < 0) continue;
    const Vec3 p = local.row(i).transpose() * hits.distance[i];
    scan.points_sensor.push_back(p);
    scan.points_world.push_back(world_from_sensor * p);
    scan.normals.push_back(hits.normals.row(i).transpose());
    scan.box_ids.push_back(hits.box_ids[i]);
  }
  return scan;
}

BoxScene CorridorScene() {
  BoxScene scene;
  const Mat3 identity = Mat3::Identity();
  scene.AddBox(Vec3(30, 0, -0.15), Vec3(80, 12, 0.3), identity);   // Floor.
  scene.AddBox(Vec3(30, 0, 3.15), Vec3(80, 12, 0.3), identity);    // Ceiling.
  scene.AddBox(Vec3(30, -6.15, 1.5), Vec3(80, 0.3, 3), identity);  // Walls.
  scene.AddBox(Vec3(30, 6.15, 1.5), Vec3(80, 0.3, 3), identity);
  scene.AddBox(Vec3(-10.15, 0, 1.5), Vec3(0.3, 12, 3), identity);
  for (int k = 0; k < 9; ++k) {
    const double x = -4.0 + 8.0 * k;
    scene.AddBox(Vec3(x, -3.5, 1.5), Vec3(0.5, 0.5, 3), identity);
    scene.AddBox(Vec3(x + 2.0, 3.5, 1.5), Vec3(0.5, 0.5, 3), identity);
  }
  // A few small boxes break the symmetry.
  scene.AddBox(Vec3(5, 4.5, 0.4), Vec3(1.0, 1.0, 0.8), identity);
  scene.AddBox(Vec3(17, -4.8, 0.6), Vec3(2.0, 1.0, 1.2), identity);
  scene.AddBox(Vec3(29, 4.0, 0.3), Vec3(0.6, 2.0, 0.6), identity);
  return scene;
}

Pose CorridorSensorPose(double x, double yaw) {
  return Pose(Eigen::AngleAxisd(yaw, Vec3::UnitZ()).toRotationMatrix(),
              Vec3(x, 0.3 * std::sin(0.2 * x), 1.9));
}

std::vector<SimBox> GarageStructure() {
  std::vector<SimBox> boxes;
  boxes.push_back({Vec3(15, 10, -0.1), Vec3(46, 32, 0.2)});  // Floor slab.
  boxes.push_back({Vec3(15, 10, 3.1), Vec3(46, 32, 0.2)});   // Ceiling slab.
  boxes.push_back({Vec3(-8, 10, 1.5), Vec3(0.3, 32, 3)});
  boxes.push_back({Vec3(38, 10, 1.5), Vec3(0.3, 32, 3)});
  boxes.push_back({Vec3(15, -6, 1.5), Vec3(46, 0.3, 3)});
  boxes.push_back({Vec3(15, 26, 1.5), Vec3(46, 0.3, 3)});
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
  std::vector<SimBox> cars = {{Vec3(4.0, -2.6, 0.75), Vec3(4.6, 1.9, 1.5)},
                              {Vec3(20.0, -2.6, 0.75), Vec3(4.6, 1.9, 1.5)},
                              {Vec3(28.0, 22.6, 0.75), Vec3(4.6, 1.9, 1.5)}};
  if (with_extra) cars.push_back({Vec3(12.0, 8.2, 0.75), Vec3(4.6, 1.9, 1.5)});
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
  double s = std::fmod(start_ + speed_ * t, length());
  if (s < 0) s += length();
  const double arc = kPi * kRadius;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  double yaw_rate = 0.0;
  if (s < kStraight) {
    x = 2.0 + s;
    y = 5.0;
  } else if (s < kStraight + arc) {
    const double turned = (s - kStraight) / kRadius;
    x = 28.0 + kRadius * std::sin(turned);
    y = 10.0 - kRadius * std::cos(turned);
    yaw = turned;
    yaw_rate = speed_ / kRadius;
  } else if (s < 2.0 * kStraight + arc) {
    x = 28.0 - (s - kStraight - arc);
    y = 15.0;
    yaw = kPi;
  } else {
    const double turned = (s - 2.0 * kStraight - arc) / kRadius;
    x = 2.0 - kRadius * std::sin(turned);
    y = 10.0 + kRadius * std::cos(turned);
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
  const int num_azimuths = static_cast<int>(std::lround(360.0 / lidar.az_res));
  const int num_rays = num_azimuths * lidar.beams;
  RowMatX3d origins(num_rays, 3);
  RowMatX3d directions(num_rays, 3);
  std::vector<Vec3> local(static_cast<size_t>(num_rays));
  std::vector<double> ray_time(static_cast<size_t>(num_azimuths));
  for (int a = 0; a < num_azimuths; ++a) {
    const double tau = -lidar.sweep + lidar.sweep * (a + 1) / num_azimuths;
    ray_time[static_cast<size_t>(a)] = tau;
    const Pose world_from_lidar =
        world_from_vehicle(stamp + tau) * vehicle_from_lidar;
    const double azimuth = DegToRad(a * lidar.az_res);
    for (int b = 0; b < lidar.beams; ++b) {
      const double elevation = DegToRad(
          lidar.el_min + (lidar.el_max - lidar.el_min) * b / (lidar.beams - 1));
      const Vec3 d(std::cos(elevation) * std::cos(azimuth),
                   std::cos(elevation) * std::sin(azimuth),
                   std::sin(elevation));
      const int r = a * lidar.beams + b;
      local[static_cast<size_t>(r)] = d;
      origins.row(r) = world_from_lidar.translation.transpose();
      directions.row(r) = (world_from_lidar.rotation * d).transpose();
    }
  }
  const RaycastHits hits = scene.Raycast(origins, directions, lidar.max_range);
  std::normal_distribution<double> noise(0.0, lidar.noise);
  Sweep sweep;
  for (int a = 0; a < num_azimuths; ++a) {
    for (int b = 0; b < lidar.beams; ++b) {
      const int r = a * lidar.beams + b;
      if (hits.box_ids[r] < 0 || hits.distance[r] < 0.5) continue;
      sweep.points.push_back(local[static_cast<size_t>(r)] *
                             (hits.distance[r] + noise(rng)));
      sweep.times.push_back(ray_time[static_cast<size_t>(a)]);
    }
  }
  return sweep;
}

Proprioception SimulateProprioception(const LoopRoute& route, double t0,
                                      double t1, const Vec6& imu_bias) {
  Proprioception out;
  const int num_imu = static_cast<int>(std::floor((t1 - t0) / 0.005 + 1e-6));
  for (int i = 0; i <= num_imu; ++i) {
    const double t = t0 + 0.005 * i;
    const double yaw_rate = route.YawRateAt(t);
    out.imu.stamps.push_back(t);
    out.imu.gyro.push_back(Vec3(0.0, 0.0, yaw_rate) + imu_bias.head<3>());
    // Centripetal acceleration plus the reaction to gravity.
    out.imu.accel.push_back(Vec3(0.0, route.speed() * yaw_rate, 9.81) +
                            imu_bias.tail<3>());
  }
  const int num_wheel = static_cast<int>(std::floor((t1 - t0) / 0.01 + 1e-6));
  for (int i = 0; i <= num_wheel; ++i) {
    out.wheel.stamps.push_back(t0 + 0.01 * i);
    out.wheel.speed.push_back(route.speed());
  }
  return out;
}

}  // namespace gr::testing
