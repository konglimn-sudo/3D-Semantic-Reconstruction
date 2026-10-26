// Scene and sensor simulation for the C++ tests: a corridor for registration
// tests, and a flat garage level (walls, column grid, parked cars) with a
// closed driving loop, a rotating LiDAR with motion distortion, IMU and wheel
// speed for the pipeline tests.

#ifndef GARAGE_RECON_GR_TESTING_SIM_GARAGE_H_
#define GARAGE_RECON_GR_TESTING_SIM_GARAGE_H_

#include <functional>
#include <random>
#include <vector>

#include "gr/core/pose.h"
#include "gr/sensors/imu.h"
#include "gr/sim/sim.h"

namespace gr::testing {

// ---------------------------------------------------------------------------
// Static scans
// ---------------------------------------------------------------------------

struct StaticScan {
  Points points_sensor;
  Points points_world;
  Points normals;  // World frame.
  std::vector<int> box_ids;
};

// 16 beams in [-15, 15] deg and 720 azimuth steps from a static sensor.
StaticScan ScanFromPose(const BoxScene& scene, const Pose& world_from_sensor,
                        double max_range = 50.0);

// 60 m corridor with columns: enough structure to constrain all 6 DoF.
BoxScene CorridorScene();
// Sensor pose along the corridor at x with a gentle lateral weave.
Pose CorridorSensorPose(double x, double yaw);

// ---------------------------------------------------------------------------
// Test garage (x in [-8, 38], y in [-6, 26], floor at z = 0, ceiling at 3 m)
// ---------------------------------------------------------------------------

struct SimBox {
  Vec3 center;
  Vec3 size;
};

std::vector<SimBox> GarageStructure();
// Parked cars; `with_extra` adds one at (12, 8.2) that only some sessions see.
std::vector<SimBox> ParkedCars(bool with_extra);
BoxScene MakeScene(const std::vector<SimBox>& boxes);

// Closed loop: +x along y = 5, left half-turn around (28, 10), -x along
// y = 15, left half-turn around (2, 10); arc length from (2, 5).
class LoopRoute {
 public:
  static constexpr double kRadius = 5.0;
  static constexpr double kStraight = 26.0;

  explicit LoopRoute(double start = 0.0, double speed = 4.0)
      : start_(start), speed_(speed) {}

  double length() const { return 2.0 * kStraight + 2.0 * kPi * kRadius; }
  double speed() const { return speed_; }
  // world_from_vehicle at time t.
  Pose PoseAt(double t) const;
  double YawRateAt(double t) const;

 private:
  struct State {
    Pose pose;
    double yaw_rate;
  };
  State StateAt(double t) const;

  double start_;
  double speed_;
};

struct LidarModel {
  int beams = 16;
  double el_min = -15.0;  // deg
  double el_max = 15.0;   // deg
  double az_res = 1.0;    // deg
  double sweep = 0.1;     // s
  double max_range = 50.0;
  double noise = 0.01;  // m
};

struct Sweep {
  Points points;              // Sensor frame at each measurement time.
  std::vector<double> times;  // Relative to the stamp (<= 0).
};

// One motion-distorted sweep ending at `stamp`.
Sweep SimulateSweep(const BoxScene& scene, const LoopRoute& route,
                    const Pose& vehicle_from_lidar, double stamp,
                    const LidarModel& lidar, std::mt19937& rng);
// The same for any trajectory (world_from_vehicle at time t).
Sweep SimulateSweep(const BoxScene& scene,
                    const std::function<Pose(double)>& world_from_vehicle,
                    const Pose& vehicle_from_lidar, double stamp,
                    const LidarModel& lidar, std::mt19937& rng);

struct Proprioception {
  ImuSamples imu;      // 200 Hz, vehicle frame.
  WheelSamples wheel;  // 100 Hz.
};

// Flat ground, constant speed; `imu_bias` ([gyro, accel]) is added to every
// IMU sample.
Proprioception SimulateProprioception(const LoopRoute& route, double t0,
                                      double t1,
                                      const Vec6& imu_bias = Vec6::Zero());

}  // namespace gr::testing

#endif  // GARAGE_RECON_GR_TESTING_SIM_GARAGE_H_
