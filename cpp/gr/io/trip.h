// A trip directory (one drive through the garage), as data/dataset.py
// reads it:
//
//   <trip>/meta.yaml                    trip_id, garage_id, optional
//                                       initial_pose (4x4 world <- vehicle)
//                                       and initial_pose_time (s)
//   <trip>/rig.yaml                     sensors (Rig below)
//   <trip>/lidar/<name>/<stamp_ns>.npz  points (N, 3) float32, or points_q
//                                       int16 * scale; intensity; time (s
//                                       relative to the stamp, <= 0)
//   <trip>/cameras/<name>/<stamp_ns>.jpg
//   <trip>/imu.csv                      t, gx, gy, gz, ax, ay, az (IMU frame)
//   <trip>/wheel.csv                    t, speed
//   <trip>/gnss.csv                     t, x, y, z, std
//   <trip>/control_points.csv           t, px, py, pz, x, y, z(, std)

#ifndef GARAGE_RECON_GR_IO_TRIP_H_
#define GARAGE_RECON_GR_IO_TRIP_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/io/value.h"
#include "gr/sensors/imu.h"

namespace gr {

// rig.yaml's sweep_duration (s) where it gives none.
inline constexpr double kDefaultSweepDuration = 0.1;

struct LidarCalibration {
  std::string name;
  Pose vehicle_from_lidar;
  double sweep_duration = kDefaultSweepDuration;  // s
};

struct CameraCalibration {
  std::string name;
  std::string model;  // pinhole, kannala_brandt or double_sphere.
  int width = 0;
  int height = 0;
  std::vector<double> params;
  Pose vehicle_from_camera;  // OpenCV camera axes.
  // s from the exposure of one image row to the next.
  double rolling_shutter_line_time = 0.0;
  // s from the image stamp to the exposure of row 0 (data/schema.py).
  double time_offset = 0.0;
};

// rig.yaml (data/schema.py).
struct Rig {
  std::vector<LidarCalibration> lidars;  // The first is the primary one.
  std::vector<CameraCalibration> cameras;
  Pose vehicle_from_imu;
  double wheel_speed_scale = 1.0;

  const LidarCalibration& primary_lidar() const;
  static Rig FromValue(const Value& value);
};

// A 4x4 matrix given as nested lists (or its 16 numbers row by row).
Pose PoseFromValue(const Value& value);

struct LidarSweep {
  int index = 0;
  double stamp = 0.0;  // s, end of the sweep.
  Points points;       // Sensor frame, non-finite points dropped.
  std::vector<float> intensity;
  std::vector<double> time;  // s relative to the stamp.
};

class TripReader {
 public:
  // Throws std::runtime_error if `root` has no rig.yaml.
  explicit TripReader(const std::string& root);

  const std::string& root() const { return root_; }
  std::string trip_id() const;  // meta trip_id, else the directory name.
  const Rig& rig() const { return rig_; }
  const Value& meta() const { return meta_; }
  std::optional<Pose> initial_pose() const;
  // Stamp of the initial pose; nullopt means the first LiDAR stamp.
  std::optional<double> initial_pose_time() const;

  // Sweeps of the primary LiDAR, by stamp.
  size_t size() const { return stamps_ns_.size(); }
  const std::vector<int64_t>& lidar_stamps_ns() const { return stamps_ns_; }
  double lidar_stamp(size_t index) const;
  LidarSweep LoadLidar(size_t index) const;

  // CSV tables (0 rows when the file is absent).
  Eigen::MatrixXd Imu() const;
  Eigen::MatrixXd Wheel() const;
  Eigen::MatrixXd Gnss() const;
  // Surveyed targets seen in the sweeps (dataset.control_points): rows
  // [t, px, py, pz (LiDAR frame), x, y, z (world)(, std)].
  Eigen::MatrixXd ControlPoints() const;
  // In the vehicle axes / with the speed scale, as dataset.imu_samples and
  // wheel_samples.
  ImuSamples ImuSamplesInVehicle() const;
  WheelSamples WheelSamplesScaled() const;
  // Wheel + gyro dead reckoning (odometry.motion_prior): empty without IMU.
  MotionPrior BuildMotionPrior() const;
  // The INS pose at the entrance moved to the first LiDAR stamp with the
  // motion prior (odometry.entrance_pose); nullopt without one.
  std::optional<Pose> EntrancePose(const MotionPrior& prior) const;

 private:
  Eigen::MatrixXd Table(const std::string& name) const;

  std::string root_;
  Value meta_;
  Rig rig_;
  std::vector<int64_t> stamps_ns_;
};

// Stamps (the file names) of the files in `directory` with one of the
// suffixes, ascending; empty if the directory does not exist.
std::vector<int64_t> StampedFiles(const std::string& directory,
                                  const std::vector<std::string>& suffixes);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_TRIP_H_
