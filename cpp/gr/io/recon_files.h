// The work directory of a reconstruction, as recon/pipeline.py lays it out,
// so that the C++ chain (gr_recon) and the Python stages can continue each
// other's work:
//
//   config.used.yaml
//   trips/<trip>/odometry.{npz,tum}    LiDAR odometry, keyframe frames
//   trips/<trip>/kf/<frame>.npz        keyframe sweeps (odometry deskew):
//                                      points float32, intensity float16
//   trips/<trip>/kf_final/<frame>.npz  ... deskewed with the pose set
//   trips/<trip>/poses.{npz,tum}       frame poses after the back-end
//   graph/kf_poses.npz, loops.json, summary.json
//   pose_set/pose_set.json, <trip>.npz
//   map/point_map.npz, levels.json, floor_grids.npz
//   provenance.json

#ifndef GARAGE_RECON_GR_IO_RECON_FILES_H_
#define GARAGE_RECON_GR_IO_RECON_FILES_H_

#include <cstdint>
#include <string>
#include <vector>

#include "gr/backend/backend.h"
#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/io/value.h"
#include "gr/mapping/levels.h"

namespace gr {

// trips/<trip>/odometry.npz (odometry.TripOdometry).
struct TripOdometryFile {
  std::vector<double> stamps;     // LiDAR frame stamps (s).
  std::vector<Pose> lidar_poses;  // world <- lidar, one per frame.
  std::vector<int64_t> kf_frames;
};

// odometry.npz and odometry.tum (world <- vehicle).
void WriteTripOdometry(const std::string& trip_dir,
                       const TripOdometryFile& odometry,
                       const Pose& vehicle_from_lidar);
TripOdometryFile ReadTripOdometry(const std::string& trip_dir);

// A keyframe sweep as odometry.load_keyframe returns it: float32 points
// widened to double, float16 intensity widened to float.
struct SweepFile {
  Points points;
  std::vector<float> intensity;
};
void WriteSweep(const std::string& path, const Points& points,
                const std::vector<float>& intensity);
SweepFile ReadSweep(const std::string& path);
std::string SweepPath(const std::string& trip_dir, const std::string& sweeps,
                      int64_t frame);  // <trip_dir>/<sweeps>/<frame>.npz

// graph/kf_poses.npz.
struct KeyframeTable {
  std::vector<Pose> poses;  // world <- lidar after the back-end.
  std::vector<Pose> odom;
  std::vector<int64_t> trip;
  std::vector<int64_t> frame;
  std::vector<double> stamp;
  InertialEstimate inertial;  // Empty without IMU factors.
};
void WriteKeyframeTable(const std::string& path, const KeyframeTable& table);
KeyframeTable ReadKeyframeTable(const std::string& path);

// backend.loop_json of every loop.
Value LoopsValue(const std::vector<LoopConstraint>& loops);
// The fields of graph/summary.json the back-end fills (pipeline.stage_graph
// adds the trip ids and IMU bias means). `trip_ids` names the trips of the
// alignments.
Value GraphSummaryValue(const GraphStageResult& result, size_t keyframes,
                        const std::vector<std::string>& trip_ids);

// trips/<trip>/poses.{npz,tum} (trajectory.save_frame_poses).
void WriteFramePoses(const std::string& trip_dir,
                     const std::vector<double>& stamps,
                     const std::vector<Pose>& lidar,
                     const Pose& vehicle_from_lidar);
// (stamps, world <- lidar) of poses.npz.
void ReadFramePoses(const std::string& trip_dir, std::vector<double>* stamps,
                    std::vector<Pose>* lidar);

// One trip of the pose set (pose_set.TripPoses).
struct TripPoses {
  std::string trip_id;
  std::string dataset;             // The trip directory.
  std::vector<double> knot_stamps;
  std::vector<Pose> knot_poses;    // world <- vehicle.
  std::vector<double> bias_stamps;
  std::vector<Vec6> biases;        // [gyro, accel].
  Value rig;                       // schema.Rig.to_dict of rig.yaml.
};

// Rig.from_dict(rig.yaml).to_dict(): the canonical rig dictionary.
Value RigDict(const Value& rig_yaml);

struct PoseSetIds {
  std::string trajectory_id;
  std::string pose_set_id;
};
// The ids PoseSet computes; `imu` / `wheel` are the trips' raw imu.csv /
// wheel.csv tables (0 rows when absent).
PoseSetIds ComputePoseSetIds(const std::vector<TripPoses>& trips,
                             const std::vector<Eigen::MatrixXd>& imu,
                             const std::vector<Eigen::MatrixXd>& wheel);
void WritePoseSet(const std::string& work_dir,
                  const std::vector<TripPoses>& trips, const PoseSetIds& ids);
// Trips of pose_set/pose_set.json with their knots and biases.
std::vector<TripPoses> ReadPoseSet(const std::string& work_dir);

// provenance.record.
void RecordProvenance(const std::string& work_dir, const std::string& stage,
                      const PoseSetIds& ids, bool uses_cameras);

// map/point_map.npz (static_map.PointMap); empty optional fields are left
// out.
struct PointMapFile {
  Points xyz;
  std::vector<float> intensity;
  std::vector<int32_t> kf;
  std::vector<int16_t> trip;
  Points normals;  // Stored as float32.
  std::vector<float> planarity;
  std::vector<float> sigma;
  std::vector<uint16_t> observations;
  std::vector<uint8_t> dynamic;
  std::vector<uint8_t> dynamic_kind;
  std::vector<int16_t> level;
  std::vector<float> height;
  std::vector<uint8_t> on_ramp;
};
void WritePointMap(const std::string& path, const PointMapFile& map);

// map/levels.json and map/floor_grids.npz (levels.LevelModel.save).
void WriteLevels(const std::string& map_dir, const LevelModel& model);

// A 4x4 matrix as nested lists of floats (numpy tolist()).
Value MatrixValue(const Pose& pose);

void WriteTextFile(const std::string& path, const std::string& text);

}  // namespace gr

#endif  // GARAGE_RECON_GR_IO_RECON_FILES_H_
