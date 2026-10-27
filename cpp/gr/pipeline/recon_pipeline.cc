#include "gr/pipeline/recon_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "gr/backend/backend.h"
#include "gr/backend/keyframes.h"
#include "gr/core/numeric.h"
#include "gr/core/voxel.h"
#include "gr/io/config.h"
#include "gr/io/recon_files.h"
#include "gr/io/tables.h"
#include "gr/mapping/levels.h"
#include "gr/mapping/static_map.h"
#include "gr/odometry/odometry.h"
#include "gr/odometry/trajectory.h"

namespace gr {
namespace {

namespace fs = std::filesystem;

constexpr char kOdometrySweeps[] = "kf";
constexpr char kFinalSweeps[] = "kf_final";

// Python's repr of a number (log lines print config values with it).
std::string Repr(double value) { return ToJson(Value::Number(value), -1); }

double ConfigDouble(const Value& value, const std::string& name) {
  if (!value.is_number()) throw std::runtime_error(name + ": number expected");
  return value.AsDouble();
}

// Median distance from `query` (every 10th point) to its nearest point of
// `reference`, ignoring distances of 0.5 m or more
// (trajectory._median_shift); NaN without such pairs.
double MedianShift(const Points& reference, const Points& query) {
  constexpr double kCell = 0.5;
  if (reference.empty() || query.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> cells;
  for (size_t k = 0; k < reference.size(); ++k) {
    cells[VoxelOf(reference[k], 1.0 / kCell)].push_back(static_cast<int>(k));
  }
  std::vector<double> distances;
  for (size_t q = 0; q < query.size(); q += 10) {
    const VoxelKey key = VoxelOf(query[q], 1.0 / kCell);
    double best = kCell;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const auto it =
              cells.find(VoxelKey{key.x + dx, key.y + dy, key.z + dz});
          if (it == cells.end()) continue;
          for (int k : it->second) {
            best = std::min(best, (reference[static_cast<size_t>(k)] -
                                   query[q])
                                      .norm());
          }
        }
      }
    }
    if (best < kCell) distances.push_back(best);
  }
  return Median(distances);
}

// vehicle <- primary LiDAR of a pose-set rig.
Pose RigLidar(const Value& rig) {
  const Value& lidar = rig["lidars"].members().front().second;
  return PoseFromValue(lidar["T_vehicle_lidar"]);
}

// pose_set.trajectory: knots, the IMU with the back-end's gyro biases, the
// wheel speed.
ContinuousTrajectory TripTrajectory(const TripPoses& poses,
                                    const TripReader& trip) {
  std::vector<Vec3> gyro;
  for (const Vec6& bias : poses.biases) gyro.push_back(bias.head<3>());
  return ContinuousTrajectory(poses.knot_stamps, poses.knot_poses,
                              trip.ImuSamplesInVehicle(), poses.bias_stamps,
                              gyro, trip.WheelSamplesScaled());
}

}  // namespace

ReconPipeline::ReconPipeline(const std::vector<std::string>& trip_dirs,
                             std::string work_dir, Value config, Logger log)
    : work_(std::move(work_dir)), config_(std::move(config)),
      log_(std::move(log)) {
  for (const std::string& dir : trip_dirs) trips_.emplace_back(dir);
  fs::create_directories(work_);
  WriteTextFile((fs::path(work_) / "config.used.yaml").string(),
                ToYaml(config_));
}

const std::vector<std::string>& ReconPipeline::StageNames() {
  static const std::vector<std::string> names = {"odometry", "graph",
                                                 "trajectory", "map"};
  return names;
}

void ReconPipeline::Run(const std::vector<std::string>& stages) {
  for (const std::string& stage : stages) {
    const auto start = std::chrono::steady_clock::now();
    if (stage == "odometry") {
      StageOdometry();
    } else if (stage == "graph") {
      StageGraph();
    } else if (stage == "trajectory") {
      StageTrajectory();
    } else if (stage == "map") {
      StageMap();
    } else {
      throw std::runtime_error("unknown stage " + stage);
    }
    const double seconds = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    LogLine(log_,
            StringPrintf("== stage %s done in %.1fs", stage.c_str(), seconds));
  }
}

std::string ReconPipeline::TripDir(size_t trip) const {
  return (fs::path(work_) / "trips" / trips_[trip].trip_id()).string();
}

std::vector<ReconPipeline::KeyframeEntry> ReconPipeline::Keyframes(
    bool final) const {
  const bool use_final =
      final && !trips_.empty() &&
      fs::is_directory(fs::path(TripDir(0)) / kFinalSweeps);
  const std::string sweeps = use_final ? kFinalSweeps : kOdometrySweeps;
  std::vector<KeyframeEntry> out;
  for (size_t t = 0; t < trips_.size(); ++t) {
    const TripOdometryFile odometry = ReadTripOdometry(TripDir(t));
    for (int64_t frame : odometry.kf_frames) {
      KeyframeEntry entry;
      entry.trip = static_cast<int>(t);
      entry.frame = frame;
      entry.stamp = odometry.stamps.at(static_cast<size_t>(frame));
      entry.odom = odometry.lidar_poses.at(static_cast<size_t>(frame));
      entry.path = SweepPath(TripDir(t), sweeps, frame);
      out.push_back(std::move(entry));
    }
  }
  return out;
}

void ReconPipeline::StageOdometry() {
  const OdometryConfig odometry_config = OdometryConfigFrom(config_);
  const KeyframeConfig keyframe_config = KeyframeConfigFrom(config_);
  const Value& external = config_["odometry"]["external_trajectory"];
  for (size_t t = 0; t < trips_.size(); ++t) {
    const TripReader& trip = trips_[t];
    const Pose vehicle_from_lidar =
        trip.rig().primary_lidar().vehicle_from_lidar;
    const MotionPrior prior = trip.BuildMotionPrior();
    const std::optional<Pose> entrance = trip.EntrancePose(prior);
    const size_t n = trip.size();
    std::vector<double> stamps(n);
    for (size_t k = 0; k < n; ++k) stamps[k] = trip.lidar_stamp(k);
    std::vector<Pose> external_poses;
    if (external.is_string() && !external.AsString().empty()) {
      // str.format(trip=<trip directory>).
      std::string path = external.AsString();
      for (size_t at = path.find("{trip}"); at != std::string::npos;
           at = path.find("{trip}", at + trip.root().size())) {
        path.replace(at, 6, trip.root());
      }
      const Trajectory given = ReadTum(path);
      external_poses = InterpolatePoses(given.stamps, given.poses, stamps);
      for (Pose& pose : external_poses) pose = pose * vehicle_from_lidar;
      LogLine(log_, "[odometry] " + trip.trip_id() +
                        ": using external trajectory " + path);
    }
    OdometryPipeline pipeline(odometry_config, keyframe_config,
                              vehicle_from_lidar, entrance.value_or(Pose()),
                              prior);
    const std::string dir = TripDir(t);
    fs::create_directories(fs::path(dir) / kOdometrySweeps);
    for (size_t k = 0; k < n; ++k) {
      const LidarSweep sweep = trip.LoadLidar(k);
      std::optional<Pose> given;
      if (!external_poses.empty()) given = external_poses[k];
      const OdometryFrame frame =
          pipeline.Process(sweep.points, sweep.intensity, sweep.time,
                           sweep.stamp, k + 1 == n, given);
      if (frame.is_keyframe) {
        WriteSweep(SweepPath(dir, kOdometrySweeps, static_cast<int64_t>(k)),
                   frame.points, frame.intensity);
      }
    }
    TripOdometryFile odometry;
    odometry.stamps = stamps;
    odometry.lidar_poses = pipeline.poses();
    for (int frame : pipeline.keyframes()) {
      odometry.kf_frames.push_back(frame);
    }
    WriteTripOdometry(dir, odometry, vehicle_from_lidar);
    double path = 0.0;
    for (size_t k = 1; k < odometry.lidar_poses.size(); ++k) {
      path += (odometry.lidar_poses[k].translation -
               odometry.lidar_poses[k - 1].translation)
                  .norm();
    }
    LogLine(log_,
            StringPrintf("[odometry] %s: %zu frames, %zu keyframes, path "
                         "%.1f m%s%s%s",
                         trip.trip_id().c_str(), n, odometry.kf_frames.size(),
                         path,
                         pipeline.uses_prior() ? " (wheel/IMU prior)" : "",
                         pipeline.uses_gyro_prior() ? " (gyro prior)" : "",
                         pipeline.uses_lio() ? " (LiDAR-inertial)" : ""));
  }
}

void ReconPipeline::StageGraph() {
  const GraphStageConfig config = GraphStageConfigFrom(config_);
  const std::vector<KeyframeEntry> entries = Keyframes(false);
  KeyframeSet keyframes(config.RequiredVoxels());
  for (const KeyframeEntry& entry : entries) {
    Keyframe keyframe;
    keyframe.trip = entry.trip;
    keyframe.frame = static_cast<int>(entry.frame);
    keyframe.stamp = entry.stamp;
    keyframe.odom = entry.odom;
    keyframes.Add(keyframe, ReadSweep(entry.path).points);
  }
  std::vector<TripInputs> inputs;
  std::vector<TripOdometryFile> odometries;
  std::vector<std::string> trip_ids;
  for (size_t t = 0; t < trips_.size(); ++t) {
    const TripReader& trip = trips_[t];
    odometries.push_back(ReadTripOdometry(TripDir(t)));
    trip_ids.push_back(trip.trip_id());
    TripInputs input;
    input.vehicle_from_lidar = trip.rig().primary_lidar().vehicle_from_lidar;
    input.initial_pose = trip.EntrancePose(trip.BuildMotionPrior());
    input.gnss = trip.Gnss();
    input.control_points = trip.ControlPoints();
    input.frame_stamps = odometries.back().stamps;
    input.frame_odometry = odometries.back().lidar_poses;
    input.imu = trip.ImuSamplesInVehicle();
    input.wheel = trip.WheelSamplesScaled();
    inputs.push_back(std::move(input));
  }
  for (const auto& [i, j] : loop_pairs_) {
    const int size = static_cast<int>(keyframes.size());
    if (i < 0 || j < 0 || i >= size || j >= size) {
      throw std::runtime_error("loop pair keyframe index out of range");
    }
  }
  const GraphStageResult result =
      RunGraphStage(keyframes, inputs, config, log_, loop_pairs_);

  KeyframeTable table;
  table.poses = result.poses;
  for (const KeyframeEntry& entry : entries) {
    table.odom.push_back(entry.odom);
    table.trip.push_back(entry.trip);
    table.frame.push_back(entry.frame);
    table.stamp.push_back(entry.stamp);
  }
  table.inertial = result.inertial;
  const fs::path out = fs::path(work_) / "graph";
  WriteKeyframeTable((out / "kf_poses.npz").string(), table);
  WriteTextFile((out / "loops.json").string(),
                ToJson(LoopsValue(result.loops), -1));
  Value summary = GraphSummaryValue(result, keyframes.size(), trip_ids);
  const std::vector<uint8_t>& valid = result.inertial.valid;
  const bool any_valid =
      std::any_of(valid.begin(), valid.end(), [](uint8_t v) { return v; });
  for (size_t t = 0; t < trips_.size() && any_valid; ++t) {
    Vec6 sum = Vec6::Zero();
    int count = 0;
    for (size_t k = 0; k < valid.size(); ++k) {
      if (!valid[k] || entries[k].trip != static_cast<int>(t)) continue;
      sum += result.inertial.biases[k];
      ++count;
    }
    if (count == 0) continue;
    const Vec6 mean = sum / count;
    // numpy round(4): rint(x * 1e4) / 1e4.
    const auto round4 = [](double x) { return std::nearbyint(x * 1e4) / 1e4; };
    Value gyro = Value::List();
    Value accel = Value::List();
    for (int k = 0; k < 3; ++k) {
      gyro.Append(Value::Number(round4(mean[k] * 180.0 / kPi)));
      accel.Append(Value::Number(round4(mean[3 + k])));
    }
    Value bias = Value::Map();
    bias.Set("gyro_deg_s", std::move(gyro));
    bias.Set("accel_m_s2", std::move(accel));
    summary.Set("imu_bias_" + trip_ids[t], std::move(bias));
  }
  WriteTextFile((out / "summary.json").string(), ToJson(summary, 2));

  for (size_t t = 0; t < trips_.size(); ++t) {
    std::vector<double> stamps;
    std::vector<Pose> odom;
    std::vector<Pose> poses;
    for (size_t k = 0; k < entries.size(); ++k) {
      if (entries[k].trip != static_cast<int>(t)) continue;
      stamps.push_back(entries[k].stamp);
      odom.push_back(entries[k].odom);
      poses.push_back(result.poses[k]);
    }
    const TripOdometryFile& odometry = odometries[t];
    const std::vector<Pose> lidar = CorrectFramePoses(
        stamps, odom, poses, odometry.stamps, odometry.lidar_poses);
    WriteFramePoses(TripDir(t), odometry.stamps, lidar,
                    trips_[t].rig().primary_lidar().vehicle_from_lidar);
  }
}

void ReconPipeline::StageTrajectory() {
  const KeyframeTable table =
      ReadKeyframeTable((fs::path(work_) / "graph" / "kf_poses.npz").string());
  const bool with_bias = table.inertial.valid.size() == table.trip.size();
  std::vector<TripPoses> pose_set;
  std::vector<Eigen::MatrixXd> imu;
  std::vector<Eigen::MatrixXd> wheel;
  for (size_t t = 0; t < trips_.size(); ++t) {
    const TripReader& trip = trips_[t];
    TripPoses poses;
    poses.trip_id = trip.trip_id();
    poses.dataset = trip.root();
    poses.rig = RigDict(ReadYamlFile((fs::path(trip.root()) / "rig.yaml")
                                         .string()));
    std::vector<Pose> lidar;
    ReadFramePoses(TripDir(t), &poses.knot_stamps, &lidar);
    const Pose lidar_from_vehicle =
        trip.rig().primary_lidar().vehicle_from_lidar.Inverse();
    for (const Pose& pose : lidar) {
      poses.knot_poses.push_back(pose * lidar_from_vehicle);
    }
    std::vector<size_t> mine;
    for (size_t k = 0; with_bias && k < table.trip.size(); ++k) {
      if (table.trip[k] == static_cast<int64_t>(t) && table.inertial.valid[k]) {
        mine.push_back(k);
      }
    }
    std::stable_sort(mine.begin(), mine.end(), [&](size_t a, size_t b) {
      return table.stamp[a] < table.stamp[b];
    });
    for (size_t k : mine) {
      poses.bias_stamps.push_back(table.stamp[k]);
      poses.biases.push_back(table.inertial.biases[k]);
    }
    pose_set.push_back(std::move(poses));
    imu.push_back(trip.Imu());
    wheel.push_back(trip.Wheel());
  }
  const PoseSetIds ids = ComputePoseSetIds(pose_set, imu, wheel);
  WritePoseSet(work_, pose_set, ids);

  const double min_range =
      ConfigDouble(config_["odometry"]["min_range"], "odometry.min_range");
  const double max_range =
      ConfigDouble(config_["odometry"]["max_range"], "odometry.max_range");
  const double voxel =
      ConfigDouble(config_["keyframes"]["voxel"], "keyframes.voxel");
  int written = 0;
  std::vector<double> shifts;
  for (size_t t = 0; t < trips_.size(); ++t) {
    const TripReader& trip = trips_[t];
    const ContinuousTrajectory trajectory = TripTrajectory(pose_set[t], trip);
    const Pose vehicle_from_lidar =
        trip.rig().primary_lidar().vehicle_from_lidar;
    const TripOdometryFile odometry = ReadTripOdometry(TripDir(t));
    for (int64_t frame : odometry.kf_frames) {
      const LidarSweep sweep = trip.LoadLidar(static_cast<size_t>(frame));
      const SweepSamples samples = DeskewKeyframeSweep(
          sweep.points, sweep.intensity, sweep.time, sweep.stamp, trajectory,
          vehicle_from_lidar, min_range, max_range, voxel);
      const std::string path = SweepPath(TripDir(t), kFinalSweeps, frame);
      WriteSweep(path, samples.points, samples.intensity);
      if (++written % 10 == 1) {  // Shift against the odometry deskew.
        const SweepFile old =
            ReadSweep(SweepPath(TripDir(t), kOdometrySweeps, frame));
        shifts.push_back(MedianShift(old.points, samples.points));
      }
    }
    LogLine(log_,
            StringPrintf("[trajectory] %s: %zu keyframe sweeps deskewed (gyro "
                         "%.0f%%, wheel %.0f%% of knot intervals)",
                         trip.trip_id().c_str(), odometry.kf_frames.size(),
                         100.0 * trajectory.gyro_coverage(),
                         100.0 * trajectory.wheel_coverage()));
  }
  shifts.erase(std::remove_if(shifts.begin(), shifts.end(),
                              [](double s) { return !std::isfinite(s); }),
               shifts.end());
  LogLine(log_, "[trajectory] pose set " + ids.pose_set_id + ": " +
                    std::to_string(written) +
                    " sweeps re-deskewed, median point shift " +
                    (shifts.empty() ? "None" : Repr(Median(shifts) * 100.0)) +
                    " cm");
  RecordProvenance(work_, "recon.trajectory", ids, false);
}

void ReconPipeline::StageMap() {
  const Value& map_config = config_["map"];
  const double voxel = ConfigDouble(map_config["voxel"], "map.voxel");
  const double max_range =
      ConfigDouble(map_config["max_range"], "map.max_range");
  const double normal_radius =
      ConfigDouble(map_config["normal_radius"], "map.normal_radius");
  const MapFusionConfig fusion = MapFusionConfigFrom(config_);

  const std::vector<TripPoses> pose_set = ReadPoseSet(work_);
  if (pose_set.size() != trips_.size()) {
    throw std::runtime_error("the pose set does not match the trips");
  }
  std::vector<Eigen::MatrixXd> imu;
  std::vector<Eigen::MatrixXd> wheel;
  std::vector<ContinuousTrajectory> trajectories;
  for (size_t t = 0; t < trips_.size(); ++t) {
    if (pose_set[t].trip_id != trips_[t].trip_id()) {
      throw std::runtime_error("the pose set does not match the trips");
    }
    trajectories.push_back(TripTrajectory(pose_set[t], trips_[t]));
    imu.push_back(trips_[t].Imu());
    wheel.push_back(trips_[t].Wheel());
  }
  const PoseSetIds ids = ComputePoseSetIds(pose_set, imu, wheel);

  // Keyframe poses from the pose set (calibration.keyframe_poses).
  const std::vector<KeyframeEntry> keyframes = Keyframes(true);
  std::vector<Pose> poses(keyframes.size());
  for (size_t t = 0; t < trips_.size(); ++t) {
    std::vector<size_t> index;
    std::vector<double> stamps;
    for (size_t k = 0; k < keyframes.size(); ++k) {
      if (keyframes[k].trip != static_cast<int>(t)) continue;
      index.push_back(k);
      stamps.push_back(keyframes[k].stamp);
    }
    const std::vector<Pose> vehicle = trajectories[t].PosesAt(stamps);
    const Pose lidar = RigLidar(pose_set[t].rig);
    for (size_t m = 0; m < index.size(); ++m) {
      poses[index[m]] = vehicle[m] * lidar;
    }
  }

  // static_map.accumulate.
  MapAccumulator accumulator(voxel, max_range, fusion);
  if (fusion.fuse) {
    for (size_t k = 0; k < keyframes.size(); ++k) {
      accumulator.AddRanges(poses[k], ReadSweep(keyframes[k].path).points);
    }
  }
  for (size_t k = 0; k < keyframes.size(); ++k) {
    const SweepFile sweep = ReadSweep(keyframes[k].path);
    accumulator.Add(static_cast<int>(k), poses[k], sweep.points,
                    sweep.intensity);
  }
  MapAccumulator::Cloud cloud = accumulator.Collect();
  PointMapFile map;
  map.xyz = cloud.points;
  map.intensity = cloud.intensity;
  map.kf.assign(cloud.keyframe.begin(), cloud.keyframe.end());
  for (int k : cloud.keyframe) {
    map.trip.push_back(
        static_cast<int16_t>(keyframes[static_cast<size_t>(k)].trip));
  }
  for (int& count : cloud.observations) count = std::min(count, 65535);
  for (int count : cloud.observations) {
    map.observations.push_back(static_cast<uint16_t>(count));
  }

  // static_map.compute_normals.
  std::vector<Vec3> origins;
  for (const Pose& pose : poses) origins.push_back(pose.translation);
  const NormalsResult normals = EstimateOrientedNormals(
      cloud.points, cloud.keyframe, origins, normal_radius, 6, 48);
  map.normals = normals.normals;
  map.planarity.assign(normals.planarity.begin(), normals.planarity.end());
  if (fusion.fuse) {
    map.sigma = PointSigmas(cloud, normals.normals, fusion.sigma_floor);
  }
  // The later steps see the stored float32 normals.
  Points stored_normals = normals.normals;
  for (Vec3& n : stored_normals) n = n.cast<float>().cast<double>();

  // levels.detect_levels and LevelModel.assign.
  std::vector<Points> paths;
  for (const TripPoses& trip : pose_set) {
    Points path;
    for (const Pose& pose : trip.knot_poses) path.push_back(pose.translation);
    paths.push_back(std::move(path));
  }
  const LevelModel levels = DetectLevels(paths, cloud.points, stored_normals,
                                         LevelConfigFrom(config_), log_);
  const LevelAssignment assignment = AssignLevels(levels, cloud.points);
  map.level = assignment.level;
  map.height = assignment.height;
  map.on_ramp = assignment.on_ramp;

  // static_map.carve_dynamic.
  const size_t n = cloud.points.size();
  const Value& enabled = map_config["dynamic"]["enabled"];
  if (!enabled.is_null() && enabled.AsBool()) {
    std::vector<int> trip(n);
    for (size_t i = 0; i < n; ++i) trip[i] = map.trip[i];
    DynamicDetector detector(cloud.points, stored_normals, trip,
                             DynamicConfigFrom(config_));
    const Value& el_res = map_config["dynamic"]["el_res"];
    if (el_res.is_null() || (el_res.is_number() && el_res.AsDouble() == 0.0)) {
      // np.linspace(0, K - 1, min(3, K)).astype(int).
      const size_t count = std::min<size_t>(3, keyframes.size());
      std::vector<Points> sample;
      for (size_t m = 0; m < count; ++m) {
        const double step =
            count > 1 ? static_cast<double>(keyframes.size() - 1) /
                            static_cast<double>(count - 1)
                      : 0.0;
        const size_t k = m + 1 == count && count > 1
                             ? keyframes.size() - 1
                             : static_cast<size_t>(m * step);
        sample.push_back(ReadSweep(keyframes[k].path).points);
      }
      detector.SetElevationResolution(EstimateBeamSpacing(sample));
    }
    for (size_t k = 0; k < keyframes.size(); ++k) {
      detector.AddSweep(keyframes[k].trip, poses[k],
                        ReadSweep(keyframes[k].path).points);
    }
    for (size_t t = 0; t < paths.size(); ++t) {
      detector.AddEgoPath(static_cast<int>(t), paths[t]);
    }
    detector.SetHeights(
        std::vector<double>(map.height.begin(), map.height.end()));
    const std::vector<DynamicKind> kinds = detector.Classify();
    size_t dynamic = 0;
    size_t moved = 0;
    size_t changed = 0;
    for (DynamicKind kind : kinds) {
      const uint8_t value = static_cast<uint8_t>(kind);
      map.dynamic_kind.push_back(value);
      map.dynamic.push_back(value > 0);
      dynamic += value > 0;
      moved += value == 1;
      changed += value == 2;
    }
    LogLine(log_, StringPrintf("[map] dynamic points: %zu / %zu (%.1f%%), "
                               "moved %zu, changed between sessions %zu",
                               dynamic, n,
                               n ? 100.0 * static_cast<double>(dynamic) /
                                       static_cast<double>(n)
                                 : 0.0,
                               moved, changed));
  } else {
    map.dynamic.assign(n, 0);
    map.dynamic_kind.assign(n, 0);
  }
  const fs::path out = fs::path(work_) / "map";
  WritePointMap((out / "point_map.npz").string(), map);
  WriteLevels(out.string(), levels);
  LogLine(log_, "[map] " + std::to_string(n) + " points at " + Repr(voxel) +
                    " m");
  RecordProvenance(work_, "recon.map", ids, false);
}

}  // namespace gr
