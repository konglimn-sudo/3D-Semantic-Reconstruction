#include "gr/pipeline/recon_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
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
#include "gr/mapping/tiled_map.h"
#include "gr/odometry/odometry.h"
#include "gr/odometry/trajectory.h"
#include "gr/sensors/imu_preintegration.h"

namespace gr {
namespace {

namespace fs = std::filesystem;

constexpr char kOdometrySweeps[] = "kf";
constexpr char kFinalSweeps[] = "kf_final";
// odometry.external_trajectory names the trip directory with this.
constexpr char kTripPlaceholder[] = "{trip}";
// graph/summary.json is written with this indentation (json.dump indent).
constexpr int kSummaryJsonIndent = 2;
// The summary rounds IMU biases to this many decimals (numpy round(4)).
constexpr double kBiasRounding = 1e4;
// The point shift against the odometry deskew is measured on every
// kShiftCheckInterval-th sweep, from the first.
constexpr int kShiftCheckInterval = 10;
// Median shift: every kShiftQueryStride-th query point, against reference
// points in the voxels this many steps around, closer than kMaxShift (m).
constexpr size_t kShiftQueryStride = 10;
constexpr int kNeighbourhoodRadius = 1;
constexpr double kMaxShift = 0.5;
constexpr double kPercent = 100.0;

// Python's repr of a number (log lines print config values with it).
std::string Repr(double value) {
  return ToJson(Value::Number(value), kJsonOneLine);
}

double ConfigDouble(const Value& value, const std::string& name) {
  if (!value.is_number()) throw std::runtime_error(name + ": number expected");
  return value.AsDouble();
}

// Median distance from `query` (every kShiftQueryStride-th point) to its
// nearest point of `reference`, ignoring distances of kMaxShift or more
// (trajectory._median_shift); NaN without such pairs.
double MedianShift(const Points& reference, const Points& query) {
  if (reference.empty() || query.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> cells;
  for (size_t point = 0; point < reference.size(); ++point) {
    cells[VoxelOf(reference[point], 1.0 / kMaxShift)].push_back(
        static_cast<int>(point));
  }
  std::vector<double> distances;
  for (size_t point = 0; point < query.size(); point += kShiftQueryStride) {
    const VoxelKey key = VoxelOf(query[point], 1.0 / kMaxShift);
    double best = kMaxShift;
    for (int dx = -kNeighbourhoodRadius; dx <= kNeighbourhoodRadius; ++dx) {
      for (int dy = -kNeighbourhoodRadius; dy <= kNeighbourhoodRadius; ++dy) {
        for (int dz = -kNeighbourhoodRadius; dz <= kNeighbourhoodRadius;
             ++dz) {
          const auto found =
              cells.find(VoxelKey{key.x + dx, key.y + dy, key.z + dz});
          if (found == cells.end()) continue;
          for (int other : found->second) {
            best = std::min(best, (reference[static_cast<size_t>(other)] -
                                   query[point])
                                      .norm());
          }
        }
      }
    }
    if (best < kMaxShift) distances.push_back(best);
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
  for (const Vec6& bias : poses.biases) {
    gyro.push_back(bias.segment<kSpaceDim>(kGyroBiasOffset));
  }
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
  for (size_t trip = 0; trip < trips_.size(); ++trip) {
    const TripOdometryFile odometry = ReadTripOdometry(TripDir(trip));
    for (int64_t frame : odometry.kf_frames) {
      KeyframeEntry entry;
      entry.trip = static_cast<int>(trip);
      entry.frame = frame;
      entry.stamp = odometry.stamps.at(static_cast<size_t>(frame));
      entry.odom = odometry.lidar_poses.at(static_cast<size_t>(frame));
      entry.path = SweepPath(TripDir(trip), sweeps, frame);
      out.push_back(std::move(entry));
    }
  }
  return out;
}

void ReconPipeline::StageOdometry() {
  const OdometryConfig odometry_config = OdometryConfigFrom(config_);
  const KeyframeConfig keyframe_config = KeyframeConfigFrom(config_);
  const Value& external = config_["odometry"]["external_trajectory"];
  const size_t placeholder_length = std::strlen(kTripPlaceholder);
  for (size_t trip_index = 0; trip_index < trips_.size(); ++trip_index) {
    const TripReader& trip = trips_[trip_index];
    const Pose vehicle_from_lidar =
        trip.rig().primary_lidar().vehicle_from_lidar;
    const MotionPrior prior = trip.BuildMotionPrior();
    const std::optional<Pose> entrance = trip.EntrancePose(prior);
    const size_t frame_count = trip.size();
    std::vector<double> stamps(frame_count);
    for (size_t k = 0; k < frame_count; ++k) stamps[k] = trip.lidar_stamp(k);
    std::vector<Pose> external_poses;
    if (external.is_string() && !external.AsString().empty()) {
      // str.format(trip=<trip directory>).
      std::string path = external.AsString();
      for (size_t at = path.find(kTripPlaceholder); at != std::string::npos;
           at = path.find(kTripPlaceholder, at + trip.root().size())) {
        path.replace(at, placeholder_length, trip.root());
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
    const std::string dir = TripDir(trip_index);
    fs::create_directories(fs::path(dir) / kOdometrySweeps);
    for (size_t k = 0; k < frame_count; ++k) {
      const LidarSweep sweep = trip.LoadLidar(k);
      std::optional<Pose> given;
      if (!external_poses.empty()) given = external_poses[k];
      const OdometryFrame frame =
          pipeline.Process(sweep.points, sweep.intensity, sweep.time,
                           sweep.stamp, k + 1 == frame_count, given);
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
                         trip.trip_id().c_str(), frame_count,
                         odometry.kf_frames.size(),
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
  for (size_t trip_index = 0; trip_index < trips_.size(); ++trip_index) {
    const TripReader& trip = trips_[trip_index];
    odometries.push_back(ReadTripOdometry(TripDir(trip_index)));
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
  for (const auto& [first, second] : loop_pairs_) {
    const int size = static_cast<int>(keyframes.size());
    if (first < 0 || second < 0 || first >= size || second >= size) {
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
                ToJson(LoopsValue(result.loops), kJsonOneLine));
  Value summary = GraphSummaryValue(result, keyframes.size(), trip_ids);
  const std::vector<uint8_t>& valid = result.inertial.valid;
  const bool any_valid =
      std::any_of(valid.begin(), valid.end(),
                  [](uint8_t flag) { return flag != 0; });
  for (size_t trip = 0; trip < trips_.size() && any_valid; ++trip) {
    Vec6 sum = Vec6::Zero();
    int count = 0;
    for (size_t k = 0; k < valid.size(); ++k) {
      if (!valid[k] || entries[k].trip != static_cast<int>(trip)) continue;
      sum += result.inertial.biases[k];
      ++count;
    }
    if (count == 0) continue;
    const Vec6 mean = sum / count;
    // numpy round(4): rint(x * 1e4) / 1e4.
    const auto round4 = [](double x) {
      return std::nearbyint(x * kBiasRounding) / kBiasRounding;
    };
    Value gyro = Value::List();
    Value accel = Value::List();
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      gyro.Append(
          Value::Number(round4(RadToDeg(mean[kGyroBiasOffset + axis]))));
      accel.Append(Value::Number(round4(mean[kAccelBiasOffset + axis])));
    }
    Value bias = Value::Map();
    bias.Set("gyro_deg_s", std::move(gyro));
    bias.Set("accel_m_s2", std::move(accel));
    summary.Set("imu_bias_" + trip_ids[trip], std::move(bias));
  }
  WriteTextFile((out / "summary.json").string(),
                ToJson(summary, kSummaryJsonIndent));

  for (size_t trip = 0; trip < trips_.size(); ++trip) {
    std::vector<double> stamps;
    std::vector<Pose> odom;
    std::vector<Pose> poses;
    for (size_t k = 0; k < entries.size(); ++k) {
      if (entries[k].trip != static_cast<int>(trip)) continue;
      stamps.push_back(entries[k].stamp);
      odom.push_back(entries[k].odom);
      poses.push_back(result.poses[k]);
    }
    const TripOdometryFile& odometry = odometries[trip];
    const std::vector<Pose> lidar = CorrectFramePoses(
        stamps, odom, poses, odometry.stamps, odometry.lidar_poses);
    WriteFramePoses(TripDir(trip), odometry.stamps, lidar,
                    trips_[trip].rig().primary_lidar().vehicle_from_lidar);
  }
}

void ReconPipeline::StageTrajectory() {
  const KeyframeTable table =
      ReadKeyframeTable((fs::path(work_) / "graph" / "kf_poses.npz").string());
  const bool with_bias = table.inertial.valid.size() == table.trip.size();
  std::vector<TripPoses> pose_set;
  std::vector<Eigen::MatrixXd> imu;
  std::vector<Eigen::MatrixXd> wheel;
  for (size_t trip_index = 0; trip_index < trips_.size(); ++trip_index) {
    const TripReader& trip = trips_[trip_index];
    TripPoses poses;
    poses.trip_id = trip.trip_id();
    poses.dataset = trip.root();
    poses.rig = RigDict(ReadYamlFile((fs::path(trip.root()) / "rig.yaml")
                                         .string()));
    std::vector<Pose> lidar;
    ReadFramePoses(TripDir(trip_index), &poses.knot_stamps, &lidar);
    const Pose lidar_from_vehicle =
        trip.rig().primary_lidar().vehicle_from_lidar.Inverse();
    for (const Pose& pose : lidar) {
      poses.knot_poses.push_back(pose * lidar_from_vehicle);
    }
    std::vector<size_t> mine;
    for (size_t k = 0; with_bias && k < table.trip.size(); ++k) {
      if (table.trip[k] == static_cast<int64_t>(trip_index) &&
          table.inertial.valid[k]) {
        mine.push_back(k);
      }
    }
    std::stable_sort(mine.begin(), mine.end(), [&](size_t left, size_t right) {
      return table.stamp[left] < table.stamp[right];
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
  for (size_t trip_index = 0; trip_index < trips_.size(); ++trip_index) {
    const TripReader& trip = trips_[trip_index];
    const ContinuousTrajectory trajectory =
        TripTrajectory(pose_set[trip_index], trip);
    const Pose vehicle_from_lidar =
        trip.rig().primary_lidar().vehicle_from_lidar;
    const std::string dir = TripDir(trip_index);
    const TripOdometryFile odometry = ReadTripOdometry(dir);
    for (int64_t frame : odometry.kf_frames) {
      const LidarSweep sweep = trip.LoadLidar(static_cast<size_t>(frame));
      const SweepSamples samples = DeskewKeyframeSweep(
          sweep.points, sweep.intensity, sweep.time, sweep.stamp, trajectory,
          vehicle_from_lidar, min_range, max_range, voxel);
      const std::string path = SweepPath(dir, kFinalSweeps, frame);
      WriteSweep(path, samples.points, samples.intensity);
      // Shift against the odometry deskew.
      if (++written % kShiftCheckInterval == 1) {
        const SweepFile old =
            ReadSweep(SweepPath(dir, kOdometrySweeps, frame));
        shifts.push_back(MedianShift(old.points, samples.points));
      }
    }
    LogLine(log_,
            StringPrintf("[trajectory] %s: %zu keyframe sweeps deskewed (gyro "
                         "%.0f%%, wheel %.0f%% of knot intervals)",
                         trip.trip_id().c_str(), odometry.kf_frames.size(),
                         kPercent * trajectory.gyro_coverage(),
                         kPercent * trajectory.wheel_coverage()));
  }
  shifts.erase(std::remove_if(shifts.begin(), shifts.end(),
                              [](double shift) {
                                return !std::isfinite(shift);
                              }),
               shifts.end());
  LogLine(log_, "[trajectory] pose set " + ids.pose_set_id + ": " +
                    std::to_string(written) +
                    " sweeps re-deskewed, median point shift " +
                    (shifts.empty()
                         ? "None"
                         : Repr(Median(shifts) * kCentimetresPerMetre)) +
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
  for (size_t trip = 0; trip < trips_.size(); ++trip) {
    if (pose_set[trip].trip_id != trips_[trip].trip_id()) {
      throw std::runtime_error("the pose set does not match the trips");
    }
    trajectories.push_back(TripTrajectory(pose_set[trip], trips_[trip]));
    imu.push_back(trips_[trip].Imu());
    wheel.push_back(trips_[trip].Wheel());
  }
  const PoseSetIds ids = ComputePoseSetIds(pose_set, imu, wheel);

  // Keyframe poses from the pose set (calibration.keyframe_poses).
  const std::vector<KeyframeEntry> keyframes = Keyframes(true);
  std::vector<Pose> poses(keyframes.size());
  for (size_t trip = 0; trip < trips_.size(); ++trip) {
    std::vector<size_t> index;
    std::vector<double> stamps;
    for (size_t k = 0; k < keyframes.size(); ++k) {
      if (keyframes[k].trip != static_cast<int>(trip)) continue;
      index.push_back(k);
      stamps.push_back(keyframes[k].stamp);
    }
    const std::vector<Pose> vehicle = trajectories[trip].PosesAt(stamps);
    const Pose lidar = RigLidar(pose_set[trip].rig);
    for (size_t k = 0; k < index.size(); ++k) {
      poses[index[k]] = vehicle[k] * lidar;
    }
  }

  // static_map.build_map: accumulation, normals and sigmas, levels and
  // dynamic points, tile by tile when map.tile_size is set.
  std::vector<Points> paths;
  for (const TripPoses& trip : pose_set) {
    Points path;
    for (const Pose& pose : trip.knot_poses) path.push_back(pose.translation);
    paths.push_back(std::move(path));
  }
  TiledMapConfig tiled;
  tiled.voxel = voxel;
  tiled.max_range = max_range;
  tiled.normal_radius = normal_radius;
  const Value& tile_size = map_config["tile_size"];
  tiled.tile_size = tile_size.is_null()
                        ? 0.0
                        : ConfigDouble(tile_size, "map.tile_size");
  tiled.fusion = fusion;
  tiled.levels = LevelConfigFrom(config_);
  const Value& enabled = map_config["dynamic"]["enabled"];
  tiled.dynamic = !enabled.is_null() && enabled.AsBool();
  tiled.dynamic_config = DynamicConfigFrom(config_);
  const Value& el_res = map_config["dynamic"]["el_res"];
  if (el_res.is_null()) tiled.dynamic_config.el_res = 0.0;
  std::vector<int> keyframe_trip;
  for (const KeyframeEntry& keyframe : keyframes) {
    keyframe_trip.push_back(keyframe.trip);
  }
  const SweepLoader load = [&](int keyframe, Points* points,
                               std::vector<float>* intensity) {
    SweepFile sweep =
        ReadSweep(keyframes[static_cast<size_t>(keyframe)].path);
    *points = std::move(sweep.points);
    *intensity = std::move(sweep.intensity);
  };
  const fs::path out = fs::path(work_) / "map";
  const fs::path tile_dir = out / "tiles";
  fs::remove_all(tile_dir);
  fs::remove(out / "tiles.json");
  fs::remove(out / "point_map.npz");
  Value tile_list = Value::List();
  const ChunkSink sink = [&](const MapTile& tile, MapChunk&& chunk) {
    PointMapFile map;
    const size_t count = chunk.size();
    map.xyz = std::move(chunk.xyz);
    map.intensity = std::move(chunk.intensity);
    map.kf = std::move(chunk.keyframe);
    map.trip = std::move(chunk.trip);
    map.normals = std::move(chunk.normals);
    map.planarity = std::move(chunk.planarity);
    map.sigma = std::move(chunk.sigma);
    map.observations = std::move(chunk.observations);
    for (uint8_t kind : chunk.dynamic_kind) {
      map.dynamic.push_back(kind != static_cast<uint8_t>(DynamicKind::kStatic));
    }
    map.dynamic_kind = std::move(chunk.dynamic_kind);
    map.level = std::move(chunk.level);
    map.height = std::move(chunk.height);
    map.on_ramp = std::move(chunk.on_ramp);
    if (tile.whole) {
      WritePointMap((out / "point_map.npz").string(), map);
      return;
    }
    map.range = std::move(chunk.range);
    map.sequence = std::move(chunk.sequence);
    const std::string name = "tile_" + std::to_string(tile.ix) + "_" +
                             std::to_string(tile.iy) + ".npz";
    WritePointMap((tile_dir / name).string(), map);
    Value entry = Value::Map();
    entry.Set("ix", Value::Integer(tile.ix));
    entry.Set("iy", Value::Integer(tile.iy));
    entry.Set("file", Value::String("tiles/" + name));
    entry.Set("points", Value::Integer(static_cast<long long>(count)));
    tile_list.Append(std::move(entry));
  };
  const TiledMapSummary summary = BuildTiledMap(poses, keyframe_trip, paths,
                                                load, tiled, sink, log_);
  const LevelModel& levels = summary.levels;
  const size_t point_count = summary.points;
  if (tiled.dynamic) {
    const size_t dynamic = summary.moved + summary.changed;
    LogLine(log_, StringPrintf("[map] dynamic points: %zu / %zu (%.1f%%), "
                               "moved %zu, changed between sessions %zu",
                               dynamic, point_count,
                               point_count
                                   ? kPercent * static_cast<double>(dynamic) /
                                         static_cast<double>(point_count)
                                   : 0.0,
                               summary.moved, summary.changed));
  }
  if (tiled.tile_size > 0.0) {
    Value index = Value::Map();
    index.Set("tile_size", Value::Number(tiled.tile_size));
    index.Set("voxel", Value::Number(voxel));
    index.Set("points", Value::Integer(static_cast<long long>(point_count)));
    index.Set("tiles", std::move(tile_list));
    WriteTextFile((out / "tiles.json").string(), ToJson(index));
  }
  WriteLevels(out.string(), levels);
  LogLine(log_, "[map] " + std::to_string(point_count) + " points at " +
                    Repr(voxel) + " m");
  RecordProvenance(work_, "recon.map", ids, false);
}

}  // namespace gr
