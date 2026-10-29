#include "gr/io/recon_files.h"

#include <cinttypes>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "gr/io/digest.h"
#include "gr/io/npy.h"
#include "gr/io/tables.h"
#include "gr/io/trip.h"

namespace gr {
namespace {

namespace fs = std::filesystem;

// Numbers of a homogeneous transform.
constexpr size_t kMatrixSize = Mat4::SizeAtCompileTime;
// Each floor grid's row of floor_grids.npz "meta": x0, y0, res.
constexpr int64_t kFloorGridMetaFields = 3;
// The indentation levels.py writes levels.json with (json.dumps(indent=2)).
constexpr int kLevelsJsonIndent = 2;
// Room for "<frame>.npz" and for a local time stamp.
constexpr size_t kSweepNameBuffer = 32;
constexpr size_t kTimeStampBuffer = 32;

// The poses' 4x4 matrices, row by row.
std::vector<double> Flat(const std::vector<Pose>& poses) {
  std::vector<double> values;
  values.reserve(kMatrixSize * poses.size());
  for (const Pose& pose : poses) {
    const Mat4 matrix = pose.Matrix();
    for (int row = 0; row < kHomogeneousDim; ++row) {
      for (int column = 0; column < kHomogeneousDim; ++column) {
        values.push_back(matrix(row, column));
      }
    }
  }
  return values;
}

NpyArray PosesArray(const std::vector<Pose>& poses) {
  return NpyArray::Of(Flat(poses), {static_cast<int64_t>(poses.size()),
                                    kHomogeneousDim, kHomogeneousDim});
}

std::vector<Pose> PosesFrom(const NpyArray& array, const std::string& name) {
  const std::vector<double> values = array.ToDoubles();
  if (values.size() % kMatrixSize != 0) {
    throw std::runtime_error(name + ": (N, 4, 4) poses expected");
  }
  std::vector<Pose> out(values.size() / kMatrixSize);
  for (size_t k = 0; k < out.size(); ++k) {
    // Row by row, as numpy stores it (Mat4 itself is column-major).
    Mat4 matrix;
    for (int row = 0; row < kHomogeneousDim; ++row) {
      for (int column = 0; column < kHomogeneousDim; ++column) {
        matrix(row, column) =
            values[kMatrixSize * k + kHomogeneousDim * row + column];
      }
    }
    out[k] = Pose::FromMatrix(matrix);
  }
  return out;
}

// The rows' coefficients, row by row.
template <typename Row>
std::vector<double> Flat(const std::vector<Row>& rows) {
  std::vector<double> values;
  values.reserve(Row::SizeAtCompileTime * rows.size());
  for (const Row& row : rows) {
    for (int k = 0; k < Row::SizeAtCompileTime; ++k) values.push_back(row[k]);
  }
  return values;
}

// An (N, size of Row) array of fixed-size vectors.
template <typename Row>
NpyArray RowsArray(const std::vector<Row>& rows) {
  return NpyArray::Of(
      Flat(rows), {static_cast<int64_t>(rows.size()), Row::SizeAtCompileTime});
}

template <typename Row>
std::vector<Row> RowsFrom(const NpyArray& array) {
  constexpr size_t kRowSize = Row::SizeAtCompileTime;
  const std::vector<double> values = array.ToDoubles();
  std::vector<Row> out(values.size() / kRowSize);
  for (size_t k = 0; k < out.size(); ++k) {
    for (size_t column = 0; column < kRowSize; ++column) {
      out[k][column] = values[kRowSize * k + column];
    }
  }
  return out;
}

const NpyArray& Member(const NpzArchive& archive, const std::string& name,
                       const std::string& path) {
  const auto found = archive.find(name);
  if (found == archive.end()) throw std::runtime_error(path + ": no " + name);
  return found->second;
}

std::vector<double> Flat(const Eigen::MatrixXd& table) {
  std::vector<double> values;  // C order.
  values.reserve(static_cast<size_t>(table.size()));
  for (Eigen::Index row = 0; row < table.rows(); ++row) {
    for (Eigen::Index column = 0; column < table.cols(); ++column) {
      values.push_back(table(row, column));
    }
  }
  return values;
}

Value Floats(const std::vector<double>& values) {
  Value out = Value::List();
  for (double value : values) out.Append(Value::Number(value));
  return out;
}

// A rig.yaml matrix (nested lists or its 16 numbers) as a 4x4 list of
// floats.
Value MatrixOf(const Value& value) {
  std::vector<double> flat;
  if (value.is_list()) {
    for (const Value& item : value.items()) {
      if (item.is_list()) {
        for (const Value& entry : item.items()) {
          flat.push_back(entry.AsDouble());
        }
      } else {
        flat.push_back(item.AsDouble());
      }
    }
  }
  if (flat.size() != kMatrixSize) {
    throw std::runtime_error("4x4 matrix expected");
  }
  Value out = Value::List();
  for (int row = 0; row < kHomogeneousDim; ++row) {
    const auto row_start = flat.begin() + kHomogeneousDim * row;
    out.Append(
        Floats(std::vector<double>(row_start, row_start + kHomogeneousDim)));
  }
  return out;
}

Value IdentityMatrix() { return MatrixValue(Pose()); }

double DoubleOr(const Value& value, double fallback) {
  return value.is_null() ? fallback : value.AsDouble();
}

}  // namespace

void WriteTextFile(const std::string& path, const std::string& text) {
  fs::create_directories(fs::path(path).parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << text;
  if (!out) throw std::runtime_error("cannot write " + path);
}

Value MatrixValue(const Pose& pose) {
  const Mat4 matrix = pose.Matrix();
  Value out = Value::List();
  for (int row = 0; row < kHomogeneousDim; ++row) {
    std::vector<double> values;
    for (int column = 0; column < kHomogeneousDim; ++column) {
      values.push_back(matrix(row, column));
    }
    out.Append(Floats(values));
  }
  return out;
}

void WriteTripOdometry(const std::string& trip_dir,
                       const TripOdometryFile& odometry,
                       const Pose& vehicle_from_lidar) {
  fs::create_directories(trip_dir);
  Trajectory vehicle;
  vehicle.stamps = odometry.stamps;
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  for (const Pose& pose : odometry.lidar_poses) {
    vehicle.poses.push_back(pose * lidar_from_vehicle);
  }
  WriteTum((fs::path(trip_dir) / "odometry.tum").string(), vehicle);
  NpzArchive archive;
  archive["stamps"] = NpyArray::Of(
      odometry.stamps, {static_cast<int64_t>(odometry.stamps.size())});
  archive["lidar_poses"] = PosesArray(odometry.lidar_poses);
  archive["kf_frames"] = NpyArray::Of(
      odometry.kf_frames, {static_cast<int64_t>(odometry.kf_frames.size())});
  WriteNpz((fs::path(trip_dir) / "odometry.npz").string(), archive, false);
}

TripOdometryFile ReadTripOdometry(const std::string& trip_dir) {
  const std::string path = (fs::path(trip_dir) / "odometry.npz").string();
  const NpzArchive archive = ReadNpz(path);
  TripOdometryFile out;
  out.stamps = Member(archive, "stamps", path).ToDoubles();
  out.lidar_poses = PosesFrom(Member(archive, "lidar_poses", path), path);
  out.kf_frames = Member(archive, "kf_frames", path).ToInts();
  if (out.stamps.size() != out.lidar_poses.size()) {
    throw std::runtime_error(path + ": one pose per stamp expected");
  }
  return out;
}

std::string SweepPath(const std::string& trip_dir, const std::string& sweeps,
                      int64_t frame) {
  char name[kSweepNameBuffer];
  std::snprintf(name, sizeof(name), "%06" PRId64 ".npz", frame);
  return (fs::path(trip_dir) / sweeps / name).string();
}

void WriteSweep(const std::string& path, const Points& points,
                const std::vector<float>& intensity) {
  fs::create_directories(fs::path(path).parent_path());
  std::vector<float> xyz;
  xyz.reserve(kSpaceDim * points.size());
  for (const Vec3& point : points) {
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      xyz.push_back(static_cast<float>(point[axis]));
    }
  }
  NpzArchive archive;
  archive["points"] =
      NpyArray::Of(xyz, {static_cast<int64_t>(points.size()), kSpaceDim});
  archive["intensity"] =
      NpyArray::FromHalfs(intensity, {static_cast<int64_t>(intensity.size())});
  WriteNpz(path, archive, false);
}

SweepFile ReadSweep(const std::string& path) {
  const NpzArchive archive = ReadNpz(path);
  const std::vector<double> xyz = Member(archive, "points", path).ToDoubles();
  SweepFile out;
  out.points.resize(xyz.size() / kSpaceDim);
  for (size_t k = 0; k < out.points.size(); ++k) {
    out.points[k] = Eigen::Map<const Vec3>(xyz.data() + kSpaceDim * k);
  }
  out.intensity = Member(archive, "intensity", path).ToFloats();
  return out;
}

void WriteKeyframeTable(const std::string& path, const KeyframeTable& table) {
  fs::create_directories(fs::path(path).parent_path());
  const int64_t count = static_cast<int64_t>(table.stamp.size());
  NpzArchive archive;
  archive["poses"] = PosesArray(table.poses);
  archive["odom"] = PosesArray(table.odom);
  archive["trip"] = NpyArray::Of(table.trip, {count});
  archive["frame"] = NpyArray::Of(table.frame, {count});
  archive["stamp"] = NpyArray::Of(table.stamp, {count});
  archive["velocity"] = RowsArray(table.inertial.velocities);
  archive["bias"] = RowsArray(table.inertial.biases);
  archive["inertial_valid"] =
      NpyArray::FromBools(table.inertial.valid,
                          {static_cast<int64_t>(table.inertial.valid.size())});
  WriteNpz(path, archive, false);
}

KeyframeTable ReadKeyframeTable(const std::string& path) {
  const NpzArchive archive = ReadNpz(path);
  KeyframeTable out;
  out.poses = PosesFrom(Member(archive, "poses", path), path);
  out.odom = PosesFrom(Member(archive, "odom", path), path);
  out.trip = Member(archive, "trip", path).ToInts();
  out.frame = Member(archive, "frame", path).ToInts();
  out.stamp = Member(archive, "stamp", path).ToDoubles();
  if (archive.count("bias")) {
    out.inertial.biases = RowsFrom<Vec6>(archive.at("bias"));
  }
  if (archive.count("velocity")) {
    out.inertial.velocities = RowsFrom<Vec3>(archive.at("velocity"));
  }
  if (archive.count("inertial_valid")) {
    for (int64_t valid : archive.at("inertial_valid").ToInts()) {
      out.inertial.valid.push_back(valid != 0);
    }
  }
  return out;
}

Value LoopsValue(const std::vector<LoopConstraint>& loops) {
  Value out = Value::List();
  for (const LoopConstraint& loop : loops) {
    Value item = Value::Map();
    item.Set("i", Value::Integer(loop.i));
    item.Set("j", Value::Integer(loop.j));
    item.Set("T_ji", MatrixValue(loop.j_from_i));
    item.Set("fitness", Value::Number(loop.fitness));
    item.Set("rmse", Value::Number(loop.rmse));
    item.Set("sc_distance", Value::Number(loop.sc_distance));
    out.Append(std::move(item));
  }
  return out;
}

Value GraphSummaryValue(const GraphStageResult& result, size_t keyframes,
                        const std::vector<std::string>& trip_ids) {
  const GraphSummary& summary = result.summary;
  Value out = Value::Map();
  out.Set("iterations", Value::Integer(summary.iterations));
  out.Set("initial_cost", Value::Number(summary.initial_cost));
  out.Set("final_cost", Value::Number(summary.final_cost));
  out.Set("converged", Value::Bool(summary.converged));
  out.Set("loop_chi2_median", Value::Number(summary.loop_chi2_median));
  out.Set("loop_chi2_max", Value::Number(summary.loop_chi2_max));
  out.Set("imu_factors", Value::Integer(summary.imu_factors));
  out.Set("imu_chi2_median", Value::Number(summary.imu_chi2_median));
  out.Set("keyframes", Value::Integer(static_cast<int64_t>(keyframes)));
  out.Set("loops", Value::Integer(static_cast<int64_t>(result.loops.size())));
  out.Set("anchors",
          Value::Integer(static_cast<int64_t>(result.anchors.size())));
  out.Set("gravity_factors",
          Value::Integer(static_cast<int64_t>(result.gravity.size())));
  out.Set("loops_tested", Value::Integer(result.loop_stats.tested));
  out.Set("loops_verified", Value::Integer(result.loops_verified));
  Value rounds = Value::List();
  for (const LidarBaRound& ba_round : result.ba.rounds) {
    Value round = Value::Map();
    round.Set("planes", Value::Integer(ba_round.planes));
    round.Set("observations", Value::Integer(ba_round.observations));
    round.Set("rms_median_before", Value::Number(ba_round.rms_median_before));
    round.Set("rms_p90_before", Value::Number(ba_round.rms_p90_before));
    round.Set("rms_median_after", Value::Number(ba_round.rms_median_after));
    round.Set("rms_p90_after", Value::Number(ba_round.rms_p90_after));
    round.Set("shift_median", Value::Number(ba_round.shift_median));
    round.Set("shift_max", Value::Number(ba_round.shift_max));
    round.Set("iterations", Value::Integer(ba_round.iterations));
    rounds.Append(std::move(round));
  }
  out.Set("ba_rounds", std::move(rounds));
  Value alignments = Value::List();
  for (const TripAlignment& alignment : result.trip_alignments) {
    Value item = Value::Map();
    item.Set("trip",
             Value::String(trip_ids.at(static_cast<size_t>(alignment.trip))));
    item.Set("aligned", Value::Bool(alignment.aligned));
    item.Set("T", MatrixValue(alignment.correction));
    item.Set("score", Value::Number(alignment.score));
    item.Set("fitness", Value::Number(alignment.fitness));
    item.Set("rmse", Value::Number(alignment.rmse));
    item.Set("runner_up", Value::Number(alignment.runner_up));
    item.Set("hypotheses", Value::Integer(alignment.hypotheses));
    alignments.Append(std::move(item));
  }
  out.Set("trip_alignments", std::move(alignments));
  return out;
}

void WriteFramePoses(const std::string& trip_dir,
                     const std::vector<double>& stamps,
                     const std::vector<Pose>& lidar,
                     const Pose& vehicle_from_lidar) {
  fs::create_directories(trip_dir);
  const Pose lidar_from_vehicle = vehicle_from_lidar.Inverse();
  Trajectory vehicle;
  vehicle.stamps = stamps;
  for (const Pose& pose : lidar) {
    vehicle.poses.push_back(pose * lidar_from_vehicle);
  }
  NpzArchive archive;
  archive["stamps"] =
      NpyArray::Of(stamps, {static_cast<int64_t>(stamps.size())});
  archive["lidar"] = PosesArray(lidar);
  archive["vehicle"] = PosesArray(vehicle.poses);
  WriteNpz((fs::path(trip_dir) / "poses.npz").string(), archive, false);
  WriteTum((fs::path(trip_dir) / "poses.tum").string(), vehicle);
}

void ReadFramePoses(const std::string& trip_dir, std::vector<double>* stamps,
                    std::vector<Pose>* lidar) {
  const std::string path = (fs::path(trip_dir) / "poses.npz").string();
  const NpzArchive archive = ReadNpz(path);
  *stamps = Member(archive, "stamps", path).ToDoubles();
  *lidar = PosesFrom(Member(archive, "lidar", path), path);
}

Value RigDict(const Value& rig) {
  Value lidars = Value::Map();
  for (const auto& [name, entry] : rig["lidars"].members()) {
    Value lidar = Value::Map();
    lidar.Set("T_vehicle_lidar", MatrixOf(entry["T_vehicle_lidar"]));
    lidar.Set("sweep_duration", Value::Number(DoubleOr(entry["sweep_duration"],
                                                       kDefaultSweepDuration)));
    lidars.Set(name, std::move(lidar));
  }
  Value cameras = Value::Map();
  if (rig["cameras"].is_map()) {
    const CameraCalibration defaults;
    for (const auto& [name, entry] : rig["cameras"].members()) {
      Value camera = Value::Map();
      camera.Set("model", Value::String(entry["model"].AsString()));
      camera.Set("width", Value::Integer(entry["width"].AsInt()));
      camera.Set("height", Value::Integer(entry["height"].AsInt()));
      camera.Set("params", Floats(entry["params"].AsDoubles()));
      camera.Set("T_vehicle_cam", MatrixOf(entry["T_vehicle_cam"]));
      camera.Set("rolling_shutter_line_time",
                 Value::Number(DoubleOr(entry["rolling_shutter_line_time"],
                                        defaults.rolling_shutter_line_time)));
      camera.Set("time_offset", Value::Number(DoubleOr(entry["time_offset"],
                                                       defaults.time_offset)));
      cameras.Set(name, std::move(camera));
    }
  }
  Value pairs = Value::List();
  if (rig["stereo_pairs"].is_list()) {
    for (const Value& pair : rig["stereo_pairs"].items()) {
      Value names = Value::List();
      for (const Value& name : pair.items()) names.Append(name);
      pairs.Append(std::move(names));
    }
  }
  Value out = Value::Map();
  out.Set("lidars", std::move(lidars));
  out.Set("cameras", std::move(cameras));
  out.Set("stereo_pairs", std::move(pairs));
  out.Set("T_vehicle_imu", rig["T_vehicle_imu"].is_null()
                               ? IdentityMatrix()
                               : MatrixOf(rig["T_vehicle_imu"]));
  out.Set("wheel_speed_scale",
          Value::Number(
              DoubleOr(rig["wheel_speed_scale"], Rig().wheel_speed_scale)));
  return out;
}

PoseSetIds ComputePoseSetIds(const std::vector<TripPoses>& trips,
                             const std::vector<Eigen::MatrixXd>& imu,
                             const std::vector<Eigen::MatrixXd>& wheel) {
  PythonDigest lidar;
  std::vector<Value> cameras;
  for (size_t trip_index = 0; trip_index < trips.size(); ++trip_index) {
    const TripPoses& trip = trips[trip_index];
    Value rig = Value::Map();
    for (const auto& [key, value] : trip.rig.members()) {
      if (key == "cameras") {
        cameras.push_back(value);
      } else {
        rig.Set(key, value);
      }
    }
    lidar.AddJson(Value::String(trip.trip_id));
    lidar.AddDoubles(trip.knot_stamps);
    lidar.AddDoubles(Flat(trip.knot_poses));
    lidar.AddDoubles(trip.bias_stamps);
    lidar.AddDoubles(Flat(trip.biases));
    lidar.AddJson(rig);
    lidar.AddDoubles(Flat(imu.at(trip_index)));
    lidar.AddDoubles(Flat(wheel.at(trip_index)));
  }
  PoseSetIds ids;
  ids.trajectory_id = lidar.Hex16();
  PythonDigest full;
  full.AddJson(Value::String(ids.trajectory_id));
  for (const Value& camera : cameras) full.AddJson(camera);
  ids.pose_set_id = full.Hex16();
  return ids;
}

void WritePoseSet(const std::string& work_dir,
                  const std::vector<TripPoses>& trips, const PoseSetIds& ids) {
  const fs::path out = fs::path(work_dir) / "pose_set";
  fs::create_directories(out);
  Value records = Value::List();
  for (const TripPoses& trip : trips) {
    NpzArchive archive;
    archive["knot_stamps"] = NpyArray::Of(
        trip.knot_stamps, {static_cast<int64_t>(trip.knot_stamps.size())});
    archive["knot_poses"] = PosesArray(trip.knot_poses);
    archive["bias_stamps"] = NpyArray::Of(
        trip.bias_stamps, {static_cast<int64_t>(trip.bias_stamps.size())});
    archive["biases"] = RowsArray(trip.biases);
    WriteNpz((out / (trip.trip_id + ".npz")).string(), archive, false);
    Value record = Value::Map();
    record.Set("trip_id", Value::String(trip.trip_id));
    record.Set("dataset", Value::String(trip.dataset));
    record.Set("rig", trip.rig);
    records.Append(std::move(record));
  }
  Value meta = Value::Map();
  meta.Set("pose_set_id", Value::String(ids.pose_set_id));
  meta.Set("trajectory_id", Value::String(ids.trajectory_id));
  meta.Set("trips", std::move(records));
  WriteTextFile((out / "pose_set.json").string(),
                ToJson(meta, kDefaultJsonIndent));
}

std::vector<TripPoses> ReadPoseSet(const std::string& work_dir) {
  const fs::path directory = fs::path(work_dir) / "pose_set";
  const Value meta = ReadJsonFile((directory / "pose_set.json").string());
  std::vector<TripPoses> out;
  for (const Value& record : meta["trips"].items()) {
    TripPoses trip;
    trip.trip_id = record["trip_id"].AsString();
    trip.dataset = record["dataset"].AsString();
    trip.rig = record["rig"];
    const std::string path = (directory / (trip.trip_id + ".npz")).string();
    const NpzArchive archive = ReadNpz(path);
    trip.knot_stamps = Member(archive, "knot_stamps", path).ToDoubles();
    trip.knot_poses = PosesFrom(Member(archive, "knot_poses", path), path);
    trip.bias_stamps = Member(archive, "bias_stamps", path).ToDoubles();
    trip.biases = RowsFrom<Vec6>(Member(archive, "biases", path));
    out.push_back(std::move(trip));
  }
  return out;
}

void RecordProvenance(const std::string& work_dir, const std::string& stage,
                      const PoseSetIds& ids, bool uses_cameras) {
  const std::string path = (fs::path(work_dir) / "provenance.json").string();
  Value entries = fs::exists(path) ? ReadJsonFile(path) : Value::Map();
  const std::time_t now = std::time(nullptr);
  char stamp[kTimeStampBuffer];
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S",
                std::localtime(&now));
  Value entry = Value::Map();
  entry.Set("pose_set_id", Value::String(ids.pose_set_id));
  entry.Set("trajectory_id", Value::String(ids.trajectory_id));
  entry.Set("uses_cameras", Value::Bool(uses_cameras));
  entry.Set("time", Value::String(stamp));
  entries.Set(stage, std::move(entry));
  WriteTextFile(path, PythonJsonDumps(entries, kDefaultJsonIndent));
}

void WritePointMap(const std::string& path, const PointMapFile& map) {
  fs::create_directories(fs::path(path).parent_path());
  const int64_t point_count = static_cast<int64_t>(map.xyz.size());
  const auto check = [&](size_t size, const char* name) {
    if (size != 0 && static_cast<int64_t>(size) != point_count) {
      throw std::runtime_error(std::string("point map: ") + name +
                               " has the wrong length");
    }
    return size != 0;
  };
  NpzArchive archive;
  archive["xyz"] = NpyArray::Of(Flat(map.xyz), {point_count, kSpaceDim});
  if (check(map.intensity.size(), "intensity")) {
    archive["intensity"] = NpyArray::Of(map.intensity, {point_count});
  }
  if (check(map.kf.size(), "kf")) {
    archive["kf"] = NpyArray::Of(map.kf, {point_count});
  }
  if (check(map.trip.size(), "trip")) {
    archive["trip"] = NpyArray::Of(map.trip, {point_count});
  }
  if (check(map.normals.size(), "normals")) {
    std::vector<float> normals;
    for (double coordinate : Flat(map.normals)) {
      normals.push_back(static_cast<float>(coordinate));
    }
    archive["normals"] = NpyArray::Of(normals, {point_count, kSpaceDim});
  }
  if (check(map.planarity.size(), "planarity")) {
    archive["planarity"] = NpyArray::Of(map.planarity, {point_count});
  }
  if (check(map.sigma.size(), "sigma")) {
    archive["sigma"] = NpyArray::Of(map.sigma, {point_count});
  }
  if (check(map.observations.size(), "observations")) {
    archive["observations"] = NpyArray::Of(map.observations, {point_count});
  }
  if (check(map.dynamic.size(), "dynamic")) {
    archive["dynamic"] = NpyArray::FromBools(map.dynamic, {point_count});
  }
  if (check(map.dynamic_kind.size(), "dynamic_kind")) {
    archive["dynamic_kind"] = NpyArray::Of(map.dynamic_kind, {point_count});
  }
  if (check(map.level.size(), "level")) {
    archive["level"] = NpyArray::Of(map.level, {point_count});
  }
  if (check(map.height.size(), "height")) {
    archive["height"] = NpyArray::Of(map.height, {point_count});
  }
  if (check(map.on_ramp.size(), "on_ramp")) {
    archive["on_ramp"] = NpyArray::FromBools(map.on_ramp, {point_count});
  }
  if (check(map.range.size(), "range")) {
    archive["range"] = NpyArray::Of(map.range, {point_count});
  }
  if (check(map.sequence.size(), "sequence")) {
    archive["sequence"] = NpyArray::Of(map.sequence, {point_count});
  }
  WriteNpz(path, archive, false);
}

void WriteLevels(const std::string& map_dir, const LevelModel& model) {
  fs::create_directories(map_dir);
  Value levels = Value::List();
  for (size_t k = 0; k < model.z.size(); ++k) {
    Value level = Value::Map();
    level.Set("id", Value::Integer(static_cast<int64_t>(k)));
    level.Set("z", Value::Number(model.z[k]));
    level.Set("ceiling", Value::Number(model.ceiling[k]));
    levels.Append(std::move(level));
  }
  Value ramps = Value::List();
  for (const Ramp& ramp : model.ramps) {
    Value item = Value::Map();
    item.Set("level_from", Value::Integer(ramp.level_from));
    item.Set("level_to", Value::Integer(ramp.level_to));
    item.Set("plane", Floats({ramp.plane.x(), ramp.plane.y(), ramp.plane.z()}));
    Value polygon = Value::List();
    for (const Vec2& vertex : ramp.polygon) {
      polygon.Append(Floats({vertex.x(), vertex.y()}));
    }
    item.Set("polygon", std::move(polygon));
    item.Set("slope", Value::Number(ramp.slope));
    ramps.Append(std::move(item));
  }
  Value json = Value::Map();
  json.Set("levels", std::move(levels));
  json.Set("ramps", std::move(ramps));
  WriteTextFile((fs::path(map_dir) / "levels.json").string(),
                ToJson(json, kLevelsJsonIndent));
  NpzArchive archive;
  std::vector<double> meta;
  for (size_t k = 0; k < model.grids.size(); ++k) {
    const FloorGrid& grid = model.grids[k];
    archive["g" + std::to_string(k)] =
        NpyArray::Of(grid.z, {grid.height, grid.width});
    meta.insert(meta.end(), {grid.x0, grid.y0, grid.res});
  }
  archive["meta"] =
      model.grids.empty()
          ? NpyArray::Of(meta, {0})
          : NpyArray::Of(meta, {static_cast<int64_t>(model.grids.size()),
                                kFloorGridMetaFields});
  WriteNpz((fs::path(map_dir) / "floor_grids.npz").string(), archive, false);
}

}  // namespace gr
