#include "gr/io/recon_files.h"

#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "gr/io/digest.h"
#include "gr/io/npy.h"
#include "gr/io/tables.h"

namespace gr {
namespace {

namespace fs = std::filesystem;

NpyArray PosesArray(const std::vector<Pose>& poses) {
  std::vector<double> values;
  values.reserve(16 * poses.size());
  for (const Pose& pose : poses) {
    const Mat4 m = pose.Matrix();
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) values.push_back(m(r, c));
    }
  }
  return NpyArray::Of(values, {static_cast<int64_t>(poses.size()), 4, 4});
}

std::vector<Pose> PosesFrom(const NpyArray& array, const std::string& name) {
  const std::vector<double> values = array.ToDoubles();
  if (values.size() % 16 != 0) {
    throw std::runtime_error(name + ": (N, 4, 4) poses expected");
  }
  std::vector<Pose> out(values.size() / 16);
  for (size_t k = 0; k < out.size(); ++k) {
    Mat4 m;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) m(r, c) = values[16 * k + 4 * r + c];
    }
    out[k] = Pose::FromMatrix(m);
  }
  return out;
}

template <int N>
NpyArray RowsArray(const std::vector<Eigen::Matrix<double, N, 1>>& rows) {
  std::vector<double> values;
  values.reserve(N * rows.size());
  for (const auto& row : rows) {
    for (int k = 0; k < N; ++k) values.push_back(row[k]);
  }
  return NpyArray::Of(values, {static_cast<int64_t>(rows.size()), N});
}

template <int N>
std::vector<Eigen::Matrix<double, N, 1>> RowsFrom(const NpyArray& array) {
  const std::vector<double> values = array.ToDoubles();
  std::vector<Eigen::Matrix<double, N, 1>> out(values.size() / N);
  for (size_t k = 0; k < out.size(); ++k) {
    for (int c = 0; c < N; ++c) out[k][c] = values[N * k + c];
  }
  return out;
}

const NpyArray& Member(const NpzArchive& archive, const std::string& name,
                       const std::string& path) {
  const auto it = archive.find(name);
  if (it == archive.end()) throw std::runtime_error(path + ": no " + name);
  return it->second;
}

std::vector<double> Flat(const Points& points) {
  std::vector<double> values;
  values.reserve(3 * points.size());
  for (const Vec3& p : points) values.insert(values.end(), {p[0], p[1], p[2]});
  return values;
}

std::vector<double> Flat(const Eigen::MatrixXd& table) {
  std::vector<double> values;  // C order.
  values.reserve(static_cast<size_t>(table.size()));
  for (Eigen::Index r = 0; r < table.rows(); ++r) {
    for (Eigen::Index c = 0; c < table.cols(); ++c) {
      values.push_back(table(r, c));
    }
  }
  return values;
}

std::vector<double> Flat(const std::vector<Pose>& poses) {
  std::vector<double> values;
  for (const Pose& pose : poses) {
    const Mat4 m = pose.Matrix();
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) values.push_back(m(r, c));
    }
  }
  return values;
}

std::vector<double> Flat(const std::vector<Vec6>& rows) {
  std::vector<double> values;
  for (const Vec6& row : rows) {
    for (int k = 0; k < 6; ++k) values.push_back(row[k]);
  }
  return values;
}

Value Floats(const std::vector<double>& values) {
  Value out = Value::List();
  for (double v : values) out.Append(Value::Number(v));
  return out;
}

// A rig.yaml matrix (nested lists or 16 numbers) as a 4x4 list of floats.
Value MatrixOf(const Value& value) {
  std::vector<double> flat;
  if (value.is_list()) {
    for (const Value& item : value.items()) {
      if (item.is_list()) {
        for (const Value& v : item.items()) flat.push_back(v.AsDouble());
      } else {
        flat.push_back(item.AsDouble());
      }
    }
  }
  if (flat.size() != 16) throw std::runtime_error("4x4 matrix expected");
  Value out = Value::List();
  for (int r = 0; r < 4; ++r) {
    out.Append(Floats(std::vector<double>(flat.begin() + 4 * r,
                                          flat.begin() + 4 * r + 4)));
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
  const Mat4 m = pose.Matrix();
  Value out = Value::List();
  for (int r = 0; r < 4; ++r) {
    out.Append(Floats({m(r, 0), m(r, 1), m(r, 2), m(r, 3)}));
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
  char name[32];
  std::snprintf(name, sizeof(name), "%06lld.npz",
                static_cast<long long>(frame));
  return (fs::path(trip_dir) / sweeps / name).string();
}

void WriteSweep(const std::string& path, const Points& points,
                const std::vector<float>& intensity) {
  fs::create_directories(fs::path(path).parent_path());
  std::vector<float> xyz;
  xyz.reserve(3 * points.size());
  for (const Vec3& p : points) {
    for (int k = 0; k < 3; ++k) xyz.push_back(static_cast<float>(p[k]));
  }
  NpzArchive archive;
  archive["points"] =
      NpyArray::Of(xyz, {static_cast<int64_t>(points.size()), 3});
  archive["intensity"] = NpyArray::FromHalfs(
      intensity, {static_cast<int64_t>(intensity.size())});
  WriteNpz(path, archive, false);
}

SweepFile ReadSweep(const std::string& path) {
  const NpzArchive archive = ReadNpz(path);
  const std::vector<double> xyz = Member(archive, "points", path).ToDoubles();
  SweepFile out;
  out.points.resize(xyz.size() / 3);
  for (size_t k = 0; k < out.points.size(); ++k) {
    out.points[k] = Vec3(xyz[3 * k], xyz[3 * k + 1], xyz[3 * k + 2]);
  }
  out.intensity = Member(archive, "intensity", path).ToFloats();
  return out;
}

void WriteKeyframeTable(const std::string& path, const KeyframeTable& table) {
  fs::create_directories(fs::path(path).parent_path());
  const int64_t n = static_cast<int64_t>(table.stamp.size());
  NpzArchive archive;
  archive["poses"] = PosesArray(table.poses);
  archive["odom"] = PosesArray(table.odom);
  archive["trip"] = NpyArray::Of(table.trip, {n});
  archive["frame"] = NpyArray::Of(table.frame, {n});
  archive["stamp"] = NpyArray::Of(table.stamp, {n});
  archive["velocity"] = RowsArray<3>(table.inertial.velocities);
  archive["bias"] = RowsArray<6>(table.inertial.biases);
  archive["inertial_valid"] = NpyArray::FromBools(
      table.inertial.valid,
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
    out.inertial.biases = RowsFrom<6>(archive.at("bias"));
  }
  if (archive.count("velocity")) {
    out.inertial.velocities = RowsFrom<3>(archive.at("velocity"));
  }
  if (archive.count("inertial_valid")) {
    for (int64_t v : archive.at("inertial_valid").ToInts()) {
      out.inertial.valid.push_back(v != 0);
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
  const GraphSummary& s = result.summary;
  Value out = Value::Map();
  out.Set("iterations", Value::Integer(s.iterations));
  out.Set("initial_cost", Value::Number(s.initial_cost));
  out.Set("final_cost", Value::Number(s.final_cost));
  out.Set("converged", Value::Bool(s.converged));
  out.Set("loop_chi2_median", Value::Number(s.loop_chi2_median));
  out.Set("loop_chi2_max", Value::Number(s.loop_chi2_max));
  out.Set("imu_factors", Value::Integer(s.imu_factors));
  out.Set("imu_chi2_median", Value::Number(s.imu_chi2_median));
  out.Set("keyframes", Value::Integer(static_cast<long long>(keyframes)));
  out.Set("loops",
          Value::Integer(static_cast<long long>(result.loops.size())));
  out.Set("anchors",
          Value::Integer(static_cast<long long>(result.anchors.size())));
  out.Set("gravity_factors",
          Value::Integer(static_cast<long long>(result.gravity.size())));
  out.Set("loops_tested", Value::Integer(result.loop_stats.tested));
  out.Set("loops_verified", Value::Integer(result.loops_verified));
  Value rounds = Value::List();
  for (const LidarBaRound& r : result.ba.rounds) {
    Value round = Value::Map();
    round.Set("planes", Value::Integer(r.planes));
    round.Set("observations", Value::Integer(r.observations));
    round.Set("rms_median_before", Value::Number(r.rms_median_before));
    round.Set("rms_p90_before", Value::Number(r.rms_p90_before));
    round.Set("rms_median_after", Value::Number(r.rms_median_after));
    round.Set("rms_p90_after", Value::Number(r.rms_p90_after));
    round.Set("shift_median", Value::Number(r.shift_median));
    round.Set("shift_max", Value::Number(r.shift_max));
    round.Set("iterations", Value::Integer(r.iterations));
    rounds.Append(std::move(round));
  }
  out.Set("ba_rounds", std::move(rounds));
  Value alignments = Value::List();
  for (const TripAlignment& a : result.trip_alignments) {
    Value item = Value::Map();
    item.Set("trip", Value::String(trip_ids.at(static_cast<size_t>(a.trip))));
    item.Set("aligned", Value::Bool(a.aligned));
    item.Set("T", MatrixValue(a.correction));
    item.Set("score", Value::Number(a.score));
    item.Set("fitness", Value::Number(a.fitness));
    item.Set("rmse", Value::Number(a.rmse));
    item.Set("runner_up", Value::Number(a.runner_up));
    item.Set("hypotheses", Value::Integer(a.hypotheses));
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
  for (const auto& [name, v] : rig["lidars"].members()) {
    Value lidar = Value::Map();
    lidar.Set("T_vehicle_lidar", MatrixOf(v["T_vehicle_lidar"]));
    lidar.Set("sweep_duration",
              Value::Number(DoubleOr(v["sweep_duration"], 0.1)));
    lidars.Set(name, std::move(lidar));
  }
  Value cameras = Value::Map();
  if (rig["cameras"].is_map()) {
    for (const auto& [name, v] : rig["cameras"].members()) {
      Value camera = Value::Map();
      camera.Set("model", Value::String(v["model"].AsString()));
      camera.Set("width", Value::Integer(v["width"].AsInt()));
      camera.Set("height", Value::Integer(v["height"].AsInt()));
      camera.Set("params", Floats(v["params"].AsDoubles()));
      camera.Set("T_vehicle_cam", MatrixOf(v["T_vehicle_cam"]));
      camera.Set("rolling_shutter_line_time",
                 Value::Number(DoubleOr(v["rolling_shutter_line_time"], 0.0)));
      camera.Set("time_offset",
                 Value::Number(DoubleOr(v["time_offset"], 0.0)));
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
          Value::Number(DoubleOr(rig["wheel_speed_scale"], 1.0)));
  return out;
}

PoseSetIds ComputePoseSetIds(const std::vector<TripPoses>& trips,
                             const std::vector<Eigen::MatrixXd>& imu,
                             const std::vector<Eigen::MatrixXd>& wheel) {
  PythonDigest lidar;
  std::vector<Value> cameras;
  for (size_t t = 0; t < trips.size(); ++t) {
    const TripPoses& trip = trips[t];
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
    lidar.AddDoubles(Flat(imu.at(t)));
    lidar.AddDoubles(Flat(wheel.at(t)));
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
    archive["biases"] = RowsArray<6>(trip.biases);
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
  WriteTextFile((out / "pose_set.json").string(), ToJson(meta, 1));
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
    trip.biases = RowsFrom<6>(Member(archive, "biases", path));
    out.push_back(std::move(trip));
  }
  return out;
}

void RecordProvenance(const std::string& work_dir, const std::string& stage,
                      const PoseSetIds& ids, bool uses_cameras) {
  const std::string path = (fs::path(work_dir) / "provenance.json").string();
  Value entries = fs::exists(path) ? ReadJsonFile(path) : Value::Map();
  const std::time_t now = std::time(nullptr);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S",
                std::localtime(&now));
  Value entry = Value::Map();
  entry.Set("pose_set_id", Value::String(ids.pose_set_id));
  entry.Set("trajectory_id", Value::String(ids.trajectory_id));
  entry.Set("uses_cameras", Value::Bool(uses_cameras));
  entry.Set("time", Value::String(stamp));
  entries.Set(stage, std::move(entry));
  WriteTextFile(path, PythonJsonDumps(entries, 1));
}

void WritePointMap(const std::string& path, const PointMapFile& map) {
  fs::create_directories(fs::path(path).parent_path());
  const int64_t n = static_cast<int64_t>(map.xyz.size());
  const auto check = [&](size_t size, const char* name) {
    if (size != 0 && static_cast<int64_t>(size) != n) {
      throw std::runtime_error(std::string("point map: ") + name +
                               " has the wrong length");
    }
    return size != 0;
  };
  NpzArchive archive;
  archive["xyz"] = NpyArray::Of(Flat(map.xyz), {n, 3});
  if (check(map.intensity.size(), "intensity")) {
    archive["intensity"] = NpyArray::Of(map.intensity, {n});
  }
  if (check(map.kf.size(), "kf")) archive["kf"] = NpyArray::Of(map.kf, {n});
  if (check(map.trip.size(), "trip")) {
    archive["trip"] = NpyArray::Of(map.trip, {n});
  }
  if (check(map.normals.size(), "normals")) {
    std::vector<float> normals;
    for (double v : Flat(map.normals)) normals.push_back(static_cast<float>(v));
    archive["normals"] = NpyArray::Of(normals, {n, 3});
  }
  if (check(map.planarity.size(), "planarity")) {
    archive["planarity"] = NpyArray::Of(map.planarity, {n});
  }
  if (check(map.sigma.size(), "sigma")) {
    archive["sigma"] = NpyArray::Of(map.sigma, {n});
  }
  if (check(map.observations.size(), "observations")) {
    archive["observations"] = NpyArray::Of(map.observations, {n});
  }
  if (check(map.dynamic.size(), "dynamic")) {
    archive["dynamic"] = NpyArray::FromBools(map.dynamic, {n});
  }
  if (check(map.dynamic_kind.size(), "dynamic_kind")) {
    archive["dynamic_kind"] = NpyArray::Of(map.dynamic_kind, {n});
  }
  if (check(map.level.size(), "level")) {
    archive["level"] = NpyArray::Of(map.level, {n});
  }
  if (check(map.height.size(), "height")) {
    archive["height"] = NpyArray::Of(map.height, {n});
  }
  if (check(map.on_ramp.size(), "on_ramp")) {
    archive["on_ramp"] = NpyArray::FromBools(map.on_ramp, {n});
  }
  WriteNpz(path, archive, false);
}

void WriteLevels(const std::string& map_dir, const LevelModel& model) {
  fs::create_directories(map_dir);
  Value levels = Value::List();
  for (size_t k = 0; k < model.z.size(); ++k) {
    Value level = Value::Map();
    level.Set("id", Value::Integer(static_cast<long long>(k)));
    level.Set("z", Value::Number(model.z[k]));
    level.Set("ceiling", Value::Number(model.ceiling[k]));
    levels.Append(std::move(level));
  }
  Value ramps = Value::List();
  for (const Ramp& ramp : model.ramps) {
    Value item = Value::Map();
    item.Set("level_from", Value::Integer(ramp.level_from));
    item.Set("level_to", Value::Integer(ramp.level_to));
    item.Set("plane",
             Floats({ramp.plane.x(), ramp.plane.y(), ramp.plane.z()}));
    Value polygon = Value::List();
    for (const Vec2& p : ramp.polygon) polygon.Append(Floats({p.x(), p.y()}));
    item.Set("polygon", std::move(polygon));
    item.Set("slope", Value::Number(ramp.slope));
    ramps.Append(std::move(item));
  }
  Value json = Value::Map();
  json.Set("levels", std::move(levels));
  json.Set("ramps", std::move(ramps));
  WriteTextFile((fs::path(map_dir) / "levels.json").string(), ToJson(json, 2));
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
          : NpyArray::Of(meta, {static_cast<int64_t>(model.grids.size()), 3});
  WriteNpz((fs::path(map_dir) / "floor_grids.npz").string(), archive, false);
}

}  // namespace gr
