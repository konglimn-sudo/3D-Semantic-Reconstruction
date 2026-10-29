#include "gr/io/trip.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <stdexcept>

#include "gr/io/npy.h"
#include "gr/io/tables.h"

namespace gr {
namespace fs = std::filesystem;
namespace {

constexpr double kSecondsPerNanosecond = 1e-9;
// Stamps closer than this (s) are the same instant.
constexpr double kSameInstant = 1e-6;

}  // namespace

Pose PoseFromValue(const Value& value) {
  std::vector<double> values;
  for (const Value& item : value.items()) {
    if (item.is_list()) {
      for (const Value& entry : item.items()) {
        values.push_back(entry.AsDouble());
      }
    } else {
      values.push_back(item.AsDouble());
    }
  }
  if (values.size() != static_cast<size_t>(Mat4::SizeAtCompileTime)) {
    throw std::runtime_error("expected a 4x4 matrix");
  }
  Mat4 matrix;
  for (int row = 0; row < kHomogeneousDim; ++row) {
    for (int column = 0; column < kHomogeneousDim; ++column) {
      matrix(row, column) = values[row * kHomogeneousDim + column];
    }
  }
  return Pose::FromMatrix(matrix);
}

const LidarCalibration& Rig::primary_lidar() const {
  if (lidars.empty()) throw std::runtime_error("the rig has no LiDAR");
  return lidars.front();
}

Rig Rig::FromValue(const Value& value) {
  Rig rig;
  for (const auto& [name, entry] : value["lidars"].members()) {
    LidarCalibration lidar;
    lidar.name = name;
    lidar.vehicle_from_lidar = PoseFromValue(entry["T_vehicle_lidar"]);
    if (!entry["sweep_duration"].is_null()) {
      lidar.sweep_duration = entry["sweep_duration"].AsDouble();
    }
    rig.lidars.push_back(lidar);
  }
  if (value["cameras"].is_map()) {
    for (const auto& [name, entry] : value["cameras"].members()) {
      CameraCalibration camera;
      camera.name = name;
      camera.model = entry["model"].AsString();
      camera.width = entry["width"].AsInt();
      camera.height = entry["height"].AsInt();
      camera.params = entry["params"].AsDoubles();
      camera.vehicle_from_camera = PoseFromValue(entry["T_vehicle_cam"]);
      if (!entry["rolling_shutter_line_time"].is_null()) {
        camera.rolling_shutter_line_time =
            entry["rolling_shutter_line_time"].AsDouble();
      }
      if (!entry["time_offset"].is_null()) {
        camera.time_offset = entry["time_offset"].AsDouble();
      }
      rig.cameras.push_back(camera);
    }
  }
  if (!value["T_vehicle_imu"].is_null()) {
    rig.vehicle_from_imu = PoseFromValue(value["T_vehicle_imu"]);
  }
  if (!value["wheel_speed_scale"].is_null()) {
    rig.wheel_speed_scale = value["wheel_speed_scale"].AsDouble();
  }
  return rig;
}

std::vector<int64_t> StampedFiles(const std::string& directory,
                                  const std::vector<std::string>& suffixes) {
  std::vector<int64_t> out;
  if (!fs::is_directory(directory)) return out;
  for (const fs::directory_entry& entry : fs::directory_iterator(directory)) {
    std::string suffix = entry.path().extension().string();
    std::transform(
        suffix.begin(), suffix.end(), suffix.begin(),
        [](unsigned char character) { return std::tolower(character); });
    if (std::find(suffixes.begin(), suffixes.end(), suffix) == suffixes.end()) {
      continue;
    }
    const std::string stem = entry.path().stem().string();
    if (stem.empty() ||
        !std::all_of(stem.begin(), stem.end(), [](char character) {
          return std::isdigit(static_cast<unsigned char>(character));
        })) {
      continue;
    }
    out.push_back(std::stoll(stem));
  }
  std::sort(out.begin(), out.end());
  return out;
}

TripReader::TripReader(const std::string& root) : root_(root) {
  const fs::path dir(root);
  if (!fs::exists(dir / "rig.yaml")) {
    throw std::runtime_error(root +
                             " is not a trip directory (rig.yaml missing)");
  }
  if (fs::exists(dir / "meta.yaml")) {
    meta_ = ReadYamlFile((dir / "meta.yaml").string());
  }
  rig_ = Rig::FromValue(ReadYamlFile((dir / "rig.yaml").string()));
  stamps_ns_ = StampedFiles(
      (dir / "lidar" / rig_.primary_lidar().name).string(), {".npz"});
}

std::string TripReader::trip_id() const {
  const Value& id = meta_["trip_id"];
  if (id.is_string()) return id.AsString();
  if (id.is_number()) return ToJson(id, kJsonOneLine);
  return fs::path(root_).filename().string();
}

std::optional<Pose> TripReader::initial_pose() const {
  const Value& pose = meta_["initial_pose"];
  if (pose.is_null()) return std::nullopt;
  return PoseFromValue(pose);
}

std::optional<double> TripReader::initial_pose_time() const {
  const Value& time = meta_["initial_pose_time"];
  if (time.is_null()) return std::nullopt;
  return time.AsDouble();
}

double TripReader::lidar_stamp(size_t index) const {
  return static_cast<double>(stamps_ns_.at(index)) * kSecondsPerNanosecond;
}

LidarSweep TripReader::LoadLidar(size_t index) const {
  const int64_t stamp_ns = stamps_ns_.at(index);
  const fs::path path = fs::path(root_) / "lidar" / rig_.primary_lidar().name /
                        (std::to_string(stamp_ns) + ".npz");
  const NpzArchive npz = ReadNpz(path.string());
  std::vector<double> xyz;
  if (npz.count("points_q")) {  // Quantised: int16 * scale.
    const double scale = npz.at("scale").ToDoubles().at(0);
    xyz = npz.at("points_q").ToDoubles();
    for (double& coordinate : xyz) coordinate *= scale;
  } else if (npz.count("points")) {
    xyz = npz.at("points").ToDoubles();
  } else {
    throw std::runtime_error(path.string() + ": no points");
  }
  if (xyz.size() % kSpaceDim) {
    throw std::runtime_error(path.string() + ": not N x 3");
  }
  const size_t count = xyz.size() / kSpaceDim;
  const std::vector<float> intensity = npz.count("intensity")
                                           ? npz.at("intensity").ToFloats()
                                           : std::vector<float>(count, 0.0f);
  const std::vector<double> time = npz.count("time")
                                       ? npz.at("time").ToDoubles()
                                       : std::vector<double>(count, 0.0);
  if (intensity.size() != count || time.size() != count) {
    throw std::runtime_error(path.string() + ": arrays of different lengths");
  }
  LidarSweep sweep;
  sweep.index = static_cast<int>(index);
  sweep.stamp = static_cast<double>(stamp_ns) * kSecondsPerNanosecond;
  for (size_t k = 0; k < count; ++k) {
    const Vec3 point = Eigen::Map<const Vec3>(xyz.data() + kSpaceDim * k);
    if (!point.allFinite()) continue;
    sweep.points.push_back(point);
    sweep.intensity.push_back(intensity[k]);
    sweep.time.push_back(time[k]);
  }
  return sweep;
}

Eigen::MatrixXd TripReader::Table(const std::string& name) const {
  const fs::path path = fs::path(root_) / name;
  if (!fs::exists(path)) return Eigen::MatrixXd(0, 0);
  return ReadCsv(path.string());
}

Eigen::MatrixXd TripReader::Imu() const { return Table("imu.csv"); }
Eigen::MatrixXd TripReader::Wheel() const { return Table("wheel.csv"); }
Eigen::MatrixXd TripReader::Gnss() const { return Table("gnss.csv"); }
Eigen::MatrixXd TripReader::ControlPoints() const {
  return Table("control_points.csv");
}

ImuSamples TripReader::ImuSamplesInVehicle() const {
  const Eigen::MatrixXd rows = Imu();
  if (rows.rows() == 0) return ImuSamples();
  if (rows.cols() < ImuSamples::kCsvColumns) {
    throw std::runtime_error("imu.csv needs " +
                             std::to_string(ImuSamples::kCsvColumns) +
                             " columns");
  }
  return ImuSamples::FromRows(rows, rig_.vehicle_from_imu.rotation);
}

WheelSamples TripReader::WheelSamplesScaled() const {
  const Eigen::MatrixXd rows = Wheel();
  if (rows.rows() == 0) return WheelSamples();
  if (rows.cols() < WheelSamples::kCsvColumns) {
    throw std::runtime_error("wheel.csv needs " +
                             std::to_string(WheelSamples::kCsvColumns) +
                             " columns");
  }
  return WheelSamples::FromRows(rows, rig_.wheel_speed_scale);
}

MotionPrior TripReader::BuildMotionPrior() const {
  ImuSamples imu = ImuSamplesInVehicle();
  if (imu.size() <= 1) return MotionPrior();
  return MotionPrior(std::move(imu), WheelSamplesScaled());
}

std::optional<Pose> TripReader::EntrancePose(const MotionPrior& prior) const {
  const std::optional<Pose> ins = initial_pose();
  const std::optional<double> ins_time = initial_pose_time();
  if (!ins || !ins_time || size() == 0) return ins;
  const double first_stamp = lidar_stamp(0);
  if (std::abs(first_stamp - *ins_time) < kSameInstant || !prior.valid()) {
    return ins;
  }
  if (*ins_time < first_stamp) {
    const std::optional<Pose> delta = prior.Delta(*ins_time, first_stamp);
    return delta ? *ins * *delta : *ins;
  }
  const std::optional<Pose> delta = prior.Delta(first_stamp, *ins_time);
  return delta ? *ins * delta->Inverse() : *ins;
}

}  // namespace gr
