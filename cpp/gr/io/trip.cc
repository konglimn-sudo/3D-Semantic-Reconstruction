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

Pose PoseFromValue(const Value& value) {
  std::vector<double> m;
  for (const Value& item : value.items()) {
    if (item.is_list()) {
      for (const Value& v : item.items()) m.push_back(v.AsDouble());
    } else {
      m.push_back(item.AsDouble());
    }
  }
  if (m.size() != 16) throw std::runtime_error("expected a 4x4 matrix");
  Mat4 matrix;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) matrix(r, c) = m[r * 4 + c];
  }
  return Pose::FromMatrix(matrix);
}

const LidarCalibration& Rig::primary_lidar() const {
  if (lidars.empty()) throw std::runtime_error("the rig has no LiDAR");
  return lidars.front();
}

Rig Rig::FromValue(const Value& value) {
  Rig rig;
  for (const auto& [name, v] : value["lidars"].members()) {
    LidarCalibration lidar;
    lidar.name = name;
    lidar.vehicle_from_lidar = PoseFromValue(v["T_vehicle_lidar"]);
    if (!v["sweep_duration"].is_null()) {
      lidar.sweep_duration = v["sweep_duration"].AsDouble();
    }
    rig.lidars.push_back(lidar);
  }
  if (value["cameras"].is_map()) {
    for (const auto& [name, v] : value["cameras"].members()) {
      CameraCalibration camera;
      camera.name = name;
      camera.model = v["model"].AsString();
      camera.width = v["width"].AsInt();
      camera.height = v["height"].AsInt();
      camera.params = v["params"].AsDoubles();
      camera.vehicle_from_camera = PoseFromValue(v["T_vehicle_cam"]);
      if (!v["rolling_shutter_line_time"].is_null()) {
        camera.rolling_shutter_line_time =
            v["rolling_shutter_line_time"].AsDouble();
      }
      if (!v["time_offset"].is_null()) {
        camera.time_offset = v["time_offset"].AsDouble();
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
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (std::find(suffixes.begin(), suffixes.end(), suffix) ==
        suffixes.end()) {
      continue;
    }
    const std::string stem = entry.path().stem().string();
    if (stem.empty() || !std::all_of(stem.begin(), stem.end(), [](char c) {
          return std::isdigit(static_cast<unsigned char>(c));
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
  if (id.is_number()) return ToJson(id, -1);
  return fs::path(root_).filename().string();
}

std::optional<Pose> TripReader::initial_pose() const {
  const Value& pose = meta_["initial_pose"];
  if (pose.is_null()) return std::nullopt;
  return PoseFromValue(pose);
}

std::optional<double> TripReader::initial_pose_time() const {
  const Value& t = meta_["initial_pose_time"];
  if (t.is_null()) return std::nullopt;
  return t.AsDouble();
}

double TripReader::lidar_stamp(size_t index) const {
  return static_cast<double>(stamps_ns_.at(index)) * 1e-9;
}

LidarSweep TripReader::LoadLidar(size_t index) const {
  const int64_t ns = stamps_ns_.at(index);
  const fs::path path = fs::path(root_) / "lidar" /
                        rig_.primary_lidar().name /
                        (std::to_string(ns) + ".npz");
  const NpzArchive npz = ReadNpz(path.string());
  std::vector<double> xyz;
  if (npz.count("points_q")) {  // Quantised: int16 * scale.
    const double scale = npz.at("scale").ToDoubles().at(0);
    xyz = npz.at("points_q").ToDoubles();
    for (double& v : xyz) v *= scale;
  } else if (npz.count("points")) {
    xyz = npz.at("points").ToDoubles();
  } else {
    throw std::runtime_error(path.string() + ": no points");
  }
  if (xyz.size() % 3) throw std::runtime_error(path.string() + ": not N x 3");
  const size_t n = xyz.size() / 3;
  const std::vector<float> intensity =
      npz.count("intensity") ? npz.at("intensity").ToFloats()
                             : std::vector<float>(n, 0.0f);
  const std::vector<double> time = npz.count("time")
                                       ? npz.at("time").ToDoubles()
                                       : std::vector<double>(n, 0.0);
  if (intensity.size() != n || time.size() != n) {
    throw std::runtime_error(path.string() + ": arrays of different lengths");
  }
  LidarSweep sweep;
  sweep.index = static_cast<int>(index);
  sweep.stamp = static_cast<double>(ns) * 1e-9;
  for (size_t k = 0; k < n; ++k) {
    const Vec3 p(xyz[3 * k], xyz[3 * k + 1], xyz[3 * k + 2]);
    if (!p.allFinite()) continue;
    sweep.points.push_back(p);
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

ImuSamples TripReader::ImuSamplesInVehicle() const {
  const Eigen::MatrixXd rows = Imu();
  if (rows.rows() == 0) return ImuSamples();
  if (rows.cols() < 7) throw std::runtime_error("imu.csv needs 7 columns");
  return ImuSamples::FromRows(rows, rig_.vehicle_from_imu.rotation);
}

WheelSamples TripReader::WheelSamplesScaled() const {
  const Eigen::MatrixXd rows = Wheel();
  if (rows.rows() == 0) return WheelSamples();
  if (rows.cols() < 2) throw std::runtime_error("wheel.csv needs 2 columns");
  return WheelSamples::FromRows(rows, rig_.wheel_speed_scale);
}

MotionPrior TripReader::BuildMotionPrior() const {
  ImuSamples imu = ImuSamplesInVehicle();
  if (imu.size() <= 1) return MotionPrior();
  return MotionPrior(std::move(imu), WheelSamplesScaled());
}

std::optional<Pose> TripReader::EntrancePose(const MotionPrior& prior) const {
  const std::optional<Pose> ins = initial_pose();
  const std::optional<double> t_ins = initial_pose_time();
  if (!ins || !t_ins || size() == 0) return ins;
  const double t_first = lidar_stamp(0);
  if (std::abs(t_first - *t_ins) < 1e-6 || !prior.valid()) return ins;
  if (*t_ins < t_first) {
    const std::optional<Pose> delta = prior.Delta(*t_ins, t_first);
    return delta ? *ins * *delta : *ins;
  }
  const std::optional<Pose> delta = prior.Delta(t_first, *t_ins);
  return delta ? *ins * delta->Inverse() : *ins;
}

}  // namespace gr
