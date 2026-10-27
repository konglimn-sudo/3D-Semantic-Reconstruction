#include "gr/io/config.h"

#include <stdexcept>

namespace gr {
namespace {

#include "gr/io/config_tables.inc"

// Sets the leaves of `table` (under `path` of `out`) that `user` does not.
void SetUnlessGiven(const Value& table, const Value& user,
                    std::vector<std::string>* path, Value* out) {
  for (const auto& [key, value] : table.members()) {
    path->push_back(key);
    if (value.is_map()) {
      SetUnlessGiven(value, user, path, out);
    } else {
      const Value* given = &user;
      for (const std::string& k : *path) {
        given = given->Find(k);
        if (given == nullptr) break;
      }
      if (given == nullptr || given->is_null()) {
        Value* section = out;
        for (size_t k = 0; k + 1 < path->size(); ++k) {
          Value* child = section->Find((*path)[k]);
          section = child != nullptr && child->is_map()
                        ? child
                        : &section->Set((*path)[k], Value::Map());
        }
        section->Set(path->back(), value);
      }
    }
    path->pop_back();
  }
}

}  // namespace

Value ApplyImuGrade(const Value& config, const Value& user) {
  const Value& grade = config["imu_grade"];
  if (grade.is_null() || (config["imu_grade_applied"].kind() ==
                              Value::Kind::kBool &&
                          config["imu_grade_applied"].AsBool())) {
    return config;
  }
  const Value* table = config["imu_grades"].Find(grade.AsString());
  if (table == nullptr) {
    throw std::runtime_error("unknown imu_grade " + grade.AsString());
  }
  Value out = config;
  std::vector<std::string> path;
  SetUnlessGiven(*table, user, &path, &out);
  out.Set("imu_grade_applied", Value::Bool(true));
  return out;
}

Value LoadConfig(const std::vector<std::string>& paths) {
  Value config = Value::Map();
  Value user = Value::Map();
  for (size_t k = 0; k < paths.size(); ++k) {
    const Value file = ReadYamlFile(paths[k]);
    if (file.is_null()) continue;
    config = DeepMerge(config, file);
    if (k > 0) user = DeepMerge(user, file);
  }
  return ApplyImuGrade(config, user);
}

OdometryConfig OdometryConfigFrom(const Value& config) {
  OdometryConfig out;
  const Value& section = config["odometry"];
  OdometryConfigFields().Fill(section, &out);
  out.lio = LioConfig();
  LioConfigFields().Fill(section["lio"], &out.lio);
  return out;
}

KeyframeConfig KeyframeConfigFrom(const Value& config) {
  KeyframeConfig out;
  KeyframeConfigFields().Fill(config["keyframes"], &out);
  return out;
}

LocalizerConfig LocalizerConfigFrom(const Value& config) {
  LocalizerConfig out;
  const Value& section = config["localization"];
  LocalizerConfigFields().Fill(section, &out);
  // As localize.localizer_config: the placement starts from its own
  // defaults, not the localizer's.
  out.placement = PlacementConfig();
  PlacementConfigFields().Fill(section["placement"], &out.placement);
  return out;
}

TripLocalizerConfig TripLocalizerConfigFrom(const Value& config) {
  TripLocalizerConfig out;
  out.odometry = OdometryConfigFrom(config);
  out.keyframes = KeyframeConfigFrom(config);
  out.localizer = LocalizerConfigFrom(config);
  const Value& window = config["localization"]["level_window"];
  if (!window.is_null()) out.level_window = window.AsDouble();
  return out;
}

SlotStateConfig SlotStateConfigFrom(const Value& config) {
  SlotStateConfig out;
  SlotStateConfigFields().Fill(config["nav"]["slot_states"], &out);
  return out;
}

ChangeConfig ChangeConfigFrom(const Value& config) {
  ChangeConfig out;
  ChangeConfigFields().Fill(config["nav"]["changes"], &out);
  return out;
}

GraphStageConfig GraphStageConfigFrom(const Value& config) {
  GraphStageConfig out;
  const Value& loops = config["loops"];
  LoopConfigFields().Fill(loops, &out.loops);
  AppearanceLoopConfigFields().Fill(loops["appearance"], &out.appearance);
  BackendConfigFields().Fill(config["backend"], &out.backend);
  TripAlignConfigFields().Fill(config["trip_align"], &out.trip_align);
  LidarBaConfigFields().Fill(config["ba"], &out.ba);
  RefineConfigFields().Fill(config["refine"], &out.refine);
  return out;
}

MapFusionConfig MapFusionConfigFrom(const Value& config) {
  MapFusionConfig out;
  MapFusionConfigFields().Fill(config["map"]["fusion"], &out);
  return out;
}

DynamicConfig DynamicConfigFrom(const Value& config) {
  DynamicConfig out;
  DynamicConfigFields().Fill(config["map"]["dynamic"], &out);
  return out;
}

LevelConfig LevelConfigFrom(const Value& config) {
  LevelConfig out;
  LevelConfigFields().Fill(config["levels"], &out);
  return out;
}

LidarImuConfig LidarImuConfigFrom(const Value& config) {
  LidarImuConfig out;
  LidarImuConfigFields().Fill(config["calibration"]["lidar_imu"], &out);
  return out;
}

double CorridorHalfWidth(const Value& config) {
  const Value& vehicle = config["nav"]["vehicle"];
  const double width =
      vehicle["width"].is_null() ? 1.9 : vehicle["width"].AsDouble();
  const double margin =
      vehicle["margin"].is_null() ? 0.3 : vehicle["margin"].AsDouble();
  return 0.5 * width + margin;
}

}  // namespace gr
