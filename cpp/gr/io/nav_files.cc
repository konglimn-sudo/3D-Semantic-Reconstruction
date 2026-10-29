#include "gr/io/nav_files.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stdexcept>

#include "gr/io/npy.h"
#include "gr/io/tables.h"

namespace gr {
namespace fs = std::filesystem;
namespace {

// Decimal digits localize.py keeps: tenths, hundredths, thousandths and
// shares to 1e-4.
constexpr int kTenths = 1;
constexpr int kHundredths = 2;
constexpr int kThousandths = 3;
constexpr int kShareDigits = 4;

// Room for a double printed with "%.*f" and at most kShareDigits digits.
constexpr size_t kRoundBuffer = 64;
// Room for one line of sweeps.csv.
constexpr size_t kSweepLineBuffer = 512;

// The grid of a level_<k>_layers.npz: [x0, y0, res, width, height, level].
constexpr size_t kGridX0 = 0;
constexpr size_t kGridY0 = 1;
constexpr size_t kGridResolution = 2;
constexpr size_t kGridWidth = 3;
constexpr size_t kGridHeight = 4;
constexpr size_t kGridLevel = 5;
constexpr size_t kGridFields = 6;

constexpr int kZ = 2;  // The z coordinate of an (x, y, z) list.

// Python's round(x, digits) (correctly rounded decimal).
double Round(double x, int digits) {
  char buffer[kRoundBuffer];
  std::snprintf(buffer, sizeof(buffer), "%.*f", digits, x);
  return std::strtod(buffer, nullptr);
}

// numpy.round(x, digits): rint(x * 10^digits) / 10^digits.
double NumpyRound(double x, int digits) {
  constexpr double kDecimal = 10.0;
  const double scale = std::pow(kDecimal, digits);
  return std::nearbyint(x * scale) / scale;
}

// [[x, y], ...] as points.
std::vector<Vec2> PointsXy(const Value& list) {
  std::vector<Vec2> out;
  for (const Value& point : list.items()) {
    out.emplace_back(point[0].AsDouble(), point[1].AsDouble());
  }
  return out;
}

// A vector rounded to `digits` as a list.
Value RoundedList(const Vec3& vector, int digits) {
  Value out = Value::List();
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    out.Append(Value::Number(Round(vector[axis], digits)));
  }
  return out;
}

}  // namespace

Points ReadLocalizationMap(const std::string& nav_dir) {
  const std::string path =
      (fs::path(nav_dir) / "localization_map.npz").string();
  const NpzArchive npz = ReadNpz(path);
  const auto found = npz.find("xyz");
  if (found == npz.end() || found->second.columns() != kSpaceDim) {
    throw std::runtime_error(path + ": no (N, 3) xyz");
  }
  const std::vector<double> xyz = found->second.ToDoubles();
  Points out(xyz.size() / kSpaceDim);
  for (size_t k = 0; k < out.size(); ++k) {
    out[k] = Eigen::Map<const Vec3>(xyz.data() + kSpaceDim * k);
  }
  return out;
}

std::vector<NavEdge> ReadNavEdges(const std::string& graph_json) {
  const Value graph = ReadJsonFile(graph_json);
  std::vector<NavEdge> out;
  for (const Value& entry : graph["edges"].items()) {
    NavEdge edge;
    edge.level = entry["level"].AsInt();
    edge.kind = entry["kind"].AsString();
    edge.points = PointsXy(entry["points"]);
    out.push_back(std::move(edge));
  }
  return out;
}

std::vector<NavSlot> ReadNavSlots(const std::string& goals_json) {
  const Value goals = ReadJsonFile(goals_json);
  std::vector<NavSlot> out;
  for (const Value& goal : goals["goals"].items()) {
    if (goal.Find("corners") == nullptr) continue;
    NavSlot slot;
    slot.level = goal["level"].AsInt();
    slot.slot_id = goal["slot_id"].AsString();
    slot.corners = PointsXy(goal["corners"]);
    out.push_back(std::move(slot));
  }
  return out;
}

std::vector<SlotGeometry> ReadSlotGeometry(const std::string& goals_json) {
  const Value goals = ReadJsonFile(goals_json);
  std::vector<SlotGeometry> out;
  for (const Value& goal : goals["goals"].items()) {
    if (goal.Find("corners") == nullptr ||
        goal["corners"].size() != kSlotCorners) {
      continue;
    }
    SlotGeometry slot;
    slot.slot_id = goal["slot_id"].AsString();
    slot.level = goal["level"].AsInt();
    const std::vector<Vec2> corners = PointsXy(goal["corners"]);
    std::copy(corners.begin(), corners.end(), slot.corners.begin());
    slot.z = goal["park"]["xyz"][kZ].AsDouble();
    out.push_back(slot);
  }
  return out;
}

Value ChangesValue(const std::vector<MapChange>& changes, int sweeps) {
  Value list = Value::List();
  int added = 0;
  int blocking = 0;
  for (const MapChange& change : changes) {
    added += change.cluster.added;
    blocking += change.cluster.added && !change.edges.empty();
    Value item = Value::Map();
    item.Set("kind", Value::String(change.cluster.added ? "added" : "removed"));
    item.Set("level", change.level ? Value::Integer(*change.level) : Value());
    item.Set("center", RoundedList(change.cluster.center, kHundredths));
    item.Set("min", RoundedList(change.cluster.min, kHundredths));
    item.Set("max", RoundedList(change.cluster.max, kHundredths));
    item.Set("voxels", Value::Integer(change.cluster.voxels));
    Value edges = Value::List();
    for (int edge : change.edges) edges.Append(Value::Integer(edge));
    item.Set("edges", edges);
    list.Append(item);
  }
  Value summary = Value::Map();
  summary.Set("sweeps", Value::Integer(sweeps));
  summary.Set("added", Value::Integer(added));
  summary.Set("removed",
              Value::Integer(static_cast<int64_t>(changes.size()) - added));
  summary.Set("added_on_routes", Value::Integer(blocking));
  Value out = Value::Map();
  out.Set("summary", summary);
  out.Set("changes", list);
  return out;
}

Value SlotStatesValue(const std::vector<SlotGeometry>& slots,
                      const SlotStateTracker& tracker) {
  static const char* const kStateNames[kSlotStateCount] = {"unknown", "free",
                                                           "occupied"};
  Value list = Value::List();
  std::array<int, kSlotStateCount> counts{};
  for (size_t k = 0; k < slots.size(); ++k) {
    const SlotStateTracker::Entry& entry = tracker.entries()[k];
    ++counts[static_cast<int>(entry.state)];
    Value slot = Value::Map();
    slot.Set("slot_id", Value::String(slots[k].slot_id));
    slot.Set("level", Value::Integer(slots[k].level));
    slot.Set("state",
             Value::String(kStateNames[static_cast<int>(entry.state)]));
    slot.Set("stamp", entry.state == SlotState::kUnknown
                          ? Value()
                          : Value::Number(Round(entry.stamp, kThousandths)));
    slot.Set("observations", Value::Integer(entry.observations));
    list.Append(slot);
  }
  Value summary = Value::Map();
  for (int state = 0; state < kSlotStateCount; ++state) {
    summary.Set(kStateNames[state], Value::Integer(counts[state]));
  }
  Value out = Value::Map();
  out.Set("summary", summary);
  out.Set("slots", list);
  return out;
}

std::vector<FloorHeights> ReadFloorHeights(const std::string& nav_dir) {
  static const std::regex kName("level_.*_layers\\.npz");
  std::vector<std::string> paths;
  for (const fs::directory_entry& entry : fs::directory_iterator(nav_dir)) {
    const std::string name = entry.path().filename().string();
    if (std::regex_match(name, kName)) paths.push_back(entry.path().string());
  }
  std::sort(paths.begin(), paths.end());
  std::vector<FloorHeights> out;
  for (const std::string& path : paths) {
    const NpzArchive npz = ReadNpz(path);
    if (!npz.count("floor_z") || !npz.count("grid")) {
      throw std::runtime_error(path + ": floor_z or grid missing");
    }
    const std::vector<double> grid = npz.at("grid").ToDoubles();
    if (grid.size() < kGridFields) {
      throw std::runtime_error(path + ": bad grid");
    }
    FloorHeights floor;
    floor.x0 = grid[kGridX0];
    floor.y0 = grid[kGridY0];
    floor.res = grid[kGridResolution];
    floor.width = static_cast<int>(grid[kGridWidth]);
    floor.height = static_cast<int>(grid[kGridHeight]);
    floor.level = static_cast<int>(grid[kGridLevel]);
    floor.z = npz.at("floor_z").ToDoubles();
    if (floor.z.size() != static_cast<size_t>(floor.width) * floor.height) {
      throw std::runtime_error(path + ": floor_z does not match the grid");
    }
    out.push_back(std::move(floor));
  }
  return out;
}

const char* StatusName(LocalizationStatus status) {
  switch (status) {
    case LocalizationStatus::kInitializing:
      return "initializing";
    case LocalizationStatus::kTracking:
      return "tracking";
    case LocalizationStatus::kLost:
      return "lost";
  }
  return "?";
}

Value SummaryValue(const TripLocalizationSummary& summary) {
  const bool tracked = summary.first_tracked >= 0;
  const auto optional = [](bool defined, double value) {
    return defined ? Value::Number(value) : Value();
  };
  Value out = Value::Map();
  out.Set("sweeps", Value::Integer(summary.sweeps));
  out.Set("first_tracked",
          tracked ? Value::Integer(summary.first_tracked) : Value());
  out.Set("init_time_s",
          optional(tracked, Round(summary.init_time_s, kHundredths)));
  out.Set("init_distance_m",
          optional(tracked, Round(summary.init_distance_m, kHundredths)));
  out.Set("placement_attempts", Value::Integer(summary.placement_attempts));
  out.Set("tracked", Value::Integer(summary.tracked));
  out.Set("registered", Value::Integer(summary.registered));
  out.Set("registered_share_of_tracked",
          optional(tracked,
                   Round(summary.registered_share_of_tracked, kShareDigits)));
  out.Set("lost_sweeps", Value::Integer(summary.lost_sweeps));
  out.Set("degenerate_share",
          optional(tracked, Round(summary.degenerate_share, kShareDigits)));
  out.Set("ms_median", Value::Number(Round(summary.ms_median, kTenths)));
  out.Set("ms_p95", Value::Number(Round(summary.ms_p95, kTenths)));
  return out;
}

Value PositionValue(const NavPosition& position) {
  Value out = Value::Map();
  Value xyz = Value::List();
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    xyz.Append(Value::Number(NumpyRound(position.xyz[axis], kThousandths)));
  }
  out.Set("xyz", xyz);
  out.Set("yaw_deg", Value::Number(Round(position.yaw_deg, kHundredths)));
  out.Set("level", position.level ? Value::Integer(*position.level) : Value());
  if (position.edge) {
    Value edge = Value::Map();
    edge.Set("edge", Value::Integer(position.edge->edge));
    edge.Set("kind", Value::String(position.edge->kind));
    edge.Set("distance_m",
             Value::Number(Round(position.edge->distance, kThousandths)));
    edge.Set("along_m",
             Value::Number(Round(position.edge->along, kHundredths)));
    edge.Set("lateral_m",
             Value::Number(Round(position.edge->lateral, kThousandths)));
    edge.Set("heading_rel_deg",
             Value::Number(Round(position.edge->heading_rel_deg, kTenths)));
    out.Set("edge", edge);
  } else {
    out.Set("edge", Value());
  }
  out.Set("slot", position.slot ? Value::String(*position.slot) : Value());
  return out;
}

void WriteLocalizationResult(const std::string& directory,
                             const std::vector<SweepLocalization>& sweeps,
                             const Value& report) {
  fs::create_directories(directory);
  const fs::path dir(directory);
  Trajectory tracked;
  for (const SweepLocalization& sweep : sweeps) {
    if (sweep.result.status != LocalizationStatus::kTracking) continue;
    tracked.stamps.push_back(sweep.stamp);
    tracked.poses.push_back(sweep.map_from_vehicle);
  }
  WriteTum((dir / "poses.tum").string(), tracked);
  std::ofstream csv(dir / "sweeps.csv");
  if (!csv) throw std::runtime_error("cannot write sweeps.csv");
  csv << "stamp,status,registered,fitness,rmse,degenerate_t,step_t,"
         "step_r,eigen_ratio_t,sigma_t,sigma_yaw_deg,ms,x,y,z\n";
  char line[kSweepLineBuffer];
  for (const SweepLocalization& sweep : sweeps) {
    const LocalizationResult& result = sweep.result;
    const Vec3& position = sweep.map_from_vehicle.translation;
    std::snprintf(line, sizeof(line),
                  "%.6f,%s,%d,%.4f,%.4f,%d,%.4f,%.3f,%.2e,%.4f,%.4f,%.1f,"
                  "%.3f,%.3f,%.3f\n",
                  sweep.stamp, StatusName(result.status),
                  result.registered ? 1 : 0, result.fitness, result.rmse,
                  result.degenerate_t, result.step_t, result.step_r,
                  result.min_eigen_ratio_t, sweep.sigma_t, sweep.sigma_yaw_deg,
                  sweep.ms, position.x(), position.y(), position.z());
    csv << line;
  }
  std::ofstream json(dir / "localization.json");
  if (!json) throw std::runtime_error("cannot write localization.json");
  json << ToJson(report, kDefaultJsonIndent);
}

}  // namespace gr
