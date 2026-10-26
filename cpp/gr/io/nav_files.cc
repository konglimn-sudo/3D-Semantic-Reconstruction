#include "gr/io/nav_files.h"

#include <algorithm>
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

// Python's round(x, digits) (correctly rounded decimal).
double Round(double x, int digits) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.*f", digits, x);
  return std::strtod(buffer, nullptr);
}

// numpy.round(x, digits): rint(x * 10^digits) / 10^digits.
double NumpyRound(double x, int digits) {
  const double scale = std::pow(10.0, digits);
  return std::nearbyint(x * scale) / scale;
}

std::vector<Vec2> PointsXy(const Value& list) {
  std::vector<Vec2> out;
  for (const Value& p : list.items()) {
    out.emplace_back(p[0].AsDouble(), p[1].AsDouble());
  }
  return out;
}

}  // namespace

Points ReadLocalizationMap(const std::string& nav_dir) {
  const std::string path =
      (fs::path(nav_dir) / "localization_map.npz").string();
  const NpzArchive npz = ReadNpz(path);
  const auto found = npz.find("xyz");
  if (found == npz.end() || found->second.columns() != 3) {
    throw std::runtime_error(path + ": no (N, 3) xyz");
  }
  const std::vector<double> xyz = found->second.ToDoubles();
  Points out(xyz.size() / 3);
  for (size_t k = 0; k < out.size(); ++k) {
    out[k] = Vec3(xyz[3 * k], xyz[3 * k + 1], xyz[3 * k + 2]);
  }
  return out;
}

std::vector<NavEdge> ReadNavEdges(const std::string& graph_json) {
  const Value graph = ReadJsonFile(graph_json);
  std::vector<NavEdge> out;
  for (const Value& e : graph["edges"].items()) {
    NavEdge edge;
    edge.level = e["level"].AsInt();
    edge.kind = e["kind"].AsString();
    edge.points = PointsXy(e["points"]);
    out.push_back(std::move(edge));
  }
  return out;
}

std::vector<NavSlot> ReadNavSlots(const std::string& goals_json) {
  const Value goals = ReadJsonFile(goals_json);
  std::vector<NavSlot> out;
  for (const Value& g : goals["goals"].items()) {
    if (g.Find("corners") == nullptr) continue;
    NavSlot slot;
    slot.level = g["level"].AsInt();
    slot.slot_id = g["slot_id"].AsString();
    slot.corners = PointsXy(g["corners"]);
    out.push_back(std::move(slot));
  }
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
    if (grid.size() < 6) throw std::runtime_error(path + ": bad grid");
    FloorHeights floor;
    floor.x0 = grid[0];
    floor.y0 = grid[1];
    floor.res = grid[2];
    floor.width = static_cast<int>(grid[3]);
    floor.height = static_cast<int>(grid[4]);
    floor.level = static_cast<int>(grid[5]);
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

Value SummaryValue(const TripLocalizationSummary& s) {
  const bool tracked = s.first_tracked >= 0;
  const auto optional = [](bool defined, double value) {
    return defined ? Value::Number(value) : Value();
  };
  Value out = Value::Map();
  out.Set("sweeps", Value::Integer(s.sweeps));
  out.Set("first_tracked",
          tracked ? Value::Integer(s.first_tracked) : Value());
  out.Set("init_time_s", optional(tracked, Round(s.init_time_s, 2)));
  out.Set("init_distance_m", optional(tracked, Round(s.init_distance_m, 2)));
  out.Set("placement_attempts", Value::Integer(s.placement_attempts));
  out.Set("tracked", Value::Integer(s.tracked));
  out.Set("registered", Value::Integer(s.registered));
  out.Set("registered_share_of_tracked",
          optional(tracked, Round(s.registered_share_of_tracked, 4)));
  out.Set("lost_sweeps", Value::Integer(s.lost_sweeps));
  out.Set("degenerate_share", optional(tracked, Round(s.degenerate_share, 4)));
  out.Set("ms_median", Value::Number(Round(s.ms_median, 1)));
  out.Set("ms_p95", Value::Number(Round(s.ms_p95, 1)));
  return out;
}

Value PositionValue(const NavPosition& p) {
  Value out = Value::Map();
  Value xyz = Value::List();
  for (int k = 0; k < 3; ++k) {
    xyz.Append(Value::Number(NumpyRound(p.xyz[k], 3)));
  }
  out.Set("xyz", xyz);
  out.Set("yaw_deg", Value::Number(Round(p.yaw_deg, 2)));
  out.Set("level", p.level ? Value::Integer(*p.level) : Value());
  if (p.edge) {
    Value edge = Value::Map();
    edge.Set("edge", Value::Integer(p.edge->edge));
    edge.Set("kind", Value::String(p.edge->kind));
    edge.Set("distance_m", Value::Number(Round(p.edge->distance, 3)));
    edge.Set("along_m", Value::Number(Round(p.edge->along, 2)));
    edge.Set("lateral_m", Value::Number(Round(p.edge->lateral, 3)));
    edge.Set("heading_rel_deg",
             Value::Number(Round(p.edge->heading_rel_deg, 1)));
    out.Set("edge", edge);
  } else {
    out.Set("edge", Value());
  }
  out.Set("slot", p.slot ? Value::String(*p.slot) : Value());
  return out;
}

void WriteLocalizationResult(const std::string& directory,
                             const std::vector<SweepLocalization>& sweeps,
                             const Value& report) {
  fs::create_directories(directory);
  const fs::path dir(directory);
  Trajectory tracked;
  for (const SweepLocalization& s : sweeps) {
    if (s.result.status != LocalizationStatus::kTracking) continue;
    tracked.stamps.push_back(s.stamp);
    tracked.poses.push_back(s.map_from_vehicle);
  }
  WriteTum((dir / "poses.tum").string(), tracked);
  std::ofstream csv(dir / "sweeps.csv");
  if (!csv) throw std::runtime_error("cannot write sweeps.csv");
  csv << "stamp,status,registered,fitness,rmse,degenerate_t,step_t,"
         "step_r,eigen_ratio_t,sigma_t,sigma_yaw_deg,ms,x,y,z\n";
  char line[512];
  for (const SweepLocalization& s : sweeps) {
    const LocalizationResult& r = s.result;
    const Vec3& t = s.map_from_vehicle.translation;
    std::snprintf(line, sizeof(line),
                  "%.6f,%s,%d,%.4f,%.4f,%d,%.4f,%.3f,%.2e,%.4f,%.4f,%.1f,"
                  "%.3f,%.3f,%.3f\n",
                  s.stamp, StatusName(r.status), r.registered ? 1 : 0,
                  r.fitness, r.rmse, r.degenerate_t, r.step_t, r.step_r,
                  r.min_eigen_ratio_t, s.sigma_t, s.sigma_yaw_deg, s.ms,
                  t.x(), t.y(), t.z());
    csv << line;
  }
  std::ofstream json(dir / "localization.json");
  if (!json) throw std::runtime_error("cannot write localization.json");
  json << ToJson(report, 1);
}

}  // namespace gr
