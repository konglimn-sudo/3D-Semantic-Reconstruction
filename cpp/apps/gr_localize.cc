// gr_localize: localizes a trip in the navigation map of a reconstruction,
// without Python (the vehicle side of `garage_recon localize`).
//
//   gr_localize --nav <work>/semantics/nav --trip <trip> --out <dir>
//               [--config a.yaml [--config b.yaml ...]]
//               [--prior none|entrance|reference [--reference poses.tum]]
//               [--stride N] [--start S] [--changes]
//
// Priors: none (global initialization), entrance (the trip's INS pose at the
// entrance) or reference (a TUM trajectory, map <- vehicle, at the first
// sweep: a diagnostic of tracking alone).
//
// The configuration files are deep-merged in order; pass the package's
// default_config.yaml first. Without --config the reconstruction's
// <work>/config.used.yaml is used. Writes poses.tum, sweeps.csv and
// localization.json as nav/localize.py does (without the errors against a
// reference, which the Python evaluation adds), and slot_states.json: the
// live occupancy of the slots the vehicle saw (gr/nav/slot_state.h). With
// --changes also changes.json: the structural changes against the map and
// the route edges they reach (gr/mapping/change_detection.h,
// gr/nav/map_update.h).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "gr/io/config.h"
#include "gr/io/nav_files.h"
#include "gr/io/tables.h"
#include "gr/io/trip.h"
#include "gr/mapping/change_detection.h"
#include "gr/nav/map_update.h"
#include "gr/nav/position.h"
#include "gr/nav/slot_state.h"
#include "gr/nav/trip_localizer.h"

namespace {

namespace fs = std::filesystem;

struct Args {
  std::string nav;
  std::string trip;
  std::string out;
  std::vector<std::string> configs;
  std::string prior = "none";
  std::string reference;
  int stride = 1;
  double start = 0.0;
  bool changes = false;
};

[[noreturn]] void Usage(const std::string& error) {
  std::fprintf(stderr,
               "%s\nusage: gr_localize --nav DIR --trip DIR --out DIR "
               "[--config FILE ...] [--prior none|entrance|reference] "
               "[--reference TUM] [--stride N] [--start S] [--changes]\n",
               error.c_str());
  std::exit(2);
}

Args Parse(int argc, char** argv) {
  Args args;
  for (int k = 1; k < argc; ++k) {
    const std::string flag = argv[k];
    if (flag == "--changes") {
      args.changes = true;
      continue;
    }
    if (k + 1 >= argc) Usage("missing value of " + flag);
    const std::string value = argv[++k];
    if (flag == "--nav") {
      args.nav = value;
    } else if (flag == "--trip") {
      args.trip = value;
    } else if (flag == "--out") {
      args.out = value;
    } else if (flag == "--config") {
      args.configs.push_back(value);
    } else if (flag == "--prior") {
      args.prior = value;
    } else if (flag == "--reference") {
      args.reference = value;
    } else if (flag == "--stride") {
      args.stride = std::atoi(value.c_str());
    } else if (flag == "--start") {
      args.start = std::atof(value.c_str());
    } else {
      Usage("unknown flag " + flag);
    }
  }
  if (args.nav.empty() || args.trip.empty() || args.out.empty()) {
    Usage("--nav, --trip and --out are required");
  }
  if (args.prior != "none" && args.prior != "entrance" &&
      args.prior != "reference") {
    Usage("--prior must be none, entrance or reference");
  }
  if (args.prior == "reference" && args.reference.empty()) {
    Usage("--prior reference needs --reference");
  }
  if (args.stride < 1) Usage("--stride must be positive");
  if (args.prior == "entrance" && args.start > 0.0) {
    Usage("the entrance prior needs --start 0");
  }
  if (args.configs.empty()) {
    // <work>/semantics/nav -> <work>/config.used.yaml
    const fs::path used =
        fs::path(args.nav).parent_path().parent_path() / "config.used.yaml";
    if (!fs::exists(used)) Usage("no --config and no " + used.string());
    args.configs.push_back(used.string());
  }
  return args;
}

void Log(const std::string& line) {
  std::printf("%s\n", line.c_str());
  std::fflush(stdout);
}

int Run(const Args& args) {
  using namespace gr;  // NOLINT(build/namespaces)
  const Value config = LoadConfig(args.configs);
  const TripLocalizerConfig trip_config = TripLocalizerConfigFrom(config);
  const TripReader trip(args.trip);
  if (trip.size() == 0) throw std::runtime_error(args.trip + ": no sweeps");
  const Points map = ReadLocalizationMap(args.nav);
  MotionPrior prior = trip.BuildMotionPrior();
  std::optional<Pose> map_from_vehicle;
  if (args.prior == "entrance") {
    map_from_vehicle = trip.EntrancePose(prior);
    if (!map_from_vehicle) {
      throw std::runtime_error(args.trip + " has no entrance pose");
    }
  }
  const std::vector<int64_t>& stamps_ns = trip.lidar_stamps_ns();
  const double t0 = trip.lidar_stamp(0) + args.start;
  // First sweep at or after t0 (numpy.searchsorted, left).
  size_t first = 0;
  while (first < trip.size() && trip.lidar_stamp(first) < t0) ++first;
  if (first == trip.size()) throw std::runtime_error("--start beyond the trip");
  if (args.prior == "reference") {
    const Trajectory reference = ReadTum(args.reference);
    if (reference.stamps.empty()) {
      throw std::runtime_error(args.reference + ": empty trajectory");
    }
    map_from_vehicle = InterpolatePoses(reference.stamps, reference.poses,
                                        {trip.lidar_stamp(first)})[0];
  }
  TripLocalizer localizer(
      map, trip_config, trip.rig().primary_lidar().vehicle_from_lidar,
      LevelledStart(trip.ImuSamplesInVehicle(), trip.lidar_stamp(first),
                    trip_config.level_window),
      std::move(prior), map_from_vehicle, Log);
  // Live slot occupancy from the localized sweeps.
  const fs::path goals = fs::path(args.nav) / "goals.json";
  const std::vector<SlotGeometry> slots =
      fs::exists(goals) ? ReadSlotGeometry(goals.string())
                        : std::vector<SlotGeometry>();
  const SlotStateConfig slot_config =
      SlotStateConfigFrom(config);
  SlotStateTracker slot_states(slots.size(), slot_config);
  std::optional<ChangeDetector> changes;
  if (args.changes) changes.emplace(map, ChangeConfigFrom(config));
  const size_t n = stamps_ns.size();
  for (size_t k = first; k < n; ++k) {
    const LidarSweep sweep = trip.LoadLidar(k);
    const std::optional<SweepLocalization> s = localizer.Process(
        sweep.points, sweep.intensity, sweep.time, sweep.stamp,
        static_cast<int>(k), k == n - 1,
        (k - first) % static_cast<size_t>(args.stride) == 0);
    if (!s || !s->result.registered ||
        s->result.status != LocalizationStatus::kTracking) {
      continue;
    }
    const Points in_map =
        TransformPoints(localizer.last_sweep(), s->result.pose);
    if (!slots.empty()) {
      slot_states.Update(s->stamp,
                         ObserveSlots(in_map, s->result.pose.translation,
                                      slots, slot_config));
    }
    if (changes) {
      changes->AddSweep(in_map, s->result.pose.translation, s->stamp);
    }
  }
  const TripLocalizationSummary summary = localizer.Summary();
  Value report = Value::Map();
  report.Set("trip", Value::String(trip.trip_id()));
  report.Set("prior", Value::String(args.prior));
  report.Set("start_s", Value::Number(args.start));
  report.Set("map_points", Value::Integer(static_cast<long long>(map.size())));
  const Value summary_value = SummaryValue(summary);
  for (const auto& [key, value] : summary_value.members()) {
    report.Set(key, value);
  }
  const std::vector<SweepLocalization>& sweeps = localizer.sweeps();
  const auto last = std::find_if(
      sweeps.rbegin(), sweeps.rend(), [](const SweepLocalization& s) {
        return s.result.status == LocalizationStatus::kTracking;
      });
  if (last != sweeps.rend()) {
    const fs::path nav(args.nav);
    report.Set("last_position",
               PositionValue(Where(
                   last->map_from_vehicle,
                   ReadNavEdges((nav / "graph.json").string()),
                   ReadNavSlots((nav / "goals.json").string()),
                   ReadFloorHeights(args.nav))));
  }
  if (changes) {
    const fs::path nav(args.nav);
    const Value found = ChangesValue(
        AssessChanges(changes->Clusters(),
                      ReadNavEdges((nav / "graph.json").string()),
                      ReadNavSlots((nav / "goals.json").string()),
                      ReadFloorHeights(args.nav), CorridorHalfWidth(config)),
        changes->sweeps());
    report.Set("changes", found["summary"]);
    fs::create_directories(args.out);
    std::ofstream file(fs::path(args.out) / "changes.json");
    file << ToJson(found, 1);
    if (!file) throw std::runtime_error("cannot write changes.json");
  }
  if (!slots.empty()) {
    const Value states = SlotStatesValue(slots, slot_states);
    report.Set("slot_states", states["summary"]);
    fs::create_directories(args.out);
    std::ofstream file(fs::path(args.out) / "slot_states.json");
    file << ToJson(states, 1);
    if (!file) throw std::runtime_error("cannot write slot_states.json");
  }
  WriteLocalizationResult(args.out, sweeps, report);
  Log(StringPrintf(
      "[localize] %s: tracked %d/%d sweeps, registered %d, lost %d, "
      "%.1f ms/sweep (median)",
      trip.trip_id().c_str(), summary.tracked, summary.sweeps,
      summary.registered, summary.lost_sweeps, summary.ms_median));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = Parse(argc, argv);
  try {
    return Run(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gr_localize: %s\n", e.what());
    return 1;
  }
}
