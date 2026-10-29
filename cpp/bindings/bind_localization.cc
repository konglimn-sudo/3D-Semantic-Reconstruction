// Bindings of the map localization (gr/nav/localization.h, per sweep;
// gr/nav/trip_localizer.h, per trip), of the position on the route graph
// (gr/nav/position.h) and of the placement of a gravity-aligned cloud on a
// reference (gr/registration/cloud_compare.h).

#include <memory>
#include <string>
#include <vector>

#include "bind_util.h"
#include "gr/nav/localization.h"
#include "gr/mapping/change_detection.h"
#include "gr/nav/map_update.h"
#include "gr/nav/position.h"
#include "gr/nav/slot_state.h"
#include "gr/nav/trip_localizer.h"
#include "gr/registration/cloud_compare.h"

namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(build/namespaces)

namespace gr::bindings {
namespace {

py::dict PlacementToDict(const Placement& placement) {
  return py::dict("placed"_a = placement.placed,
                  "verdict"_a = placement.verdict,
                  "T"_a = placement.reference_from_source.Matrix(),
                  "score"_a = placement.score,
                  "fitness"_a = placement.fitness, "rmse"_a = placement.rmse,
                  "runner_up"_a = placement.runner_up,
                  "hypotheses"_a = placement.hypotheses);
}

const char* const kStatus[] = {"initializing", "tracking", "lost"};

py::dict ResultToDict(const LocalizationResult& result) {
  return py::dict(
      "status"_a = kStatus[static_cast<int>(result.status)],
      "T"_a = result.pose.Matrix(), "registered"_a = result.registered,
      "initialized"_a = result.initialized, "fitness"_a = result.fitness,
      "rmse"_a = result.rmse, "degenerate_t"_a = result.degenerate_t,
      "degenerate_r"_a = result.degenerate_r,
      "min_eigen_ratio_t"_a = result.min_eigen_ratio_t,
      "step_t"_a = result.step_t, "step_r"_a = result.step_r,
      "covariance"_a = result.covariance, "failures"_a = result.failures);
}

// A TripLocalizer with the Python logger it calls (kept alive and at a fixed
// address: pybind11 holds the object by pointer).
struct PyTripLocalizer {
  py::object log;
  std::exception_ptr log_error;
  std::unique_ptr<TripLocalizer> localizer;
};

std::vector<Vec2> Xy(const Eigen::MatrixXd& points) {
  Require(points.rows() == 0 || points.cols() >= kPlaneDim,
          "expected (M, 2) points");
  std::vector<Vec2> out;
  for (Eigen::Index row = 0; row < points.rows(); ++row) {
    out.emplace_back(points(row, 0), points(row, 1));
  }
  return out;
}

// Route edges as [(level, kind, (M, 2) points)].
std::vector<NavEdge> NavEdgesFrom(const py::list& edges) {
  std::vector<NavEdge> out;
  for (const py::handle& item : edges) {
    const auto [level, kind, points] =
        item.cast<std::tuple<int, std::string, Eigen::MatrixXd>>();
    out.push_back({level, kind, Xy(points)});
  }
  return out;
}

// Slots as [(level, slot_id, (K, 2) corners)].
std::vector<NavSlot> NavSlotsFrom(const py::list& slots) {
  std::vector<NavSlot> out;
  for (const py::handle& item : slots) {
    const auto [level, slot_id, corners] =
        item.cast<std::tuple<int, std::string, Eigen::MatrixXd>>();
    out.push_back({level, slot_id, Xy(corners)});
  }
  return out;
}

// Floor heights as [(level, x0, y0, res, floor_z (H, W))].
std::vector<FloorHeights> FloorHeightsFrom(const py::list& floors) {
  std::vector<FloorHeights> out;
  for (const py::handle& item : floors) {
    const auto [level, x0, y0, res, z] =
        item.cast<std::tuple<int, double, double, double, Eigen::MatrixXd>>();
    FloorHeights floor;
    floor.level = level;
    floor.x0 = x0;
    floor.y0 = y0;
    floor.res = res;
    floor.height = static_cast<int>(z.rows());
    floor.width = static_cast<int>(z.cols());
    for (Eigen::Index row = 0; row < z.rows(); ++row) {
      for (Eigen::Index column = 0; column < z.cols(); ++column) {
        floor.z.push_back(z(row, column));
      }
    }
    out.push_back(std::move(floor));
  }
  return out;
}

}  // namespace

void BindLocalization(py::module_& m) {
  py::class_<PlacementConfig>(m, "PlacementConfig")
      .def(py::init<>())
      .def_readwrite("voxel", &PlacementConfig::voxel)
      .def_readwrite("icp_schedule", &PlacementConfig::icp_schedule)
      .def_readwrite("icp_iters", &PlacementConfig::icp_iters)
      .def_readwrite("min_score", &PlacementConfig::min_score)
      .def_readwrite("z_peak_ratio", &PlacementConfig::z_peak_ratio)
      .def_readwrite("max_candidates", &PlacementConfig::max_candidates)
      .def_readwrite("min_fitness", &PlacementConfig::min_fitness)
      .def_readwrite("max_rmse", &PlacementConfig::max_rmse)
      .def_readwrite("min_fitness_gap", &PlacementConfig::min_fitness_gap)
      .def_readwrite("distinct_t", &PlacementConfig::distinct_t)
      .def_readwrite("distinct_r", &PlacementConfig::distinct_r);

  m.def(
      "place_gravity_aligned",
      [](const PointsIn& source, const PointsIn& reference,
         const PlacementConfig& config) {
        Require(config.voxel > 0.0, "voxel must be positive");
        const Points source_points = ToPoints(source);
        const Points reference_points = ToPoints(reference);
        Placement placement;
        {
          py::gil_scoped_release release;
          placement =
              PlaceGravityAligned(source_points, reference_points, config);
        }
        return PlacementToDict(placement);
      },
      "source"_a, "reference"_a, "config"_a = PlacementConfig(),
      "Places a gravity-aligned cloud on a reference without a guess: dict "
      "of placed, verdict, T (reference <- source), score, fitness, rmse, "
      "runner_up, hypotheses.");

  py::class_<LocalizerConfig>(m, "LocalizerConfig")
      .def(py::init<>())
      .def_readwrite("map_voxel", &LocalizerConfig::map_voxel)
      .def_readwrite("hash_voxel", &LocalizerConfig::hash_voxel)
      .def_readwrite("scan_voxel", &LocalizerConfig::scan_voxel)
      .def_readwrite("min_range", &LocalizerConfig::min_range)
      .def_readwrite("max_range", &LocalizerConfig::max_range)
      .def_readwrite("deskew", &LocalizerConfig::deskew)
      .def_readwrite("gates", &LocalizerConfig::gates)
      .def_readwrite("icp_iters", &LocalizerConfig::icp_iters)
      .def_readwrite("min_fitness", &LocalizerConfig::min_fitness)
      .def_readwrite("max_rmse", &LocalizerConfig::max_rmse)
      .def_readwrite("max_jump_t", &LocalizerConfig::max_jump_t)
      .def_readwrite("max_jump_r", &LocalizerConfig::max_jump_r)
      .def_readwrite("degenerate_ratio", &LocalizerConfig::degenerate_ratio)
      .def_readwrite("degenerate_sigma_t",
                     &LocalizerConfig::degenerate_sigma_t)
      .def_readwrite("degenerate_sigma_r",
                     &LocalizerConfig::degenerate_sigma_r)
      .def_readwrite("max_failures", &LocalizerConfig::max_failures)
      .def_readwrite("prior_schedule", &LocalizerConfig::prior_schedule)
      .def_readwrite("init_voxel", &LocalizerConfig::init_voxel)
      .def_readwrite("init_spacing", &LocalizerConfig::init_spacing)
      .def_readwrite("init_distance", &LocalizerConfig::init_distance)
      .def_readwrite("init_retry", &LocalizerConfig::init_retry)
      .def_readwrite("init_max_distance", &LocalizerConfig::init_max_distance)
      .def_readwrite("placement", &LocalizerConfig::placement);

  py::class_<Localizer>(m, "Localizer")
      .def(py::init([](const PointsIn& map_points,
                       const LocalizerConfig& config) {
             Require(map_points.rows() > 0, "the map is empty");
             Require(config.map_voxel > 0.0 && config.hash_voxel > 0.0 &&
                         config.scan_voxel > 0.0 && config.init_voxel > 0.0 &&
                         config.placement.voxel > 0.0,
                     "voxel sizes must be positive");
             Require(!config.gates.empty(), "gates must not be empty");
             const Points points = ToPoints(map_points);
             py::gil_scoped_release release;
             return Localizer(points, config);
           }),
           "map_points"_a, "config"_a = LocalizerConfig())
      .def(
          "set_prior",
          [](Localizer& localizer, const Mat4& map_from_sensor) {
            localizer.SetPrior(Pose::FromMatrix(map_from_sensor));
          },
          "T_map_sensor"_a)
      .def(
          "process",
          [](Localizer& localizer, const PointsIn& points,
             const py::object& times, double stamp,
             const Mat4& odom_from_sensor) {
            const Points cloud = ToPoints(points);
            const std::vector<double> point_times = ToDoubleVector(times);
            Require(point_times.empty() || point_times.size() == cloud.size(),
                    "times must be empty or match the points");
            LocalizationResult result;
            {
              py::gil_scoped_release release;
              result = localizer.Process(cloud, point_times, stamp,
                                    Pose::FromMatrix(odom_from_sensor));
            }
            return ResultToDict(result);
          },
          "points"_a, "times"_a, "stamp"_a, "T_odom_sensor"_a,
          "One sweep (sensor frame) with its odometry pose: dict of status "
          "(initializing / tracking / lost), T (map <- sensor), registered, "
          "initialized, fitness, rmse, degenerate_t / _r, min_eigen_ratio_t, "
          "step_t (m) / step_r (deg), "
          "covariance (6x6, [t, phi]), failures.")
      .def_property_readonly("T_map_odom",
                             [](const Localizer& localizer) {
                               return localizer.map_from_odom().Matrix();
                             })
      .def_property_readonly(
          "last_placement",
          [](const Localizer& localizer) {
            return PlacementToDict(localizer.last_placement());
          })
      .def_property_readonly("placement_attempts",
                             &Localizer::placement_attempts)
      .def_property_readonly("travelled", &Localizer::travelled)
      .def_property_readonly("map_size", &Localizer::map_size);

  py::class_<TripLocalizerConfig>(m, "TripLocalizerConfig")
      .def(py::init<>())
      .def_readwrite("odometry", &TripLocalizerConfig::odometry)
      .def_readwrite("keyframes", &TripLocalizerConfig::keyframes)
      .def_readwrite("localizer", &TripLocalizerConfig::localizer)
      .def_readwrite("level_window", &TripLocalizerConfig::level_window);

  m.def(
      "levelled_start",
      [](const ImuSamples& imu, double stamp, double window) {
        return LevelledStart(imu, stamp, window).Matrix();
      },
      "imu"_a, "stamp"_a, "window"_a,
      "world <- vehicle at `stamp` with the roll and pitch of gravity (mean "
      "specific force over `window` s); identity without IMU samples.");

  py::class_<PyTripLocalizer>(m, "TripLocalizer")
      .def(py::init([](const PointsIn& map_points,
                       const TripLocalizerConfig& config,
                       const Mat4& vehicle_from_lidar,
                       const Mat4& odom_from_vehicle0,
                       const MotionPrior& prior, const py::object& map_prior,
                       const py::object& log) {
             Require(map_points.rows() > 0, "the map is empty");
             Require(!config.localizer.gates.empty(),
                     "gates must not be empty");
             auto out = std::make_unique<PyTripLocalizer>();
             out->log = log;
             const Logger logger = MakeLogger(out->log, &out->log_error);
             std::optional<Pose> prior_pose;
             if (!map_prior.is_none()) {
               prior_pose = Pose::FromMatrix(map_prior.cast<Mat4>());
             }
             const Points points = ToPoints(map_points);
             {
               py::gil_scoped_release release;
               out->localizer = std::make_unique<TripLocalizer>(
                   points, config, Pose::FromMatrix(vehicle_from_lidar),
                   Pose::FromMatrix(odom_from_vehicle0), prior, prior_pose,
                   logger);
             }
             RethrowIfSet(out->log_error);
             return out;
           }),
           "map_points"_a, "config"_a, "T_vehicle_lidar"_a,
           "T_odom_vehicle0"_a, "motion_prior"_a,
           "T_map_vehicle"_a = py::none(), "log"_a = py::none())
      .def(
          "process",
          [](PyTripLocalizer& self, const PointsIn& points,
             const py::object& intensity, const py::object& times,
             double stamp, int index, bool last,
             bool localize) -> py::object {
            const Points cloud = ToPoints(points);
            const std::vector<float> point_intensity = ToFloatVector(intensity);
            const std::vector<double> point_times = ToDoubleVector(times);
            Require(point_times.empty() || point_times.size() == cloud.size(),
                    "times must be empty or match the points");
            std::optional<SweepLocalization> localized;
            {
              py::gil_scoped_release release;
              localized =
                  self.localizer->Process(cloud, point_intensity, point_times,
                                          stamp, index, last, localize);
            }
            RethrowIfSet(self.log_error);
            if (!localized) return py::none();
            py::dict out = ResultToDict(localized->result);
            out["index"] = localized->index;
            out["stamp"] = localized->stamp;
            out["T_map_vehicle"] = localized->map_from_vehicle.Matrix();
            out["T_odom_vehicle"] = localized->odom_from_vehicle.Matrix();
            out["sigma_t"] = localized->sigma_t;
            out["sigma_yaw_deg"] = localized->sigma_yaw_deg;
            out["ms"] = localized->ms;
            return std::move(out);
          },
          "points"_a, "intensity"_a, "times"_a, "stamp"_a, "index"_a,
          "last"_a, "localize"_a,
          "One sweep through the odometry and, with `localize`, the "
          "localizer: None, or the localizer's dict plus index, stamp, "
          "T_map_vehicle, T_odom_vehicle, sigma_t, sigma_yaw_deg, ms.")
      .def("summary",
           [](const PyTripLocalizer& self) {
             const TripLocalizationSummary summary = self.localizer->Summary();
             const bool tracked = summary.first_tracked !=
                                  TripLocalizationSummary::kNeverTracked;
             const auto optional = [&](double value) -> py::object {
               return tracked ? py::cast(value) : py::none();
             };
             return py::dict(
                 "sweeps"_a = summary.sweeps,
                 "first_tracked"_a =
                     tracked ? py::cast(summary.first_tracked) : py::none(),
                 "init_time_s"_a = optional(summary.init_time_s),
                 "init_distance_m"_a = optional(summary.init_distance_m),
                 "placement_attempts"_a = summary.placement_attempts,
                 "tracked"_a = summary.tracked,
                 "registered"_a = summary.registered,
                 "registered_share_of_tracked"_a =
                     optional(summary.registered_share_of_tracked),
                 "lost_sweeps"_a = summary.lost_sweeps,
                 "degenerate_share"_a = optional(summary.degenerate_share),
                 "ms_median"_a = summary.ms_median,
                 "ms_p95"_a = summary.ms_p95);
           })
      .def_property_readonly("map_size",
                             [](const PyTripLocalizer& self) {
                               return self.localizer->localizer().map_size();
                             })
      .def_property_readonly("last_sweep", [](const PyTripLocalizer& self) {
        return ToMatrix(self.localizer->last_sweep());
      });

  py::class_<ChangeConfig>(m, "ChangeConfig")
      .def(py::init<>())
      .def_readwrite("voxel", &ChangeConfig::voxel)
      .def_readwrite("max_range", &ChangeConfig::max_range)
      .def_readwrite("ray_voxel", &ChangeConfig::ray_voxel)
      .def_readwrite("min_hits", &ChangeConfig::min_hits)
      .def_readwrite("min_hit_span", &ChangeConfig::min_hit_span)
      .def_readwrite("min_free", &ChangeConfig::min_free)
      .def_readwrite("free_margin", &ChangeConfig::free_margin)
      .def_readwrite("min_cluster_voxels", &ChangeConfig::min_cluster_voxels)
      .def_readwrite("flat_thickness", &ChangeConfig::flat_thickness)
      .def_readwrite("vehicle_min_length", &ChangeConfig::vehicle_min_length)
      .def_readwrite("vehicle_max_length", &ChangeConfig::vehicle_max_length)
      .def_readwrite("vehicle_min_width", &ChangeConfig::vehicle_min_width)
      .def_readwrite("vehicle_max_width", &ChangeConfig::vehicle_max_width)
      .def_readwrite("vehicle_max_height",
                     &ChangeConfig::vehicle_max_height);

  py::class_<ChangeDetector>(m, "ChangeDetector")
      .def(py::init([](const PointsIn& map_points, const ChangeConfig& config) {
             Require(config.voxel > 0.0 && config.ray_voxel > 0.0,
                     "voxel sizes must be positive");
             const Points points = ToPoints(map_points);
             py::gil_scoped_release release;
             return ChangeDetector(points, config);
           }),
           "map_points"_a, "config"_a = ChangeConfig())
      .def(
          "add_sweep",
          [](ChangeDetector& detector, const PointsIn& sweep,
             const Mat4& map_from_sensor, double stamp) {
            const Points sweep_points = ToPoints(sweep);
            const Pose pose = Pose::FromMatrix(map_from_sensor);
            py::gil_scoped_release release;
            detector.AddSweep(TransformPoints(sweep_points, pose),
                              pose.translation, stamp);
          },
          "sweep"_a, "T_map_sensor"_a, "stamp"_a,
          "One localized sweep (sensor frame) with its pose in the map.")
      .def_property_readonly("sweeps", &ChangeDetector::sweeps)
      .def(
          "assess",
          [](const ChangeDetector& detector, const py::list& edges,
             const py::list& slots, const py::list& floors,
             double half_width) {
            const std::vector<NavEdge> nav_edges = NavEdgesFrom(edges);
            const std::vector<NavSlot> nav_slots = NavSlotsFrom(slots);
            const std::vector<FloorHeights> floor_heights =
                FloorHeightsFrom(floors);
            py::list out;
            for (const MapChange& change :
                 AssessChanges(detector.Clusters(), nav_edges, nav_slots,
                               floor_heights, half_width)) {
              out.append(py::dict(
                  "added"_a = change.cluster.added,
                  "level"_a =
                      change.level ? py::cast(*change.level) : py::none(),
                  "center"_a = change.cluster.center,
                  "min"_a = change.cluster.min, "max"_a = change.cluster.max,
                  "voxels"_a = change.cluster.voxels,
                  "edges"_a = change.edges));
            }
            return out;
          },
          "edges"_a, "slots"_a, "floors"_a, "half_width"_a,
          "The structural changes (C++ AssessChanges): dicts of added, "
          "level, center, min, max, voxels, edges (drivable edges whose "
          "corridor they reach); edges / slots / floors as nav_position.");

  py::class_<SlotStateConfig>(m, "SlotStateConfig")
      .def(py::init<>())
      .def_readwrite("max_range", &SlotStateConfig::max_range)
      .def_readwrite("inset", &SlotStateConfig::inset)
      .def_readwrite("min_height", &SlotStateConfig::min_height)
      .def_readwrite("max_height", &SlotStateConfig::max_height)
      .def_readwrite("floor_band", &SlotStateConfig::floor_band)
      .def_readwrite("min_body_points", &SlotStateConfig::min_body_points)
      .def_readwrite("max_free_points", &SlotStateConfig::max_free_points)
      .def_readwrite("cell", &SlotStateConfig::cell)
      .def_readwrite("min_floor_share", &SlotStateConfig::min_floor_share)
      .def_readwrite("confirm", &SlotStateConfig::confirm);

  // Slots as (slot_id, level, (4, 2) corners, z).
  const auto slots_from = [](const py::list& slots) {
    std::vector<SlotGeometry> out;
    for (const py::handle& item : slots) {
      const auto [slot_id, level, corner_rows, z] = item.cast<
          std::tuple<std::string, int, Eigen::MatrixXd, double>>();
      const std::vector<Vec2> corners = Xy(corner_rows);
      Require(corners.size() == kSlotCorners, "a slot has four corners");
      SlotGeometry slot;
      slot.slot_id = slot_id;
      slot.level = level;
      std::copy(corners.begin(), corners.end(), slot.corners.begin());
      slot.z = z;
      out.push_back(slot);
    }
    return out;
  };

  py::class_<SlotStateTracker>(m, "SlotStateTracker")
      .def(py::init([slots_from](const py::list& slots,
                                 const SlotStateConfig& config) {
             return SlotStateTracker(slots_from(slots).size(), config);
           }),
           "slots"_a, "config"_a = SlotStateConfig())
      .def(
          "observe",
          [slots_from](SlotStateTracker& tracker, const py::list& slots,
                       double stamp, const PointsIn& sweep,
                       const Mat4& map_from_sensor,
                       const SlotStateConfig& config) {
            const std::vector<SlotGeometry> slot_geometry = slots_from(slots);
            Require(slot_geometry.size() == tracker.entries().size(),
                    "the tracker was made for other slots");
            const Points sweep_points = ToPoints(sweep);
            const Pose pose = Pose::FromMatrix(map_from_sensor);
            py::gil_scoped_release release;
            tracker.Update(
                stamp, ObserveSlots(TransformPoints(sweep_points, pose),
                                    pose.translation, slot_geometry, config));
          },
          "slots"_a, "stamp"_a, "sweep"_a, "T_map_sensor"_a, "config"_a,
          "One localized sweep (sensor frame) with its pose in the map.")
      .def("states", [](const SlotStateTracker& tracker) {
        static const char* const kState[] = {"unknown", "free", "occupied"};
        py::list out;
        for (const SlotStateTracker::Entry& entry : tracker.entries()) {
          out.append(py::dict("state"_a = kState[static_cast<int>(entry.state)],
                              "stamp"_a = entry.state == SlotState::kUnknown
                                              ? py::none()
                                              : py::cast(entry.stamp),
                              "observations"_a = entry.observations));
        }
        return out;
      });

  m.def(
      "nav_position",
      [](const Mat4& map_from_vehicle, const py::list& edges,
         const py::list& slots, const py::list& floors) -> py::dict {
        const NavPosition position =
            Where(Pose::FromMatrix(map_from_vehicle), NavEdgesFrom(edges),
                  NavSlotsFrom(slots), FloorHeightsFrom(floors));
        py::object edge = py::none();
        if (position.edge) {
          const EdgePosition& on_edge = *position.edge;
          edge = py::dict("edge"_a = on_edge.edge, "kind"_a = on_edge.kind,
                          "distance"_a = on_edge.distance,
                          "along"_a = on_edge.along,
                          "lateral"_a = on_edge.lateral,
                          "heading_rel_deg"_a = on_edge.heading_rel_deg);
        }
        return py::dict(
            "xyz"_a = position.xyz, "yaw_deg"_a = position.yaw_deg,
            "level"_a = position.level ? py::cast(*position.level)
                                       : py::none(),
            "edge"_a = edge,
            "slot"_a = position.slot ? py::cast(*position.slot) : py::none());
      },
      "T_map_vehicle"_a, "edges"_a, "slots"_a, "floors"_a,
      "Where a vehicle is on the route graph. edges: [(level, kind, (M, 2))], "
      "slots: [(level, slot_id, corners (K, 2))], floors: [(level, x0, y0, "
      "res, floor_z (H, W))]. Dict of xyz, yaw_deg, level, edge (edge, kind, "
      "distance, along, lateral, heading_rel_deg) or None, slot or None.");
}

}  // namespace gr::bindings
