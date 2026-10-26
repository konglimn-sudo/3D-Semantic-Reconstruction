// Bindings of the map localization (gr/nav/localization.h, per sweep;
// gr/nav/trip_localizer.h, per trip), of the position on the route graph
// (gr/nav/position.h) and of the placement of a gravity-aligned cloud on a
// reference (gr/registration/cloud_compare.h).

#include <memory>
#include <string>
#include <vector>

#include "bind_util.h"
#include "gr/nav/localization.h"
#include "gr/nav/position.h"
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

py::dict ResultToDict(const LocalizationResult& r) {
  return py::dict(
      "status"_a = kStatus[static_cast<int>(r.status)],
      "T"_a = r.pose.Matrix(), "registered"_a = r.registered,
      "initialized"_a = r.initialized, "fitness"_a = r.fitness,
      "rmse"_a = r.rmse, "degenerate_t"_a = r.degenerate_t,
      "degenerate_r"_a = r.degenerate_r,
      "min_eigen_ratio_t"_a = r.min_eigen_ratio_t, "step_t"_a = r.step_t,
      "step_r"_a = r.step_r, "covariance"_a = r.covariance,
      "failures"_a = r.failures);
}

// A TripLocalizer with the Python logger it calls (kept alive and at a fixed
// address: pybind11 holds the object by pointer).
struct PyTripLocalizer {
  py::object log;
  std::exception_ptr log_error;
  std::unique_ptr<TripLocalizer> localizer;
};

std::vector<Vec2> Xy(const Eigen::MatrixXd& points) {
  Require(points.rows() == 0 || points.cols() >= 2, "expected (M, 2) points");
  std::vector<Vec2> out;
  for (Eigen::Index r = 0; r < points.rows(); ++r) {
    out.emplace_back(points(r, 0), points(r, 1));
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
        const Points s = ToPoints(source);
        const Points r = ToPoints(reference);
        Placement placement;
        {
          py::gil_scoped_release release;
          placement = PlaceGravityAligned(s, r, config);
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
            const Points p = ToPoints(points);
            const std::vector<double> t = ToDoubleVector(times);
            Require(t.empty() || t.size() == p.size(),
                    "times must be empty or match the points");
            LocalizationResult r;
            {
              py::gil_scoped_release release;
              r = localizer.Process(p, t, stamp,
                                    Pose::FromMatrix(odom_from_sensor));
            }
            return ResultToDict(r);
          },
          "points"_a, "times"_a, "stamp"_a, "T_odom_sensor"_a,
          "One sweep (sensor frame) with its odometry pose: dict of status "
          "(initializing / tracking / lost), T (map <- sensor), registered, "
          "initialized, fitness, rmse, degenerate_t / _r, min_eigen_ratio_t, "
          "step_t (m) / step_r (deg), "
          "covariance (6x6, [t, phi]), failures.")
      .def_property_readonly(
          "T_map_odom",
          [](const Localizer& l) { return l.map_from_odom().Matrix(); })
      .def_property_readonly("last_placement",
                             [](const Localizer& l) {
                               return PlacementToDict(l.last_placement());
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
            const Points p = ToPoints(points);
            const std::vector<float> i = ToFloatVector(intensity);
            const std::vector<double> t = ToDoubleVector(times);
            Require(t.empty() || t.size() == p.size(),
                    "times must be empty or match the points");
            std::optional<SweepLocalization> s;
            {
              py::gil_scoped_release release;
              s = self.localizer->Process(p, i, t, stamp, index, last,
                                          localize);
            }
            RethrowIfSet(self.log_error);
            if (!s) return py::none();
            py::dict out = ResultToDict(s->result);
            out["index"] = s->index;
            out["stamp"] = s->stamp;
            out["T_map_vehicle"] = s->map_from_vehicle.Matrix();
            out["T_odom_vehicle"] = s->odom_from_vehicle.Matrix();
            out["sigma_t"] = s->sigma_t;
            out["sigma_yaw_deg"] = s->sigma_yaw_deg;
            out["ms"] = s->ms;
            return std::move(out);
          },
          "points"_a, "intensity"_a, "times"_a, "stamp"_a, "index"_a,
          "last"_a, "localize"_a,
          "One sweep through the odometry and, with `localize`, the "
          "localizer: None, or the localizer's dict plus index, stamp, "
          "T_map_vehicle, T_odom_vehicle, sigma_t, sigma_yaw_deg, ms.")
      .def("summary",
           [](const PyTripLocalizer& self) {
             const TripLocalizationSummary s = self.localizer->Summary();
             const bool tracked = s.first_tracked >= 0;
             const auto optional = [&](double value) -> py::object {
               return tracked ? py::cast(value) : py::none();
             };
             return py::dict(
                 "sweeps"_a = s.sweeps,
                 "first_tracked"_a = tracked ? py::cast(s.first_tracked)
                                             : py::none(),
                 "init_time_s"_a = optional(s.init_time_s),
                 "init_distance_m"_a = optional(s.init_distance_m),
                 "placement_attempts"_a = s.placement_attempts,
                 "tracked"_a = s.tracked, "registered"_a = s.registered,
                 "registered_share_of_tracked"_a =
                     optional(s.registered_share_of_tracked),
                 "lost_sweeps"_a = s.lost_sweeps,
                 "degenerate_share"_a = optional(s.degenerate_share),
                 "ms_median"_a = s.ms_median, "ms_p95"_a = s.ms_p95);
           })
      .def_property_readonly("map_size",
                             [](const PyTripLocalizer& self) {
                               return self.localizer->localizer().map_size();
                             });

  m.def(
      "nav_position",
      [](const Mat4& map_from_vehicle, const py::list& edges,
         const py::list& slots, const py::list& floors) -> py::dict {
        std::vector<NavEdge> e;
        for (const py::handle& item : edges) {
          const auto t = item.cast<
              std::tuple<int, std::string, Eigen::MatrixXd>>();
          e.push_back({std::get<0>(t), std::get<1>(t), Xy(std::get<2>(t))});
        }
        std::vector<NavSlot> s;
        for (const py::handle& item : slots) {
          const auto t = item.cast<
              std::tuple<int, std::string, Eigen::MatrixXd>>();
          s.push_back({std::get<0>(t), std::get<1>(t), Xy(std::get<2>(t))});
        }
        std::vector<FloorHeights> f;
        for (const py::handle& item : floors) {
          const auto t = item.cast<std::tuple<int, double, double, double,
                                              Eigen::MatrixXd>>();
          FloorHeights floor;
          floor.level = std::get<0>(t);
          floor.x0 = std::get<1>(t);
          floor.y0 = std::get<2>(t);
          floor.res = std::get<3>(t);
          const Eigen::MatrixXd& z = std::get<4>(t);
          floor.height = static_cast<int>(z.rows());
          floor.width = static_cast<int>(z.cols());
          for (Eigen::Index r = 0; r < z.rows(); ++r) {
            for (Eigen::Index c = 0; c < z.cols(); ++c) {
              floor.z.push_back(z(r, c));
            }
          }
          f.push_back(std::move(floor));
        }
        const NavPosition p =
            Where(Pose::FromMatrix(map_from_vehicle), e, s, f);
        py::object edge = py::none();
        if (p.edge) {
          edge = py::dict("edge"_a = p.edge->edge, "kind"_a = p.edge->kind,
                          "distance"_a = p.edge->distance,
                          "along"_a = p.edge->along,
                          "lateral"_a = p.edge->lateral,
                          "heading_rel_deg"_a = p.edge->heading_rel_deg);
        }
        return py::dict(
            "xyz"_a = p.xyz, "yaw_deg"_a = p.yaw_deg,
            "level"_a = p.level ? py::cast(*p.level) : py::none(),
            "edge"_a = edge,
            "slot"_a = p.slot ? py::cast(*p.slot) : py::none());
      },
      "T_map_vehicle"_a, "edges"_a, "slots"_a, "floors"_a,
      "Where a vehicle is on the route graph. edges: [(level, kind, (M, 2))], "
      "slots: [(level, slot_id, corners (K, 2))], floors: [(level, x0, y0, "
      "res, floor_z (H, W))]. Dict of xyz, yaw_deg, level, edge (edge, kind, "
      "distance, along, lateral, heading_rel_deg) or None, slot or None.");
}

}  // namespace gr::bindings
