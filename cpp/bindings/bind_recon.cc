// Bindings of section 4: IMU / wheel prior, odometry pipeline, keyframes,
// loop closure, the back-end (graph stage, LiDAR bundle adjustment), the
// continuous-time trajectory (deskewing, camera rows), camera calibration,
// map accumulation and dynamic points, and the comparison with a reference
// cloud (acceptance against the handheld ground truth).

#include <algorithm>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bind_util.h"
#include "gr/backend/backend.h"
#include "gr/backend/keyframes.h"
#include "gr/backend/lidar_ba.h"
#include "gr/backend/loop_closure.h"
#include "gr/calibration/camera_calibration.h"
#include "gr/calibration/clock_sync.h"
#include "gr/calibration/lidar_imu.h"
#include "gr/mapping/static_map.h"
#include "gr/odometry/odometry.h"
#include "gr/odometry/trajectory.h"
#include "gr/registration/cloud_compare.h"
#include "gr/sensors/camera.h"
#include "gr/sensors/imu.h"

namespace gr::bindings {
namespace {

namespace py = ::pybind11;
using ::pybind11::literals::operator""_a;

py::object PoseOrNone(const std::optional<Pose>& pose) {
  if (!pose) return py::none();
  return py::cast(pose->Matrix());
}

py::dict SummaryDict(const GraphSummary& summary) {
  return py::dict("iterations"_a = summary.iterations,
                  "initial_cost"_a = summary.initial_cost,
                  "final_cost"_a = summary.final_cost,
                  "converged"_a = summary.converged,
                  "loop_chi2_median"_a = summary.loop_chi2_median,
                  "loop_chi2_max"_a = summary.loop_chi2_max,
                  "imu_factors"_a = summary.imu_factors,
                  "imu_chi2_median"_a = summary.imu_chi2_median);
}

py::list BaRounds(const LidarBaSummary& summary) {
  py::list rounds;
  for (const LidarBaRound& round : summary.rounds) {
    rounds.append(py::dict(
        "planes"_a = round.planes, "observations"_a = round.observations,
        "rms_median_before"_a = round.rms_median_before,
        "rms_p90_before"_a = round.rms_p90_before,
        "rms_median_after"_a = round.rms_median_after,
        "rms_p90_after"_a = round.rms_p90_after,
        "shift_median"_a = round.shift_median, "shift_max"_a = round.shift_max,
        "iterations"_a = round.iterations));
  }
  return rounds;
}

py::list TripAlignments(const std::vector<TripAlignment>& alignments) {
  py::list out;
  for (const TripAlignment& alignment : alignments) {
    out.append(py::dict(
        "trip"_a = alignment.trip, "aligned"_a = alignment.aligned,
        "T"_a = alignment.correction.Matrix(), "score"_a = alignment.score,
        "fitness"_a = alignment.fitness, "rmse"_a = alignment.rmse,
        "runner_up"_a = alignment.runner_up,
        "hypotheses"_a = alignment.hypotheses));
  }
  return out;
}

// Elements of a Vec6 (pose or IMU bias).
constexpr int kVec6Size = static_cast<int>(Vec6::SizeAtCompileTime);
// Clock-sync windows are reported as [center, lag, residual, samples] rows.
constexpr int kClockWindowColumns = 4;
// Pairs of keyframe indices.
constexpr int kPairSize = 2;
// Camera tracks are [image, u, v] rows.
constexpr int kTrackColumns = 3;
constexpr int kTrackImageColumn = 0;
constexpr int kTrackPixelColumn = 1;

py::array_t<double> Vec6Array(const std::vector<Vec6>& values) {
  py::array_t<double> array({static_cast<py::ssize_t>(values.size()),
                             static_cast<py::ssize_t>(kVec6Size)});
  double* data = array.mutable_data();
  for (size_t i = 0; i < values.size(); ++i) {
    for (int element = 0; element < kVec6Size; ++element) {
      data[kVec6Size * i + static_cast<size_t>(element)] = values[i][element];
    }
  }
  return array;
}

std::vector<Vec3> Vec3sFrom(const py::object& values) {
  const Points points = PointsFrom(values);
  return std::vector<Vec3>(points.begin(), points.end());
}

void RequirePoses(const KeyframeSet& keyframes,
                  const std::vector<Pose>& poses) {
  Require(poses.size() == keyframes.size(), "one pose per keyframe expected");
}

void RequireLoops(const KeyframeSet& keyframes,
                  const std::vector<LoopConstraint>& loops) {
  const int keyframe_count = static_cast<int>(keyframes.size());
  for (const LoopConstraint& loop : loops) {
    RequireIndex(loop.i >= 0 && loop.i < keyframe_count && loop.j >= 0 &&
                     loop.j < keyframe_count,
                 "loop refers to an unknown keyframe");
  }
}

void BindImu(py::module_& m) {
  py::class_<ImuSamples>(m, "ImuSamples")
      .def(py::init<>())
      .def(py::init([](const Eigen::MatrixXd& rows,
                       const Mat3& vehicle_from_imu, const Vec3& position) {
             Require(rows.rows() == 0 || rows.cols() >= ImuSamples::kCsvColumns,
                     "expected rows of [t, gx, gy, gz, ax, ay, az]");
             return ImuSamples::FromRows(rows, vehicle_from_imu, position);
           }),
           "rows"_a, "R_vehicle_imu"_a = Mat3::Identity(),
           "position"_a = Vec3::Zero(),
           "rows: (N, 7) [t, gx, gy, gz, ax, ay, az] in the IMU frame; stored "
           "in the vehicle frame. position: the IMU origin in the vehicle "
           "frame (T_vehicle_imu translation), the body origin of inertial "
           "factors.")
      .def("__len__", &ImuSamples::size)
      .def_readonly("position", &ImuSamples::position)
      .def_property_readonly(
          "t",
          [](const ImuSamples& samples) { return ToArray(samples.stamps); })
      .def_property_readonly(
          "gyro",
          [](const ImuSamples& samples) { return PointsToArray(samples.gyro); })
      .def_property_readonly("acc", [](const ImuSamples& samples) {
        return PointsToArray(samples.accel);
      });

  py::class_<WheelSamples>(m, "WheelSamples")
      .def(py::init<>())
      .def(py::init([](const Eigen::MatrixXd& rows, double speed_scale) {
             Require(
                 rows.rows() == 0 || rows.cols() >= WheelSamples::kCsvColumns,
                 "expected rows of [t, speed]");
             return WheelSamples::FromRows(rows, speed_scale);
           }),
           "rows"_a, "speed_scale"_a = 1.0, "rows: (M, 2) [t, speed (m/s)].")
      .def("__len__", &WheelSamples::size);

  py::class_<ClockSyncConfig>(m, "ClockSyncConfig")
      .def(py::init<>())
      .def_readwrite("window", &ClockSyncConfig::window)
      .def_readwrite("max_lag", &ClockSyncConfig::max_lag)
      .def_readwrite("step", &ClockSyncConfig::step)
      .def_readwrite("smoothing", &ClockSyncConfig::smoothing)
      .def_readwrite("min_rate", &ClockSyncConfig::min_rate)
      .def_readwrite("max_residual", &ClockSyncConfig::max_residual)
      .def_readwrite("min_samples", &ClockSyncConfig::min_samples);

  m.def(
      "estimate_clock_offset",
      [](const std::vector<double>& reference_stamps,
         const PointsIn& reference_rates,
         const std::vector<double>& other_stamps, const PointsIn& other_rates,
         const ClockSyncConfig& config) {
        if (static_cast<size_t>(reference_rates.rows()) !=
                reference_stamps.size() ||
            static_cast<size_t>(other_rates.rows()) != other_stamps.size()) {
          throw py::value_error("stamps and rates differ in length");
        }
        if (!std::is_sorted(reference_stamps.begin(), reference_stamps.end()) ||
            !std::is_sorted(other_stamps.begin(), other_stamps.end())) {
          throw py::value_error("stamps must be ascending");
        }
        ClockSyncResult result;
        {
          const Points reference = ToPoints(reference_rates);
          const Points other = ToPoints(other_rates);
          py::gil_scoped_release release;
          result = EstimateClockOffset(reference_stamps, reference,
                                       other_stamps, other, config);
        }
        Eigen::MatrixXd windows(
            static_cast<Eigen::Index>(result.windows.size()),
            kClockWindowColumns);
        for (size_t i = 0; i < result.windows.size(); ++i) {
          const ClockSyncWindow& window = result.windows[i];
          windows.row(static_cast<Eigen::Index>(i)) << window.center,
              window.lag, window.residual, window.samples;
        }
        py::dict out;
        out["valid"] = result.valid;
        out["windows"] = windows;
        out["offset"] = result.offset;
        out["drift"] = result.drift;
        out["reference_time"] = result.reference_time;
        out["rotation"] = Mat3(result.rotation);
        return out;
      },
      "reference_stamps"_a, "reference_rates"_a, "other_stamps"_a,
      "other_rates"_a, "config"_a = ClockSyncConfig(),
      "Clock offset of `other` against `reference` from two gyros: dict with "
      "windows (M, 4) [center, lag, residual, samples], offset, drift, "
      "reference_time, rotation (reference <- other), valid. An `other` "
      "sample stamped t was taken at reference time t + lag(t).");

  py::class_<MotionPrior>(m, "MotionPrior")
      .def(py::init<>())
      .def(py::init<ImuSamples, WheelSamples>(), "imu"_a, "wheel"_a)
      .def_property_readonly("valid", &MotionPrior::valid)
      .def_property_readonly("has_gyro", &MotionPrior::has_gyro)
      .def(
          "delta",
          [](const MotionPrior& prior, double t0, double t1) {
            return PoseOrNone(prior.Delta(t0, t1));
          },
          "t0"_a, "t1"_a,
          "Vehicle-frame motion T_v(t0)^-1 T_v(t1), or None outside the IMU "
          "span.");

  m.def(
      "deskew_with_prior",
      [](const PointsIn& points, const py::object& times, double stamp,
         const MotionPrior& prior, const Mat4& vehicle_from_lidar,
         int samples) -> py::object {
        const Points cloud = ToPoints(points);
        const std::vector<double> point_times = ToDoubleVector(times);
        std::optional<Points> out;
        {
          py::gil_scoped_release release;
          out = DeskewWithPrior(cloud, point_times, stamp, prior,
                                Pose::FromMatrix(vehicle_from_lidar), samples);
        }
        if (!out) return py::none();
        return py::cast(ToMatrix(*out));
      },
      "points"_a, "times"_a, "stamp"_a, "prior"_a, "T_vehicle_lidar"_a,
      "samples"_a = 16,
      "IMU / wheel motion compensation of one sweep; None if the prior does "
      "not cover it.");

  py::class_<GravityConfig>(m, "GravityConfig")
      .def(py::init<>())
      .def_readwrite("window", &GravityConfig::window)
      .def_readwrite("max_dev", &GravityConfig::max_dev)
      .def_readwrite("max_yaw_rate", &GravityConfig::max_yaw_rate)
      .def_readwrite("sigma_deg", &GravityConfig::sigma_deg);

  py::class_<GravityMeasurement>(m, "GravityMeasurement")
      .def_readonly("index", &GravityMeasurement::index)
      .def_readonly("up", &GravityMeasurement::up)
      .def_readonly("sigma_rad", &GravityMeasurement::sigma_rad);

  m.def("gravity_measurements", &MeasureGravity, "imu"_a, "wheel"_a,
        "R_sensor_vehicle"_a, "stamps"_a, "config"_a = GravityConfig(),
        "World +z in the sensor frame around each stamp (straight driving "
        "only).");
}

void BindOdometryPipeline(py::module_& m) {
  py::class_<KeyframeConfig>(m, "KeyframeConfig")
      .def(py::init<>())
      .def_readwrite("distance", &KeyframeConfig::distance)
      .def_readwrite("angle", &KeyframeConfig::angle)
      .def_readwrite("voxel", &KeyframeConfig::voxel);

  py::class_<OdometryPipeline>(m, "OdometryPipeline")
      .def(py::init(
               [](const OdometryConfig& config, const KeyframeConfig& keyframes,
                  const Mat4& vehicle_from_lidar,
                  const Mat4& world_from_vehicle0, const MotionPrior& prior) {
                 Require(config.voxel_size > 0.0,
                         "voxel_size must be positive");
                 return OdometryPipeline(
                     config, keyframes, Pose::FromMatrix(vehicle_from_lidar),
                     Pose::FromMatrix(world_from_vehicle0), prior);
               }),
           "config"_a, "keyframes"_a, "T_vehicle_lidar"_a,
           "T_world_vehicle0"_a = Mat4::Identity(), "prior"_a = MotionPrior())
      .def(
          "process",
          [](OdometryPipeline& pipeline, const PointsIn& points,
             const py::object& intensity, const py::object& times, double stamp,
             bool force_keyframe, const py::object& external_pose,
             bool fill_sweep) -> py::tuple {
            const Points cloud = ToPoints(points);
            const std::vector<float> point_intensity = ToFloatVector(intensity);
            const std::vector<double> point_times = ToDoubleVector(times);
            std::optional<Pose> external;
            if (!external_pose.is_none()) {
              external = Pose::FromMatrix(external_pose.cast<Mat4>());
            }
            OdometryFrame frame;
            {
              py::gil_scoped_release release;
              frame =
                  pipeline.Process(cloud, point_intensity, point_times, stamp,
                                   force_keyframe, external, fill_sweep);
            }
            if (!frame.is_keyframe && !fill_sweep) {
              return py::make_tuple(frame.pose.Matrix(), false, py::none(),
                                    py::none());
            }
            return py::make_tuple(frame.pose.Matrix(), frame.is_keyframe,
                                  ToMatrix(frame.points),
                                  ToArray(frame.intensity));
          },
          "points"_a, "intensity"_a, "times"_a, "stamp"_a,
          "force_keyframe"_a = false, "external_pose"_a = py::none(),
          "fill_sweep"_a = false,
          "One sweep -> (world<-lidar pose, is_keyframe, compensated sweep "
          "(keyframes, or every frame with fill_sweep) or None, its "
          "intensity or None).")
      .def("poses",
           [](const OdometryPipeline& pipeline) {
             return PosesToArray(pipeline.poses());
           })
      .def("stamps",
           [](const OdometryPipeline& pipeline) {
             return ToArray(pipeline.stamps());
           })
      .def("keyframes",
           [](const OdometryPipeline& pipeline) {
             return ToArray(pipeline.keyframes());
           })
      .def_property_readonly("uses_prior", &OdometryPipeline::uses_prior)
      .def_property_readonly("uses_gyro_prior",
                             &OdometryPipeline::uses_gyro_prior)
      .def_property_readonly("uses_lio", &OdometryPipeline::uses_lio)
      .def(
          "lio_stats",
          [](const OdometryPipeline& pipeline) {
            Require(pipeline.uses_lio(), "the pipeline does not run the LIO");
            const LioStats& stats = pipeline.lio_stats();
            py::dict out;
            out["correspondences"] = stats.correspondences;
            out["iterations"] = stats.iterations;
            out["rmse"] = stats.rmse;
            out["min_translation_information"] =
                stats.min_translation_information;
            out["point_residuals"] = stats.point_residuals;
            out["stationary"] = stats.stationary;
            out["velocity"] = stats.velocity;
            return out;
          },
          "Statistics of the last LIO update.");
}

void BindKeyframes(py::module_& m) {
  py::class_<KeyframeSet>(m, "KeyframeSet")
      .def(py::init([](const std::vector<double>& voxels) {
             for (double voxel : voxels) {
               Require(voxel > 0.0,
                       "KeyframeSet: voxel sizes must be positive");
             }
             return KeyframeSet(voxels);
           }),
           "voxels"_a = std::vector<double>{KeyframeSet::kDefaultVoxel},
           "Keyframe table; sweeps are kept downsampled at each voxel size of "
           "`voxels`.")
      .def(
          "add",
          [](KeyframeSet& keyframes, int trip, int frame, double stamp,
             const Mat4& odom, const py::object& sweep) {
            Keyframe keyframe;
            keyframe.trip = trip;
            keyframe.frame = frame;
            keyframe.stamp = stamp;
            keyframe.odom = Pose::FromMatrix(odom);
            const Points points = PointsFrom(sweep);
            py::gil_scoped_release release;
            return keyframes.Add(keyframe, points);
          },
          "trip"_a, "frame"_a, "stamp"_a, "odom"_a, "sweep"_a)
      .def("__len__", &KeyframeSet::size)
      .def_property_readonly("voxels", &KeyframeSet::voxels)
      .def_property_readonly("num_trips", &KeyframeSet::num_trips)
      .def_property_readonly("trip",
                             [](const KeyframeSet& keyframes) {
                               std::vector<int> trips;
                               for (const Keyframe& keyframe :
                                    keyframes.keyframes()) {
                                 trips.push_back(keyframe.trip);
                               }
                               return ToArray(trips);
                             })
      .def_property_readonly("stamp",
                             [](const KeyframeSet& keyframes) {
                               std::vector<double> stamps;
                               for (const Keyframe& keyframe :
                                    keyframes.keyframes()) {
                                 stamps.push_back(keyframe.stamp);
                               }
                               return ToArray(stamps);
                             })
      .def(
          "indices_of_trip",
          [](const KeyframeSet& keyframes, int trip) {
            return ToArray(keyframes.TripIndices(trip));
          },
          "trip"_a)
      .def("odom_poses",
           [](const KeyframeSet& keyframes) {
             return PosesToArray(keyframes.OdomPoses());
           })
      .def(
          "cloud",
          [](const KeyframeSet& keyframes, size_t k, double voxel) {
            RequireIndex(k < keyframes.size(), "keyframe index out of range");
            Require(keyframes.HasVoxel(voxel),
                    "sweeps are not kept at this voxel size");
            return ToMatrix(keyframes.Cloud(k, voxel));
          },
          "k"_a, "voxel"_a)
      .def(
          "path_distance",
          [](const KeyframeSet& keyframes, const py::object& poses) {
            const std::vector<Pose> keyframe_poses = PosesFrom(poses);
            RequirePoses(keyframes, keyframe_poses);
            return ToArray(keyframes.PathDistance(keyframe_poses));
          },
          "poses"_a)
      .def(
          "submap",
          [](const KeyframeSet& keyframes, size_t k, int half_window,
             double voxel, const py::object& poses) {
            const std::vector<Pose> keyframe_poses = PosesFrom(poses);
            RequirePoses(keyframes, keyframe_poses);
            RequireIndex(k < keyframes.size(), "keyframe index out of range");
            Require(keyframes.HasVoxel(voxel),
                    "sweeps are not kept at this voxel size");
            return ToMatrix(
                keyframes.Submap(k, half_window, voxel, keyframe_poses));
          },
          "k"_a, "half_window"_a, "voxel"_a, "poses"_a);
}

void BindLoops(py::module_& m) {
  py::class_<LoopConfig>(m, "LoopConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &LoopConfig::enabled)
      .def_readwrite("stride", &LoopConfig::stride)
      .def_readwrite("min_index_gap", &LoopConfig::min_index_gap)
      .def_readwrite("search_radius", &LoopConfig::search_radius)
      .def_readwrite("level_gate", &LoopConfig::level_gate)
      .def_readwrite("sc_rings", &LoopConfig::sc_rings)
      .def_readwrite("sc_sectors", &LoopConfig::sc_sectors)
      .def_readwrite("sc_max_radius", &LoopConfig::sc_max_radius)
      .def_readwrite("sc_voxel", &LoopConfig::sc_voxel)
      .def_readwrite("sc_threshold", &LoopConfig::sc_threshold)
      .def_readwrite("top_k", &LoopConfig::top_k)
      .def_readwrite("submap_half_window", &LoopConfig::submap_half_window)
      .def_readwrite("icp_voxel", &LoopConfig::icp_voxel)
      .def_readwrite("icp_schedule", &LoopConfig::icp_schedule)
      .def_readwrite("icp_iters", &LoopConfig::icp_iters)
      .def_readwrite("min_fitness", &LoopConfig::min_fitness)
      .def_readwrite("max_rmse", &LoopConfig::max_rmse)
      .def_readwrite("max_correction", &LoopConfig::max_correction)
      .def_readwrite("pcm_threshold", &LoopConfig::pcm_threshold)
      .def_readwrite("pcm_drift_per_m", &LoopConfig::pcm_drift_per_m);

  py::class_<LoopConstraint>(m, "LoopConstraint")
      .def(py::init([](int i, int j, const Mat4& j_from_i, double fitness,
                       double rmse, double sc_distance) {
             LoopConstraint loop;
             loop.i = i;
             loop.j = j;
             loop.j_from_i = Pose::FromMatrix(j_from_i);
             loop.fitness = fitness;
             loop.rmse = rmse;
             loop.sc_distance = sc_distance;
             return loop;
           }),
           "i"_a, "j"_a, "T_ji"_a, "fitness"_a = 1.0, "rmse"_a = 0.0,
           "sc_distance"_a = 0.0)
      .def_readwrite("i", &LoopConstraint::i)
      .def_readwrite("j", &LoopConstraint::j)
      .def_property(
          "T_ji",
          [](const LoopConstraint& loop) { return loop.j_from_i.Matrix(); },
          [](LoopConstraint& loop, const Mat4& matrix) {
            loop.j_from_i = Pose::FromMatrix(matrix);
          })
      .def_readwrite("fitness", &LoopConstraint::fitness)
      .def_readwrite("rmse", &LoopConstraint::rmse)
      .def_readwrite("sc_distance", &LoopConstraint::sc_distance);

  m.def(
      "detect_loops",
      [](const KeyframeSet& keyframes, const py::object& poses,
         const LoopConfig& config) {
        const std::vector<Pose> keyframe_poses = PosesFrom(poses);
        if (config.enabled &&
            keyframes.size() >= static_cast<size_t>(kMinLoopKeyframes)) {
          RequirePoses(keyframes, keyframe_poses);
          Require(keyframes.HasVoxel(config.sc_voxel) &&
                      keyframes.HasVoxel(config.icp_voxel),
                  "keyframes must keep sweeps at sc_voxel and icp_voxel");
        }
        LoopStats stats;
        std::vector<LoopConstraint> loops;
        {
          py::gil_scoped_release release;
          loops = DetectLoops(keyframes, keyframe_poses, config, &stats);
        }
        return py::make_tuple(loops, py::dict("queries"_a = stats.queries,
                                              "tested"_a = stats.tested));
      },
      "keyframes"_a, "poses"_a, "config"_a, "Returns (loops, stats).");
  m.def(
      "pcm_filter",
      [](const std::vector<LoopConstraint>& loops, const KeyframeSet& keyframes,
         const py::object& poses, const LoopConfig& config) {
        const std::vector<Pose> keyframe_poses = PosesFrom(poses);
        RequireLoops(keyframes, loops);
        if (loops.size() >= kMinConsistencyLoops)
          RequirePoses(keyframes, keyframe_poses);
        return SelectConsistentLoops(loops, keyframes, keyframe_poses, config);
      },
      "loops"_a, "keyframes"_a, "poses"_a, "config"_a);
}

void BindBackend(py::module_& m) {
  py::class_<BackendConfig>(m, "BackendConfig")
      .def(py::init<>())
      .def_readwrite("odom_sigma_t", &BackendConfig::odom_sigma_t)
      .def_readwrite("odom_sigma_t_per_m", &BackendConfig::odom_sigma_t_per_m)
      .def_readwrite("odom_sigma_r", &BackendConfig::odom_sigma_r)
      .def_readwrite("odom_sigma_r_per_m", &BackendConfig::odom_sigma_r_per_m)
      .def_readwrite("loop_sigma_t", &BackendConfig::loop_sigma_t)
      .def_readwrite("loop_sigma_r", &BackendConfig::loop_sigma_r)
      .def_readwrite("loop_robust_scale", &BackendConfig::loop_robust_scale)
      .def_readwrite("anchor_sigma_t", &BackendConfig::anchor_sigma_t)
      .def_readwrite("anchor_sigma_r", &BackendConfig::anchor_sigma_r)
      .def_readwrite("gnss_sigma_scale", &BackendConfig::gnss_sigma_scale)
      .def_readwrite("gnss_max_dt", &BackendConfig::gnss_max_dt)
      .def_readwrite("control_max_dt", &BackendConfig::control_max_dt)
      .def_readwrite("gravity", &BackendConfig::gravity)
      .def_readwrite("gravity_sigma_deg", &BackendConfig::gravity_sigma_deg)
      .def_readwrite("gravity_window", &BackendConfig::gravity_window)
      .def_readwrite("gravity_max_dev", &BackendConfig::gravity_max_dev)
      .def_readwrite("imu_factors", &BackendConfig::imu_factors)
      .def_readwrite("imu_gyro_noise", &BackendConfig::imu_gyro_noise)
      .def_readwrite("imu_accel_noise", &BackendConfig::imu_accel_noise)
      .def_readwrite("imu_gyro_bias_walk", &BackendConfig::imu_gyro_bias_walk)
      .def_readwrite("imu_accel_bias_walk", &BackendConfig::imu_accel_bias_walk)
      .def_readwrite("gyro_bias_prior", &BackendConfig::gyro_bias_prior)
      .def_readwrite("accel_bias_prior", &BackendConfig::accel_bias_prior)
      .def_readwrite("imu_max_dt", &BackendConfig::imu_max_dt)
      .def_readwrite("wheel_factors", &BackendConfig::wheel_factors)
      .def_readwrite("wheel_sigma", &BackendConfig::wheel_sigma)
      .def_readwrite("wheel_lateral_sigma", &BackendConfig::wheel_lateral_sigma)
      .def_readwrite("gravity_max_yaw_rate",
                     &BackendConfig::gravity_max_yaw_rate)
      .def_readwrite("iterations", &BackendConfig::iterations);

  py::class_<TripAlignConfig>(m, "TripAlignConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &TripAlignConfig::enabled)
      .def_readwrite("voxel", &TripAlignConfig::voxel)
      .def_readwrite("icp_schedule", &TripAlignConfig::icp_schedule)
      .def_readwrite("icp_iters", &TripAlignConfig::icp_iters)
      .def_readwrite("min_score", &TripAlignConfig::min_score)
      .def_readwrite("z_peak_ratio", &TripAlignConfig::z_peak_ratio)
      .def_readwrite("max_candidates", &TripAlignConfig::max_candidates)
      .def_readwrite("min_fitness", &TripAlignConfig::min_fitness)
      .def_readwrite("max_rmse", &TripAlignConfig::max_rmse)
      .def_readwrite("min_fitness_gap", &TripAlignConfig::min_fitness_gap)
      .def_readwrite("distinct_t", &TripAlignConfig::distinct_t)
      .def_readwrite("distinct_r", &TripAlignConfig::distinct_r)
      .def_readwrite("sigma_t", &TripAlignConfig::sigma_t)
      .def_readwrite("sigma_r", &TripAlignConfig::sigma_r);

  py::class_<RefineConfig>(m, "RefineConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &RefineConfig::enabled)
      .def_readwrite("iterations", &RefineConfig::iterations)
      .def_readwrite("voxel", &RefineConfig::voxel)
      .def_readwrite("max_corr", &RefineConfig::max_corr)
      .def_readwrite("sigma_t", &RefineConfig::sigma_t)
      .def_readwrite("sigma_r", &RefineConfig::sigma_r)
      .def_readwrite("min_fitness", &RefineConfig::min_fitness)
      .def_readwrite("icp_iters", &RefineConfig::icp_iters);

  py::class_<LidarBaConfig>(m, "LidarBaConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &LidarBaConfig::enabled)
      .def_readwrite("sweep_voxel", &LidarBaConfig::sweep_voxel)
      .def_readwrite("min_range", &LidarBaConfig::min_range)
      .def_readwrite("max_range", &LidarBaConfig::max_range)
      .def_readwrite("voxel", &LidarBaConfig::voxel)
      .def_readwrite("max_depth", &LidarBaConfig::max_depth)
      .def_readwrite("max_thickness", &LidarBaConfig::max_thickness)
      .def_readwrite("min_extent", &LidarBaConfig::min_extent)
      .def_readwrite("inlier_gate", &LidarBaConfig::inlier_gate)
      .def_readwrite("min_inlier_fraction", &LidarBaConfig::min_inlier_fraction)
      .def_readwrite("max_rms", &LidarBaConfig::max_rms)
      .def_readwrite("min_points", &LidarBaConfig::min_points)
      .def_readwrite("min_keyframes", &LidarBaConfig::min_keyframes)
      .def_readwrite("max_keyframes", &LidarBaConfig::max_keyframes)
      .def_readwrite("point_sigma", &LidarBaConfig::point_sigma)
      .def_readwrite("angular_sigma", &LidarBaConfig::angular_sigma)
      .def_readwrite("max_cluster_weight", &LidarBaConfig::max_cluster_weight)
      .def_readwrite("robust_scale", &LidarBaConfig::robust_scale)
      .def_readwrite("rounds", &LidarBaConfig::rounds)
      .def_readwrite("iterations", &LidarBaConfig::iterations)
      .def_readwrite("hierarchical_above", &LidarBaConfig::hierarchical_above)
      .def_readwrite("window", &LidarBaConfig::window)
      .def_readwrite("window_prior_sigma_t",
                     &LidarBaConfig::window_prior_sigma_t)
      .def_readwrite("window_prior_sigma_r",
                     &LidarBaConfig::window_prior_sigma_r)
      .def_readwrite("window_gravity_sigma_deg",
                     &LidarBaConfig::window_gravity_sigma_deg);

  py::class_<AppearanceLoopConfig>(m, "AppearanceLoopConfig")
      .def(py::init<>())
      .def_readwrite("coarse_voxel", &AppearanceLoopConfig::coarse_voxel)
      .def_readwrite("z_peak_ratio", &AppearanceLoopConfig::z_peak_ratio)
      .def_readwrite("max_hypotheses", &AppearanceLoopConfig::max_hypotheses)
      .def_readwrite("min_fitness_gap", &AppearanceLoopConfig::min_fitness_gap)
      .def_readwrite("distinct_t", &AppearanceLoopConfig::distinct_t)
      .def_readwrite("distinct_r", &AppearanceLoopConfig::distinct_r);

  py::class_<GraphStageConfig>(m, "GraphStageConfig")
      .def(py::init<>())
      .def_readwrite("loops", &GraphStageConfig::loops)
      .def_readwrite("appearance", &GraphStageConfig::appearance)
      .def_readwrite("backend", &GraphStageConfig::backend)
      .def_readwrite("trip_align", &GraphStageConfig::trip_align)
      .def_readwrite("ba", &GraphStageConfig::ba)
      .def_readwrite("refine", &GraphStageConfig::refine)
      .def("required_voxels", &GraphStageConfig::RequiredVoxels,
           "Voxel sizes at which a KeyframeSet must keep sweeps.");

  py::class_<TripInputs>(m, "TripInputs")
      .def(py::init<>())
      .def_property(
          "T_vehicle_lidar",
          [](const TripInputs& trip) {
            return trip.vehicle_from_lidar.Matrix();
          },
          [](TripInputs& trip, const Mat4& matrix) {
            trip.vehicle_from_lidar = Pose::FromMatrix(matrix);
          })
      .def_property(
          "initial_pose",
          [](const TripInputs& trip) { return PoseOrNone(trip.initial_pose); },
          [](TripInputs& trip, const py::object& matrix) {
            trip.initial_pose.reset();
            if (!matrix.is_none()) {
              trip.initial_pose = Pose::FromMatrix(matrix.cast<Mat4>());
            }
          },
          "World <- vehicle at the entrance (INS), or None.")
      .def_property(
          "gnss", [](const TripInputs& trip) { return trip.gnss; },
          [](TripInputs& trip, const py::object& rows) {
            trip.gnss = rows.is_none() ? Eigen::MatrixXd()
                                       : rows.cast<Eigen::MatrixXd>();
            Require(trip.gnss.rows() == 0 || trip.gnss.cols() >= kGnssStdColumn,
                    "gnss rows must be [t, x, y, z(, std)]");
          },
          "(M, 4|5) rows [t, x, y, z(, std)] or None.")
      .def_property(
          "control_points",
          [](const TripInputs& trip) { return trip.control_points; },
          [](TripInputs& trip, const py::object& rows) {
            trip.control_points = rows.is_none()
                                      ? Eigen::MatrixXd()
                                      : rows.cast<Eigen::MatrixXd>();
            Require(trip.control_points.rows() == 0 ||
                        trip.control_points.cols() >= kControlStdColumn,
                    "control point rows must be [t, px, py, pz, x, y, z(, "
                    "std)]");
          },
          "(M, 7|8) rows [t, px, py, pz (LiDAR frame), x, y, z (world), "
          "std] or None.")
      .def_readwrite("frame_stamps", &TripInputs::frame_stamps)
      .def_property(
          "frame_odometry",
          [](const TripInputs& trip) {
            return PosesToArray(trip.frame_odometry);
          },
          [](TripInputs& trip, const py::object& poses) {
            trip.frame_odometry = PosesFrom(poses);
          })
      .def_readwrite("imu", &TripInputs::imu)
      .def_readwrite("wheel", &TripInputs::wheel);

  py::class_<Anchor>(m, "Anchor")
      .def_property_readonly(
          "type",
          [](const Anchor& anchor) {
            return anchor.type == Anchor::Type::kPose ? "pose" : "position";
          })
      .def_readonly("k", &Anchor::keyframe)
      .def_property_readonly(
          "T", [](const Anchor& anchor) { return anchor.pose.Matrix(); })
      .def_readonly("p", &Anchor::position)
      .def_readonly("lever", &Anchor::lever)
      .def_readonly("sigma_t", &Anchor::sigma_t)
      .def_readonly("sigma_r", &Anchor::sigma_r)
      .def_readonly("sigma", &Anchor::sigma)
      .def_readonly("weak", &Anchor::weak);

  m.def(
      "run_graph_stage",
      [](const KeyframeSet& keyframes, const std::vector<TripInputs>& trips,
         const GraphStageConfig& config, const py::object& log,
         const py::object& appearance) {
        std::vector<std::pair<int, int>> pairs;
        if (!appearance.is_none()) {
          auto array = py::array_t<int, py::array::c_style |
                                            py::array::forcecast>::ensure(
              appearance);
          Require(array && (array.size() == 0 ||
                            (array.ndim() == 2 && array.shape(1) == kPairSize)),
                  "appearance must be (N, 2) keyframe pairs");
          for (py::ssize_t row = 0; array.size() && row < array.shape(0);
               ++row) {
            pairs.emplace_back(array.at(row, 0), array.at(row, 1));
          }
        }
        for (double voxel : config.RequiredVoxels()) {
          Require(keyframes.HasVoxel(voxel),
                  "keyframes must keep sweeps at every voxel of "
                  "config.required_voxels()");
        }
        for (const TripInputs& trip : trips) {
          Require(trip.frame_stamps.size() == trip.frame_odometry.size(),
                  "frame_stamps and frame_odometry must have equal length");
          Require(trip.imu.gyro.size() == trip.imu.stamps.size() &&
                      trip.wheel.speed.size() == trip.wheel.stamps.size(),
                  "inconsistent IMU / wheel samples");
        }
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        GraphStageResult result;
        {
          py::gil_scoped_release release;
          result = RunGraphStage(keyframes, trips, config, logger, pairs);
        }
        RethrowIfSet(log_error);
        const InertialEstimate& inertial = result.inertial;
        return py::dict(
            "poses"_a = PosesToArray(result.poses), "loops"_a = result.loops,
            "anchors"_a = result.anchors, "gravity"_a = result.gravity,
            "summary"_a = SummaryDict(result.summary),
            "loops_tested"_a = result.loop_stats.tested,
            "loops_verified"_a = result.loops_verified,
            "ba_rounds"_a = BaRounds(result.ba),
            "trip_alignments"_a = TripAlignments(result.trip_alignments),
            "velocities"_a = PointsToArray(
                Points(inertial.velocities.begin(), inertial.velocities.end())),
            "biases"_a = Vec6Array(inertial.biases),
            "inertial_valid"_a = ToBoolArray(inertial.valid));
      },
      "keyframes"_a, "trips"_a, "config"_a, "log"_a = py::none(),
      "appearance"_a = py::none(),
      "Whole back-end: anchors, IMU / wheel / bias (or gravity) factors, graph "
      "without loops, loop detection + PCM, graph with loops, LiDAR bundle "
      "adjustment (or scan-to-map refinement). Returns a dict with poses "
      "(K, 4, 4), loops, anchors, gravity, summary, ba_rounds (statistics "
      "per bundle-adjustment round), and per keyframe velocities (K, 3), "
      "biases (K, 6) [gyro, accel] and inertial_valid (K,) (empty without "
      "IMU factors).");

  m.def(
      "correct_frame_poses",
      [](const std::vector<double>& kf_stamps, const py::object& kf_odom,
         const py::object& kf_poses, const std::vector<double>& frame_stamps,
         const py::object& frame_odom) {
        const std::vector<Pose> odom = PosesFrom(kf_odom);
        const std::vector<Pose> poses = PosesFrom(kf_poses);
        const std::vector<Pose> frames = PosesFrom(frame_odom);
        Require(
            kf_stamps.size() == odom.size() && kf_stamps.size() == poses.size(),
            "keyframe stamps and poses must have equal length");
        Require(frame_stamps.size() == frames.size(),
                "frame stamps and poses must have equal length");
        return PosesToArray(
            CorrectFramePoses(kf_stamps, odom, poses, frame_stamps, frames));
      },
      "kf_stamps"_a, "kf_odom"_a, "kf_poses"_a, "frame_stamps"_a,
      "frame_odom"_a,
      "All-frame poses of a trip: odometry corrected by the interpolated "
      "keyframe corrections.");
}

void BindMap(py::module_& m) {
  py::class_<MapFusionConfig>(m, "MapFusionConfig")
      .def(py::init<>())
      .def_readwrite("fuse", &MapFusionConfig::fuse)
      .def_readwrite("range_ratio", &MapFusionConfig::range_ratio)
      .def_readwrite("range_margin", &MapFusionConfig::range_margin)
      .def_readwrite("range_sigma", &MapFusionConfig::range_sigma)
      .def_readwrite("angle_sigma_deg", &MapFusionConfig::angle_sigma_deg)
      .def_readwrite("sigma_floor", &MapFusionConfig::sigma_floor);

  py::class_<MapAccumulator>(m, "MapAccumulator")
      .def(py::init([](double voxel, double max_range,
                       const MapFusionConfig& fusion) {
             Require(voxel > 0.0, "voxel must be positive");
             return MapAccumulator(voxel, max_range, fusion);
           }),
           "voxel"_a, "max_range"_a, "fusion"_a = MapFusionConfig())
      .def(
          "add_ranges",
          [](MapAccumulator& accumulator, const Mat4& pose,
             const py::object& sweep) {
            const Points points = PointsFrom(sweep);
            py::gil_scoped_release release;
            accumulator.AddRanges(Pose::FromMatrix(pose), points);
          },
          "T_world_lidar"_a, "sweep"_a,
          "Fusion: first pass over every sweep (nearest range per voxel).")
      .def(
          "add",
          [](MapAccumulator& accumulator, int keyframe, const Mat4& pose,
             const py::object& sweep, const py::object& intensity) {
            const Points points = PointsFrom(sweep);
            const std::vector<float> values = ToFloatVector(intensity);
            py::gil_scoped_release release;
            accumulator.Add(keyframe, Pose::FromMatrix(pose), points, values);
          },
          "keyframe"_a, "T_world_lidar"_a, "sweep"_a,
          "intensity"_a = py::none())
      .def("__len__", &MapAccumulator::size)
      .def(
          "result",
          [](const MapAccumulator& accumulator) {
            const MapAccumulator::Cloud cloud = accumulator.Collect();
            py::array_t<float> spread(
                {static_cast<py::ssize_t>(cloud.spread.size()),
                 static_cast<py::ssize_t>(kSymmetricTerms)});
            float* data = spread.mutable_data();
            for (size_t i = 0; i < cloud.spread.size(); ++i) {
              std::copy(cloud.spread[i].begin(), cloud.spread[i].end(),
                        data + kSymmetricTerms * i);
            }
            return py::dict("xyz"_a = ToMatrix(cloud.points),
                            "intensity"_a = ToArray(cloud.intensity),
                            "keyframe"_a = ToArray(cloud.keyframe),
                            "observations"_a = ToArray(cloud.observations),
                            "spread"_a = spread,
                            "model_sigma"_a = ToArray(cloud.model_sigma));
          },
          "Dict with xyz, intensity, keyframe (of the nearest observation), "
          "observations, spread (N, 6: xx xy xz yy yz zz) and model_sigma; "
          "nearest observations first.");

  m.def(
      "point_sigmas",
      [](const PointsIn& normals, const py::object& observations,
         const py::object& spread, const py::object& model_sigma,
         double sigma_floor) {
        MapAccumulator::Cloud cloud;
        const Points point_normals = ToPoints(normals);
        const std::vector<int> count = ToIntVector(observations);
        const Eigen::MatrixXd spread_rows = spread.cast<Eigen::MatrixXd>();
        const std::vector<double> model = ToDoubleVector(model_sigma);
        Require(count.size() == point_normals.size() &&
                    model.size() == point_normals.size() &&
                    static_cast<size_t>(spread_rows.rows()) ==
                        point_normals.size() &&
                    spread_rows.cols() == kSymmetricTerms,
                "point_sigmas: one normal, count, spread row (6) and model "
                "sigma per point");
        cloud.points.assign(point_normals.size(), Vec3::Zero());
        cloud.observations = count;
        cloud.spread.resize(point_normals.size());
        cloud.model_sigma.resize(point_normals.size());
        for (size_t i = 0; i < point_normals.size(); ++i) {
          for (int term = 0; term < kSymmetricTerms; ++term) {
            cloud.spread[i][static_cast<size_t>(term)] = static_cast<float>(
                spread_rows(static_cast<Eigen::Index>(i), term));
          }
          cloud.model_sigma[i] = static_cast<float>(model[i]);
        }
        return ToArray(PointSigmas(cloud, point_normals, sigma_floor));
      },
      "normals"_a, "observations"_a, "spread"_a, "model_sigma"_a,
      "sigma_floor"_a, "Sigma (m) of fused map points along their normals.");

  m.def(
      "map_normals",
      [](const PointsIn& xyz, const py::object& keyframe,
         const PointsIn& origins, double radius, int min_neighbors,
         int max_neighbors) {
        const Points points = ToPoints(xyz);
        const Points keyframe_origins = ToPoints(origins);
        const std::vector<int> owner = ToIntVector(keyframe);
        Require(owner.size() == points.size(),
                "map_normals: one keyframe index per point");
        for (int keyframe : owner) {
          RequireIndex(keyframe >= 0 &&
                           keyframe < static_cast<int>(keyframe_origins.size()),
                       "map_normals: keyframe index out of range");
        }
        Require(radius > 0.0, "radius must be positive");
        NormalsResult normals;
        {
          py::gil_scoped_release release;
          normals =
              EstimateOrientedNormals(points, owner, keyframe_origins, radius,
                                      min_neighbors, max_neighbors);
        }
        return py::make_tuple(ToMatrix(normals.normals),
                              ToArray(normals.planarity));
      },
      "xyz"_a, "keyframe"_a, "kf_origins"_a, "radius"_a, "min_neighbors"_a = 6,
      "max_neighbors"_a = 48,
      "Normals flipped towards the observing keyframe; returns (normals, "
      "planarity).");

  m.def(
      "estimate_beam_spacing",
      [](const std::vector<py::object>& sweeps) {
        std::vector<Points> points;
        for (const py::object& sweep : sweeps)
          points.push_back(PointsFrom(sweep));
        return EstimateBeamSpacing(points);
      },
      "sweeps"_a, "Median elevation gap between LiDAR beams (deg).");

  py::class_<DynamicConfig>(m, "DynamicConfig")
      .def(py::init<>())
      .def_readwrite("enabled", &DynamicConfig::enabled)
      .def_readwrite("ray_voxel", &DynamicConfig::ray_voxel)
      .def_readwrite("az_res", &DynamicConfig::az_res)
      .def_readwrite("el_res", &DynamicConfig::el_res)
      .def_readwrite("el_min", &DynamicConfig::el_min)
      .def_readwrite("el_max", &DynamicConfig::el_max)
      .def_readwrite("max_range", &DynamicConfig::max_range)
      .def_readwrite("margin_abs", &DynamicConfig::margin_abs)
      .def_readwrite("margin_rel", &DynamicConfig::margin_rel)
      .def_readwrite("min_cos", &DynamicConfig::min_cos)
      .def_readwrite("min_through", &DynamicConfig::min_through)
      .def_readwrite("min_ratio", &DynamicConfig::min_ratio)
      .def_readwrite("min_height", &DynamicConfig::min_height)
      .def_readwrite("ego_radius", &DynamicConfig::ego_radius)
      .def_readwrite("ego_height", &DynamicConfig::ego_height)
      .def_readwrite("bucket", &DynamicConfig::bucket);

  py::class_<DynamicDetector>(m, "DynamicDetector")
      .def(py::init([](const PointsIn& map, const PointsIn& normals,
                       const py::object& trip, const DynamicConfig& config) {
             const std::vector<int> trips = ToIntVector(trip);
             Require(normals.rows() == map.rows() &&
                         static_cast<Eigen::Index>(trips.size()) == map.rows(),
                     "map, normals and trip must have equal length");
             Require(config.az_res > 0.0 && config.el_max > config.el_min &&
                         config.bucket > 0.0,
                     "invalid DynamicConfig");
             return DynamicDetector(ToPoints(map), ToPoints(normals), trips,
                                    config);
           }),
           "map"_a, "normals"_a, "trip"_a, "config"_a)
      .def("set_elevation_resolution", &DynamicDetector::SetElevationResolution,
           "deg"_a)
      .def_property_readonly("elevation_resolution",
                             &DynamicDetector::elevation_resolution)
      .def(
          "set_heights",
          [](DynamicDetector& detector, const py::object& height) {
            const std::vector<double> heights = ToDoubleVector(height);
            Require(heights.size() == detector.num_points(),
                    "one height per map point expected");
            detector.SetHeights(heights);
          },
          "height"_a)
      .def(
          "add_sweep",
          [](DynamicDetector& detector, int trip, const Mat4& world_from_lidar,
             const py::object& sweep) {
            const Points points = PointsFrom(sweep);
            py::gil_scoped_release release;
            detector.AddSweep(trip, Pose::FromMatrix(world_from_lidar), points);
          },
          "trip"_a, "T_world_lidar"_a, "sweep"_a)
      .def(
          "add_ego_path",
          [](DynamicDetector& detector, int trip, const py::object& positions) {
            detector.AddEgoPath(trip, PointsFrom(positions));
          },
          "trip"_a, "vehicle_positions"_a)
      .def(
          "classify",
          [](const DynamicDetector& detector) {
            std::vector<DynamicKind> kind;
            {
              py::gil_scoped_release release;
              kind = detector.Classify();
            }
            std::vector<uint8_t> out(kind.size());
            for (size_t i = 0; i < kind.size(); ++i) {
              out[i] = static_cast<uint8_t>(kind[i]);
            }
            return ToArray(out);
          },
          "Per map point: 0 static, 1 moved within its session, 2 changed "
          "between sessions.")
      .def(
          "ego_swept",
          [](const DynamicDetector& detector, int trip) {
            return ToBoolArray(detector.EgoSwept(trip));
          },
          "trip"_a);
}

void BindTrajectory(py::module_& m) {
  py::class_<ContinuousTrajectory>(m, "ContinuousTrajectory")
      .def(py::init([](const std::vector<double>& stamps,
                       const py::object& poses, const py::object& imu,
                       const py::object& bias_stamps,
                       const py::object& gyro_biases, const py::object& wheel,
                       double table_step) {
             std::vector<Pose> knots = PosesFrom(poses);
             Require(!stamps.empty() && stamps.size() == knots.size(),
                     "one pose per stamp (at least one) expected");
             for (size_t k = 1; k < stamps.size(); ++k) {
               Require(stamps[k] > stamps[k - 1],
                       "stamps must be strictly increasing");
             }
             const std::vector<double> bias_t = ToDoubleVector(bias_stamps);
             const std::vector<Vec3> biases = Vec3sFrom(gyro_biases);
             Require(bias_t.size() == biases.size(),
                     "one gyro bias per bias stamp expected");
             for (size_t k = 1; k < bias_t.size(); ++k) {
               Require(bias_t[k] >= bias_t[k - 1],
                       "bias stamps must be ascending");
             }
             Require(table_step > 0.0, "table_step must be positive");
             const ImuSamples samples =
                 imu.is_none() ? ImuSamples() : imu.cast<ImuSamples>();
             const WheelSamples speeds =
                 wheel.is_none() ? WheelSamples() : wheel.cast<WheelSamples>();
             TrajectoryOptions options;
             options.table_step = table_step;
             return ContinuousTrajectory(stamps, std::move(knots), samples,
                                         bias_t, biases, speeds, options);
           }),
           "stamps"_a, "poses"_a, "imu"_a = py::none(),
           "bias_stamps"_a = py::none(), "gyro_biases"_a = py::none(),
           "wheel"_a = py::none(),
           "table_step"_a = TrajectoryOptions().table_step,
           "world_from_vehicle(t) through the pose knots (N, 4, 4); between "
           "knots the bias-corrected gyro and the wheel speed shape the "
           "curve (slerp / Hermite without them).")
      .def(
          "at",
          [](const ContinuousTrajectory& trajectory, double time) {
            return trajectory.At(time).Matrix();
          },
          "t"_a, "world_from_vehicle (4, 4) at t, clamped to the knots.")
      .def(
          "poses_at",
          [](const ContinuousTrajectory& trajectory, const py::object& times) {
            const std::vector<double> query_times = ToDoubleVector(times);
            std::vector<Pose> poses;
            {
              py::gil_scoped_release release;
              poses = trajectory.PosesAt(query_times);
            }
            return PosesToArray(poses);
          },
          "times"_a)
      .def_property_readonly("start_time", &ContinuousTrajectory::start_time)
      .def_property_readonly("end_time", &ContinuousTrajectory::end_time)
      .def_property_readonly("gyro_coverage",
                             &ContinuousTrajectory::gyro_coverage)
      .def_property_readonly("wheel_coverage",
                             &ContinuousTrajectory::wheel_coverage)
      .def_property_readonly("knot_stamps",
                             [](const ContinuousTrajectory& trajectory) {
                               return ToArray(trajectory.knot_stamps());
                             })
      .def_property_readonly("knot_poses",
                             [](const ContinuousTrajectory& trajectory) {
                               return PosesToArray(trajectory.knot_poses());
                             });

  m.def(
      "deskew_sweep",
      [](const PointsIn& points, const py::object& times, double stamp,
         const ContinuousTrajectory& trajectory, const Mat4& vehicle_from_lidar,
         int samples) {
        const Points cloud = ToPoints(points);
        const std::vector<double> point_times = ToDoubleVector(times);
        Require(point_times.size() == cloud.size(),
                "one time per point expected");
        Require(samples >= kMinDeskewSamples, "samples must be at least 2");
        Points out;
        {
          py::gil_scoped_release release;
          out = DeskewSweep(cloud, point_times, stamp, trajectory,
                            Pose::FromMatrix(vehicle_from_lidar), samples);
        }
        return ToMatrix(out);
      },
      "points"_a, "times"_a, "stamp"_a, "trajectory"_a, "T_vehicle_lidar"_a,
      "samples"_a = 32,
      "Points of a sweep in the LiDAR frame at `stamp` (its end), moved with "
      "the trajectory; times are relative to the stamp.");

  m.def(
      "deskew_keyframe_sweep",
      [](const PointsIn& points, const py::object& intensity,
         const py::object& times, double stamp,
         const ContinuousTrajectory& trajectory, const Mat4& vehicle_from_lidar,
         double min_range, double max_range, double voxel) {
        const Points cloud = ToPoints(points);
        const std::vector<float> point_intensity = ToFloatVector(intensity);
        const std::vector<double> point_times = ToDoubleVector(times);
        Require(point_times.empty() || point_times.size() == cloud.size(),
                "times must be empty or one per point");
        SweepSamples out;
        {
          py::gil_scoped_release release;
          out = DeskewKeyframeSweep(cloud, point_intensity, point_times, stamp,
                                    trajectory,
                                    Pose::FromMatrix(vehicle_from_lidar),
                                    min_range, max_range, voxel);
        }
        return py::make_tuple(ToMatrix(out.points), ToArray(out.intensity));
      },
      "points"_a, "intensity"_a, "times"_a, "stamp"_a, "trajectory"_a,
      "T_vehicle_lidar"_a, "min_range"_a, "max_range"_a, "voxel"_a,
      "Keyframe sweep for the map: deskewed with the trajectory, range-cropped "
      "and voxel-downsampled. Returns (points (M, 3), intensity (M,)).");
}

void BindCalibration(py::module_& m) {
  py::class_<CameraCalibrationConfig>(m, "CameraCalibrationConfig")
      .def(py::init<>())
      .def_readwrite("estimate_time_offset",
                     &CameraCalibrationConfig::estimate_time_offset)
      .def_readwrite("estimate_rotation",
                     &CameraCalibrationConfig::estimate_rotation)
      .def_readwrite("estimate_translation",
                     &CameraCalibrationConfig::estimate_translation)
      .def_readwrite("pixel_sigma", &CameraCalibrationConfig::pixel_sigma)
      .def_readwrite("huber", &CameraCalibrationConfig::huber)
      .def_readwrite("max_reprojection",
                     &CameraCalibrationConfig::max_reprojection)
      .def_readwrite("min_parallax_deg",
                     &CameraCalibrationConfig::min_parallax_deg)
      .def_readwrite("plane_sigma", &CameraCalibrationConfig::plane_sigma)
      .def_readwrite("plane_radius", &CameraCalibrationConfig::plane_radius)
      .def_readwrite("time_offset_prior",
                     &CameraCalibrationConfig::time_offset_prior)
      .def_readwrite("rotation_prior_deg",
                     &CameraCalibrationConfig::rotation_prior_deg)
      .def_readwrite("translation_prior",
                     &CameraCalibrationConfig::translation_prior)
      .def_readwrite("iterations", &CameraCalibrationConfig::iterations);

  m.def(
      "calibrate_camera",
      [](const CameraModel& camera, const Mat4& vehicle_from_camera,
         double time_offset, double line_time,
         const std::vector<const ContinuousTrajectory*>& trajectories,
         const py::object& image_trajectory, const py::object& image_stamps,
         const std::vector<Eigen::MatrixXd>& tracks, const py::object& surface,
         const py::object& normals, const CameraCalibrationConfig& config) {
        const std::vector<int> trajectory_index = ToIntVector(image_trajectory);
        const std::vector<double> stamps = ToDoubleVector(image_stamps);
        Require(trajectory_index.size() == stamps.size(),
                "one trajectory index per image stamp expected");
        std::vector<CalibrationImage> images(stamps.size());
        for (size_t image_index = 0; image_index < images.size();
             ++image_index) {
          const int trajectory = trajectory_index[image_index];
          RequireIndex(trajectory >= 0 &&
                           trajectory < static_cast<int>(trajectories.size()),
                       "image refers to an unknown trajectory");
          Require(trajectories[static_cast<size_t>(trajectory)] != nullptr &&
                      !trajectories[static_cast<size_t>(trajectory)]->empty(),
                  "trajectories must not be empty");
          images[image_index].trajectory =
              trajectories[static_cast<size_t>(trajectory)];
          images[image_index].stamp = stamps[image_index];
        }
        std::vector<Track> track_list;
        track_list.reserve(tracks.size());
        for (const Eigen::MatrixXd& rows : tracks) {
          Require(rows.rows() == 0 || rows.cols() == kTrackColumns,
                  "tracks must be (K, 3) rows of [image, u, v]");
          Track track;
          for (Eigen::Index row = 0; row < rows.rows(); ++row) {
            const int image = static_cast<int>(rows(row, kTrackImageColumn));
            RequireIndex(image >= 0 && image < static_cast<int>(images.size()),
                         "track refers to an unknown image");
            track.push_back({image, Vec2(rows(row, kTrackPixelColumn),
                                         rows(row, kTrackPixelColumn + 1))});
          }
          track_list.push_back(std::move(track));
        }
        const Points surface_points = PointsFrom(surface);
        const Points surface_normals = PointsFrom(normals);
        Require(surface_points.size() == surface_normals.size(),
                "one normal per surface point expected");
        Require(config.pixel_sigma > 0.0 && config.plane_sigma > 0.0 &&
                    config.huber > 0.0 && config.time_offset_prior > 0.0 &&
                    config.rotation_prior_deg > 0.0 &&
                    config.translation_prior > 0.0,
                "calibration sigmas must be positive");
        CameraCalibration initial;
        initial.vehicle_from_camera = Pose::FromMatrix(vehicle_from_camera);
        initial.time_offset = time_offset;
        initial.line_time = line_time;
        CameraCalibrationResult result;
        {
          py::gil_scoped_release release;
          const PointCloudIndex index(surface_points);
          const bool with_surface = !surface_points.empty();
          result = CalibrateCamera(camera, initial, images, track_list,
                                   with_surface ? &index : nullptr,
                                   with_surface ? &surface_normals : nullptr,
                                   config);
        }
        return py::dict(
            "T_vehicle_cam"_a = result.calibration.vehicle_from_camera.Matrix(),
            "time_offset"_a = result.calibration.time_offset,
            "rms_before"_a = result.rms_before,
            "rms_after"_a = result.rms_after, "landmarks"_a = result.landmarks,
            "observations"_a = result.observations,
            "iterations"_a = result.iterations,
            "time_offset_sigma"_a = result.time_offset_sigma,
            "rotation_sigma"_a = result.rotation_sigma,
            "translation_sigma"_a = result.translation_sigma);
      },
      "camera"_a, "T_vehicle_cam"_a, "time_offset"_a, "line_time"_a,
      "trajectories"_a, "image_trajectory"_a, "image_stamps"_a, "tracks"_a,
      "surface"_a = py::none(), "normals"_a = py::none(),
      "config"_a = CameraCalibrationConfig(),
      "Refines T_vehicle_cam and the time offset of one camera on the shared "
      "trajectories from feature tracks ((K, 3) rows [image, u, v]) and the "
      "LiDAR surface (points and normals).");

  m.def(
      "project_with_calibration",
      [](const CameraModel& camera, const Mat4& vehicle_from_camera,
         double time_offset, double line_time,
         const ContinuousTrajectory& trajectory, double stamp,
         const PointsIn& points) {
        Require(!trajectory.empty(), "trajectory must not be empty");
        CameraCalibration calibration;
        calibration.vehicle_from_camera = Pose::FromMatrix(vehicle_from_camera);
        calibration.time_offset = time_offset;
        calibration.line_time = line_time;
        const CalibrationImage image{&trajectory, stamp};
        const Points cloud = ToPoints(points);
        RowMatX2d uv = RowMatX2d::Constant(
            static_cast<Eigen::Index>(cloud.size()), kPlaneDim,
            std::numeric_limits<double>::quiet_NaN());
        {
          py::gil_scoped_release release;
          for (size_t i = 0; i < cloud.size(); ++i) {
            const std::optional<Vec2> pixel =
                ProjectWithCalibration(camera, calibration, image, cloud[i]);
            if (pixel)
              uv.row(static_cast<Eigen::Index>(i)) = pixel->transpose();
          }
        }
        return uv;
      },
      "camera"_a, "T_vehicle_cam"_a, "time_offset"_a, "line_time"_a,
      "trajectory"_a, "stamp"_a, "points"_a,
      "Pixels (N, 2) of world points in an image stamped `stamp`, with the "
      "rolling-shutter row solved per point (NaN where they do not "
      "project).");
}

py::array_t<double> DoubleArray(const std::vector<double>& values) {
  py::array_t<double> array(static_cast<py::ssize_t>(values.size()));
  std::copy(values.begin(), values.end(), array.mutable_data());
  return array;
}

void BindAcceptance(py::module_& m) {
  py::class_<CoarseAlignConfig>(m, "CoarseAlignConfig")
      .def(py::init<>())
      .def_readwrite("voxel", &CoarseAlignConfig::voxel)
      .def_readwrite("normal_radius", &CoarseAlignConfig::normal_radius)
      .def_readwrite("vertical_max_nz", &CoarseAlignConfig::vertical_max_nz)
      .def_readwrite("horizontal_min_nz", &CoarseAlignConfig::horizontal_min_nz)
      .def_readwrite("yaw_bin_deg", &CoarseAlignConfig::yaw_bin_deg)
      .def_readwrite("coarse_cell", &CoarseAlignConfig::coarse_cell)
      .def_readwrite("fine_cell", &CoarseAlignConfig::fine_cell)
      .def_readwrite("z_bin", &CoarseAlignConfig::z_bin);

  m.def(
      "align_gravity_aligned",
      [](const PointsIn& source, const PointsIn& reference,
         const CoarseAlignConfig& config) -> py::object {
        Require(config.coarse_cell > config.fine_cell && config.fine_cell > 0.0,
                "need coarse_cell > fine_cell > 0");
        const Points source_points = ToPoints(source);
        const Points reference_points = ToPoints(reference);
        std::optional<CoarseAlignment> result;
        {
          py::gil_scoped_release release;
          result = AlignGravityAligned(source_points, reference_points, config);
        }
        if (!result) return py::none();
        return py::dict("T"_a = result->reference_from_source.Matrix(),
                        "score"_a = result->score,
                        "ambiguity"_a = result->ambiguity);
      },
      "source"_a, "reference"_a, "config"_a = CoarseAlignConfig(),
      "Coarse registration of two gravity-aligned clouds (yaw, x, y, z). "
      "Returns {T: reference <- source (4, 4), score, ambiguity} or None.");

  m.def(
      "surface_distances",
      [](const PointsIn& query, const PointsIn& reference, double radius,
         double max_distance) {
        Require(radius > 0.0 && max_distance > 0.0,
                "radius and max_distance must be positive");
        const Points query_points = ToPoints(query);
        const Points reference_points = ToPoints(reference);
        SurfaceDistances result;
        {
          py::gil_scoped_release release;
          result = MeasureSurfaceDistance(query_points, reference_points,
                                          radius, max_distance);
        }
        return py::make_tuple(DoubleArray(result.signed_distance),
                              DoubleArray(result.nearest),
                              PointsToArray(result.normals));
      },
      "query"_a, "reference"_a, "radius"_a = 0.15, "max_distance"_a = 0.3,
      "(signed_distance (N,), nearest (N,), normals (N, 3)) of the query "
      "points to the reference surface: along the normal of the local "
      "reference plane (horizontal normals point up; NaN without a plane) "
      "and to the nearest reference point (NaN beyond max_distance).");

  m.def(
      "extract_planes",
      [](const KeyframeSet& keyframes, const py::object& poses,
         const LidarBaConfig& config) {
        const std::vector<Pose> keyframe_poses = PosesFrom(poses);
        Require(keyframe_poses.size() == keyframes.size(),
                "one pose per keyframe");
        Require(keyframes.HasVoxel(config.sweep_voxel),
                "keyframes must keep sweeps at config.sweep_voxel");
        std::vector<BaPlane> planes;
        {
          py::gil_scoped_release release;
          planes = ExtractPlanes(keyframes, keyframe_poses, config);
        }
        Points centroids;
        Points normals;
        std::vector<double> rms;
        std::vector<double> size;
        std::vector<double> nodes;
        std::vector<double> offsets = {0.0};
        for (const BaPlane& plane : planes) {
          centroids.push_back(plane.centroid);
          normals.push_back(plane.normal);
          rms.push_back(plane.rms);
          size.push_back(plane.size);
          for (const PlaneObservation& observation : plane.observations) {
            nodes.push_back(observation.node);
          }
          offsets.push_back(static_cast<double>(nodes.size()));
        }
        return py::dict(
            "centroid"_a = PointsToArray(centroids),
            "normal"_a = PointsToArray(normals), "rms"_a = DoubleArray(rms),
            "size"_a = DoubleArray(size), "keyframes"_a = DoubleArray(nodes),
            "offsets"_a = DoubleArray(offsets));
      },
      "keyframes"_a, "poses"_a, "config"_a,
      "Planes of the bundle adjustment at the given keyframe poses: centroid, "
      "normal, rms, voxel size, and the observing keyframes of plane i in "
      "keyframes[offsets[i]:offsets[i + 1]].");
}

}  // namespace

void BindRecon(py::module_& m) {
  BindImu(m);
  BindOdometryPipeline(m);
  BindKeyframes(m);
  BindLoops(m);
  BindBackend(m);
  BindTrajectory(m);
  BindCalibration(m);
  BindMap(m);
  BindAcceptance(m);

  py::class_<LidarImuConfig>(m, "LidarImuConfig")
      .def(py::init<>())
      .def_readwrite("half_window", &LidarImuConfig::half_window)
      .def_readwrite("stride", &LidarImuConfig::stride)
      .def_readwrite("estimate_scale", &LidarImuConfig::estimate_scale)
      .def_readwrite("estimate_time_offset",
                     &LidarImuConfig::estimate_time_offset)
      .def_readwrite("max_time_offset", &LidarImuConfig::max_time_offset)
      .def_readwrite("estimate_lever_arm", &LidarImuConfig::estimate_lever_arm)
      .def_readwrite("lever_arm_sigma", &LidarImuConfig::lever_arm_sigma)
      .def_readwrite("gyro_sigma", &LidarImuConfig::gyro_sigma)
      .def_readwrite("accel_sigma", &LidarImuConfig::accel_sigma)
      .def_readwrite("huber", &LidarImuConfig::huber)
      .def_readwrite("max_iterations", &LidarImuConfig::max_iterations)
      .def_readwrite("gravity", &LidarImuConfig::gravity);

  m.def(
      "calibrate_lidar_imu",
      [](const std::vector<double>& stamps, const py::object& poses,
         const ImuSamples& imu, const Mat4& initial,
         const LidarImuConfig& config) {
        const std::vector<Pose> lidar_poses = PosesFrom(poses);
        Require(lidar_poses.size() == stamps.size() &&
                    stamps.size() >= kMinLidarImuSamples,
                "need 3+ poses, one per stamp");
        LidarImuResult result;
        {
          py::gil_scoped_release release;
          result = CalibrateLidarImu(stamps, lidar_poses, imu,
                                     Pose::FromMatrix(initial), config);
        }
        return py::dict(
            "converged"_a = result.converged, "samples"_a = result.samples,
            "R_imu_lidar"_a = result.imu_from_lidar,
            "gyro_bias"_a = result.gyro_bias,
            "accel_bias"_a = result.accel_bias,
            "accel_scale"_a = result.accel_scale,
            "gravity_world"_a = result.gravity_world,
            "time_offset"_a = result.time_offset,
            "lever_arm"_a = result.lever_arm,
            "sigma_lever_arm"_a = result.sigma_lever_arm,
            "sigma_time_offset"_a = result.sigma_time_offset,
            "sigma_rotation_deg"_a = result.sigma_rotation_deg,
            "sigma_gyro_bias"_a = result.sigma_gyro_bias,
            "sigma_accel_bias"_a = result.sigma_accel_bias,
            "sigma_accel_scale"_a = result.sigma_accel_scale,
            "gyro_rms"_a = result.gyro_rms, "accel_rms"_a = result.accel_rms);
      },
      "stamps"_a, "T_world_lidar"_a, "imu"_a, "T_imu_lidar"_a,
      "config"_a = LidarImuConfig(),
      "LiDAR-IMU rotation, biases and accelerometer scales from a trip "
      "(gr/calibration/lidar_imu.h). `imu` in the IMU's own axes.");
}

}  // namespace gr::bindings
