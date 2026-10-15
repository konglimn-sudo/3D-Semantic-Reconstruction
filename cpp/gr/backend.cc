#include "gr/backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include "gr/check.h"
#include "gr/numeric.h"
#include "gr/pose_graph.h"
#include "gr/registration.h"

namespace gr {
namespace {

// LiDAR pose of a trip's odometry at time `t`: interpolated between the
// frames; before the first / after the last frame extended with the wheel /
// gyro prior (nullopt if the prior does not reach `t`).
std::optional<Pose> OdometryAt(const TripInputs& inputs,
                               const MotionPrior& prior, double t) {
  const std::vector<double>& stamps = inputs.frame_stamps;
  if (t >= stamps.front() && t <= stamps.back()) {
    return InterpolatePose(stamps, inputs.frame_odometry, t);
  }
  const Pose lidar_from_vehicle = inputs.vehicle_from_lidar.Inverse();
  if (t < stamps.front()) {
    // vehicle(t)^-1 vehicle(t_0); the LiDAR at t seen from the LiDAR at t_0.
    const std::optional<Pose> delta = prior.Delta(t, stamps.front());
    if (!delta) return std::nullopt;
    return inputs.frame_odometry.front() * lidar_from_vehicle *
           delta->Inverse() * inputs.vehicle_from_lidar;
  }
  const std::optional<Pose> delta = prior.Delta(stamps.back(), t);
  if (!delta) return std::nullopt;
  return inputs.frame_odometry.back() * lidar_from_vehicle * *delta *
         inputs.vehicle_from_lidar;
}

// Position anchors from the GNSS fixes of one trip.
void AddGnssAnchors(const KeyframeSet& keyframes, const std::vector<int>& trip,
                    const TripInputs& inputs, const Vec3& lever,
                    const BackendConfig& config, std::vector<Anchor>& anchors) {
  const Eigen::MatrixXd& gnss = inputs.gnss;
  if (gnss.rows() == 0 || gnss.cols() < 4 || inputs.frame_stamps.empty()) {
    return;
  }
  const MotionPrior prior(inputs.imu, inputs.wheel);
  for (Eigen::Index r = 0; r < gnss.rows(); ++r) {
    const double fix_time = gnss(r, 0);
    const Vec3 position(gnss(r, 1), gnss(r, 2), gnss(r, 3));
    const double std_m = gnss.cols() > 4 ? gnss(r, 4) : 0.05;
    const auto time_gap = [&](int k) {
      return std::abs(keyframes[static_cast<size_t>(k)].stamp - fix_time);
    };
    size_t best = 0;
    for (size_t a = 1; a < trip.size(); ++a) {
      if (time_gap(trip[a]) < time_gap(trip[best])) best = a;
    }
    if (time_gap(trip[best]) > config.gnss_max_dt) continue;
    const Keyframe& keyframe = keyframes[static_cast<size_t>(trip[best])];
    // Move the fix to the keyframe time with the odometry increment. A fix
    // before the first LiDAR frame (the drive starts with GNSS) needs the
    // wheel / gyro prior; without it the fix is dropped rather than misplaced.
    const std::optional<Pose> at_fix = OdometryAt(inputs, prior, fix_time);
    if (!at_fix) continue;
    Anchor anchor;
    anchor.type = Anchor::Type::kPosition;
    anchor.keyframe = trip[best];
    anchor.position = position + (keyframe.odom * lever - *at_fix * lever);
    anchor.lever = lever;
    anchor.sigma = std::max(std_m, 0.01) * config.gnss_sigma_scale;
    anchors.push_back(anchor);
  }
}

// Start velocity of a keyframe without an estimate: the wheel speed along the
// vehicle x axis, else the finite difference of neighbouring keyframes.
Vec3 InitialVelocity(const KeyframeSet& keyframes, const std::vector<int>& trip,
                     size_t a, const std::vector<Pose>& poses,
                     const TripInputs& inputs, const Pose& lidar_from_vehicle) {
  const int k = trip[a];
  const Pose vehicle = poses[static_cast<size_t>(k)] * lidar_from_vehicle;
  const double stamp = keyframes[static_cast<size_t>(k)].stamp;
  const WheelSamples& wheel = inputs.wheel;
  if (wheel.size() > 1 && stamp >= wheel.stamps.front() &&
      stamp <= wheel.stamps.back()) {
    return vehicle.rotation *
           Vec3(Interp1(wheel.stamps, wheel.speed, stamp), 0.0, 0.0);
  }
  if (trip.size() < 2) return Vec3::Zero();
  const size_t a0 = a == 0 ? 0 : a - 1;
  const size_t a1 = a + 1 == trip.size() ? a : a + 1;
  const size_t k0 = static_cast<size_t>(trip[a0]);
  const size_t k1 = static_cast<size_t>(trip[a1]);
  const double dt = keyframes[k1].stamp - keyframes[k0].stamp;
  if (dt <= 0.0) return Vec3::Zero();
  return ((poses[k1] * lidar_from_vehicle).translation -
          (poses[k0] * lidar_from_vehicle).translation) /
         dt;
}

// Inertial states for the keyframes inside each trip's IMU span, with IMU
// preintegration, wheel velocity and bias random-walk factors. Returns the
// number of IMU factors.
int AddInertialFactors(const KeyframeSet& keyframes,
                       const std::vector<TripInputs>& trips,
                       const BackendConfig& config,
                       const std::vector<Pose>& poses,
                       const InertialEstimate* start, PoseGraph& graph) {
  const ImuNoise noise = config.ToImuNoise();
  const bool has_start = start != nullptr &&
                         start->valid.size() == keyframes.size() &&
                         start->velocities.size() == keyframes.size() &&
                         start->biases.size() == keyframes.size();
  int num_imu = 0;
  for (size_t t = 0; t < trips.size(); ++t) {
    const TripInputs& inputs = trips[t];
    const std::vector<int>& trip = keyframes.TripIndices(static_cast<int>(t));
    const ImuSamples& imu = inputs.imu;
    if (trip.empty() || imu.size() < 2) continue;
    const Pose lidar_from_vehicle = inputs.vehicle_from_lidar.Inverse();
    const WheelSamples& wheel = inputs.wheel;
    std::vector<int> nodes;
    for (size_t a = 0; a < trip.size(); ++a) {
      const int k = trip[a];
      const size_t kk = static_cast<size_t>(k);
      const double stamp = keyframes[kk].stamp;
      if (stamp < imu.stamps.front() || stamp > imu.stamps.back()) continue;
      if (has_start && start->valid[kk]) {
        graph.AddInertialState(k, start->velocities[kk], start->biases[kk]);
      } else {
        graph.AddInertialState(k,
                               InitialVelocity(keyframes, trip, a, poses,
                                               inputs, lidar_from_vehicle),
                               Vec6::Zero());
      }
      nodes.push_back(k);
      if (config.wheel_factors && wheel.size() > 1 &&
          stamp >= wheel.stamps.front() && stamp <= wheel.stamps.back()) {
        graph.AddBodyVelocity(
            k, Vec3(Interp1(wheel.stamps, wheel.speed, stamp), 0.0, 0.0),
            Vec3(config.wheel_sigma, config.wheel_lateral_sigma,
                 config.wheel_lateral_sigma),
            lidar_from_vehicle);
      }
    }
    if (nodes.empty()) continue;
    Vec6 prior_sigma;
    prior_sigma << Vec3::Constant(config.gyro_bias_prior),
        Vec3::Constant(config.accel_bias_prior);
    graph.AddBiasPrior(nodes.front(), Vec6::Zero(), prior_sigma);
    for (size_t a = 1; a < nodes.size(); ++a) {
      const int i = nodes[a - 1];
      const int j = nodes[a];
      const double t0 = keyframes[static_cast<size_t>(i)].stamp;
      const double t1 = keyframes[static_cast<size_t>(j)].stamp;
      const double root_dt = std::sqrt(std::max(t1 - t0, 1e-3));
      Vec6 walk;
      walk << Vec3::Constant(noise.gyro_bias_walk * root_dt),
          Vec3::Constant(noise.accel_bias_walk * root_dt);
      graph.AddBiasBetween(i, j, walk);
      if (t1 - t0 > config.imu_max_dt) continue;
      const std::optional<ImuPreintegration> preintegration =
          PreintegrateImu(imu, t0, t1, graph.Bias(i), noise);
      if (!preintegration) continue;
      graph.AddImu(i, j, *preintegration, lidar_from_vehicle);
      ++num_imu;
    }
  }
  return num_imu;
}

}  // namespace

ImuNoise BackendConfig::ToImuNoise() const {
  ImuNoise noise;
  noise.gyro_noise = imu_gyro_noise;
  noise.accel_noise = imu_accel_noise;
  noise.gyro_bias_walk = imu_gyro_bias_walk;
  noise.accel_bias_walk = imu_accel_bias_walk;
  return noise;
}

GravityConfig BackendConfig::ToGravityConfig() const {
  GravityConfig config;
  config.window = gravity_window;
  config.max_dev = gravity_max_dev;
  config.max_yaw_rate = gravity_max_yaw_rate;
  config.sigma_deg = gravity_sigma_deg;
  return config;
}

std::vector<Anchor> MakeAnchors(const KeyframeSet& keyframes,
                                const std::vector<TripInputs>& trips,
                                const BackendConfig& config,
                                const Logger& log) {
  std::vector<Anchor> anchors;
  for (size_t t = 0; t < trips.size(); ++t) {
    const std::vector<int>& trip = keyframes.TripIndices(static_cast<int>(t));
    if (trip.empty()) continue;
    const TripInputs& inputs = trips[t];
    // The vehicle origin in the LiDAR frame.
    const Vec3 lever = inputs.vehicle_from_lidar.Inverse().translation;
    const size_t before = anchors.size();
    if (inputs.initial_pose) {
      Anchor anchor;
      anchor.type = Anchor::Type::kPose;
      anchor.keyframe = trip.front();
      anchor.pose = *inputs.initial_pose * inputs.vehicle_from_lidar;
      anchor.sigma_t = config.anchor_sigma_t;
      anchor.sigma_r = config.anchor_sigma_r;
      anchors.push_back(anchor);
    }
    AddGnssAnchors(keyframes, trip, inputs, lever, config, anchors);
    if (anchors.size() == before) {
      Anchor anchor;
      anchor.type = Anchor::Type::kPose;
      anchor.keyframe = trip.front();
      anchor.pose = keyframes[static_cast<size_t>(trip.front())].odom;
      anchor.sigma_t = 50.0;
      anchor.sigma_r = 90.0;
      anchors.push_back(anchor);
      LogLine(log, StringPrintf("[backend] trip %zu: no entrance anchor, "
                                "relying on inter-trip loops",
                                t));
    }
  }
  return anchors;
}

std::vector<GravityMeasurement> MeasureKeyframeGravity(
    const KeyframeSet& keyframes, const std::vector<TripInputs>& trips,
    const BackendConfig& config) {
  std::vector<GravityMeasurement> out;
  for (size_t t = 0; t < trips.size(); ++t) {
    const std::vector<int>& trip = keyframes.TripIndices(static_cast<int>(t));
    if (trip.empty()) continue;
    std::vector<double> stamps;
    stamps.reserve(trip.size());
    for (int k : trip)
      stamps.push_back(keyframes[static_cast<size_t>(k)].stamp);
    const Mat3 lidar_from_vehicle_rotation =
        trips[t].vehicle_from_lidar.Inverse().rotation;
    for (GravityMeasurement g : MeasureGravity(
             trips[t].imu, trips[t].wheel, lidar_from_vehicle_rotation, stamps,
             config.ToGravityConfig())) {
      g.index = trip[static_cast<size_t>(g.index)];
      out.push_back(g);
    }
  }
  return out;
}

namespace {

// The graph of OptimizeGraph before optimisation; reports how many odometry
// and IMU factors it holds.
PoseGraph BuildGraph(
    const KeyframeSet& keyframes, const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity, const BackendConfig& config,
    const std::vector<Pose>* initial_poses,
    const std::vector<UnaryPrior>* unary, const std::vector<TripInputs>* trips,
    const InertialEstimate* inertial, size_t& num_odometry, int& num_imu) {
  const size_t num_keyframes = keyframes.size();
  GR_CHECK(initial_poses == nullptr || initial_poses->size() == num_keyframes);
  PoseGraph graph;
  for (size_t k = 0; k < num_keyframes; ++k) {
    graph.AddNode(initial_poses != nullptr ? (*initial_poses)[k]
                                           : keyframes[k].odom);
  }
  num_imu = 0;
  if (config.imu_factors && trips != nullptr) {
    num_imu = AddInertialFactors(keyframes, *trips, config, graph.poses(),
                                 inertial, graph);
  }
  num_odometry = 0;
  for (int t = 0; t < keyframes.num_trips(); ++t) {
    const std::vector<int>& trip = keyframes.TripIndices(t);
    for (size_t a = 1; a < trip.size(); ++a) {
      const int i = trip[a - 1];
      const int j = trip[a];
      const Pose step = keyframes[static_cast<size_t>(i)].odom.Inverse() *
                        keyframes[static_cast<size_t>(j)].odom;
      const double d = step.translation.norm();
      graph.AddBetween(
          i, j, step,
          PoseInformation(config.odom_sigma_t + config.odom_sigma_t_per_m * d,
                          config.odom_sigma_r + config.odom_sigma_r_per_m * d));
      ++num_odometry;
    }
  }
  for (const LoopConstraint& loop : loops) {
    graph.AddBetween(loop.j, loop.i, loop.j_from_i,
                     PoseInformation(config.loop_sigma_t, config.loop_sigma_r),
                     config.loop_robust_scale);
  }
  for (const Anchor& anchor : anchors) {
    if (anchor.type == Anchor::Type::kPose) {
      graph.AddPrior(anchor.keyframe, anchor.pose,
                     PoseInformation(anchor.sigma_t, anchor.sigma_r));
    } else {
      graph.AddPosition(anchor.keyframe, anchor.position,
                        Mat3::Identity() / (anchor.sigma * anchor.sigma),
                        anchor.lever);
    }
  }
  if (unary != nullptr) {
    for (const UnaryPrior& prior : *unary) {
      graph.AddPrior(prior.keyframe, prior.pose, prior.information);
    }
  }
  for (const GravityMeasurement& g : gravity) {
    graph.AddGravity(g.index, g.up, g.sigma_rad);
  }
  return graph;
}

// Copies the inertial states and the summary of an optimised graph.
void CollectGraph(const PoseGraph& graph, const PoseGraphSummary& result,
                  size_t num_odometry, size_t num_loops, int num_imu,
                  GraphSummary* summary, InertialEstimate* inertial) {
  const size_t num_keyframes = graph.num_nodes();
  if (inertial != nullptr) {
    inertial->velocities.assign(num_keyframes, Vec3::Zero());
    inertial->biases.assign(num_keyframes, Vec6::Zero());
    inertial->valid.assign(num_keyframes, 0);
    for (size_t k = 0; k < num_keyframes; ++k) {
      const int node = static_cast<int>(k);
      if (!graph.HasInertialState(node)) continue;
      inertial->velocities[k] = graph.Velocity(node);
      inertial->biases[k] = graph.Bias(node);
      inertial->valid[k] = 1;
    }
  }
  if (summary == nullptr) return;
  summary->imu_factors = num_imu;
  const std::vector<double> imu_chi2 = graph.ImuChi2();
  summary->imu_chi2_median = imu_chi2.empty() ? 0.0 : Median(imu_chi2);
  summary->iterations = result.iterations;
  summary->initial_cost = result.initial_cost;
  summary->final_cost = result.final_cost;
  summary->converged = result.converged;
  const std::vector<double> chi2 = graph.BetweenChi2();
  const auto first = chi2.begin() + static_cast<int64_t>(num_odometry);
  const std::vector<double> loop_chi2(first,
                                      first + static_cast<int64_t>(num_loops));
  summary->loop_chi2_median = loop_chi2.empty() ? 0.0 : Median(loop_chi2);
  summary->loop_chi2_max =
      loop_chi2.empty() ? 0.0
                        : *std::max_element(loop_chi2.begin(), loop_chi2.end());
}

}  // namespace

std::vector<Pose> OptimizeGraph(
    const KeyframeSet& keyframes, const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity, const BackendConfig& config,
    const std::vector<Pose>* initial_poses,
    const std::vector<UnaryPrior>* unary, GraphSummary* summary,
    const std::vector<TripInputs>* trips, InertialEstimate* inertial) {
  size_t num_odometry = 0;
  int num_imu = 0;
  PoseGraph graph =
      BuildGraph(keyframes, loops, anchors, gravity, config, initial_poses,
                 unary, trips, inertial, num_odometry, num_imu);
  PoseGraphOptions options;
  options.max_iterations = config.iterations;
  const PoseGraphSummary result = graph.Optimize(options);
  CollectGraph(graph, result, num_odometry, loops.size(), num_imu, summary,
               inertial);
  return graph.poses();
}

std::vector<Pose> RunLidarBa(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity,
    const BackendConfig& backend_config, const LidarBaConfig& ba_config,
    GraphSummary* summary, LidarBaSummary* ba_summary, const Logger& log,
    const std::vector<TripInputs>* trips, InertialEstimate* inertial) {
  GR_CHECK(poses.size() == keyframes.size());
  std::vector<Pose> current = poses;
  for (int round = 0; round < ba_config.rounds; ++round) {
    const std::vector<BaPlane> planes =
        ExtractPlanes(keyframes, current, ba_config);
    size_t num_odometry = 0;
    int num_imu = 0;
    PoseGraph graph =
        BuildGraph(keyframes, loops, anchors, gravity, backend_config, &current,
                   nullptr, trips, inertial, num_odometry, num_imu);
    LidarBaRound stats;
    for (const BaPlane& plane : planes) {
      graph.AddPlane(plane.observations, ba_config.point_sigma,
                     ba_config.robust_scale);
      stats.observations += static_cast<int>(plane.observations.size());
    }
    stats.planes = static_cast<int>(planes.size());
    const std::vector<double> rms_before = graph.PlaneRms();
    PoseGraphOptions options;
    options.max_iterations = ba_config.iterations;
    const PoseGraphSummary result = graph.Optimize(options);
    const std::vector<double> rms_after = graph.PlaneRms();
    if (!planes.empty()) {
      stats.rms_median_before = Median(rms_before);
      stats.rms_p90_before = Percentile(rms_before, 90.0);
      stats.rms_median_after = Median(rms_after);
      stats.rms_p90_after = Percentile(rms_after, 90.0);
    }
    stats.iterations = result.iterations;
    std::vector<double> shifts;
    for (size_t k = 0; k < current.size(); ++k) {
      shifts.push_back(
          (graph.poses()[k].translation - current[k].translation).norm());
    }
    if (!shifts.empty()) {
      stats.shift_median = Median(shifts);
      stats.shift_max = *std::max_element(shifts.begin(), shifts.end());
    }
    CollectGraph(graph, result, num_odometry, loops.size(), num_imu, summary,
                 inertial);
    current = graph.poses();
    if (ba_summary != nullptr) ba_summary->rounds.push_back(stats);
    LogLine(log,
            StringPrintf(
                "[ba] round %d: %d planes (%d scan clusters), plane "
                "RMS median %.2f -> %.2f cm, p90 %.2f -> %.2f cm, "
                "keyframe shift median %.2f cm, max %.2f cm, %d it",
                round + 1, stats.planes, stats.observations,
                100.0 * stats.rms_median_before, 100.0 * stats.rms_median_after,
                100.0 * stats.rms_p90_before, 100.0 * stats.rms_p90_after,
                100.0 * stats.shift_median, 100.0 * stats.shift_max,
                stats.iterations));
  }
  return current;
}

std::vector<Pose> RefineScanToMap(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity,
    const BackendConfig& backend_config, const RefineConfig& refine_config,
    GraphSummary* summary, const Logger& log,
    const std::vector<TripInputs>* trips, InertialEstimate* inertial) {
  std::vector<Pose> refined = poses;
  const int64_t num_keyframes = static_cast<int64_t>(keyframes.size());
  const double voxel = refine_config.voxel;
  const Mat6 information =
      PoseInformation(refine_config.sigma_t, refine_config.sigma_r);
  for (int iteration = 0; iteration < refine_config.iterations; ++iteration) {
    VoxelHashMap map(voxel, 1e9, 60);
    for (int64_t k = 0; k < num_keyframes; ++k) {
      const size_t kk = static_cast<size_t>(k);
      map.AddPoints(TransformPoints(keyframes.Cloud(kk, voxel), refined[kk]),
                    static_cast<int>(k));
    }
    std::vector<IcpResult> alignments(static_cast<size_t>(num_keyframes));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int64_t k = 0; k < num_keyframes; ++k) {
      const size_t kk = static_cast<size_t>(k);
      IcpOptions options;
      options.max_corr_dist = refine_config.max_corr;
      options.kernel = refine_config.max_corr / 3.0;
      options.max_iters = refine_config.icp_iters;
      options.exclude_frame = static_cast<int>(k);  // Other keyframes only.
      options.parallel = false;
      alignments[kk] =
          AlignToMap(keyframes.Cloud(kk, voxel), map, refined[kk], options);
    }
    std::vector<UnaryPrior> unary;
    std::vector<double> shifts;
    for (size_t k = 0; k < alignments.size(); ++k) {
      const IcpResult& alignment = alignments[k];
      if (alignment.fitness < refine_config.min_fitness) continue;
      UnaryPrior prior;
      prior.keyframe = static_cast<int>(k);
      prior.pose = alignment.pose;
      prior.information = information;
      unary.push_back(prior);
      shifts.push_back(
          (alignment.pose.translation - refined[k].translation).norm());
    }
    refined = OptimizeGraph(keyframes, loops, anchors, gravity, backend_config,
                            &refined, &unary, summary, trips, inertial);
    LogLine(log, StringPrintf("[refine] iteration %d: median scan-to-map shift "
                              "%.1f cm, %zu/%zu keyframes aligned",
                              iteration + 1, Median(shifts) * 100.0,
                              unary.size(), keyframes.size()));
  }
  return refined;
}

GraphStageResult RunGraphStage(const KeyframeSet& keyframes,
                               const std::vector<TripInputs>& trips,
                               const GraphStageConfig& config,
                               const Logger& log) {
  for (double voxel : config.RequiredVoxels()) {
    GR_CHECK(keyframes.HasVoxel(voxel));
  }
  GraphStageResult result;
  const auto log_cost = [&](const GraphSummary& s, size_t num_loops) {
    LogLine(log, StringPrintf("[backend] cost %.1f -> %.1f in %d it, %zu loops",
                              s.initial_cost, s.final_cost, s.iterations,
                              num_loops));
  };
  result.anchors = MakeAnchors(keyframes, trips, config.backend, log);
  if (config.backend.gravity) {
    result.gravity = MeasureKeyframeGravity(keyframes, trips, config.backend);
  }
  if (config.backend.gravity) {
    LogLine(log, StringPrintf("[backend] gravity (roll/pitch) factors on "
                              "%zu/%zu keyframes",
                              result.gravity.size(), keyframes.size()));
  }

  const std::vector<TripInputs>* imu_trips =
      config.backend.imu_factors ? &trips : nullptr;
  GraphSummary initial_summary;
  std::vector<Pose> poses = OptimizeGraph(
      keyframes, {}, result.anchors, result.gravity, config.backend, nullptr,
      nullptr, &initial_summary, imu_trips, &result.inertial);
  log_cost(initial_summary, 0);
  if (imu_trips != nullptr) {
    LogLine(log, StringPrintf("[backend] IMU factors between %d keyframe "
                              "pairs, median chi2 %.2f",
                              initial_summary.imu_factors,
                              initial_summary.imu_chi2_median));
  }

  const std::vector<LoopConstraint> verified =
      DetectLoops(keyframes, poses, config.loops, &result.loop_stats);
  result.loops_verified = static_cast<int>(verified.size());
  LogLine(log, StringPrintf("[loops] %d candidates verified by ICP, %zu "
                            "accepted",
                            result.loop_stats.tested, verified.size()));
  result.loops =
      SelectConsistentLoops(verified, keyframes, poses, config.loops);
  LogLine(log, StringPrintf("[loops] PCM kept %zu/%zu", result.loops.size(),
                            verified.size()));

  poses = OptimizeGraph(keyframes, result.loops, result.anchors, result.gravity,
                        config.backend, &poses, nullptr, &result.summary,
                        imu_trips, &result.inertial);
  log_cost(result.summary, result.loops.size());
  if (config.ba.enabled) {
    poses =
        RunLidarBa(keyframes, poses, result.loops, result.anchors,
                   result.gravity, config.backend, config.ba, &result.summary,
                   &result.ba, log, imu_trips, &result.inertial);
  } else if (config.refine.enabled) {
    poses = RefineScanToMap(keyframes, poses, result.loops, result.anchors,
                            result.gravity, config.backend, config.refine,
                            &result.summary, log, imu_trips, &result.inertial);
  }
  result.poses = std::move(poses);
  return result;
}

std::vector<Pose> CorrectFramePoses(const std::vector<double>& kf_stamps,
                                    const std::vector<Pose>& kf_odom,
                                    const std::vector<Pose>& kf_poses,
                                    const std::vector<double>& frame_stamps,
                                    const std::vector<Pose>& frame_odom) {
  GR_CHECK(kf_stamps.size() == kf_odom.size() &&
           kf_stamps.size() == kf_poses.size());
  GR_CHECK(frame_stamps.size() == frame_odom.size());
  if (kf_stamps.empty()) return frame_odom;
  // World-frame corrections at the keyframes.
  std::vector<Pose> corrections(kf_stamps.size());
  for (size_t a = 0; a < corrections.size(); ++a) {
    corrections[a] = kf_poses[a] * kf_odom[a].Inverse();
  }
  std::vector<Pose> out(frame_stamps.size());
  for (size_t f = 0; f < out.size(); ++f) {
    out[f] = InterpolatePose(kf_stamps, corrections, frame_stamps[f]) *
             frame_odom[f];
  }
  return out;
}

}  // namespace gr
