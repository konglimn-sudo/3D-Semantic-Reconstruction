#include "gr/backend/backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>

#include "gr/backend/pose_graph.h"
#include "gr/core/check.h"
#include "gr/core/numeric.h"
#include "gr/core/voxel.h"
#include "gr/registration/cloud_compare.h"
#include "gr/registration/registration.h"

namespace gr {
namespace {

// GNSS fixes without a std column (TripInputs) have kDefaultGnssStd; no fix
// is trusted beyond kMinGnssStd (m).
constexpr double kDefaultGnssStd = 0.05;
constexpr double kMinGnssStd = 0.01;
// The same defaults of the control points (m).
constexpr double kDefaultControlStd = 0.01;
constexpr double kMinControlStd = 1e-3;
// Interpolation and finite differences need two samples.
constexpr size_t kMinSamples = 2;
// Bias random walks span at least this long (s).
constexpr double kMinBiasWalkInterval = 1e-3;
// The prior of a trip without any anchor: anywhere near its odometry start.
constexpr double kWeakAnchorSigmaT = 50.0;  // m
constexpr double kWeakAnchorSigmaR = 90.0;  // deg
// Hierarchical BA: windows hold kMinWindowSize keyframes at least, and only
// windows of kMinRefinedWindowSize get their own BA (a shorter trip tail
// joins the window before it). Their first keyframe is held by a stiff prior.
constexpr int kMinWindowSize = 2;
constexpr size_t kMinRefinedWindowSize = 3;
constexpr double kHoldSigmaT = 1e-4;  // m
constexpr double kHoldSigmaR = 1e-3;  // deg
// A window_of entry before the windows are cut.
constexpr int kNoWindow = -1;
// Scan-to-map refinement: points per map voxel, and the ICP kernel as a
// fraction of the correspondence gate.
constexpr int kRefineMaxPointsPerVoxel = 60;
constexpr double kGatePerKernel = 3.0;
// Upper percentile of the BA plane RMS reports.
constexpr double kReportPercentile = 90.0;

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
                    const BackendConfig& config, std::vector<Anchor>* anchors) {
  const Eigen::MatrixXd& gnss = inputs.gnss;
  if (gnss.rows() == 0 || gnss.cols() < kGnssStdColumn ||
      inputs.frame_stamps.empty()) {
    return;
  }
  const MotionPrior prior(inputs.imu, inputs.wheel);
  for (Eigen::Index row = 0; row < gnss.rows(); ++row) {
    const double fix_time = gnss(row, kGnssTimeColumn);
    const Vec3 position =
        gnss.block<1, kSpaceDim>(row, kGnssPositionColumn).transpose();
    const double std_dev = gnss.cols() > kGnssStdColumn
                               ? gnss(row, kGnssStdColumn)
                               : kDefaultGnssStd;
    const auto time_gap = [&](int index) {
      return std::abs(keyframes[static_cast<size_t>(index)].stamp - fix_time);
    };
    size_t best = 0;
    for (size_t member = 1; member < trip.size(); ++member) {
      if (time_gap(trip[member]) < time_gap(trip[best])) best = member;
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
    anchor.sigma = std::max(std_dev, kMinGnssStd) * config.gnss_sigma_scale;
    anchors->push_back(anchor);
  }
}

// Position anchors from the control points of one trip: the target's
// LiDAR point moved into the nearest keyframe's frame.
void AddControlPoints(const KeyframeSet& keyframes,
                      const std::vector<int>& trip, const TripInputs& inputs,
                      const BackendConfig& config,
                      std::vector<Anchor>* anchors) {
  const Eigen::MatrixXd& control = inputs.control_points;
  if (control.rows() == 0 || control.cols() < kControlStdColumn ||
      inputs.frame_stamps.empty()) {
    return;
  }
  const MotionPrior prior(inputs.imu, inputs.wheel);
  for (Eigen::Index row = 0; row < control.rows(); ++row) {
    const double stamp = control(row, kControlTimeColumn);
    const auto time_gap = [&](int index) {
      return std::abs(keyframes[static_cast<size_t>(index)].stamp - stamp);
    };
    size_t best = 0;
    for (size_t member = 1; member < trip.size(); ++member) {
      if (time_gap(trip[member]) < time_gap(trip[best])) best = member;
    }
    if (time_gap(trip[best]) > config.control_max_dt) continue;
    const std::optional<Pose> seen_from = OdometryAt(inputs, prior, stamp);
    if (!seen_from) continue;
    const Keyframe& keyframe = keyframes[static_cast<size_t>(trip[best])];
    const Vec3 target_lidar =
        control.block<1, kSpaceDim>(row, kControlLidarColumn).transpose();
    Anchor anchor;
    anchor.type = Anchor::Type::kPosition;
    anchor.keyframe = trip[best];
    anchor.position =
        control.block<1, kSpaceDim>(row, kControlWorldColumn).transpose();
    anchor.lever = keyframe.odom.Inverse() * (*seen_from * target_lidar);
    anchor.sigma =
        control.cols() > kControlStdColumn
            ? std::max(control(row, kControlStdColumn), kMinControlStd)
            : kDefaultControlStd;
    anchors->push_back(anchor);
  }
}

// The wheel speed (vehicle x axis at the vehicle origin) moved to the
// inertial body origin, the IMU: v + w x p, in the vehicle axes.
Vec3 BodyVelocityFromWheel(const TripInputs& inputs, double stamp) {
  const Vec3 wheel(Interp1(inputs.wheel.stamps, inputs.wheel.speed, stamp), 0.0,
                   0.0);
  const Vec3 omega =
      InterpolateSeries(inputs.imu.stamps, inputs.imu.gyro, stamp);
  return wheel + omega.cross(inputs.imu.position);
}

// Start velocity of a keyframe's inertial body without an estimate: from the
// wheel speed, else the finite difference of neighbouring keyframes.
// `member` is the keyframe's position in `trip`.
Vec3 InitialVelocity(const KeyframeSet& keyframes, const std::vector<int>& trip,
                     size_t member, const std::vector<Pose>& poses,
                     const TripInputs& inputs, const Pose& lidar_from_body) {
  const size_t keyframe = static_cast<size_t>(trip[member]);
  const Pose body = poses[keyframe] * lidar_from_body;
  const double stamp = keyframes[keyframe].stamp;
  const WheelSamples& wheel = inputs.wheel;
  if (wheel.size() >= kMinSamples && stamp >= wheel.stamps.front() &&
      stamp <= wheel.stamps.back()) {
    return body.rotation * BodyVelocityFromWheel(inputs, stamp);
  }
  if (trip.size() < kMinSamples) return Vec3::Zero();
  const size_t previous_member = member == 0 ? 0 : member - 1;
  const size_t next_member = member + 1 == trip.size() ? member : member + 1;
  const size_t previous = static_cast<size_t>(trip[previous_member]);
  const size_t next = static_cast<size_t>(trip[next_member]);
  const double interval = keyframes[next].stamp - keyframes[previous].stamp;
  if (interval <= 0.0) return Vec3::Zero();
  return ((poses[next] * lidar_from_body).translation -
          (poses[previous] * lidar_from_body).translation) /
         interval;
}

// Inertial states for the keyframes inside each trip's IMU span, with IMU
// preintegration, wheel velocity and bias random-walk factors. Returns the
// number of IMU factors.
int AddInertialFactors(const KeyframeSet& keyframes,
                       const std::vector<TripInputs>& trips,
                       const BackendConfig& config,
                       const std::vector<Pose>& poses,
                       const InertialEstimate* start, PoseGraph* graph) {
  const ImuNoise noise = config.ToImuNoise();
  const bool has_start = start != nullptr &&
                         start->valid.size() == keyframes.size() &&
                         start->velocities.size() == keyframes.size() &&
                         start->biases.size() == keyframes.size();
  int num_imu = 0;
  for (size_t trip_index = 0; trip_index < trips.size(); ++trip_index) {
    const TripInputs& inputs = trips[trip_index];
    const std::vector<int>& trip =
        keyframes.TripIndices(static_cast<int>(trip_index));
    const ImuSamples& imu = inputs.imu;
    if (trip.empty() || imu.size() < kMinSamples) continue;
    // The inertial body: vehicle axes at the IMU, where the accelerometer
    // measures (ImuSamples::position).
    const Pose lidar_from_body =
        inputs.vehicle_from_lidar.Inverse() * imu.VehicleFromBody();
    const WheelSamples& wheel = inputs.wheel;
    std::vector<int> nodes;
    for (size_t member = 0; member < trip.size(); ++member) {
      const int node = trip[member];
      const size_t keyframe = static_cast<size_t>(node);
      const double stamp = keyframes[keyframe].stamp;
      if (stamp < imu.stamps.front() || stamp > imu.stamps.back()) continue;
      if (has_start && start->valid[keyframe]) {
        graph->AddInertialState(node, start->velocities[keyframe],
                                start->biases[keyframe]);
      } else {
        graph->AddInertialState(node,
                                InitialVelocity(keyframes, trip, member, poses,
                                                inputs, lidar_from_body),
                                Vec6::Zero());
      }
      nodes.push_back(node);
      if (config.wheel_factors && wheel.size() >= kMinSamples &&
          stamp >= wheel.stamps.front() && stamp <= wheel.stamps.back()) {
        graph->AddBodyVelocity(
            node, BodyVelocityFromWheel(inputs, stamp),
            Vec3(config.wheel_sigma, config.wheel_lateral_sigma,
                 config.wheel_lateral_sigma),
            lidar_from_body);
      }
    }
    if (nodes.empty()) continue;
    Vec6 prior_sigma;
    prior_sigma << Vec3::Constant(config.gyro_bias_prior),
        Vec3::Constant(config.accel_bias_prior);
    graph->AddBiasPrior(nodes.front(), Vec6::Zero(), prior_sigma);
    for (size_t member = 1; member < nodes.size(); ++member) {
      const int node_i = nodes[member - 1];
      const int node_j = nodes[member];
      const double start_stamp = keyframes[static_cast<size_t>(node_i)].stamp;
      const double end_stamp = keyframes[static_cast<size_t>(node_j)].stamp;
      const double root_interval =
          std::sqrt(std::max(end_stamp - start_stamp, kMinBiasWalkInterval));
      Vec6 walk;
      walk << Vec3::Constant(noise.gyro_bias_walk * root_interval),
          Vec3::Constant(noise.accel_bias_walk * root_interval);
      graph->AddBiasBetween(node_i, node_j, walk);
      if (end_stamp - start_stamp > config.imu_max_dt) continue;
      const std::optional<ImuPreintegration> preintegration = PreintegrateImu(
          imu, start_stamp, end_stamp, graph->Bias(node_i), noise);
      if (!preintegration) continue;
      graph->AddImu(node_i, node_j, *preintegration, lidar_from_body);
      ++num_imu;
    }
  }
  return num_imu;
}

}  // namespace

PlacementConfig TripAlignConfig::ToPlacementConfig() const {
  PlacementConfig placement;
  placement.voxel = voxel;
  placement.icp_schedule = icp_schedule;
  placement.icp_iters = icp_iters;
  placement.min_score = min_score;
  placement.z_peak_ratio = z_peak_ratio;
  placement.max_candidates = max_candidates;
  placement.min_fitness = min_fitness;
  placement.max_rmse = max_rmse;
  placement.min_fitness_gap = min_fitness_gap;
  placement.distinct_t = distinct_t;
  placement.distinct_r = distinct_r;
  return placement;
}

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
  for (size_t trip_index = 0; trip_index < trips.size(); ++trip_index) {
    const std::vector<int>& trip =
        keyframes.TripIndices(static_cast<int>(trip_index));
    if (trip.empty()) continue;
    const TripInputs& inputs = trips[trip_index];
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
    AddGnssAnchors(keyframes, trip, inputs, lever, config, &anchors);
    AddControlPoints(keyframes, trip, inputs, config, &anchors);
    if (anchors.size() == before) {
      Anchor anchor;
      anchor.type = Anchor::Type::kPose;
      anchor.keyframe = trip.front();
      anchor.pose = keyframes[static_cast<size_t>(trip.front())].odom;
      anchor.sigma_t = kWeakAnchorSigmaT;
      anchor.sigma_r = kWeakAnchorSigmaR;
      anchor.weak = true;
      anchors.push_back(anchor);
      LogLine(log, StringPrintf("[backend] trip %zu: no entrance anchor",
                                trip_index));
    }
  }
  return anchors;
}

namespace {

// The keyframe sweeps of `trips` at `poses`, merged and downsampled.
Points TripMap(const KeyframeSet& keyframes, const std::vector<Pose>& poses,
               const std::vector<int>& trips, double voxel) {
  Points merged;
  for (int trip : trips) {
    for (int index : keyframes.TripIndices(trip)) {
      const size_t keyframe = static_cast<size_t>(index);
      const Points world =
          TransformPoints(keyframes.Cloud(keyframe, voxel), poses[keyframe]);
      merged.insert(merged.end(), world.begin(), world.end());
    }
  }
  return VoxelDownsample(merged, voxel);
}

}  // namespace

std::vector<TripAlignment> AlignTrips(const KeyframeSet& keyframes,
                                      const TripAlignConfig& config,
                                      std::vector<Pose>* poses,
                                      std::vector<Anchor>* anchors,
                                      const Logger& log) {
  GR_CHECK(poses->size() == keyframes.size());
  GR_CHECK(keyframes.HasVoxel(config.voxel));
  const int num_trips = keyframes.num_trips();
  std::vector<uint8_t> anchored(static_cast<size_t>(num_trips), 0);
  for (const Anchor& anchor : *anchors) {
    if (!anchor.weak) {
      anchored[static_cast<size_t>(
          keyframes[static_cast<size_t>(anchor.keyframe)].trip)] = 1;
    }
  }
  std::vector<int> placed;
  for (int trip_index = 0; trip_index < num_trips; ++trip_index) {
    if (anchored[static_cast<size_t>(trip_index)] &&
        !keyframes.TripIndices(trip_index).empty()) {
      placed.push_back(trip_index);
    }
  }
  std::vector<TripAlignment> alignments;
  for (int trip_index = 0; trip_index < num_trips; ++trip_index) {
    const std::vector<int>& trip = keyframes.TripIndices(trip_index);
    if (anchored[static_cast<size_t>(trip_index)] || trip.empty()) continue;
    if (placed.empty()) {
      placed.push_back(trip_index);
      LogLine(log, StringPrintf("[align] trip %d: reference frame (no trip "
                                "has an anchor)",
                                trip_index));
      continue;
    }
    TripAlignment alignment;
    alignment.trip = trip_index;
    const Points source =
        TripMap(keyframes, *poses, {trip_index}, config.voxel);
    const Points reference = TripMap(keyframes, *poses, placed, config.voxel);
    const Placement placement =
        PlaceGravityAligned(source, reference, config.ToPlacementConfig());
    alignment.hypotheses = placement.hypotheses;
    alignment.score = placement.score;
    alignment.fitness = placement.fitness;
    alignment.rmse = placement.rmse;
    alignment.runner_up = placement.runner_up;
    alignment.correction = placement.reference_from_source;
    alignment.aligned = placement.placed;
    const std::string& verdict = placement.verdict;
    const double moved = alignment.correction.translation.norm();
    const double turned =
        RadToDeg(Eigen::AngleAxisd(alignment.correction.rotation).angle());
    LogLine(log,
            StringPrintf("[align] trip %d onto %zu trip(s): %d "
                         "hypotheses, best ICP fitness %.3f (coarse "
                         "score %.2f, rmse %.1f cm), distinct runner-up "
                         "%.3f: %s (%.2f m, %.1f deg)",
                         trip_index, placed.size(), alignment.hypotheses,
                         alignment.fitness, alignment.score,
                         kCentimetresPerMetre * alignment.rmse,
                         alignment.runner_up, verdict.c_str(), moved, turned));
    if (alignment.aligned) {
      for (int index : trip) {
        const size_t keyframe = static_cast<size_t>(index);
        (*poses)[keyframe] = alignment.correction * (*poses)[keyframe];
      }
      for (Anchor& anchor : *anchors) {
        if (!anchor.weak ||
            keyframes[static_cast<size_t>(anchor.keyframe)].trip !=
                trip_index) {
          continue;
        }
        anchor.pose = (*poses)[static_cast<size_t>(anchor.keyframe)];
        anchor.sigma_t = config.sigma_t;
        anchor.sigma_r = config.sigma_r;
      }
      placed.push_back(trip_index);
    }
    alignments.push_back(alignment);
  }
  return alignments;
}

std::vector<GravityMeasurement> MeasureKeyframeGravity(
    const KeyframeSet& keyframes, const std::vector<TripInputs>& trips,
    const BackendConfig& config) {
  std::vector<GravityMeasurement> measurements;
  for (size_t trip_index = 0; trip_index < trips.size(); ++trip_index) {
    const std::vector<int>& trip =
        keyframes.TripIndices(static_cast<int>(trip_index));
    if (trip.empty()) continue;
    std::vector<double> stamps;
    stamps.reserve(trip.size());
    for (int keyframe : trip) {
      stamps.push_back(keyframes[static_cast<size_t>(keyframe)].stamp);
    }
    const TripInputs& inputs = trips[trip_index];
    const Mat3 lidar_from_vehicle_rotation =
        inputs.vehicle_from_lidar.Inverse().rotation;
    for (GravityMeasurement measurement :
         MeasureGravity(inputs.imu, inputs.wheel, lidar_from_vehicle_rotation,
                        stamps, config.ToGravityConfig())) {
      measurement.index = trip[static_cast<size_t>(measurement.index)];
      measurements.push_back(measurement);
    }
  }
  return measurements;
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
    const InertialEstimate* inertial, size_t* num_odometry, int* num_imu) {
  const size_t num_keyframes = keyframes.size();
  GR_CHECK(initial_poses == nullptr || initial_poses->size() == num_keyframes);
  PoseGraph graph;
  for (size_t keyframe = 0; keyframe < num_keyframes; ++keyframe) {
    graph.AddNode(initial_poses != nullptr ? (*initial_poses)[keyframe]
                                           : keyframes[keyframe].odom);
  }
  *num_imu = 0;
  if (config.imu_factors && trips != nullptr) {
    *num_imu = AddInertialFactors(keyframes, *trips, config, graph.poses(),
                                  inertial, &graph);
  }
  *num_odometry = 0;
  for (int trip_index = 0; trip_index < keyframes.num_trips(); ++trip_index) {
    const std::vector<int>& trip = keyframes.TripIndices(trip_index);
    for (size_t member = 1; member < trip.size(); ++member) {
      const int node_i = trip[member - 1];
      const int node_j = trip[member];
      const Pose step = keyframes[static_cast<size_t>(node_i)].odom.Inverse() *
                        keyframes[static_cast<size_t>(node_j)].odom;
      const double distance = step.translation.norm();
      graph.AddBetween(
          node_i, node_j, step,
          PoseInformation(
              config.odom_sigma_t + config.odom_sigma_t_per_m * distance,
              config.odom_sigma_r + config.odom_sigma_r_per_m * distance));
      ++*num_odometry;
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
  for (const GravityMeasurement& measurement : gravity) {
    graph.AddGravity(measurement.index, measurement.up, measurement.sigma_rad);
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
    for (size_t keyframe = 0; keyframe < num_keyframes; ++keyframe) {
      const int node = static_cast<int>(keyframe);
      if (!graph.HasInertialState(node)) continue;
      inertial->velocities[keyframe] = graph.Velocity(node);
      inertial->biases[keyframe] = graph.Bias(node);
      inertial->valid[keyframe] = 1;
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
                 unary, trips, inertial, &num_odometry, &num_imu);
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
                   nullptr, trips, inertial, &num_odometry, &num_imu);
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
      stats.rms_p90_before = Percentile(rms_before, kReportPercentile);
      stats.rms_median_after = Median(rms_after);
      stats.rms_p90_after = Percentile(rms_after, kReportPercentile);
    }
    stats.iterations = result.iterations;
    std::vector<double> shifts;
    for (size_t keyframe = 0; keyframe < current.size(); ++keyframe) {
      shifts.push_back(
          (graph.poses()[keyframe].translation - current[keyframe].translation)
              .norm());
    }
    if (!shifts.empty()) {
      stats.shift_median = Median(shifts);
      stats.shift_max = *std::max_element(shifts.begin(), shifts.end());
    }
    CollectGraph(graph, result, num_odometry, loops.size(), num_imu, summary,
                 inertial);
    current = graph.poses();
    if (ba_summary != nullptr) ba_summary->rounds.push_back(stats);
    LogLine(log, StringPrintf(
                     "[ba] round %d: %d planes (%d scan clusters), plane "
                     "RMS median %.2f -> %.2f cm, p90 %.2f -> %.2f cm, "
                     "keyframe shift median %.2f cm, max %.2f cm, %d it",
                     round + 1, stats.planes, stats.observations,
                     kCentimetresPerMetre * stats.rms_median_before,
                     kCentimetresPerMetre * stats.rms_median_after,
                     kCentimetresPerMetre * stats.rms_p90_before,
                     kCentimetresPerMetre * stats.rms_p90_after,
                     kCentimetresPerMetre * stats.shift_median,
                     kCentimetresPerMetre * stats.shift_max, stats.iterations));
  }
  return current;
}

std::vector<Pose> HierarchicalLidarBa(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity,
    const BackendConfig& backend_config, const LidarBaConfig& ba_config,
    GraphSummary* summary, LidarBaSummary* ba_summary, const Logger& log,
    const std::vector<TripInputs>* trips, InertialEstimate* inertial) {
  GR_CHECK(poses.size() == keyframes.size());
  const double voxel = ba_config.sweep_voxel;
  // 1. Windows of consecutive keyframes per trip, each refined alone.
  std::vector<std::vector<int>> windows;
  std::vector<int> window_of(keyframes.size(), kNoWindow);
  const size_t window_size =
      static_cast<size_t>(std::max(ba_config.window, kMinWindowSize));
  for (int trip_index = 0; trip_index < keyframes.num_trips(); ++trip_index) {
    const std::vector<int>& trip = keyframes.TripIndices(trip_index);
    size_t begin = 0;
    while (begin < trip.size()) {
      // A short tail joins the window before it.
      size_t end = std::min(begin + window_size, trip.size());
      if (trip.size() - end <
          std::max(kMinRefinedWindowSize, window_size / 2)) {
        end = trip.size();
      }
      std::vector<int> members(trip.begin() + static_cast<int64_t>(begin),
                               trip.begin() + static_cast<int64_t>(end));
      for (int member : members) {
        window_of[static_cast<size_t>(member)] =
            static_cast<int>(windows.size());
      }
      windows.push_back(std::move(members));
      begin = end;
    }
  }
  std::vector<Pose> local = poses;  // Refined within their window.
  LidarBaConfig window_config = ba_config;
  window_config.rounds = std::max(1, ba_config.rounds);
  const int64_t num_windows = static_cast<int64_t>(windows.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t window = 0; window < num_windows; ++window) {
    const std::vector<int>& members = windows[static_cast<size_t>(window)];
    if (members.size() < kMinRefinedWindowSize) continue;
    KeyframeSet window_keyframes({voxel});
    std::vector<Pose> start;
    for (int member : members) {
      const size_t index = static_cast<size_t>(member);
      Keyframe keyframe = keyframes[index];
      keyframe.trip = 0;
      keyframe.odom = poses[index];
      window_keyframes.Add(keyframe, keyframes.Cloud(index, voxel));
      start.push_back(poses[index]);
    }
    Anchor hold;  // The first keyframe stays.
    hold.keyframe = 0;
    hold.pose = start.front();
    hold.sigma_t = kHoldSigmaT;
    hold.sigma_r = kHoldSigmaR;
    const std::vector<Pose> refined = RunLidarBa(
        window_keyframes, start, {}, {hold}, {}, backend_config, window_config);
    for (size_t member = 0; member < members.size(); ++member) {
      local[static_cast<size_t>(members[member])] = refined[member];
    }
  }
  // 2. The windows as rigid super-keyframes.
  std::vector<Pose> frames;  // world_from_window: its first keyframe.
  KeyframeSet super({voxel});
  for (const std::vector<int>& members : windows) {
    const Pose frame = local[static_cast<size_t>(members.front())];
    const Pose window_from_world = frame.Inverse();
    Points merged;
    for (int member : members) {
      const Points cloud = TransformPoints(
          keyframes.Cloud(static_cast<size_t>(member), voxel),
          window_from_world * local[static_cast<size_t>(member)]);
      merged.insert(merged.end(), cloud.begin(), cloud.end());
    }
    Keyframe keyframe = keyframes[static_cast<size_t>(members.front())];
    keyframe.odom = frame;
    super.Add(keyframe, VoxelDownsample(merged, voxel));
    frames.push_back(frame);
  }
  // window_from_keyframe within its window.
  const auto in_window = [&](int keyframe) {
    const int window = window_of[static_cast<size_t>(keyframe)];
    return frames[static_cast<size_t>(window)].Inverse() *
           local[static_cast<size_t>(keyframe)];
  };
  std::vector<LoopConstraint> super_loops;
  for (const LoopConstraint& loop : loops) {
    const int window_i = window_of[static_cast<size_t>(loop.i)];
    const int window_j = window_of[static_cast<size_t>(loop.j)];
    if (window_i == window_j) continue;
    LoopConstraint moved = loop;
    moved.i = window_i;
    moved.j = window_j;
    // T_b^-1 T_a = (T_b^-1 T_j)(T_j^-1 T_i)(T_i^-1 T_a).
    moved.j_from_i =
        in_window(loop.j) * loop.j_from_i * in_window(loop.i).Inverse();
    super_loops.push_back(moved);
  }
  std::vector<Anchor> super_anchors;
  for (const Anchor& anchor : anchors) {
    Anchor moved = anchor;
    moved.keyframe = window_of[static_cast<size_t>(anchor.keyframe)];
    const Pose keyframe_in_window = in_window(anchor.keyframe);
    if (anchor.type == Anchor::Type::kPose) {
      moved.pose = anchor.pose * keyframe_in_window.Inverse();
    } else {
      moved.lever = keyframe_in_window * anchor.lever;
    }
    super_anchors.push_back(moved);
  }
  std::vector<GravityMeasurement> super_gravity;
  std::vector<uint8_t> has_gravity(windows.size(), 0);
  for (const GravityMeasurement& measurement : gravity) {
    const int window = window_of[static_cast<size_t>(measurement.index)];
    if (has_gravity[static_cast<size_t>(window)]) continue;  // One per window.
    has_gravity[static_cast<size_t>(window)] = 1;
    GravityMeasurement moved = measurement;
    moved.index = window;
    moved.up = in_window(measurement.index).rotation * measurement.up;
    super_gravity.push_back(moved);
  }
  if (trips != nullptr && backend_config.imu_factors &&
      ba_config.window_gravity_sigma_deg > 0.0) {
    // The super-keyframes have no IMU factors: each keeps the roll / pitch
    // the inertial graph gave its keyframes (their mean up direction in the
    // window frame; single keyframes are off by up to 0.2 deg).
    for (size_t window = 0; window < windows.size(); ++window) {
      if (has_gravity[window]) continue;
      Vec3 up = Vec3::Zero();
      for (int member : windows[window]) {
        up += in_window(member).rotation *
              (poses[static_cast<size_t>(member)].rotation.transpose() *
               Vec3::UnitZ());
      }
      GravityMeasurement tilt;
      tilt.index = static_cast<int>(window);
      tilt.up = up.normalized();
      tilt.sigma_rad = DegToRad(ba_config.window_gravity_sigma_deg);
      super_gravity.push_back(tilt);
    }
  }
  LogLine(log, StringPrintf("[ba] hierarchical: %zu keyframes in %zu windows",
                            keyframes.size(), windows.size()));
  const std::vector<Pose> placed =
      RunLidarBa(super, frames, super_loops, super_anchors, super_gravity,
                 backend_config, ba_config, nullptr, ba_summary, log);
  // 3. Every keyframe near its placed window, smoothed by the full graph.
  std::vector<Pose> rigid(keyframes.size());
  std::vector<UnaryPrior> priors;
  const Mat6 information = PoseInformation(ba_config.window_prior_sigma_t,
                                           ba_config.window_prior_sigma_r);
  for (size_t keyframe = 0; keyframe < keyframes.size(); ++keyframe) {
    const int window = window_of[keyframe];
    const int node = static_cast<int>(keyframe);
    rigid[keyframe] = placed[static_cast<size_t>(window)] * in_window(node);
    priors.push_back({node, rigid[keyframe], information});
  }
  return OptimizeGraph(keyframes, loops, anchors, gravity, backend_config,
                       &rigid, &priors, summary, trips, inertial);
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
    VoxelHashMap map(voxel, VoxelHashMap::kUnbounded, kRefineMaxPointsPerVoxel);
    for (int64_t frame = 0; frame < num_keyframes; ++frame) {
      const size_t keyframe = static_cast<size_t>(frame);
      map.AddPoints(
          TransformPoints(keyframes.Cloud(keyframe, voxel), refined[keyframe]),
          static_cast<int>(frame));
    }
    std::vector<IcpResult> alignments(static_cast<size_t>(num_keyframes));
#ifdef _OPENMP
    constexpr int kKeyframesPerChunk = 4;
#pragma omp parallel for schedule(dynamic, kKeyframesPerChunk)
#endif
    for (int64_t frame = 0; frame < num_keyframes; ++frame) {
      const size_t keyframe = static_cast<size_t>(frame);
      IcpOptions options;
      options.max_corr_dist = refine_config.max_corr;
      options.kernel = refine_config.max_corr / kGatePerKernel;
      options.max_iters = refine_config.icp_iters;
      options.exclude_frame = static_cast<int>(frame);  // Other keyframes only.
      options.parallel = false;
      alignments[keyframe] = AlignToMap(keyframes.Cloud(keyframe, voxel), map,
                                        refined[keyframe], options);
    }
    std::vector<UnaryPrior> unary;
    std::vector<double> shifts;
    for (size_t keyframe = 0; keyframe < alignments.size(); ++keyframe) {
      const IcpResult& alignment = alignments[keyframe];
      if (alignment.fitness < refine_config.min_fitness) continue;
      UnaryPrior prior;
      prior.keyframe = static_cast<int>(keyframe);
      prior.pose = alignment.pose;
      prior.information = information;
      unary.push_back(prior);
      shifts.push_back(
          (alignment.pose.translation - refined[keyframe].translation).norm());
    }
    refined = OptimizeGraph(keyframes, loops, anchors, gravity, backend_config,
                            &refined, &unary, summary, trips, inertial);
    LogLine(log,
            StringPrintf("[refine] iteration %d: median scan-to-map shift "
                         "%.1f cm, %zu/%zu keyframes aligned",
                         iteration + 1, Median(shifts) * kCentimetresPerMetre,
                         unary.size(), keyframes.size()));
  }
  return refined;
}

GraphStageResult RunGraphStage(
    const KeyframeSet& keyframes, const std::vector<TripInputs>& trips,
    const GraphStageConfig& config, const Logger& log,
    const std::vector<std::pair<int, int>>& appearance) {
  for (double voxel : config.RequiredVoxels()) {
    GR_CHECK(keyframes.HasVoxel(voxel));
  }
  GraphStageResult result;
  const auto log_cost = [&](const GraphSummary& summary, size_t num_loops) {
    LogLine(log, StringPrintf("[backend] cost %.1f -> %.1f in %d it, %zu loops",
                              summary.initial_cost, summary.final_cost,
                              summary.iterations, num_loops));
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
  if (config.trip_align.enabled) {
    result.trip_alignments =
        AlignTrips(keyframes, config.trip_align, &poses, &result.anchors, log);
    const bool moved = std::any_of(
        result.trip_alignments.begin(), result.trip_alignments.end(),
        [](const TripAlignment& alignment) { return alignment.aligned; });
    if (moved) {
      // The velocities of the moved trips are in their old frame.
      result.inertial = InertialEstimate();
      poses = OptimizeGraph(keyframes, {}, result.anchors, result.gravity,
                            config.backend, &poses, nullptr, &initial_summary,
                            imu_trips, &result.inertial);
      log_cost(initial_summary, 0);
    }
  }

  std::vector<LoopConstraint> verified =
      DetectLoops(keyframes, poses, config.loops, &result.loop_stats);
  LogLine(log, StringPrintf("[loops] %d candidates verified by ICP, %zu "
                            "accepted",
                            result.loop_stats.tested, verified.size()));
  if (config.loops.enabled && !appearance.empty()) {
    std::set<std::pair<int, int>> known;
    for (const LoopConstraint& loop : verified) known.insert({loop.i, loop.j});
    std::vector<std::pair<int, int>> pairs;
    for (const auto& pair : appearance) {
      if (!known.count(pair)) pairs.push_back(pair);
    }
    int runs = 0;
    const std::vector<LoopConstraint> found = VerifyAppearanceLoops(
        keyframes, poses, pairs, config.loops, config.appearance, &runs);
    LogLine(log, StringPrintf("[loops] %zu appearance candidates: %d ICP "
                              "runs, %zu accepted",
                              pairs.size(), runs, found.size()));
    verified.insert(verified.end(), found.begin(), found.end());
  }
  result.loops_verified = static_cast<int>(verified.size());
  result.loops =
      SelectConsistentLoops(verified, keyframes, poses, config.loops);
  LogLine(log, StringPrintf("[loops] PCM kept %zu/%zu", result.loops.size(),
                            verified.size()));

  poses = OptimizeGraph(keyframes, result.loops, result.anchors, result.gravity,
                        config.backend, &poses, nullptr, &result.summary,
                        imu_trips, &result.inertial);
  log_cost(result.summary, result.loops.size());
  const bool hierarchical =
      config.ba.hierarchical_above >= 0 &&
      static_cast<int>(keyframes.size()) > config.ba.hierarchical_above;
  if (config.ba.enabled && hierarchical) {
    poses = HierarchicalLidarBa(keyframes, poses, result.loops, result.anchors,
                                result.gravity, config.backend, config.ba,
                                &result.summary, &result.ba, log, imu_trips,
                                &result.inertial);
  } else if (config.ba.enabled) {
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
  for (size_t keyframe = 0; keyframe < corrections.size(); ++keyframe) {
    corrections[keyframe] = kf_poses[keyframe] * kf_odom[keyframe].Inverse();
  }
  std::vector<Pose> frame_poses(frame_stamps.size());
  for (size_t frame = 0; frame < frame_poses.size(); ++frame) {
    frame_poses[frame] =
        InterpolatePose(kf_stamps, corrections, frame_stamps[frame]) *
        frame_odom[frame];
  }
  return frame_poses;
}

}  // namespace gr
