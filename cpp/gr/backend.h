// Pose-graph back-end (sections 4.1 / 4.2): odometry chains of all trips,
// loop closures, entrance anchors (INS pose / GNSS fixes), IMU preintegration
// with a velocity and an IMU bias per keyframe (or, alternatively,
// accelerometer gravity factors for roll / pitch), wheel-speed velocity
// factors, and the LiDAR bundle adjustment (plane factors, lidar_ba.h) or the
// older scan-to-map refinement. RunGraphStage is the whole stage: graph
// without loops -> loop detection + PCM -> graph with loops -> bundle
// adjustment.

#ifndef GARAGE_RECON_GR_BACKEND_H_
#define GARAGE_RECON_GR_BACKEND_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "gr/imu.h"
#include "gr/imu_preintegration.h"
#include "gr/keyframes.h"
#include "gr/lidar_ba.h"
#include "gr/logging.h"
#include "gr/loop_closure.h"
#include "gr/pose.h"

namespace gr {

struct BackendConfig {
  double odom_sigma_t = 0.02;  // Per keyframe step, plus drift per metre.
  double odom_sigma_t_per_m = 0.005;
  double odom_sigma_r = 0.2;  // deg
  double odom_sigma_r_per_m = 0.05;
  double loop_sigma_t = 0.05;
  double loop_sigma_r = 0.5;
  double loop_robust_scale = 1.0;  // Cauchy, sqrt(chi2) units.
  double anchor_sigma_t = 0.05;    // Initial pose (INS at the entrance).
  double anchor_sigma_r = 0.3;
  double gnss_sigma_scale = 1.0;
  double gnss_max_dt = 1.5;  // s between a fix and its keyframe.
  // Accelerometer roll / pitch factors. They ignore the accelerometer bias;
  // the IMU factors below model it and replace them.
  bool gravity = false;
  double gravity_sigma_deg = 0.2;
  double gravity_window = 0.25;
  double gravity_max_dev = 0.3;
  double gravity_max_yaw_rate = 3.0;
  // IMU preintegration between consecutive keyframes of a trip. The noise
  // densities are effective values: the sensor's white noise inflated for
  // vibration, scale, misalignment and timing errors.
  bool imu_factors = true;
  double imu_gyro_noise = 1e-2;       // rad/s/sqrt(Hz)
  double imu_accel_noise = 0.2;       // m/s^2/sqrt(Hz)
  double imu_gyro_bias_walk = 2e-5;   // rad/s^2/sqrt(Hz)
  double imu_accel_bias_walk = 2e-4;  // m/s^3/sqrt(Hz)
  double gyro_bias_prior = 0.01;      // rad/s, sigma around zero.
  double accel_bias_prior = 0.2;      // m/s^2, sigma around zero.
  double imu_max_dt = 5.0;            // s; longer gaps get no IMU factor.
  // Wheel speed as the vehicle's body velocity (x), with zero lateral and
  // vertical velocity (non-holonomic constraint).
  bool wheel_factors = true;
  double wheel_sigma = 0.05;  // m/s
  // Loose: side-slip, and the vehicle frame is not exactly aligned with the
  // direction of motion (turns, ramp transitions).
  double wheel_lateral_sigma = 0.3;  // m/s
  int iterations = 50;

  GravityConfig ToGravityConfig() const;
  ImuNoise ToImuNoise() const;
};

// Scan-to-map refinement: every keyframe is aligned to the map of all other
// keyframes and the graph is re-optimised with these alignments as priors (a
// lightweight stand-in for the LiDAR bundle adjustment, which replaces it by
// default).
struct RefineConfig {
  bool enabled = false;
  int iterations = 2;
  double voxel = 0.2;
  double max_corr = 0.5;
  double sigma_t = 0.02;
  double sigma_r = 0.2;
  double min_fitness = 0.3;
  int icp_iters = 20;
};

struct Anchor {
  enum class Type { kPose, kPosition };

  Type type = Type::kPose;
  int keyframe = 0;
  Pose pose;                     // kPose: world_from_lidar.
  Vec3 position = Vec3::Zero();  // kPosition: world position of `lever`.
  Vec3 lever = Vec3::Zero();     // LiDAR-frame point (the vehicle origin).
  double sigma_t = 1.0;          // kPose (m)
  double sigma_r = 1.0;          // kPose (deg)
  double sigma = 1.0;            // kPosition (m)
};

// Everything the back-end needs from one trip besides its keyframes.
struct TripInputs {
  Pose vehicle_from_lidar;
  std::optional<Pose> initial_pose;  // world_from_vehicle, INS at the entrance.
  Eigen::MatrixXd gnss;              // Rows [t, x, y, z(, std)], local ENU (m).
  std::vector<double> frame_stamps;  // Every LiDAR frame of the trip.
  std::vector<Pose> frame_odometry;  // world_from_lidar per frame (odometry).
  ImuSamples imu;                    // Vehicle frame.
  WheelSamples wheel;
};

struct UnaryPrior {
  int keyframe = 0;
  Pose pose;
  Mat6 information = Mat6::Identity();
};

struct GraphSummary {
  int iterations = 0;
  double initial_cost = 0.0;
  double final_cost = 0.0;
  bool converged = false;
  double loop_chi2_median = 0.0;
  double loop_chi2_max = 0.0;
  int imu_factors = 0;
  double imu_chi2_median = 0.0;
};

// Velocity (world frame, of the vehicle origin) and IMU bias [gyro, accel]
// per keyframe; `valid` marks keyframes with an inertial state.
struct InertialEstimate {
  std::vector<Vec3> velocities;
  std::vector<Vec6> biases;
  std::vector<uint8_t> valid;
};

// INS pose at the entrance and GNSS fixes (moved to their keyframe with the
// odometry increment); trips without any anchor get a weak prior so that the
// graph stays well posed (they are then placed by inter-trip loops).
std::vector<Anchor> MakeAnchors(const KeyframeSet& keyframes,
                                const std::vector<TripInputs>& trips,
                                const BackendConfig& config,
                                const Logger& log = nullptr);

// Gravity measurements of every keyframe (index = keyframe), in its LiDAR
// frame.
std::vector<GravityMeasurement> MeasureKeyframeGravity(
    const KeyframeSet& keyframes, const std::vector<TripInputs>& trips,
    const BackendConfig& config);

// Builds and optimises the graph. `initial_poses` defaults to the odometry
// poses; `unary` adds absolute pose priors; `summary` may be null. With
// `trips` and config.imu_factors, keyframes inside a trip's IMU span get
// inertial states and IMU / wheel / bias factors; `inertial` (optional)
// provides their start values when it holds an estimate and receives the
// optimised ones.
std::vector<Pose> OptimizeGraph(
    const KeyframeSet& keyframes, const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity, const BackendConfig& config,
    const std::vector<Pose>* initial_poses = nullptr,
    const std::vector<UnaryPrior>* unary = nullptr,
    GraphSummary* summary = nullptr,
    const std::vector<TripInputs>* trips = nullptr,
    InertialEstimate* inertial = nullptr);

// `keyframes` must keep sweeps at `refine_config.voxel`. `trips` and
// `inertial` are passed on to OptimizeGraph.
std::vector<Pose> RefineScanToMap(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity,
    const BackendConfig& backend_config, const RefineConfig& refine_config,
    GraphSummary* summary = nullptr, const Logger& log = nullptr,
    const std::vector<TripInputs>* trips = nullptr,
    InertialEstimate* inertial = nullptr);

// LiDAR bundle adjustment: `ba_config.rounds` times, planes are extracted at
// the current poses (ExtractPlanes) and added as plane factors to the graph
// of OptimizeGraph, which is then optimised jointly (odometry, loops,
// anchors, inertial factors). `keyframes` must keep sweeps at
// ba_config.sweep_voxel; `summary`, `ba_summary`, `trips` and `inertial` as
// in OptimizeGraph (may be null).
std::vector<Pose> RunLidarBa(
    const KeyframeSet& keyframes, const std::vector<Pose>& poses,
    const std::vector<LoopConstraint>& loops,
    const std::vector<Anchor>& anchors,
    const std::vector<GravityMeasurement>& gravity,
    const BackendConfig& backend_config, const LidarBaConfig& ba_config,
    GraphSummary* summary = nullptr, LidarBaSummary* ba_summary = nullptr,
    const Logger& log = nullptr, const std::vector<TripInputs>* trips = nullptr,
    InertialEstimate* inertial = nullptr);

struct GraphStageConfig {
  LoopConfig loops;
  BackendConfig backend;
  LidarBaConfig ba;
  RefineConfig refine;  // Used when the bundle adjustment is disabled.

  // Voxel sizes at which a KeyframeSet must keep sweeps for this stage.
  std::vector<double> RequiredVoxels() const {
    std::vector<double> voxels = {loops.sc_voxel, loops.icp_voxel};
    if (ba.enabled) voxels.push_back(ba.sweep_voxel);
    if (refine.enabled) voxels.push_back(refine.voxel);
    return voxels;
  }
};

struct GraphStageResult {
  std::vector<Pose> poses;    // Keyframes, world_from_lidar.
  InertialEstimate inertial;  // Empty without IMU factors.
  std::vector<LoopConstraint> loops;
  std::vector<Anchor> anchors;
  std::vector<GravityMeasurement> gravity;
  GraphSummary summary;  // Of the last optimisation.
  LidarBaSummary ba;
  LoopStats loop_stats;
  int loops_verified = 0;  // Before PCM.
};

// `keyframes` must keep sweeps at every voxel of config.RequiredVoxels().
GraphStageResult RunGraphStage(const KeyframeSet& keyframes,
                               const std::vector<TripInputs>& trips,
                               const GraphStageConfig& config,
                               const Logger& log = nullptr);

// Poses of all frames of a trip: the odometry corrected by the keyframe
// corrections (kf_pose * kf_odom^-1), interpolated in time.
std::vector<Pose> CorrectFramePoses(const std::vector<double>& kf_stamps,
                                    const std::vector<Pose>& kf_odom,
                                    const std::vector<Pose>& kf_poses,
                                    const std::vector<double>& frame_stamps,
                                    const std::vector<Pose>& frame_odom);

}  // namespace gr

#endif  // GARAGE_RECON_GR_BACKEND_H_
