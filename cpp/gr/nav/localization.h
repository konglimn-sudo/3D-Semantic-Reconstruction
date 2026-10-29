// Localization of a vehicle in a prior map: the measurement point cloud of an
// earlier reconstruction, reduced to the permanent structure (walls, columns,
// floors, ceilings; parked cars change between visits).
//
// An odometry front-end (odometry.h, lio.h) supplies the motion of the
// sensor in its own gravity-aligned frame; the localizer keeps the
// correction map_from_odom and refines it on every sweep:
//
//   initialization  with a prior pose (garage entrance, GNSS before the
//                   ramp): multi-scale ICP of the first sweep from the prior.
//                   Without one, the odometry submap of the last
//                   `init_distance` metres is placed on the map by the
//                   hypothesis search of cloud_compare.h (PlaceGravityAligned)
//                   and accepted only if no distinct placement fits nearly as
//                   well; a regular garage needs a longer submap, so the
//                   search is retried every `init_retry` metres up to
//                   `init_max_distance`. Either way the current sweep must
//                   then register at the new pose, or the initialization is
//                   discarded.
//   tracking        point-to-plane ICP of the deskewed sweep against the map
//                   from the predicted pose (correction * odometry). Along
//                   directions the sweep does not constrain (eigenvalues of
//                   the translation or rotation block of the ICP Hessian
//                   below `degenerate_ratio` of the largest, e.g. along a
//                   featureless corridor) the prediction is kept. A sweep is
//                   rejected if the fit is poor or the correction jumps.
//   lost            after `max_failures` consecutive rejections the
//                   localizer re-initializes globally.

#ifndef GARAGE_RECON_GR_NAV_LOCALIZATION_H_
#define GARAGE_RECON_GR_NAV_LOCALIZATION_H_

#include <deque>
#include <limits>
#include <optional>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/registration/cloud_compare.h"
#include "gr/registration/registration.h"

namespace gr {

struct LocalizerConfig {
  double map_voxel = 0.15;     // Map points kept for tracking (m).
  double hash_voxel = 0.5;     // Voxel of the tracking hash map (m).
  double scan_voxel = 0.25;    // Sweep downsampling (m).
  double min_range = 1.0;      // Sweep points used (m).
  double max_range = 40.0;
  bool deskew = true;          // Constant velocity from the odometry.
  std::vector<double> gates = {1.0, 0.3};  // ICP passes, correspondence gate.
  int icp_iters = 30;
  // Inliers / sweep points at the last gate; parked cars are not in the map.
  double min_fitness = 0.4;
  double max_rmse = 0.12;      // m
  double max_jump_t = 0.5;     // Correction beyond the prediction (m) ...
  double max_jump_r = 3.0;     // ... and deg.
  double degenerate_ratio = 0.02;
  // Standard deviation reported along degenerate directions (m, deg).
  double degenerate_sigma_t = 1.0;
  double degenerate_sigma_r = 5.0;
  int max_failures = 10;
  // Initialization from a prior pose.
  std::vector<double> prior_schedule = {2.0, 1.0, 0.5, 0.3};
  // Global initialization.
  double init_voxel = 0.2;          // Submap sweeps (m).
  double init_spacing = 0.5;        // A sweep enters the submap every ... m.
  double init_distance = 15.0;      // First attempt after this travel (m).
  double init_retry = 5.0;          // Then every ... m.
  double init_max_distance = 60.0;  // Submap length kept (m).
  // A correct placement of the submap matches nearly all of its structure;
  // wrong ones on a regular garage still reach about 0.5 (floors, ceilings,
  // most columns), hence a stricter fitness than for whole trips.
  static constexpr double kSubmapPlacementMinFitness = 0.7;
  PlacementConfig placement = [] {
    PlacementConfig config;
    config.min_fitness = kSubmapPlacementMinFitness;
    return config;
  }();
};

enum class LocalizationStatus { kInitializing = 0, kTracking = 1, kLost = 2 };

struct LocalizationResult {
  LocalizationStatus status = LocalizationStatus::kInitializing;
  // map_from_sensor: registered, or predicted from the odometry when the
  // sweep was rejected; identity while initializing.
  Pose pose;
  bool registered = false;   // This sweep was aligned and accepted.
  bool initialized = false;  // The correction was (re)set on this sweep.
  double fitness = 0.0;
  double rmse = 0.0;
  // Degenerate directions whose correction was replaced by the prediction:
  // translation and rotation.
  int degenerate_t = 0;
  int degenerate_r = 0;
  // Smallest eigenvalue ratio of the translation block (degeneracy margin).
  double min_eigen_ratio_t = 0.0;
  // Correction of the prediction by the registration (m, deg), after the
  // degenerate directions were dropped.
  double step_t = 0.0;
  double step_r = 0.0;
  // Covariance of the sensor position and of a rotation (world axes) about
  // it, order [t, phi], from the ICP Hessian and RMSE; degenerate
  // directions get degenerate_sigma_*.
  Mat6 covariance = Mat6::Identity();
  int failures = 0;  // Consecutive rejected sweeps.
};

class Localizer {
 public:
  Localizer(const Points& map_points, const LocalizerConfig& config);

  // A prior pose of the sensor for the next sweep (map_from_sensor), e.g.
  // the garage entrance. Resets the localizer.
  void SetPrior(const Pose& map_from_sensor);

  // One sweep in the sensor frame; `times` are relative to `stamp` (may be
  // empty), `odom_from_sensor` is the odometry pose at `stamp`.
  LocalizationResult Process(const Points& points,
                             const std::vector<double>& times, double stamp,
                             const Pose& odom_from_sensor);

  LocalizationStatus status() const { return status_; }
  const Pose& map_from_odom() const { return map_from_odom_; }
  // The last global placement attempt.
  const Placement& last_placement() const { return last_placement_; }
  int placement_attempts() const { return placement_attempts_; }
  double travelled() const { return travelled_; }
  size_t map_size() const { return map_points_.size(); }

 private:
  struct SubmapSweep {
    Pose odom_from_sensor;
    Points points;  // Sensor frame, downsampled.
    double travelled;
  };

  Points PrepareSweep(const Points& points, const std::vector<double>& times,
                      double stamp, const Pose& odom_from_sensor) const;
  bool TryPrior(const Points& sweep, const Pose& odom_from_sensor,
                LocalizationResult& result);
  bool TryGlobal(const Points& sweep, const Pose& odom_from_sensor,
                 LocalizationResult& result);
  void Track(const Points& sweep, const Pose& odom_from_sensor,
             LocalizationResult& result);
  void AddToSubmap(const Points& sweep, const Pose& odom_from_sensor);

  LocalizerConfig config_;
  Points map_points_;    // At config.map_voxel.
  Points coarse_map_;    // At config.placement.voxel, for the global search.
  VoxelHashMap map_;
  LocalizationStatus status_ = LocalizationStatus::kInitializing;
  Pose map_from_odom_;
  std::optional<Pose> prior_;
  std::deque<SubmapSweep> submap_;
  std::optional<Pose> last_odom_;
  double last_stamp_ = 0.0;
  double travelled_ = 0.0;
  // Travel at the last global placement attempt (m); none yet.
  double last_attempt_ = -std::numeric_limits<double>::infinity();
  int failures_ = 0;
  int placement_attempts_ = 0;
  Placement last_placement_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_LOCALIZATION_H_
