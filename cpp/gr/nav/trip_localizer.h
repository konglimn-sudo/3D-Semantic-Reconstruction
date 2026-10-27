// Localization of a whole trip in a prior map: the odometry front-end
// (OdometryPipeline, in its own gravity-aligned frame) motion-compensates
// every sweep with the IMU / wheel prior and feeds the Localizer
// (localization.h). nav/localize.py and the gr_localize program both drive
// this loop.

#ifndef GARAGE_RECON_GR_NAV_TRIP_LOCALIZER_H_
#define GARAGE_RECON_GR_NAV_TRIP_LOCALIZER_H_

#include <optional>
#include <vector>

#include "gr/core/logging.h"
#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/nav/localization.h"
#include "gr/odometry/odometry.h"
#include "gr/sensors/imu.h"

namespace gr {

struct TripLocalizerConfig {
  OdometryConfig odometry;
  KeyframeConfig keyframes;
  LocalizerConfig localizer;
  double level_window = 1.0;  // s of accelerometer that level the start.
};

// world_from_vehicle with the roll and pitch of gravity at `stamp`. The
// global placement needs a gravity-aligned odometry frame, but a vehicle may
// be switched on anywhere, on a ramp too: the mean specific force (vehicle
// axes) over `window` seconds around `stamp` gives the up direction; the yaw
// stays arbitrary. Identity without IMU samples there.
Pose LevelledStart(const ImuSamples& imu, double stamp, double window);

struct SweepLocalization {
  int index = 0;
  double stamp = 0.0;
  LocalizationResult result;  // result.pose: map_from_sensor.
  Pose map_from_vehicle;
  Pose odom_from_vehicle;
  double sigma_t = 0.0;        // sqrt(trace of the translation covariance), m.
  double sigma_yaw_deg = 0.0;
  double ms = 0.0;             // Localizer time of the sweep.
};

struct TripLocalizationSummary {
  int sweeps = 0;
  int first_tracked = -1;     // Index into the sweeps; -1: never tracked.
  double init_time_s = 0.0;   // From the first sweep to the first tracked.
  double init_distance_m = 0.0;  // Odometry travel until then.
  int placement_attempts = 0;
  int tracked = 0;
  int registered = 0;
  double registered_share_of_tracked = 0.0;  // Defined if tracked > 0.
  int lost_sweeps = 0;
  double degenerate_share = 0.0;  // Of the tracked sweeps.
  double ms_median = 0.0;
  double ms_p95 = 0.0;
};

class TripLocalizer {
 public:
  // `odom_from_vehicle0`: the start of the odometry frame (LevelledStart);
  // `map_from_vehicle_prior`: an optional pose at the first sweep.
  TripLocalizer(const Points& map_points, const TripLocalizerConfig& config,
                const Pose& vehicle_from_lidar,
                const Pose& odom_from_vehicle0, MotionPrior prior,
                const std::optional<Pose>& map_from_vehicle_prior =
                    std::nullopt,
                Logger log = {});

  // One sweep (sensor frame; `times` relative to `stamp`). Every sweep goes
  // through the odometry (`last` for the final one); with `localize` it is
  // also localized in the map and recorded.
  std::optional<SweepLocalization> Process(
      const Points& points, const std::vector<float>& intensity,
      const std::vector<double>& times, double stamp, int index, bool last,
      bool localize);

  const std::vector<SweepLocalization>& sweeps() const { return sweeps_; }
  const Localizer& localizer() const { return localizer_; }
  // The motion-compensated sweep of the last localized sweep (sensor frame,
  // cropped and downsampled by the odometry).
  const Points& last_sweep() const { return last_sweep_; }
  TripLocalizationSummary Summary() const;

 private:
  Pose vehicle_from_lidar_;
  OdometryPipeline odometry_;
  Localizer localizer_;
  bool has_prior_ = false;
  Logger log_;
  int attempts_ = 0;
  std::vector<SweepLocalization> sweeps_;
  Points last_sweep_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_TRIP_LOCALIZER_H_
