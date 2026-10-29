#include "gr/nav/trip_localizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include "gr/core/numeric.h"

namespace gr {
namespace {

// Vectors shorter than this have no direction.
constexpr double kMinNorm = 1e-9;
constexpr double kMillisecondsPerSecond = 1000.0;
// The yaw (rotation about z) in a [dt, dphi] covariance.
constexpr int kYawIndex = kRotationOffset + 2;
// The summary reports this percentile of the localizer times.
constexpr double kTimePercentile = 95.0;

double Seconds(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       since)
      .count();
}

std::string PlacementText(const Placement& placement) {
  return StringPrintf("%s (fitness %.3f, runner-up %.3f",
                      placement.verdict.c_str(), placement.fitness,
                      placement.runner_up);
}

}  // namespace

Pose LevelledStart(const ImuSamples& imu, double stamp, double window) {
  Vec3 sum = Vec3::Zero();
  int count = 0;
  for (size_t k = 0; k < imu.size(); ++k) {
    if (std::abs(imu.stamps[k] - stamp) <= window / 2) {
      sum += imu.accel[k];
      ++count;
    }
  }
  if (count == 0) return Pose();
  Vec3 up = sum / count;
  up /= std::max(up.norm(), kMinNorm);
  // Smallest rotation taking `up` (vehicle frame) to +z (world).
  const Vec3 axis = up.cross(Vec3::UnitZ());
  const double angle = std::atan2(axis.norm(), up.z());
  if (axis.norm() <= kMinNorm) return Pose();
  return Pose(So3Exp(axis / axis.norm() * angle), Vec3::Zero());
}

TripLocalizer::TripLocalizer(const Points& map_points,
                             const TripLocalizerConfig& config,
                             const Pose& vehicle_from_lidar,
                             const Pose& odom_from_vehicle0,
                             MotionPrior prior,
                             const std::optional<Pose>& map_from_vehicle_prior,
                             Logger log)
    : vehicle_from_lidar_(vehicle_from_lidar),
      odometry_(config.odometry, config.keyframes, vehicle_from_lidar,
                odom_from_vehicle0, std::move(prior)),
      localizer_([&] {
        const auto started = std::chrono::steady_clock::now();
        Localizer localizer(map_points, config.localizer);
        LogLine(log, StringPrintf("[localize] map %zu points (%.1f s)",
                                  localizer.map_size(), Seconds(started)));
        return localizer;
      }()),
      has_prior_(map_from_vehicle_prior.has_value()),
      log_(std::move(log)) {
  if (map_from_vehicle_prior) {
    localizer_.SetPrior(*map_from_vehicle_prior * vehicle_from_lidar_);
  }
}

std::optional<SweepLocalization> TripLocalizer::Process(
    const Points& points, const std::vector<float>& intensity,
    const std::vector<double>& times, double stamp, int index, bool last,
    bool localize) {
  // The odometry compensates the sweep with the IMU / wheel motion during
  // it (better than the localizer's constant velocity in turns).
  const OdometryFrame frame = odometry_.Process(
      points, intensity, times, stamp, last, std::nullopt, localize);
  if (!localize) return std::nullopt;
  const Pose lidar_from_vehicle = vehicle_from_lidar_.Inverse();
  SweepLocalization out;
  out.index = index;
  out.stamp = stamp;
  const auto started = std::chrono::steady_clock::now();
  out.result = localizer_.Process(frame.points, {}, stamp, frame.pose);
  last_sweep_ = frame.points;
  out.ms = kMillisecondsPerSecond * Seconds(started);
  out.map_from_vehicle = out.result.pose * lidar_from_vehicle;
  out.odom_from_vehicle = frame.pose * lidar_from_vehicle;
  const Mat6& covariance = out.result.covariance;
  out.sigma_t = std::sqrt(std::max(
      covariance.topLeftCorner<kSpaceDim, kSpaceDim>().trace(), 0.0));
  out.sigma_yaw_deg = RadToDeg(
      std::sqrt(std::max(covariance(kYawIndex, kYawIndex), 0.0)));
  if (localizer_.placement_attempts() > attempts_) {
    attempts_ = localizer_.placement_attempts();
    const Placement& placement = localizer_.last_placement();
    LogLine(log_, StringPrintf("[localize] sweep %d: placement attempt %d "
                               "after %.1f m: %s, %d hypotheses)",
                               index, attempts_, localizer_.travelled(),
                               PlacementText(placement).c_str(),
                               placement.hypotheses));
  }
  if (out.result.initialized) {
    const std::string how =
        has_prior_ && localizer_.placement_attempts() == 0
            ? std::string("prior")
            : "placement " + PlacementText(localizer_.last_placement()) + ")";
    LogLine(log_, StringPrintf("[localize] sweep %d: initialized after %.1f "
                               "m by %s",
                               index, localizer_.travelled(), how.c_str()));
  }
  sweeps_.push_back(out);
  return out;
}

TripLocalizationSummary TripLocalizer::Summary() const {
  TripLocalizationSummary summary;
  summary.sweeps = static_cast<int>(sweeps_.size());
  summary.placement_attempts = localizer_.placement_attempts();
  if (sweeps_.empty()) return summary;
  double travelled = 0.0;
  int registered_tracked = 0;
  int degenerate = 0;
  std::vector<double> ms;
  for (size_t k = 0; k < sweeps_.size(); ++k) {
    const SweepLocalization& sweep = sweeps_[k];
    if (k > 0) {
      travelled += (sweep.odom_from_vehicle.translation -
                    sweeps_[k - 1].odom_from_vehicle.translation)
                       .norm();
    }
    const bool tracking =
        sweep.result.status == LocalizationStatus::kTracking;
    if (tracking &&
        summary.first_tracked == TripLocalizationSummary::kNeverTracked) {
      summary.first_tracked = static_cast<int>(k);
      summary.init_time_s = sweep.stamp - sweeps_[0].stamp;
      summary.init_distance_m = travelled;
    }
    summary.tracked += tracking;
    summary.registered += sweep.result.registered;
    registered_tracked += tracking && sweep.result.registered;
    degenerate += tracking && sweep.result.degenerate_t > 0;
    summary.lost_sweeps += sweep.result.status == LocalizationStatus::kLost;
    ms.push_back(sweep.ms);
  }
  if (summary.tracked > 0) {
    summary.registered_share_of_tracked =
        static_cast<double>(registered_tracked) / summary.tracked;
    summary.degenerate_share =
        static_cast<double>(degenerate) / summary.tracked;
  }
  summary.ms_median = Median(ms);
  summary.ms_p95 = Percentile(ms, kTimePercentile);
  return summary;
}

}  // namespace gr
