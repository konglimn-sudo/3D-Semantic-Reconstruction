// Clock offset between two sensors that both measure angular rate.
//
// Many LiDARs (Ouster, Hesai, Livox) carry an IMU stamped by the LiDAR's own
// clock. Correlating its gyro with the main IMU's gyro gives the offset
// between the two clocks, per time window, to about a millisecond: the
// correlation does not involve the LiDAR registration, whose deskew depends
// on the very offset being estimated. The offsets of a recording are not a
// constant: they differ between recordings by tens of milliseconds and can
// drift within one (FusionPortableV2: 2-82 ms, up to 13 ms of drift in 400 s).

#ifndef GARAGE_RECON_GR_CLOCK_SYNC_H_
#define GARAGE_RECON_GR_CLOCK_SYNC_H_

#include <vector>

#include "gr/types.h"

namespace gr {

struct ClockSyncConfig {
  double window = 30.0;   // s of `other` samples per estimate.
  double max_lag = 0.15;  // s searched on either side of zero.
  double step = 0.001;    // s, grid step (refined by a parabola).
  // Both rates are compared as means over +-`smoothing` (from their
  // integrals): interpolating noisy rates directly favours lags that fall
  // between reference samples, where interpolation averages the noise.
  double smoothing = 0.01;
  // Windows whose 90th percentile rate is below `min_rate` (rad/s) carry no
  // timing information; windows with an RMS residual above `max_residual`
  // (rad/s) at the best lag are rejected.
  double min_rate = 0.1;
  double max_residual = 0.1;
  int min_samples = 200;
};

struct ClockSyncWindow {
  double center = 0.0;    // `other` clock (s).
  double lag = 0.0;       // s.
  double residual = 0.0;  // RMS rate residual at the lag (rad/s).
  int samples = 0;
};

// The `other` sample stamped t was taken at reference time t + lag(t), with
// lag(t) = offset + drift * (t - reference_time) fitted robustly to the
// windows (the windows themselves allow piecewise models).
struct ClockSyncResult {
  std::vector<ClockSyncWindow> windows;  // Accepted windows, in time order.
  double offset = 0.0;
  double drift = 0.0;  // s per s.
  double reference_time = 0.0;
  Mat3 rotation = Mat3::Identity();  // reference_frame <- other_frame.
  bool valid = false;
};

// `reference_stamps` must be ascending with matching `reference_rates`; the
// same for `other`. Rates in rad/s, each in its own sensor frame (the
// rotation between them is estimated per window).
ClockSyncResult EstimateClockOffset(const std::vector<double>& reference_stamps,
                                    const std::vector<Vec3>& reference_rates,
                                    const std::vector<double>& other_stamps,
                                    const std::vector<Vec3>& other_rates,
                                    const ClockSyncConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CLOCK_SYNC_H_
