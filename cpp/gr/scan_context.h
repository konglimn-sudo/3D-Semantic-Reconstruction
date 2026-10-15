// Scan Context place descriptor (Kim & Kim, IROS 2018): polar grid of maximum
// point heights. Used to rank loop-closure candidates; verification is done by
// ICP afterwards.

#ifndef GARAGE_RECON_GR_SCAN_CONTEXT_H_
#define GARAGE_RECON_GR_SCAN_CONTEXT_H_

#include "gr/types.h"

namespace gr {

struct ScanContextConfig {
  int rings = 20;
  int sectors = 60;
  double max_radius = 30.0;
  double lidar_height = 2.0;  // Added to z so that bin values stay positive.
};

// rings x sectors matrix; empty bins are 0.
Eigen::MatrixXd ComputeScanContext(const Points& points,
                                   const ScanContextConfig& config);
// Rotation-invariant ring key (mean of each ring).
Eigen::VectorXd ScanContextRingKey(const Eigen::MatrixXd& descriptor);

struct ScanContextMatch {
  double distance = 1.0;  // 0 = identical, 1 = unrelated.
  int shift = 0;          // Columns: a ~= roll(b, shift).
  double yaw = 0.0;       // Yaw of b's frame relative to a's (rad, (-pi, pi]).
};

ScanContextMatch MatchScanContexts(const Eigen::MatrixXd& a,
                                   const Eigen::MatrixXd& b);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SCAN_CONTEXT_H_
