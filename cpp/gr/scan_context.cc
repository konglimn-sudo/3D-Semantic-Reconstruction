#include "gr/scan_context.h"

#include <algorithm>
#include <cmath>

namespace gr {

Eigen::MatrixXd ComputeScanContext(const Points& points,
                                   const ScanContextConfig& config) {
  Eigen::MatrixXd descriptor =
      Eigen::MatrixXd::Zero(config.rings, config.sectors);
  for (const Vec3& p : points) {
    const double r = std::hypot(p.x(), p.y());
    if (r >= config.max_radius || r < 1e-6) continue;
    const double theta = std::atan2(p.y(), p.x()) + kPi;  // [0, 2 pi]
    const int ring =
        std::min(config.rings - 1,
                 static_cast<int>(r / config.max_radius * config.rings));
    const int sector =
        std::min(config.sectors - 1,
                 static_cast<int>(theta / (2.0 * kPi) * config.sectors));
    const double height = std::max(0.0, p.z() + config.lidar_height);
    descriptor(ring, sector) = std::max(descriptor(ring, sector), height);
  }
  return descriptor;
}

Eigen::VectorXd ScanContextRingKey(const Eigen::MatrixXd& descriptor) {
  return descriptor.rowwise().mean();
}

ScanContextMatch MatchScanContexts(const Eigen::MatrixXd& a,
                                   const Eigen::MatrixXd& b) {
  ScanContextMatch best;
  if (a.rows() != b.rows() || a.cols() != b.cols() || a.cols() == 0) {
    return best;
  }
  const int sectors = static_cast<int>(a.cols());
  const Eigen::VectorXd norm_a = a.colwise().norm().transpose();
  const Eigen::VectorXd norm_b = b.colwise().norm().transpose();
  for (int shift = 0; shift < sectors; ++shift) {
    double sum = 0.0;
    int valid = 0;
    for (int c = 0; c < sectors; ++c) {
      // roll(b, shift)[:, c] == b[:, c - shift]
      const int cb = ((c - shift) % sectors + sectors) % sectors;
      if (norm_a[c] < 1e-9 || norm_b[cb] < 1e-9) continue;
      sum += a.col(c).dot(b.col(cb)) / (norm_a[c] * norm_b[cb]);
      ++valid;
    }
    if (valid == 0) continue;
    const double distance = 1.0 - sum / valid;
    if (distance < best.distance) {
      best.distance = distance;
      best.shift = shift;
    }
  }
  double yaw = best.shift * 2.0 * kPi / sectors;
  if (yaw > kPi) yaw -= 2.0 * kPi;
  best.yaw = yaw;
  return best;
}

}  // namespace gr
