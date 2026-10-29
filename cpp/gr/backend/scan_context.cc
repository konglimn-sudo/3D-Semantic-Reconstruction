#include "gr/backend/scan_context.h"

#include <algorithm>
#include <cmath>

namespace gr {
namespace {

constexpr double kFullTurn = 2.0 * kPi;
// Points this close (m) to the sensor axis have no azimuth.
constexpr double kMinRadius = 1e-6;
// Columns with a smaller norm are empty.
constexpr double kMinColumnNorm = 1e-9;

}  // namespace

Eigen::MatrixXd ComputeScanContext(const Points& points,
                                   const ScanContextConfig& config) {
  Eigen::MatrixXd descriptor =
      Eigen::MatrixXd::Zero(config.rings, config.sectors);
  for (const Vec3& point : points) {
    const double radius = std::hypot(point.x(), point.y());
    if (radius >= config.max_radius || radius < kMinRadius) continue;
    const double theta = std::atan2(point.y(), point.x()) + kPi;  // [0, 2 pi]
    const int ring =
        std::min(config.rings - 1,
                 static_cast<int>(radius / config.max_radius * config.rings));
    const int sector =
        std::min(config.sectors - 1,
                 static_cast<int>(theta / (2.0 * kPi) * config.sectors));
    const double height = std::max(0.0, point.z() + config.lidar_height);
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
    for (int column = 0; column < sectors; ++column) {
      // roll(b, shift)[:, column] == b[:, column - shift]
      const int b_column = ((column - shift) % sectors + sectors) % sectors;
      if (norm_a[column] < kMinColumnNorm ||
          norm_b[b_column] < kMinColumnNorm) {
        continue;
      }
      sum += a.col(column).dot(b.col(b_column)) /
             (norm_a[column] * norm_b[b_column]);
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
  if (yaw > kPi) yaw -= kFullTurn;
  best.yaw = yaw;
  return best;
}

}  // namespace gr
