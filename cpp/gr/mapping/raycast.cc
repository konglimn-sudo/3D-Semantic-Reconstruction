#include "gr/mapping/raycast.h"

#include <algorithm>

#include "gr/core/check.h"

namespace gr {
namespace {

int CellsAlong(double extent) {
  return std::max(1, static_cast<int>(std::ceil(extent)));
}

}  // namespace

int64_t FreeSpaceCarver::GridSize(const Vec3& min_bound, const Vec3& max_bound,
                                  double voxel) {
  const Vec3 extent = (max_bound - min_bound) * (1.0 / voxel);
  return static_cast<int64_t>(CellsAlong(extent.x())) * CellsAlong(extent.y()) *
         CellsAlong(extent.z());
}

FreeSpaceCarver::FreeSpaceCarver(const Vec3& min_bound, const Vec3& max_bound,
                                 double voxel, double endpoint_margin)
    : min_bound_(min_bound),
      voxel_(voxel),
      inv_voxel_(1.0 / voxel),
      margin_(endpoint_margin) {
  const Vec3 extent = (max_bound - min_bound) * inv_voxel_;
  nx_ = CellsAlong(extent.x());
  ny_ = CellsAlong(extent.y());
  nz_ = CellsAlong(extent.z());
  const int64_t n = static_cast<int64_t>(nx_) * ny_ * nz_;
  GR_CHECK(n <= kMaxVoxels);
  hits_.assign(static_cast<size_t>(n), 0);
  misses_.assign(static_cast<size_t>(n), 0);
}

std::optional<Eigen::Vector3i> FreeSpaceCarver::IndexOf(
    const Vec3& point) const {
  const Vec3 g = (point - min_bound_) * inv_voxel_;
  const Eigen::Vector3i ijk(static_cast<int>(std::floor(g.x())),
                            static_cast<int>(std::floor(g.y())),
                            static_cast<int>(std::floor(g.z())));
  if (ijk.x() < 0 || ijk.y() < 0 || ijk.z() < 0 || ijk.x() >= nx_ ||
      ijk.y() >= ny_ || ijk.z() >= nz_) {
    return std::nullopt;
  }
  return ijk;
}

void FreeSpaceCarver::Integrate(const Vec3& origin, const Points& endpoints) {
  const Vec3 origin_grid = (origin - min_bound_) * inv_voxel_;
  for (const Vec3& p : endpoints) {
    const Vec3 delta = (p - min_bound_) * inv_voxel_ - origin_grid;
    const double length = delta.norm();
    if (length < 1e-9) continue;
    const double t_end = length - margin_ * inv_voxel_;
    if (t_end > 0.0) {
      TraverseGrid(origin_grid, delta / length, t_end, nx_, ny_, nz_,
                   [&](int i, int j, int k) {
                     uint16_t& misses = misses_[Linear(i, j, k)];
                     if (misses < 65535) ++misses;
                     return true;
                   });
    }
    if (const auto ijk = IndexOf(p)) {
      uint16_t& hits = hits_[Linear(ijk->x(), ijk->y(), ijk->z())];
      if (hits < 65535) ++hits;
    }
  }
}

FreeSpaceCarver::Counts FreeSpaceCarver::Query(const Points& points) const {
  Counts counts;
  counts.hits.assign(points.size(), 0);
  counts.misses.assign(points.size(), 0);
  for (size_t n = 0; n < points.size(); ++n) {
    const auto ijk = IndexOf(points[n]);
    if (!ijk) continue;
    const size_t cell = Linear(ijk->x(), ijk->y(), ijk->z());
    counts.hits[n] = hits_[cell];
    counts.misses[n] = misses_[cell];
  }
  return counts;
}

std::vector<uint8_t> FreeSpaceCarver::FreeMask(int min_misses,
                                               int max_hits) const {
  std::vector<uint8_t> free(misses_.size(), 0);
  for (size_t c = 0; c < misses_.size(); ++c) {
    free[c] = misses_[c] >= min_misses && hits_[c] <= max_hits ? 1 : 0;
  }
  return free;
}

}  // namespace gr
