#include "gr/mapping/raycast.h"

#include <algorithm>
#include <limits>

#include "gr/core/check.h"

namespace gr {
namespace {

// Rays shorter than this (voxels) have no direction.
constexpr double kMinRayLength = 1e-9;
// Counters saturate.
constexpr uint16_t kMaxCount = std::numeric_limits<uint16_t>::max();

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
      inverse_voxel_(1.0 / voxel),
      margin_(endpoint_margin) {
  const Vec3 extent = (max_bound - min_bound) * inverse_voxel_;
  size_ = Eigen::Vector3i(CellsAlong(extent.x()), CellsAlong(extent.y()),
                          CellsAlong(extent.z()));
  const int64_t voxels =
      static_cast<int64_t>(size_.x()) * size_.y() * size_.z();
  GR_CHECK(voxels <= kMaxVoxels);
  hits_.assign(static_cast<size_t>(voxels), 0);
  misses_.assign(static_cast<size_t>(voxels), 0);
}

std::optional<Eigen::Vector3i> FreeSpaceCarver::IndexOf(
    const Vec3& point) const {
  const Vec3 grid_position = (point - min_bound_) * inverse_voxel_;
  const Eigen::Vector3i cell(static_cast<int>(std::floor(grid_position.x())),
                             static_cast<int>(std::floor(grid_position.y())),
                             static_cast<int>(std::floor(grid_position.z())));
  if (cell.x() < 0 || cell.y() < 0 || cell.z() < 0 || cell.x() >= size_.x() ||
      cell.y() >= size_.y() || cell.z() >= size_.z()) {
    return std::nullopt;
  }
  return cell;
}

void FreeSpaceCarver::Integrate(const Vec3& origin, const Points& endpoints) {
  const Vec3 origin_grid = (origin - min_bound_) * inverse_voxel_;
  for (const Vec3& endpoint : endpoints) {
    const Vec3 delta = (endpoint - min_bound_) * inverse_voxel_ - origin_grid;
    const double length = delta.norm();
    if (length < kMinRayLength) continue;
    const double t_end = length - margin_ * inverse_voxel_;
    if (t_end > 0.0) {
      TraverseGrid(origin_grid, delta / length, t_end, size_,
                   [&](int x, int y, int z) {
                     uint16_t& misses = misses_[Linear(x, y, z)];
                     if (misses < kMaxCount) ++misses;
                     return true;
                   });
    }
    if (const auto cell = IndexOf(endpoint)) {
      uint16_t& hits = hits_[Linear(cell->x(), cell->y(), cell->z())];
      if (hits < kMaxCount) ++hits;
    }
  }
}

FreeSpaceCarver::Counts FreeSpaceCarver::Query(const Points& points) const {
  Counts counts;
  counts.hits.assign(points.size(), 0);
  counts.misses.assign(points.size(), 0);
  for (size_t index = 0; index < points.size(); ++index) {
    const auto cell = IndexOf(points[index]);
    if (!cell) continue;
    const size_t voxel = Linear(cell->x(), cell->y(), cell->z());
    counts.hits[index] = hits_[voxel];
    counts.misses[index] = misses_[voxel];
  }
  return counts;
}

std::vector<uint8_t> FreeSpaceCarver::FreeMask(int min_misses,
                                               int max_hits) const {
  std::vector<uint8_t> free(misses_.size(), 0);
  for (size_t voxel = 0; voxel < misses_.size(); ++voxel) {
    free[voxel] =
        misses_[voxel] >= min_misses && hits_[voxel] <= max_hits ? 1 : 0;
  }
  return free;
}

}  // namespace gr
