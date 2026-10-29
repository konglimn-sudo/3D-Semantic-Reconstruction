// Voxel traversal (Amanatides & Woo) and dense free-space carving.

#ifndef GARAGE_RECON_GR_MAPPING_RAYCAST_H_
#define GARAGE_RECON_GR_MAPPING_RAYCAST_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "gr/core/types.h"

namespace gr {

// Walks the voxels of a grid of `size` voxels along a ray given in grid units
// (`origin`, unit `direction`) up to parameter `t_end`. `visit(x, y, z)` is
// called for the voxels inside the grid; returning false stops the walk.
template <typename Visit>
void TraverseGrid(const Vec3& origin, const Vec3& direction, double t_end,
                  const Eigen::Vector3i& size, Visit&& visit) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  int cell[kSpaceDim];
  int step[kSpaceDim];
  double t_max[kSpaceDim];    // Ray parameter of the next voxel boundary.
  double t_delta[kSpaceDim];  // Ray parameter between two boundaries.
  for (int axis = 0; axis < kSpaceDim; ++axis) {
    cell[axis] = static_cast<int>(std::floor(origin[axis]));
    if (direction[axis] > 0.0) {
      step[axis] = 1;
      t_max[axis] = (cell[axis] + 1 - origin[axis]) / direction[axis];
      t_delta[axis] = 1.0 / direction[axis];
    } else if (direction[axis] < 0.0) {
      step[axis] = -1;
      t_max[axis] = (cell[axis] - origin[axis]) / direction[axis];
      t_delta[axis] = -1.0 / direction[axis];
    } else {
      step[axis] = 0;
      t_max[axis] = kInf;
      t_delta[axis] = kInf;
    }
  }
  while (true) {
    bool inside = true;
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      inside = inside && cell[axis] >= 0 && cell[axis] < size[axis];
    }
    if (inside && !visit(cell[0], cell[1], cell[2])) return;
    int axis = 0;  // The axis whose boundary comes first.
    for (int other = 1; other < kSpaceDim; ++other) {
      if (t_max[other] < t_max[axis]) axis = other;
    }
    if (t_max[axis] > t_end) return;
    cell[axis] += step[axis];
    t_max[axis] += t_delta[axis];
  }
}

// Dense hit / miss statistics on a bounded grid. A voxel that holds points of
// one sweep but is traversed by rays of other sweeps is evidence of a dynamic
// (or since-removed) object.
class FreeSpaceCarver {
 public:
  // Grids larger than kMaxVoxels voxels are a precondition violation.
  static constexpr int64_t kMaxVoxels = 800000000;
  // Rays stop this far (m) before their endpoint, so that the surface they
  // hit is not carved.
  static constexpr double kDefaultEndpointMargin = 0.2;

  FreeSpaceCarver(const Vec3& min_bound, const Vec3& max_bound, double voxel,
                  double endpoint_margin = kDefaultEndpointMargin);
  FreeSpaceCarver(const FreeSpaceCarver&) = default;
  FreeSpaceCarver& operator=(const FreeSpaceCarver&) = default;

  // Number of voxels of the grid spanning the given bounds.
  static int64_t GridSize(const Vec3& min_bound, const Vec3& max_bound,
                          double voxel);

  // Integrates rays from `origin` to every endpoint (world frame).
  void Integrate(const Vec3& origin, const Points& endpoints);

  struct Counts {
    std::vector<int> hits;
    std::vector<int> misses;
  };
  // Hit / miss counters of the voxel containing each point (0 outside).
  Counts Query(const Points& points) const;

  // Dense flags over dims(), x-major: 1 where a voxel was traversed by at
  // least `min_misses` rays and holds at most `max_hits` endpoints.
  std::vector<uint8_t> FreeMask(int min_misses, int max_hits) const;
  const Vec3& min_bound() const { return min_bound_; }

  // Voxels along x, y and z.
  const Eigen::Vector3i& dims() const { return size_; }
  double voxel() const { return voxel_; }

 private:
  std::optional<Eigen::Vector3i> IndexOf(const Vec3& point) const;
  size_t Linear(int x, int y, int z) const {
    return (static_cast<size_t>(x) * static_cast<size_t>(size_.y()) +
            static_cast<size_t>(y)) *
               static_cast<size_t>(size_.z()) +
           static_cast<size_t>(z);
  }

  Vec3 min_bound_;
  double voxel_;
  double inverse_voxel_;
  double margin_;
  Eigen::Vector3i size_;
  std::vector<uint16_t> hits_;
  std::vector<uint16_t> misses_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_MAPPING_RAYCAST_H_
