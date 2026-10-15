// Voxel traversal (Amanatides & Woo) and dense free-space carving.

#ifndef GARAGE_RECON_GR_RAYCAST_H_
#define GARAGE_RECON_GR_RAYCAST_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "gr/types.h"

namespace gr {

// Walks the voxels of an (nx, ny, nz) grid along a ray given in grid units
// (origin `o`, unit direction `d`) up to parameter `t_end`. `visit(i, j, k)` is
// called for voxels inside the grid; returning false stops the walk.
template <typename Visit>
void TraverseGrid(const Vec3& o, const Vec3& d, double t_end, int nx, int ny,
                  int nz, Visit&& visit) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  int ijk[3] = {static_cast<int>(std::floor(o.x())),
                static_cast<int>(std::floor(o.y())),
                static_cast<int>(std::floor(o.z()))};
  const int dims[3] = {nx, ny, nz};
  int step[3];
  double t_max[3];
  double t_delta[3];
  for (int a = 0; a < 3; ++a) {
    if (d[a] > 0.0) {
      step[a] = 1;
      t_max[a] = (ijk[a] + 1 - o[a]) / d[a];
      t_delta[a] = 1.0 / d[a];
    } else if (d[a] < 0.0) {
      step[a] = -1;
      t_max[a] = (ijk[a] - o[a]) / d[a];
      t_delta[a] = -1.0 / d[a];
    } else {
      step[a] = 0;
      t_max[a] = kInf;
      t_delta[a] = kInf;
    }
  }
  while (true) {
    const bool inside = ijk[0] >= 0 && ijk[1] >= 0 && ijk[2] >= 0 &&
                        ijk[0] < dims[0] && ijk[1] < dims[1] &&
                        ijk[2] < dims[2];
    if (inside && !visit(ijk[0], ijk[1], ijk[2])) return;
    int axis = 0;
    if (t_max[1] < t_max[axis]) axis = 1;
    if (t_max[2] < t_max[axis]) axis = 2;
    if (t_max[axis] > t_end) return;
    ijk[axis] += step[axis];
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

  FreeSpaceCarver(const Vec3& min_bound, const Vec3& max_bound, double voxel,
                  double endpoint_margin = 0.2);
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

  // Dense (nx, ny, nz) flags, x-major: 1 where a voxel was traversed by at
  // least `min_misses` rays and holds at most `max_hits` endpoints.
  std::vector<uint8_t> FreeMask(int min_misses, int max_hits) const;
  const Vec3& min_bound() const { return min_bound_; }

  Eigen::Vector3i dims() const { return {nx_, ny_, nz_}; }
  double voxel() const { return voxel_; }

 private:
  std::optional<Eigen::Vector3i> IndexOf(const Vec3& point) const;
  size_t Linear(int i, int j, int k) const {
    return (static_cast<size_t>(i) * static_cast<size_t>(ny_) +
            static_cast<size_t>(j)) *
               static_cast<size_t>(nz_) +
           static_cast<size_t>(k);
  }

  Vec3 min_bound_;
  double voxel_;
  double inv_voxel_;
  double margin_;
  int nx_;
  int ny_;
  int nz_;
  std::vector<uint16_t> hits_;
  std::vector<uint16_t> misses_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_RAYCAST_H_
