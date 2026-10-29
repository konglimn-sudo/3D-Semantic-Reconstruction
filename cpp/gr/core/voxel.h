// Voxel and xy-cell hashing, and voxel downsampling.

#ifndef GARAGE_RECON_GR_CORE_VOXEL_H_
#define GARAGE_RECON_GR_CORE_VOXEL_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gr/core/types.h"

namespace gr {
namespace internal {

// Primes of the spatial hash of Teschner et al., "Optimized Spatial Hashing
// for Collision Detection of Deformable Objects" (2003), one per axis.
inline constexpr size_t kHashPrimeX = 73856093u;
inline constexpr size_t kHashPrimeY = 19349663u;
inline constexpr size_t kHashPrimeZ = 83492791u;

// A grid coordinate as the unsigned value the hash multiplies.
inline size_t HashCoordinate(int coordinate) {
  return static_cast<size_t>(static_cast<uint32_t>(coordinate));
}

}  // namespace internal

// A voxel of a 3D grid.
struct VoxelKey {
  int x = 0;
  int y = 0;
  int z = 0;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  size_t operator()(const VoxelKey& key) const {
    return (internal::HashCoordinate(key.x) * internal::kHashPrimeX) ^
           (internal::HashCoordinate(key.y) * internal::kHashPrimeY) ^
           (internal::HashCoordinate(key.z) * internal::kHashPrimeZ);
  }
};

// Voxel containing `point` for voxels of size 1 / `inverse_size`.
inline VoxelKey VoxelOf(const Vec3& point, double inverse_size) {
  return {static_cast<int>(std::floor(point.x() * inverse_size)),
          static_cast<int>(std::floor(point.y() * inverse_size)),
          static_cast<int>(std::floor(point.z() * inverse_size))};
}

// A cell of a 2D grid in the xy plane.
struct CellKey {
  int x = 0;
  int y = 0;

  bool operator==(const CellKey& other) const {
    return x == other.x && y == other.y;
  }
};

struct CellKeyHash {
  size_t operator()(const CellKey& key) const {
    return (internal::HashCoordinate(key.x) * internal::kHashPrimeX) ^
           (internal::HashCoordinate(key.y) * internal::kHashPrimeY);
  }
};

// xy cell of size `cell_size` containing `point`.
inline CellKey CellOf(const Vec3& point, double cell_size) {
  return {static_cast<int>(std::floor(point.x() / cell_size)),
          static_cast<int>(std::floor(point.y() / cell_size))};
}

// Index of the first point that falls into every occupied voxel, in input
// order. A non-positive voxel size keeps every point.
std::vector<int> VoxelDownsampleIndices(const Points& points,
                                        double voxel_size);

// The points VoxelDownsampleIndices keeps.
Points VoxelDownsample(const Points& points, double voxel_size);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_VOXEL_H_
