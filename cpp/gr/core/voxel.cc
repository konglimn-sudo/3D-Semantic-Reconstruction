#include "gr/core/voxel.h"

#include <unordered_set>

namespace gr {

std::vector<int> VoxelDownsampleIndices(const Points& points,
                                        double voxel_size) {
  std::vector<int> indices;
  if (points.empty() || voxel_size <= 0.0) {
    indices.resize(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
      indices[i] = static_cast<int>(i);
    }
    return indices;
  }
  // Room for one voxel per two points; more voxels grow the containers.
  constexpr size_t kPointsPerVoxelGuess = 2;
  const double inverse_voxel_size = 1.0 / voxel_size;
  std::unordered_set<VoxelKey, VoxelKeyHash> occupied;
  occupied.reserve(points.size() / kPointsPerVoxelGuess + 1);
  indices.reserve(points.size() / kPointsPerVoxelGuess + 1);
  for (size_t i = 0; i < points.size(); ++i) {
    if (occupied.insert(VoxelOf(points[i], inverse_voxel_size)).second) {
      indices.push_back(static_cast<int>(i));
    }
  }
  return indices;
}

Points VoxelDownsample(const Points& points, double voxel_size) {
  const std::vector<int> indices = VoxelDownsampleIndices(points, voxel_size);
  Points out;
  out.reserve(indices.size());
  for (int index : indices) out.push_back(points[static_cast<size_t>(index)]);
  return out;
}

}  // namespace gr
