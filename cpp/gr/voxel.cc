#include "gr/voxel.h"

#include <unordered_map>

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
  const double inv = 1.0 / voxel_size;
  std::unordered_map<VoxelKey, int, VoxelKeyHash> seen;
  seen.reserve(points.size() / 2 + 1);
  indices.reserve(points.size() / 2 + 1);
  for (size_t i = 0; i < points.size(); ++i) {
    if (seen.emplace(VoxelOf(points[i], inv), static_cast<int>(i)).second) {
      indices.push_back(static_cast<int>(i));
    }
  }
  return indices;
}

Points VoxelDownsample(const Points& points, double voxel_size) {
  const std::vector<int> indices = VoxelDownsampleIndices(points, voxel_size);
  Points out;
  out.reserve(indices.size());
  for (int i : indices) out.push_back(points[static_cast<size_t>(i)]);
  return out;
}

}  // namespace gr
