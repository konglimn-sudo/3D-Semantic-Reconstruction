// Changes of the permanent structure between the map and a new trip
// localized in it (map updates: construction fences, removed ducts, new
// walls).
//
// Both are voxelized (`voxel`). From the new trip's sweeps (map frame, with
// their sensor positions):
//
//   added     voxels hit in at least `min_hits` sweeps spread over at least
//             `min_hit_span` seconds (a passing car stays in a voxel for a
//             few sweeps only) that are not in the map and have no map
//             voxel among their 26 neighbours (the registration's
//             tolerance);
//   removed   map voxels the new rays pass through (stopping `free_margin`
//             short of their returns) in at least `min_free` sweeps, with
//             no voxel hit by the new trip among them and their neighbours.
//
// Changed voxels are clustered (26-connected). Clusters of fewer than
// `min_cluster_voxels` are noise; clusters at most `flat_thickness` high
// are floor (seen for the first time under a car parked during the mapping,
// or grazed by low rays); clusters with a parked car's footprint and height
// are occupancy, not structure (`vehicle_like`).

#ifndef GARAGE_RECON_GR_MAPPING_CHANGE_DETECTION_H_
#define GARAGE_RECON_GR_MAPPING_CHANGE_DETECTION_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gr/core/types.h"
#include "gr/core/voxel.h"

namespace gr {

struct ChangeConfig {
  double voxel = 0.2;
  double max_range = 20.0;  // Returns used (m).
  double ray_voxel = 0.3;   // Sweeps downsampled for the rays (m).
  int min_hits = 3;
  double min_hit_span = 5.0;  // s
  int min_free = 3;
  double free_margin = 0.4;  // m
  int min_cluster_voxels = 12;
  double flat_thickness = 0.25;  // m
  // Parked-car clusters: footprint length / width and height range (m).
  double vehicle_min_length = 3.0;
  double vehicle_max_length = 6.0;
  double vehicle_min_width = 1.3;
  double vehicle_max_width = 2.6;
  double vehicle_max_height = 2.4;
};

struct ChangeCluster {
  bool added = true;  // Else removed.
  Vec3 min = Vec3::Zero();
  Vec3 max = Vec3::Zero();
  Vec3 center = Vec3::Zero();
  int voxels = 0;
  bool vehicle_like = false;
};

class ChangeDetector {
 public:
  ChangeDetector(const Points& map_points, const ChangeConfig& config);

  // One localized sweep in the map frame, its sensor position and stamp.
  void AddSweep(const Points& sweep, const Vec3& sensor, double stamp);

  // Clusters of the changes so far, largest first.
  std::vector<ChangeCluster> Clusters() const;
  int sweeps() const { return sweeps_; }

 private:
  ChangeConfig config_;
  double inv_;
  std::unordered_set<VoxelKey, VoxelKeyHash> map_;
  struct Hits {
    int sweeps = 0;
    double first = 0.0;
    double last = 0.0;
  };
  // Per voxel: sweeps that hit it (new trip) / passed through it (map).
  std::unordered_map<VoxelKey, Hits, VoxelKeyHash> hits_;
  std::unordered_map<VoxelKey, int, VoxelKeyHash> free_;
  int sweeps_ = 0;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_MAPPING_CHANGE_DETECTION_H_
