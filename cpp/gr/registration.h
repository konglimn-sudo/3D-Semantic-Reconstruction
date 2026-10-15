// Voxel hash map with nearest-neighbour / local-plane queries and robust ICP
// (point-to-plane with point-to-point fallback, Geman-McClure kernel, left
// perturbation on R^3 x SO(3)).

#ifndef GARAGE_RECON_GR_REGISTRATION_H_
#define GARAGE_RECON_GR_REGISTRATION_H_

#include <optional>
#include <unordered_map>
#include <vector>

#include "gr/pose.h"
#include "gr/types.h"
#include "gr/voxel.h"

namespace gr {

class VoxelHashMap {
 public:
  explicit VoxelHashMap(double voxel_size, double max_distance = 1e9,
                        int max_points_per_voxel = 20);
  VoxelHashMap(const VoxelHashMap&) = default;
  VoxelHashMap& operator=(const VoxelHashMap&) = default;
  VoxelHashMap(VoxelHashMap&&) = default;
  VoxelHashMap& operator=(VoxelHashMap&&) = default;

  struct Neighbor {
    Vec3 point;
    double distance_squared;
  };

  // Principal components of a neighbourhood.
  struct LocalPlane {
    Vec3 centroid;
    Vec3 normal;       // Unit.
    double planarity;  // (l2 - l3) / l1, eigenvalues l1 >= l2 >= l3.
    double linearity;  // (l1 - l2) / l1
    double thickness;  // sqrt(l3)
  };

  void Clear() { map_.clear(); }
  bool empty() const { return map_.empty(); }
  size_t num_voxels() const { return map_.size(); }
  size_t CountPoints() const;
  double voxel_size() const { return voxel_size_; }

  // Adds world-frame points; `frame_id` lets scan-to-map alignment ignore a
  // frame's own points.
  void AddPoints(const Points& points, int frame_id = -1);
  // Drops voxels farther than `max_distance` from `origin`.
  void RemoveFar(const Vec3& origin);
  Points PointCloud() const;

  // Nearest map point within `max_distance` of `query`, searching at most
  // `max_search_voxels` voxels around it in each direction (KISS-ICP searches
  // only the 27-neighbourhood); a huge gate is clipped.
  std::optional<Neighbor> Nearest(const Vec3& query, double max_distance,
                                  int exclude_frame = -1,
                                  int max_search_voxels = 2) const;

  // PCA of the map points within `radius` of `query`; nullopt with fewer than
  // `min_points` neighbours.
  std::optional<LocalPlane> FitLocalPlane(const Vec3& query, double radius,
                                          int min_points,
                                          int exclude_frame = -1) const;

 private:
  struct Block {
    std::vector<Vec3> points;
    std::vector<int> frames;
  };

  double voxel_size_;
  double inv_voxel_size_;
  double max_distance_;
  int max_points_per_voxel_;
  std::unordered_map<VoxelKey, Block, VoxelKeyHash> map_;
};

struct IcpOptions {
  double max_corr_dist = 1.0;  // Correspondence gate (m).
  double kernel = 0.3;         // Geman-McClure scale (m).
  int max_iters = 50;
  double conv_eps = 1e-4;
  int exclude_frame = -1;
  // Point-to-plane residuals where the map is locally planar; this removes
  // the bias of point-to-point matching towards zero motion on sparse LiDAR
  // rings (floors, ceilings, walls). Non-planar neighbourhoods fall back to
  // point-to-point.
  bool point_to_plane = true;
  double plane_radius = 0.0;  // <= 0: the map voxel size.
  // The radius doubles up to this until a plane spans several rings.
  double max_plane_radius = 1.2;
  int min_plane_points = 6;
  double min_planarity = 0.5;
  double max_plane_thickness = 0.05;
  // Neighbourhoods that stay line-like (a lone ring segment) at the largest
  // radius carry no unbiased constraint and are skipped.
  double max_linearity = 0.85;
  // Weight of point-to-point residuals used where no plane is found
  // (corners, clutter); 0 makes the alignment purely point-to-plane.
  double point_weight = 0.05;
  int max_search_voxels = 2;  // Neighbour-search window (voxels).
  // Parallelise over source points; callers that already run many alignments
  // in parallel (loop verification, scan-to-map refinement) switch it off.
  bool parallel = true;
};

struct IcpResult {
  Pose pose;             // world_from_source
  double rmse = 0.0;     // Over inliers (residual < max_corr_dist).
  double fitness = 0.0;  // Inliers / source points.
  int inliers = 0;
  int iterations = 0;
  bool converged = false;
  Mat6 hessian = Mat6::Zero();  // Sum of J^T J over inliers, order [t, phi].
};

// Aligns `source` (in the frame of `initial_guess`) to the map.
IcpResult AlignToMap(const Points& source, const VoxelHashMap& map,
                     const Pose& initial_guess, const IcpOptions& options);

// Coarse-to-fine ICP of `source` against the cloud `target`: one alignment
// per correspondence gate of `schedule` (kernel = gate / 3; the map voxel
// follows the gate so that the neighbour search stays local), then the
// statistics of the full-resolution source at the finest gate.
IcpResult AlignMultiScale(const Points& source, const Points& target,
                          const Pose& initial_guess,
                          const std::vector<double>& schedule,
                          double base_voxel, int max_iterations = 40,
                          bool parallel = true);

}  // namespace gr

#endif  // GARAGE_RECON_GR_REGISTRATION_H_
