// Local geometry: radius-neighbourhood normals, RANSAC planes and the
// "plane thickness" crispness metric used as a ground-truth-free map QA
// signal.

#ifndef GARAGE_RECON_GR_GEOMETRY_H_
#define GARAGE_RECON_GR_GEOMETRY_H_

#include <vector>

#include "gr/types.h"

namespace gr {

struct NormalsResult {
  std::vector<Vec3> normals;      // Unit; zero if too few neighbours.
  std::vector<double> planarity;  // (l2 - l3) / l1, eigenvalues l1 >= l2 >= l3.
  std::vector<double> thickness;  // sqrt(l3): spread along the normal (m).
  std::vector<int> neighbors;
  std::vector<Vec3> centroids;  // Mean of the neighbours.
};

// Normals from the neighbours within `radius` (the closest `max_neighbors`).
// Normals are not oriented; callers flip them towards the sensor if needed.
NormalsResult EstimateNormals(const Points& points, double radius,
                              int min_neighbors, int max_neighbors = 64);

// Each point moved along its local normal onto the plane of its neighbours
// within `radius` (moving least squares, one step), where the neighbourhood
// is planar (planarity >= `min_planarity`) and the move is at most
// `max_shift`; other points stay. A rendering of flat discs on a noisy layer
// of points stops at the front of the noise, so the anchors of the 3DGS start
// from the denoised surface. `shift` (optional) receives the signed moves.
Points ProjectToLocalPlanes(const Points& points, double radius,
                            int min_neighbors, int max_neighbors,
                            double min_planarity, double max_shift,
                            std::vector<double>* shift = nullptr);

struct PlaneFit {
  Vec3 normal = Vec3::UnitZ();
  double d = 0.0;  // normal . x + d = 0
  std::vector<int> inliers;
};

// RANSAC plane followed by a least-squares refit on the inliers.
PlaneFit FitPlaneRansac(const Points& points, double threshold, int iterations,
                        int seed = 1);

struct ThicknessStats {
  int num_voxels = 0;
  double mean = 0.0;
  double median = 0.0;
  double p90 = 0.0;
};

// Splits the cloud into voxels, keeps the planar ones (planarity >=
// `min_planarity`) and reports the distribution of their thickness.
ThicknessStats MeasurePlaneThickness(const Points& points, double voxel,
                                     int min_points, double min_planarity);

}  // namespace gr

#endif  // GARAGE_RECON_GR_GEOMETRY_H_
