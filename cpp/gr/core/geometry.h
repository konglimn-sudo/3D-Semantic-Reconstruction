// Local geometry: radius-neighbourhood normals, RANSAC planes and the
// "plane thickness" crispness metric used as a ground-truth-free map QA
// signal.

#ifndef GARAGE_RECON_GR_CORE_GEOMETRY_H_
#define GARAGE_RECON_GR_CORE_GEOMETRY_H_

#include <vector>

#include "gr/core/types.h"

namespace gr {

// EstimateNormals uses at most this many neighbours per point by default.
inline constexpr int kDefaultMaxNormalNeighbors = 64;

struct NormalsResult {
  std::vector<Vec3> normals;      // Unit; zero if too few neighbours.
  std::vector<double> planarity;  // (l2 - l3) / l1, eigenvalues l1 >= l2 >= l3.
  std::vector<double> thickness;  // sqrt(l3): spread along the normal (m).
  std::vector<int> neighbors;     // Neighbours used per point.
  std::vector<Vec3> centroids;    // Mean of the neighbours.
};

// Normals from the neighbours within `radius` (the closest `max_neighbors`).
// Normals are not oriented; callers flip them towards the sensor if needed.
NormalsResult EstimateNormals(const Points& points, double radius,
                              int min_neighbors,
                              int max_neighbors = kDefaultMaxNormalNeighbors);

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

// The plane normal . x + offset = 0 and the points within the threshold.
struct PlaneFit {
  Vec3 normal = Vec3::UnitZ();
  double offset = 0.0;
  std::vector<int> inliers;
};

// RANSAC plane followed by a least-squares refit on the inliers: points
// closer than `threshold` to the plane are inliers.
PlaneFit FitPlaneRansac(const Points& points, double threshold, int iterations,
                        int seed = 1);

// Convex hull of points in the plane, counter-clockwise from the lowest-x
// (then lowest-y) point, without collinear points; fewer than three distinct
// points are returned as they are.
std::vector<Vec2> ConvexHull2D(std::vector<Vec2> points);

// The minimum-area rectangle around 2D points (as cv2.minAreaRect, by
// rotating calipers over the convex hull edges): its centre, the direction
// of its long side (rad, in (-pi/2, pi/2]) and its long and short sides.
struct RotatedRect {
  Vec2 center = Vec2::Zero();
  double yaw = 0.0;
  double length = 0.0;
  double width = 0.0;
};
RotatedRect MinAreaRect(const std::vector<Vec2>& points);

// Whether `point` lies inside the polygon (even-odd rule; points on the
// boundary may go either way).
bool PointInPolygon(const std::vector<Vec2>& polygon, const Vec2& point);

// Signed area of a polygon (positive counter-clockwise).
double PolygonArea(const std::vector<Vec2>& polygon);

// Intersection of two convex polygons (any orientation); empty if they do
// not overlap.
std::vector<Vec2> ClipConvex(const std::vector<Vec2>& subject,
                             const std::vector<Vec2>& clip);
// The area of ClipConvex(a, b).
double ConvexIntersectionArea(const std::vector<Vec2>& a,
                              const std::vector<Vec2>& b);

// A convex polygon shrunk by `distance` (every edge moved inwards, as a
// negative buffer); empty if nothing is left.
std::vector<Vec2> InsetConvex(const std::vector<Vec2>& polygon,
                              double distance);

struct ThicknessStats {
  int num_voxels = 0;  // The planar voxels measured.
  double mean = 0.0;
  double median = 0.0;
  double p90 = 0.0;  // 90th percentile.
};

// Splits the cloud into voxels of `voxel_size`, keeps those with at least
// `min_points` points that are planar (planarity >= `min_planarity`) and
// reports the distribution of their thickness.
ThicknessStats MeasurePlaneThickness(const Points& points, double voxel_size,
                                     int min_points, double min_planarity);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_GEOMETRY_H_
