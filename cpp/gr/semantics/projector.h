// Point-splatting z-buffer renderer and visibility test for any CameraModel
// (pinhole or fisheye) and exposure (global or rolling shutter). Depth is
// stored as Euclidean range so that fisheye rays beyond 90 degrees are handled
// consistently. PointCloudIndex keeps a static map bucketed in xy so that
// per-frame work only touches nearby points.

#ifndef GARAGE_RECON_GR_SEMANTICS_PROJECTOR_H_
#define GARAGE_RECON_GR_SEMANTICS_PROJECTOR_H_

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/core/voxel.h"
#include "gr/sensors/camera.h"
#include "gr/sensors/exposure.h"

namespace gr {

struct RenderResult {
  int width = 0;
  int height = 0;
  std::vector<float> depth;    // Range (m), +inf where empty; row-major H x W.
  std::vector<int32_t> label;  // -1 where empty.
  std::vector<int32_t> index;  // Index of the winning point, -1 where empty.
};

// Incremental z-buffer. The splat radius is clamp(round(focal *
// point_spacing / range), min_radius, max_radius); the nearest point wins
// (the first one on ties).
class ZBuffer {
 public:
  // `camera` and `exposure` must outlive the buffer.
  ZBuffer(const CameraModel& camera, const CameraExposure& exposure,
          double point_spacing, int min_radius, int max_radius,
          double min_range, double max_range);
  ZBuffer(const ZBuffer&) = delete;
  ZBuffer& operator=(const ZBuffer&) = delete;

  void Splat(const Vec3& point_world, int label, int index);
  const RenderResult& result() const { return result_; }
  RenderResult TakeResult() { return std::move(result_); }

 private:
  const CameraModel& camera_;
  const CameraExposure& exposure_;
  double point_spacing_;
  double min_range_;
  double max_range_;
  double focal_;
  int min_radius_;
  int max_radius_;
  RenderResult result_;
};

RenderResult RenderPoints(const CameraModel& camera,
                          const CameraExposure& exposure,
                          const Points& points_world,
                          const std::vector<int>& labels, double point_spacing,
                          int min_radius, int max_radius, double min_range,
                          double max_range);

// Range of the first surface per pixel, as the mean of the points in it.
// The z-buffer above keeps the nearest point of a pixel; on a surface whose
// points scatter by a few cm (noise, roughness) or that is seen at a grazing
// angle, that nearest point is biased towards the camera, and so are depth
// targets taken from it. Here the z-buffer (splats as in RenderPoints)
// decides which surface is first; the range is then the mean over the points
// within `abs_tol` + `rel_tol` x range behind the nearest one, each splatted
// over at most `mean_radius` pixels (NaN where no point falls).
std::vector<float> RenderSurfaceDepth(const CameraModel& camera,
                                      const CameraExposure& exposure,
                                      const Points& points_world,
                                      double point_spacing, int min_radius,
                                      int max_radius, double min_range,
                                      double max_range, double abs_tol,
                                      double rel_tol, int mean_radius);

struct VisibilityResult {
  RowMatX2d uv;  // NaN where the point does not project.
  std::vector<uint8_t> in_image;
  std::vector<uint8_t> visible;  // In the image and not occluded.
  std::vector<double> range;
};

// A query point is visible if it projects into the image and its range is at
// most depth(u, v) * (1 + rel_tol) + abs_tol. An empty depth buffer means "no
// occlusion test".
VisibilityResult ProjectVisibility(const CameraModel& camera,
                                   const CameraExposure& exposure,
                                   const Points& query_world,
                                   const std::vector<float>& depth,
                                   double abs_tol, double rel_tol);

class PointCloudIndex {
 public:
  explicit PointCloudIndex(Points points, double bucket = 4.0);
  PointCloudIndex(const PointCloudIndex&) = default;
  PointCloudIndex& operator=(const PointCloudIndex&) = default;
  PointCloudIndex(PointCloudIndex&&) = default;
  PointCloudIndex& operator=(PointCloudIndex&&) = default;

  const Points& points() const { return points_; }
  size_t size() const { return points_.size(); }
  // Indices (ascending) of the points closer than `radius` to `center`, in 3D
  // or in the xy plane.
  std::vector<int> RadiusSearch(const Vec3& center, double radius) const;
  std::vector<int> RadiusSearchXy(const Vec2& center, double radius) const;

 private:
  template <typename Accept>
  std::vector<int> Search(const Vec2& center, double radius,
                          Accept&& accept) const;

  Points points_;
  double bucket_;
  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> buckets_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_PROJECTOR_H_
