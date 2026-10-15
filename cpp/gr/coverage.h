// Camera coverage of the LiDAR map and training masks of the 3DGS stage
// (plan section 3.4).
//
// CameraCoverage z-buffers the map in every camera frame and counts, per
// point, the views that see it unoccluded on a usable pixel (neither dark nor
// saturated, not masked). Points with too few such views cannot learn
// appearance; their Gaussians keep the LiDAR geometry and are frozen. The
// same pass gives a view-weighted mean colour, the initial colour of the
// Gaussians, and the best ground sample distance. Without an image every
// visible point counts, which evaluates a planned camera rig on an existing
// trajectory.
//
// RenderTrainingMask marks the pixels that must not supervise the static
// Gaussians: moving objects (points and boxes) and, when changed objects are
// masked rather than trained per trip, the objects that changed between
// trips.

#ifndef GARAGE_RECON_GR_COVERAGE_H_
#define GARAGE_RECON_GR_COVERAGE_H_

#include <cstdint>
#include <vector>

#include "gr/camera.h"
#include "gr/exposure.h"
#include "gr/image.h"
#include "gr/objects.h"
#include "gr/projector.h"
#include "gr/types.h"

namespace gr {

struct CoverageConfig {
  double max_range = 40.0;
  double point_spacing = 0.05;  // Splat size of the map points (m).
  int min_radius = 1;
  int max_radius = 3;
  double depth_abs_tol = 0.05;
  double depth_rel_tol = 0.02;
  int dark = 15;        // Usable pixels: dark < luminance < saturated.
  int saturated = 245;
  double min_cos = 0.1;  // Grazing views below this cosine do not count.
};

struct CoverageResult {
  std::vector<int> views;       // Unoccluded views in the image.
  std::vector<int> good_views;  // ... on usable, unmasked pixels.
  std::vector<float> best_gsd;  // Best metres per pixel (+inf if never).
  std::vector<Eigen::Vector3f> color;  // Weighted mean RGB in [0, 1].
  std::vector<float> color_weight;     // Sum of the colour weights.
};

class CameraCoverage {
 public:
  // `normals` may be empty (no incidence test).
  CameraCoverage(Points points, Points normals,
                 const CoverageConfig& config = {});

  // One frame. `image` (RGB) may be empty; `mask` (H x W, nonzero =
  // excluded) may be null.
  void AddFrame(const CameraModel& camera, const CameraExposure& exposure,
                const ImageView& image, const uint8_t* mask);

  CoverageResult Result() const;
  int frames() const { return frames_; }

 private:
  PointCloudIndex index_;
  Points normals_;
  CoverageConfig config_;
  int frames_ = 0;
  std::vector<int> views_;
  std::vector<int> good_views_;
  std::vector<float> best_gsd_;
  std::vector<Eigen::Vector3d> color_sum_;
  std::vector<double> weight_sum_;
};

struct TrainingMaskConfig {
  double max_range = 40.0;
  double point_spacing = 0.05;
  int min_radius = 1;
  int max_radius = 4;
  double box_step = 0.05;  // Samples of the box surfaces (m).
  int dilate = 4;          // Pixels.
};

// Bits of a training-mask pixel.
enum TrainingMaskBit : uint8_t {
  kMaskMoving = 1,   // Moving object (point or box).
  kMaskChanged = 2,  // Object that changed between trips.
};

// H x W mask of one frame. The static points occlude; `moving` and `changed`
// points and the surfaces of `boxes` set their bit where they are the
// nearest surface. Each bit is dilated by `dilate` pixels.
std::vector<uint8_t> RenderTrainingMask(const CameraModel& camera,
                                        const CameraExposure& exposure,
                                        const PointCloudIndex& static_points,
                                        const Points& moving,
                                        const Points& changed,
                                        const std::vector<ObjectBox>& boxes,
                                        const TrainingMaskConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_COVERAGE_H_
