// Labels by projection (section 5): the 3D map is labelled once and every
// camera frame inherits consistent labels (FrameLabeler); conversely, any
// per-pixel 2D result is lifted onto the map by multi-view voting
// (LabelLifter).

#ifndef GARAGE_RECON_GR_SEMANTICS_LABELING_H_
#define GARAGE_RECON_GR_SEMANTICS_LABELING_H_

#include <array>
#include <cstdint>
#include <vector>

#include "gr/core/image.h"
#include "gr/core/pose.h"
#include "gr/semantics/objects.h"
#include "gr/semantics/projector.h"
#include "gr/sensors/camera.h"
#include "gr/sensors/exposure.h"

namespace gr {

struct FrameLabelConfig {
  double max_range = 12.0;
  double sample_step = 0.02;  // Dense 3D samples along outlines / markings.
  double depth_abs_tol = 0.15;
  double depth_rel_tol = 0.03;
  double render_spacing = 0.05;
  double box_render_step = 0.05;  // Box surfaces in the label image.
  double box_sample_step = 0.1;   // Box surfaces for the 2D boxes.
};

struct SlotGeometry {
  std::array<Vec2, 4> corners;  // corners[0], corners[1]: entrance edge.
  double z = 0.0;               // Floor height.
};

struct MarkingGeometry {
  Vec2 p0 = Vec2::Zero();
  Vec2 p1 = Vec2::Zero();
  double width = 0.1;
};

using Polyline2 = std::vector<Vec2>;

struct SlotProjection {
  int index = 0;
  std::array<Vec2, 4> corners_uv;  // (-1, -1) where a corner does not project.
  std::array<bool, 4> corner_visible{};
  double entrance_visible = 0.0;  // Visible fraction of the entrance edge.
  double visible_fraction = 0.0;
  std::vector<Polyline2> polylines;  // Visible runs of the outline.
};

struct MarkingProjection {
  int index = 0;
  std::vector<Polyline2> polylines;
};

struct BoxProjection {
  int index = 0;
  // u_min, v_min, u_max, v_max of the visible surface.
  Eigen::Vector4d bbox = Eigen::Vector4d::Zero();
  double visible_fraction = 0.0;
};

struct FrameLabels {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> label_image;  // H x W class ids (0 = none).
  std::vector<SlotProjection> slots;
  std::vector<MarkingProjection> markings;
  std::vector<BoxProjection> boxes;
  // Mean image-gradient strength along the projected marking edges; low
  // values reveal pose, extrinsic or time-sync errors. NaN without an image.
  double consistency_score = 0.0;
};

class FrameLabeler {
 public:
  // One label per static point.
  FrameLabeler(const Points& static_world,
               const std::vector<int>& static_labels,
               std::vector<SlotGeometry> slots,
               std::vector<MarkingGeometry> markings,
               const FrameLabelConfig& config);
  FrameLabeler(const FrameLabeler&) = default;
  FrameLabeler& operator=(const FrameLabeler&) = default;
  FrameLabeler(FrameLabeler&&) = default;
  FrameLabeler& operator=(FrameLabeler&&) = default;

  // z-buffer of the static map plus this frame's boxes -> label image and
  // occlusion test; slots, markings and boxes are sampled densely in 3D,
  // projected (fisheye-safe) and kept where visible. `image` (optional) feeds
  // the consistency score.
  FrameLabels Label(const CameraModel& camera, const CameraExposure& exposure,
                    const std::vector<ObjectBox>& boxes,
                    const ImageView& image = ImageView()) const;

 private:
  class Projector;  // Per-frame projection with the occlusion test.

  void ProjectSlots(const Projector& projector, FrameLabels& labels) const;
  void ProjectMarkings(const Projector& projector,
                       const std::vector<float>& edges, const ImageView& image,
                       FrameLabels& labels) const;
  void ProjectBoxes(const Projector& projector,
                    const std::vector<ObjectBox>& boxes,
                    FrameLabels& labels) const;

  FrameLabelConfig config_;
  PointCloudIndex index_;
  std::vector<int> labels_;
  std::vector<SlotGeometry> slots_;
  std::vector<MarkingGeometry> markings_;
  std::vector<double> marking_z_;
};

// Sobel gradient magnitude of the channel mean, normalised by its 95th
// percentile (H x W).
std::vector<float> EdgeStrength(const ImageView& image);

// Multi-view voting: every view renders a z-buffer of the points, the winning
// point of each pixel receives that pixel's label, pixels on label boundaries
// are skipped and votes are accumulated; a point needs `min_views` views.
class LabelLifter {
 public:
  // `num_classes` must be positive.
  LabelLifter(const Points& points, int num_classes, double spacing = 0.05,
              double max_range = 15.0, int boundary = 1, int min_views = 2);
  LabelLifter(const LabelLifter&) = default;
  LabelLifter& operator=(const LabelLifter&) = default;
  LabelLifter(LabelLifter&&) = default;
  LabelLifter& operator=(LabelLifter&&) = default;

  // `labels`: camera-sized H x W image (-1 = none); `weights`: H x W or null.
  void AddView(const CameraModel& camera, const CameraExposure& exposure,
               const int32_t* labels, const float* weights = nullptr);

  struct Lifted {
    std::vector<int> labels;         // -1 = unlabelled.
    std::vector<double> confidence;  // Winning share of the votes.
  };
  Lifted Lift() const;
  const std::vector<float>& votes() const { return votes_; }  // N x classes.
  int num_classes() const { return num_classes_; }

 private:
  PointCloudIndex index_;
  int num_classes_;
  double spacing_;
  double max_range_;
  int boundary_;
  int min_views_;
  std::vector<float> votes_;
  std::vector<int> num_views_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_LABELING_H_
