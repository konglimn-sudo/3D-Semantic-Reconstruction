// Cross-view identity association of 2D masks through the shared geometry.
//
// ILGS / Gaussian Grouping obtain multi-view consistent identity labels from a
// zero-shot video tracker (DEVA), which loses or swaps identities over
// thousands of frames of repetitive garage structure. Here every view renders
// a z-buffer of the carrier points (the LiDAR map) with its exposure from the
// shared trajectory; every carrier point that passes the occlusion test is
// covered by the mask at its pixel, so each 2D mask covers a set of 3D points.
// With normals, points facing away from the camera are culled (the other
// face of a slab or a wall), and the depth tolerance grows with the pixel
// footprint at grazing incidence.
//
// Views are processed in order. A mask joins the identity that owns most of
// its already-identified points; when it covers several identities with
// substantial shares, they are one object seen in pieces and are merged,
// unless two of them were separate masks of one view (a cannot-link). A mask
// with too few identified points opens a new identity. The mask's points then
// vote for its identity, and every point ends with the identity of the most
// votes.

#ifndef GARAGE_RECON_GR_SEMANTICS_IDENTITY_H_
#define GARAGE_RECON_GR_SEMANTICS_IDENTITY_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "gr/core/types.h"
#include "gr/semantics/projector.h"
#include "gr/sensors/camera.h"
#include "gr/sensors/exposure.h"

namespace gr {

struct IdentityConfig {
  double max_range = 15.0;
  double point_spacing = 0.05;  // Splat size of the carrier points (m).
  double depth_abs_tol = 0.15;  // Occlusion test of the covered points.
  double depth_rel_tol = 0.03;
  // With normals: extra tolerance of this many pixel footprints divided by
  // the cosine of the incidence angle (floors seen at grazing angles).
  double grazing_pixels = 2.0;
  int boundary = 1;         // Skip pixels this close to a mask boundary.
  int min_mask_points = 8;  // Smaller masks get no identity.
  int min_common_points = 5;
  // A mask joins identity I if I owns this share of its identified points,
  // and its identified points are at least `min_known_fraction` of it.
  double min_share = 0.6;
  double min_known_fraction = 0.1;
  // Identities owning at least this share of a mask's identified points are
  // merged into one, if the mask also covers at least `merge_coverage` of
  // each merged identity's points visible in the view (a mask that only
  // bleeds over an object's border must not merge it).
  double merge_share = 0.2;
  double merge_coverage = 0.5;
};

// Geometry of one mask in one view (inputs to view weighting downstream).
struct MaskRecord {
  int view = 0;
  int mask = 0;  // Mask id within the view.
  int identity = -1;
  int points = 0;  // Carrier points covered.
  int pixels = 0;
  double mean_range = 0.0;  // m
  bool touches_border = false;
};

class IdentityAssociator {
 public:
  // `normals` (one per point, oriented towards the sensors) may be empty.
  IdentityAssociator(Points points, Points normals,
                     const IdentityConfig& config);
  IdentityAssociator(const IdentityAssociator&) = default;
  IdentityAssociator& operator=(const IdentityAssociator&) = default;
  IdentityAssociator(IdentityAssociator&&) = default;
  IdentityAssociator& operator=(IdentityAssociator&&) = default;

  // Adds one view. `masks` is a camera-sized H x W image of mask ids (-1 =
  // none). Returns the records of the view's masks.
  std::vector<MaskRecord> AddView(const CameraModel& camera,
                                  const CameraExposure& exposure,
                                  const int32_t* masks);

  struct Result {
    std::vector<int> point_identity;      // -1 = never covered by a mask.
    std::vector<float> point_confidence;  // Winning share of the votes.
    std::vector<MaskRecord> masks;        // Every mask of every view.
    int num_identities = 0;
  };
  Result Finish() const;

  size_t num_points() const { return index_.size(); }
  int num_views() const { return num_views_; }

 private:
  // Root of an identity after merges (union-find).
  int Find(int identity) const;
  int NewIdentity();
  // Merges the identities into the smallest root; returns it.
  int Merge(const std::vector<int>& roots);
  bool CannotLink(int a, int b) const;
  // Root identity with the most votes of a point, -1 without votes.
  int CurrentIdentity(size_t point) const;
  void Vote(size_t point, int identity);

  IdentityConfig config_;
  PointCloudIndex index_;
  Points normals_;
  // Votes per point as (identity, count); identities resolve through Find.
  std::vector<std::vector<std::pair<int, float>>> votes_;
  mutable std::vector<int> parent_;
  // Identities seen as separate masks of one view (kept at the roots).
  std::vector<std::vector<int>> cannot_link_;
  std::vector<MaskRecord> records_;
  int num_views_ = 0;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_IDENTITY_H_
