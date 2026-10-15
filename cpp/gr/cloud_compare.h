// Comparison of a reconstruction with a reference cloud in another frame
// (plan section 3.5 / 7: acceptance against the handheld ground truth).
//
// Both clouds are assumed gravity aligned (z up), as LiDAR-inertial maps and
// handheld scanner exports are. The coarse registration then has four
// degrees of freedom: yaw from the dominant directions of the vertical
// surfaces (four candidates, 90 deg apart), x / y by correlating plan-view
// occupancy images of the vertical structure, z from the heights of the
// horizontal surfaces. Fine registration is the multi-scale ICP of
// registration.h; MeasureSurfaceDistance gives the point-to-surface
// distances of the accuracy and completeness reports.

#ifndef GARAGE_RECON_GR_CLOUD_COMPARE_H_
#define GARAGE_RECON_GR_CLOUD_COMPARE_H_

#include <optional>
#include <vector>

#include "gr/pose.h"
#include "gr/types.h"

namespace gr {

struct CoarseAlignConfig {
  double voxel = 0.1;              // Downsampling of both clouds (m).
  double normal_radius = 0.3;      // For the surface orientation (m).
  double vertical_max_nz = 0.3;    // |n_z| below this: vertical surface.
  double horizontal_min_nz = 0.9;  // |n_z| above this: horizontal surface.
  double yaw_bin_deg = 0.5;        // Histogram of wall directions (mod 90 deg).
  double coarse_cell = 1.0;        // Occupancy correlation, coarse (m) ...
  double fine_cell = 0.25;         // ... then fine around the coarse optimum.
  double z_bin = 0.05;             // Height histogram of horizontal surfaces.
};

struct CoarseAlignment {
  Pose reference_from_source;  // Rotation about z and a translation.
  double score = 0.0;          // Fraction of source structure cells matched.
  double ambiguity = 0.0;  // Best score of the other yaw candidates / score.
};

// nullopt if either cloud has too little vertical structure.
std::optional<CoarseAlignment> AlignGravityAligned(
    const Points& source, const Points& reference,
    const CoarseAlignConfig& config = {});

struct SurfaceDistances {
  // Along the normal of the reference surface fitted around the nearest
  // reference points (positive on the side the normal points to); NaN where
  // the reference has no local plane within `radius`.
  std::vector<double> signed_distance;
  // To the nearest reference point; NaN beyond `max_distance`.
  std::vector<double> nearest;
  Points normals;  // Reference surface normal (zero without a plane).
};

// Distances of `query` points to the `reference` cloud (same frame).
SurfaceDistances MeasureSurfaceDistance(const Points& query,
                                        const Points& reference, double radius,
                                        double max_distance);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CLOUD_COMPARE_H_
