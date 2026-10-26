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

#ifndef GARAGE_RECON_GR_REGISTRATION_CLOUD_COMPARE_H_
#define GARAGE_RECON_GR_REGISTRATION_CLOUD_COMPARE_H_

#include <optional>
#include <string>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"

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

// Every hypothesis of the search above for a verification by ICP: the four
// yaw candidates (best occupancy score first), each with the height shifts
// whose correlation is a local maximum reaching `z_peak_ratio` of the best
// (levels of the same layout give one peak per level spacing; 1 keeps the
// best only). `ambiguity` is that of the best yaw. Empty if either cloud has
// too little vertical structure.
std::vector<CoarseAlignment> AlignGravityAlignedCandidates(
    const Points& source, const Points& reference,
    const CoarseAlignConfig& config = {}, double z_peak_ratio = 0.5);

// Placement of a gravity-aligned cloud on a reference cloud without any
// initial guess: every hypothesis of AlignGravityAlignedCandidates above
// `min_score` (best first, at most `max_candidates`) is refined by the
// multi-scale ICP of registration.h, and the best ICP fit is accepted only if
// it is good enough and no distinct hypothesis (farther than distinct_t or
// distinct_r from it) fits nearly as well. Regular garages saturate the
// fitness: a wrong placement still matches floors, ceilings and most columns.
// Used to place trips without an anchor (backend.h, AlignTrips) and to
// localize a vehicle that has no prior pose (localization.h).
struct PlacementConfig {
  double voxel = 0.2;  // Both clouds are downsampled to it.
  std::vector<double> icp_schedule = {2.0, 1.0, 0.5, 0.3};
  int icp_iters = 40;
  double min_score = 0.3;     // Coarse: fraction of structure cells matched.
  double z_peak_ratio = 0.5;  // Height hypotheses per yaw.
  int max_candidates = 8;
  double min_fitness = 0.5;  // ICP: inliers / source points.
  double max_rmse = 0.2;     // m
  double min_fitness_gap = 0.03;
  double distinct_t = 1.0;  // m
  double distinct_r = 5.0;  // deg
};

struct Placement {
  bool placed = false;
  // "placed", "ambiguous", "ICP rejected" or "no coarse hypothesis".
  std::string verdict = "no coarse hypothesis";
  Pose reference_from_source;  // Best ICP fit (also when not placed).
  double score = 0.0;      // Coarse score of the best fit.
  double fitness = 0.0;    // Its ICP fitness and RMSE.
  double rmse = 0.0;
  double runner_up = 0.0;  // Best ICP fitness of a distinct hypothesis.
  int hypotheses = 0;      // Refined by ICP.
};

Placement PlaceGravityAligned(const Points& source, const Points& reference,
                              const PlacementConfig& config = {});

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

#endif  // GARAGE_RECON_GR_REGISTRATION_CLOUD_COMPARE_H_
