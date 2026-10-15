// Loop closure within and across trips (section 4.1).
//
// Repeated column grids and identical levels make appearance-only place
// recognition unreliable, so candidates are gated by the current position
// estimate and by level (|dz|), ranked by Scan Context, verified by
// coarse-to-fine ICP against a local submap and finally filtered for
// pairwise consistency (PCM).

#ifndef GARAGE_RECON_GR_LOOP_CLOSURE_H_
#define GARAGE_RECON_GR_LOOP_CLOSURE_H_

#include <vector>

#include "gr/keyframes.h"
#include "gr/pose.h"

namespace gr {

struct LoopConfig {
  bool enabled = true;
  int stride = 2;          // Test every n-th keyframe.
  int min_index_gap = 30;  // Same-trip separation (keyframes).
  double search_radius = 12.0;
  double level_gate = 1.5;  // |dz| allowed between candidates.
  int sc_rings = 20;
  int sc_sectors = 60;
  double sc_max_radius = 30.0;
  double sc_voxel = 0.2;  // Sweep resolution for the descriptors.
  double sc_threshold = 0.35;
  int top_k = 3;
  int submap_half_window = 3;  // Keyframes on each side of the target.
  double icp_voxel = 0.2;
  std::vector<double> icp_schedule = {2.0, 1.0, 0.5, 0.3};
  int icp_iters = 40;
  double min_fitness = 0.55;
  double max_rmse = 0.12;
  double max_correction = 4.0;  // m between the graph prior and the ICP result.
  double pcm_threshold = 0.5;   // Metres-equivalent cycle error.
  double pcm_drift_per_m = 0.01;
};

struct LoopConstraint {
  int i = 0;      // Source keyframe (the later one).
  int j = 0;      // Target keyframe.
  Pose j_from_i;  // Measured T_j^-1 T_i (LiDAR frames).
  double fitness = 0.0;
  double rmse = 0.0;
  double sc_distance = 1.0;
};

struct LoopStats {
  int queries = 0;  // Keyframes that had candidates.
  int tested = 0;   // Candidates verified by ICP.
};

// At most one verified loop per query keyframe; queries run in parallel.
// `poses` holds one pose per keyframe; `stats` may be null.
std::vector<LoopConstraint> DetectLoops(const KeyframeSet& keyframes,
                                        const std::vector<Pose>& poses,
                                        const LoopConfig& config,
                                        LoopStats* stats = nullptr);

// Pairwise consistency maximisation (Mangelson et al., ICRA 2018) with a
// greedy clique: two loops between the same pair of trips must agree with
// the odometry between their ends, within a tolerance that grows with the
// distance travelled.
std::vector<LoopConstraint> SelectConsistentLoops(
    const std::vector<LoopConstraint>& loops, const KeyframeSet& keyframes,
    const std::vector<Pose>& poses, const LoopConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_LOOP_CLOSURE_H_
