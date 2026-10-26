// Range-image visibility voting for dynamic-point removal (in the spirit of
// Removert, Kim & Kim, IROS 2020): a map point is "seen through" by a sweep
// when the sweep measured farther than the point in the point's direction,
// i.e. the point was absent at that time. Grazing rays are ignored, which is
// what makes this robust on floors, ceilings and long walls.

#ifndef GARAGE_RECON_GR_MAPPING_VISIBILITY_H_
#define GARAGE_RECON_GR_MAPPING_VISIBILITY_H_

#include <map>
#include <unordered_map>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/core/voxel.h"

namespace gr {

struct RangeImageSpec {
  double az_res_deg = 1.0;
  double el_res_deg = 1.0;
  double el_min_deg = -30.0;
  double el_max_deg = 30.0;
};

struct VisibilityVoteOptions {
  RangeImageSpec image;
  double max_range = 30.0;
  // A return within margin_abs + margin_rel * range of a point "sees" it.
  double margin_abs = 0.3;
  double margin_rel = 0.02;
  // Rays with |normal . direction| below this are grazing and do not vote.
  double min_cos = 0.25;
};

class VisibilityVoter {
 public:
  // `normals` holds one unit normal per map point.
  VisibilityVoter(const Points& map, const Points& normals,
                  double bucket_size = 4.0);
  VisibilityVoter(const VisibilityVoter&) = default;
  VisibilityVoter& operator=(const VisibilityVoter&) = default;
  VisibilityVoter(VisibilityVoter&&) = default;
  VisibilityVoter& operator=(VisibilityVoter&&) = default;

  // Adds the votes of one sweep (LiDAR frame) to the counters of `session`
  // (e.g. a trip).
  void AddSweep(const Pose& world_from_lidar, const Points& sweep,
                const VisibilityVoteOptions& options, int session = 0);
  void ResetCounts() { counts_.clear(); }

  // Votes per map point of one session (all zero if it has none).
  const std::vector<int>& SeenCounts(int session = 0) const;
  const std::vector<int>& ThroughCounts(int session = 0) const;
  std::vector<int> Sessions() const;

 private:
  struct Counts {
    std::vector<int> seen;
    std::vector<int> through;
  };

  Points map_;
  Points normals_;
  double bucket_size_;
  std::unordered_map<CellKey, std::vector<int>, CellKeyHash> buckets_;
  std::map<int, Counts> counts_;
  std::vector<int> zeros_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_MAPPING_VISIBILITY_H_
