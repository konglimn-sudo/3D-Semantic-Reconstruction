// Object boxes and their timeline: parked objects of a trip plus moving tracks
// interpolated between keyframes, which is what occupancy and per-frame labels
// need at any sensor time.

#ifndef GARAGE_RECON_GR_OBJECTS_H_
#define GARAGE_RECON_GR_OBJECTS_H_

#include <map>
#include <vector>

#include "gr/types.h"

namespace gr {

struct ObjectBox {
  int track_id = -1;
  int label = 0;  // Semantic class id.
  Vec3 center = Vec3::Zero();
  Vec3 size = Vec3::Ones();  // Length (x), width (y), height (z), box frame.
  double yaw = 0.0;          // Rotation about z.
  bool is_static = true;
  Vec3 velocity = Vec3::Zero();
  int num_points = 0;
  double score = 1.0;

  bool Contains(const Vec3& point, double margin = 0.0) const;
  // Samples of the four sides and the top (no bottom), about `step` apart.
  Points SampleSurface(double step) const;
};

class BoxTimeline {
 public:
  BoxTimeline() = default;
  BoxTimeline(const BoxTimeline&) = default;
  BoxTimeline& operator=(const BoxTimeline&) = default;
  BoxTimeline(BoxTimeline&&) = default;
  BoxTimeline& operator=(BoxTimeline&&) = default;

  void AddStatic(int trip, const ObjectBox& box);
  // One keyframe observation of a moving track.
  void AddMoving(int trip, double stamp, const ObjectBox& box);

  // Boxes at `stamp`: the trip's parked objects, then every track of the trip
  // whose time span (+- slack) covers the stamp, interpolated (linear centre,
  // shortest-arc yaw).
  std::vector<ObjectBox> At(int trip, double stamp, double slack = 0.5) const;

 private:
  struct Observation {
    double stamp;
    int trip;
    ObjectBox box;
  };

  std::map<int, std::vector<ObjectBox>> static_boxes_;
  std::vector<int> track_order_;  // First-seen order.
  std::map<int, std::vector<Observation>> tracks_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_OBJECTS_H_
