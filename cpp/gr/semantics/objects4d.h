// 4D object labels (section 5, module 6): parked objects from the
// aggregated map, per-keyframe detections, multi-object tracking and
// track-level boxes (semantics/objects4d.py).
//
// Parked objects: object points of a level (not structure, in the object
// height band) are clustered on a BEV grid; a cluster spanning several
// slots is split by slot, and each part is judged as a whole, because
// point-level visibility evidence is incomplete: mostly points that moved
// within their session make a moving smear (its points become dynamic),
// mostly points that changed between sessions make an object parked only in
// those sessions, anything else is parked in every session. Its box comes
// from the slot it stands in or from the garage grid, completed to the
// typical vehicle size away from the driven path.
//
// Moving objects: each keyframe sweep is clustered the same way (the
// geometric baseline detector); detections not explained by a parked object
// are associated across keyframes (Hungarian assignment on constant-velocity
// predictions), fragments of one object are stitched, and moving tracks get
// smoothed centres, velocity-aligned yaw and one object-frame size.

#ifndef GARAGE_RECON_GR_SEMANTICS_OBJECTS4D_H_
#define GARAGE_RECON_GR_SEMANTICS_OBJECTS4D_H_

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#include "gr/core/logging.h"
#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/mapping/levels.h"
#include "gr/semantics/objects.h"
#include "gr/semantics/slots.h"

namespace gr {

struct ObjectsConfig {
  int frame_stride = 1;  // Keyframes of a trip that are detected.
  double cluster_res = 0.15;
  int min_points = 30;
  double height_min = 0.3;  // Object band above the floor (m).
  double height_max = 2.6;
  double detect_max_range = 20.0;
  double match_gate = 2.0;
  int max_missed = 4;
  int min_hits = 2;
  double static_displacement = 1.0;
  double typical_length = 4.6;
  double typical_width = 1.85;
  double moving_fraction = 0.15;
  double streak_min_moved = 0.05;
  int parked_min_points = 150;
  double parked_min_extent = 1.0;
  double vehicle_max_height = 2.3;
  double max_flat_height_span = 1.2;
  double nms_radius = 2.5;
  int structure_dilation = 1;
  double structure_voxel = 0.1;
  // Taxonomy ids.
  int vehicle_id = 12;
  int vehicle_moving_id = 13;
  int pedestrian_id = 14;
  int two_wheeler_id = 15;
  int other_id = 20;
};

// ClassifyObject's result for sizes of no class.
inline constexpr int kUnclassified = -1;

// The class of a box size: vehicle, pedestrian, two-wheeler, other, or
// kUnclassified. `partial`: a vehicle seen from one side may be narrow.
int ClassifyObject(const Vec3& size, bool partial, const ObjectsConfig& config);

// "Is this location static structure?" on voxels of the labelled map.
class StructureIndex {
 public:
  StructureIndex(const Points& structure_points, double voxel);

  // Whether a structure voxel lies within `dilate` voxels of each point.
  std::vector<uint8_t> IsStructure(const Points& points, int dilate) const;

 private:
  using Key = std::array<int64_t, kSpaceDim>;
  struct KeyHash {
    size_t operator()(const Key& key) const;
  };
  double voxel_;
  std::unordered_set<Key, KeyHash> keys_;
};

struct SlotOutline {
  int level = 0;
  // Entrance left / right, back right / left.
  std::array<Vec2, kParkingSlotCorners> corners;
  double width = 0.0;
};

// The map points the parked-object search needs.
struct ObjectMapView {
  const Points* xyz = nullptr;
  const std::vector<float>* height = nullptr;  // Above the floor (NaN none).
  const std::vector<int16_t>* level = nullptr;
  const std::vector<uint8_t>* dynamic_kind = nullptr;  // DynamicKind.
  const std::vector<int16_t>* trip = nullptr;
  const std::vector<float>* normal_z = nullptr;
  const std::vector<uint8_t>* labels = nullptr;  // Semantic class per point.
  int floor_id = 1;
  int ramp_id = 2;
};

struct ParkedObjects {
  std::vector<std::vector<ObjectBox>> boxes;  // Per trip.
  std::vector<uint8_t> moving;  // Map points re-labelled as moving.
};

// Parked objects of the aggregated map (see the file comment). `yaw` is the
// garage grid orientation, `paths_by_level` the driven paths (xy).
ParkedObjects ParkedObjectsFromMap(const ObjectMapView& map,
                                   const LevelModel& levels, double yaw,
                                   const std::map<int, std::vector<Vec2>>&
                                       paths_by_level,
                                   int num_trips,
                                   const std::vector<SlotOutline>& slots,
                                   const ObjectsConfig& config,
                                   const Logger& log = nullptr);

struct Detection {
  int keyframe = 0;
  double stamp = 0.0;
  Vec3 center = Vec3::Zero();
  Vec3 size = Vec3::Ones();  // Length, width, height.
  double yaw = 0.0;
  int label = 0;
  double score = 0.0;
  Points points;
};

// Object clusters of one sweep (world frame), within detect_max_range of the
// sensor `origin`.
std::vector<Detection> DetectObjects(const Points& world_points, int keyframe,
                                     double stamp, const Vec3& origin,
                                     const LevelModel& levels,
                                     const StructureIndex& structure,
                                     const ObjectsConfig& config);

struct Track {
  int track_id = 0;
  std::vector<Detection> detections;
  Vec2 velocity = Vec2::Zero();
  int missed = 0;
};

// Hungarian association with constant-velocity prediction over the frames'
// detections, then stitching; tracks with at least min_hits detections.
std::vector<Track> TrackDetections(
    const std::vector<std::vector<Detection>>& frames,
    const ObjectsConfig& config, int start_id);

// A moving track's box per keyframe; empty for a static track.
std::map<int, ObjectBox> MovingTrackBoxes(const Track& track,
                                          const ObjectsConfig& config);

struct ObjectTrack {
  int trip = 0;
  int track_id = 0;
  bool is_static = false;
  int label = 0;
  std::vector<int> keyframes;  // Moving tracks.
  ObjectBox box;               // Parked objects.
};

struct ObjectsResult {
  std::vector<ObjectTrack> tracks;
  std::vector<std::vector<ObjectBox>> parked;    // Per trip.
  std::map<int, std::vector<ObjectBox>> moving;  // Per keyframe.
  std::vector<uint8_t> moving_points;            // Map points.
};

// A keyframe's sweep in the LiDAR frame.
using KeyframeSweep = std::function<Points(int keyframe)>;

// The objects stage: parked objects from the map, then per trip the
// detections of every frame_stride-th keyframe, tracked.
ObjectsResult RunObjects(const ObjectMapView& map, const LevelModel& levels,
                         const StructureIndex& structure, double yaw,
                         const std::map<int, std::vector<Vec2>>&
                             paths_by_level,
                         const std::vector<SlotOutline>& slots,
                         const std::vector<Pose>& keyframe_poses,
                         const std::vector<int>& keyframe_trip,
                         const std::vector<double>& keyframe_stamp,
                         const std::vector<std::string>& trip_names,
                         const KeyframeSweep& load,
                         const ObjectsConfig& config,
                         const Logger& log = nullptr);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_OBJECTS4D_H_
