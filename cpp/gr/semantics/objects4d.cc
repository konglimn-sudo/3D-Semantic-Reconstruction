#include "gr/semantics/objects4d.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/geometry.h"
#include "gr/core/morphology.h"
#include "gr/core/numeric.h"
#include "gr/core/voxel.h"
#include "gr/mapping/static_map.h"

namespace gr {
namespace {

constexpr double kGated = 1e6;  // Cost of a pair outside the match gate.
constexpr double kInf = std::numeric_limits<double>::infinity();
// Clustering closes the BEV grid once with a cross first.
constexpr int kClusterClosingIterations = 1;
// The floor support grid: kSupportRes (m) cells with a margin of
// kSupportMarginCells cells, closed kSupportClosingIterations times.
constexpr double kSupportRes = 0.5;
constexpr int kSupportMarginCells = 2;
constexpr int kSupportClosingIterations = 4;
// Floor points within this (m) of a level's floor belong to it.
constexpr double kLevelFloorTolerance = 1.0;
// Box extents: the kExtentLow / kExtentHigh percentiles of the points;
// heights: the kTopPercentile of the points above the floor.
constexpr double kExtentLow = 1.0;
constexpr double kExtentHigh = 99.0;
constexpr double kTopPercentile = 98.0;
// Boxes shorter than the typical car by more than kSizeSlack (m) are
// extended to it; in a slot at most kBackOverhang (m) beyond the back line
// and kMaxEntranceOverhang (m) before the entrance.
constexpr double kSizeSlack = 0.3;
constexpr double kBackOverhang = 0.2;
constexpr double kMaxEntranceOverhang = 0.5;
// Directions of shorter vectors have no meaning.
constexpr double kMinNorm = 1e-9;
// Without a path the aisle is assumed kNoPathAisleOffset (m) towards -y.
constexpr double kNoPathAisleOffset = 5.0;
// A grid box at least kElongatedLength long and at most kElongatedWidth
// wide (m) shows its length axis; else the aisle direction decides, over
// kTangentHalfWindow path samples to both sides (needing kMinPathSamples).
constexpr double kElongatedLength = 3.0;
constexpr double kElongatedWidth = 2.6;
constexpr size_t kTangentHalfWindow = 5;
constexpr size_t kMinPathSamples = 2;
// Stitching: a fragment may start up to kMaxStitchOverlap (s) before the
// other ends and kDefaultMaxStitchGap (s) after it, within kStitchGateFactor
// match gates of the prediction.
constexpr double kMaxStitchOverlap = 1.0;
constexpr double kDefaultMaxStitchGap = 3.0;
constexpr double kStitchGateFactor = 1.5;
// Class sizes (m): vehicles kVehicleMinLength..kVehicleMaxLength long, at
// least kVehicleMinHeight high and kVehicleMinWidth..kVehicleMaxWidth wide
// (seen partly: at least kPartialVehicleMinWidth); pedestrians at most
// kPedestrianMaxLength long and kPedestrianMaxWidth wide,
// kPedestrianMinHeight..kPedestrianMaxHeight high; two-wheelers longer, up
// to kVehicleMinLength, at most kTwoWheelerMaxWidth wide and
// kTwoWheelerMinHeight..kTwoWheelerMaxHeight high; other objects at least
// kMinObjectSize long and high.
constexpr double kVehicleMinLength = 2.5;
constexpr double kVehicleMaxLength = 6.5;
constexpr double kVehicleMinHeight = 0.9;
constexpr double kVehicleMinWidth = 1.2;
constexpr double kPartialVehicleMinWidth = 0.3;
constexpr double kVehicleMaxWidth = 2.8;
constexpr double kPedestrianMaxLength = 1.2;
constexpr double kPedestrianMaxWidth = 1.0;
constexpr double kPedestrianMinHeight = 1.0;
constexpr double kPedestrianMaxHeight = 2.1;
constexpr double kTwoWheelerMaxWidth = 1.0;
constexpr double kTwoWheelerMinHeight = 0.8;
constexpr double kTwoWheelerMaxHeight = 1.8;
constexpr double kMinObjectSize = 0.3;
// Detection boxes are at least this (m) along each axis; a detection of
// kFullScorePoints points scores 1.
constexpr double kMinBoxSize = 0.1;
constexpr double kFullScorePoints = 200.0;
// Track velocities: over at least kMinTrackInterval (s), the mean of the
// previous and the new estimate.
constexpr double kMinTrackInterval = 1e-3;
constexpr double kPreviousVelocityWeight = 0.5;
constexpr double kNewVelocityWeight = 0.5;
// Moving tracks: centres smoothed over up to kMaxSmoothingWindow (at least
// kMinSmoothingWindow) detections; headings from speeds above
// kMinHeadingSpeed (m/s); extents from the kTrackExtentLow..kTrackExtentHigh
// percentiles of the points.
constexpr int kMaxSmoothingWindow = 5;
constexpr int kMinSmoothingWindow = 3;
constexpr double kMinHeadingSpeed = 0.3;
constexpr double kTrackExtentLow = 2.0;
constexpr double kTrackExtentHigh = 98.0;
// Parked objects: ids from kFirstParkedId; object points carry
// kUnlabeledClass; a cluster part needs kMinSlotShare of its points in a
// slot to belong to it, kMinChangedShare changed points to be trip-specific
// and kMinTripShare of those to appear in a trip.
constexpr int kFirstParkedId = 50000;
constexpr uint8_t kUnlabeledClass = 0;
constexpr int kNoSlot = -1;
constexpr double kMinSlotShare = 0.2;
constexpr double kMinChangedShare = 0.5;
constexpr double kMinTripShare = 0.2;
// Clusters spanning more than this (m) are streaks or rows, not objects.
constexpr double kMaxObjectSpan = 7.0;
// Flat surfaces face up more than kFlatNormalZ; with more than
// kMinFlatShare of the points flat, the kFlatSpreadLow..kFlatSpreadHigh
// percentile spread of their heights is checked.
constexpr float kFlatNormalZ = 0.8f;
constexpr double kMinFlatShare = 0.3;
constexpr double kFlatSpreadLow = 5.0;
constexpr double kFlatSpreadHigh = 95.0;
// Moving track ids of trip t start at t * kTrackIdsPerTrip.
constexpr int kTrackIdsPerTrip = 10000;
// Detections inside a parked box (plus this margin, m) are explained.
constexpr double kExplainedMargin = 0.5;
constexpr int kMinFrameStride = 1;

uint8_t KindCode(DynamicKind kind) { return static_cast<uint8_t>(kind); }

// Rotation by `angle` (structure.rot2).
Vec2 Rotate(const Vec2& point, double angle) {
  const double cos_angle = std::cos(angle);
  const double sin_angle = std::sin(angle);
  return Vec2(cos_angle * point.x() - sin_angle * point.y(),
              sin_angle * point.x() + cos_angle * point.y());
}

// The 8-connected clusters of points on a BEV grid of `res` (a cross
// closing first joins cells one apart), in label order; each cluster lists
// its point indices in order.
std::vector<std::vector<int>> ClusterPoints(const Points& points, double res) {
  if (points.empty()) return {};
  Vec2 lower = points[0].head<2>();
  for (const Vec3& point : points) lower = lower.cwiseMin(point.head<2>());
  std::vector<std::pair<int, int>> cell_of(points.size());
  int width = 0;
  int height = 0;
  for (size_t k = 0; k < points.size(); ++k) {
    // astype(int) truncates; the offsets are not negative.
    cell_of[k] = {static_cast<int>((points[k].x() - lower.x()) / res),
                  static_cast<int>((points[k].y() - lower.y()) / res)};
    width = std::max(width, cell_of[k].first + 1);
    height = std::max(height, cell_of[k].second + 1);
  }
  Mask grid(static_cast<size_t>(width) * height, 0);
  for (const auto& [column, row] : cell_of) {
    grid[static_cast<size_t>(row) * width + column] = 1;
  }
  Mask closed = CloseCross(grid, width, height, kClusterClosingIterations);
  for (size_t cell = 0; cell < closed.size(); ++cell) {
    closed[cell] |= grid[cell];
  }
  const Components components = Label8(closed, width, height);
  std::vector<std::vector<int>> clusters(components.cells.size());
  for (size_t k = 0; k < points.size(); ++k) {
    const int label =
        components.label[static_cast<size_t>(cell_of[k].second) * width +
                         cell_of[k].first];
    clusters[static_cast<size_t>(label - 1)].push_back(static_cast<int>(k));
  }
  return clusters;
}

std::vector<Vec2> Xy(const Points& points, const std::vector<int>& index) {
  std::vector<Vec2> out;
  out.reserve(index.size());
  for (int k : index) out.push_back(points[static_cast<size_t>(k)].head<2>());
  return out;
}

ObjectBox MakeBox(int track_id, int label, const Vec3& center,
                  const Vec3& size, double yaw, bool is_static,
                  const Vec3& velocity, int num_points, double score) {
  ObjectBox box;
  box.track_id = track_id;
  box.label = label;
  box.center = center;
  box.size = size;
  box.yaw = yaw;
  box.is_static = is_static;
  box.velocity = velocity;
  box.num_points = num_points;
  box.score = score;
  return box;
}

// The mean of a slot's corners.
Vec2 SlotCenter(const std::vector<Vec2>& polygon) {
  Vec2 sum = polygon[0];
  for (int corner = 1; corner < kParkingSlotCorners; ++corner) {
    sum += polygon[static_cast<size_t>(corner)];
  }
  return sum / static_cast<double>(kParkingSlotCorners);
}

// Whether an xy location is over (dilated) observed floor (_support_grid).
class FloorSupport {
 public:
  FloorSupport(const std::vector<Vec2>& floor, double res) : res_(res) {
    if (floor.empty()) return;
    lower_ = floor[0];
    for (const Vec2& point : floor) lower_ = lower_.cwiseMin(point);
    lower_ -= Vec2::Constant(kSupportMarginCells * res);
    std::vector<std::pair<int, int>> cell_of;
    int max_column = 0;
    int max_row = 0;
    for (const Vec2& point : floor) {
      cell_of.emplace_back(static_cast<int>((point.x() - lower_.x()) / res),
                           static_cast<int>((point.y() - lower_.y()) / res));
      max_column = std::max(max_column, cell_of.back().first);
      max_row = std::max(max_row, cell_of.back().second);
    }
    width_ = max_column + 1 + kSupportMarginCells;
    height_ = max_row + 1 + kSupportMarginCells;
    Mask grid(static_cast<size_t>(width_) * height_, 0);
    for (const auto& [column, row] : cell_of) {
      grid[static_cast<size_t>(row) * width_ + column] = 1;
    }
    Mask closed =
        CloseCross(grid, width_, height_, kSupportClosingIterations);
    for (size_t cell = 0; cell < closed.size(); ++cell) {
      closed[cell] |= grid[cell];
    }
    // Parked cars hide the floor under them.
    grid_ = FillHolesCross(closed, width_, height_);
    any_ = true;
  }

  bool Inside(const Vec2& xy) const {
    if (!any_) return true;
    const int column = static_cast<int>((xy.x() - lower_.x()) / res_);
    const int row = static_cast<int>((xy.y() - lower_.y()) / res_);
    return row >= 0 && row < height_ && column >= 0 && column < width_ &&
           grid_[static_cast<size_t>(row) * width_ + column] != 0;
  }

 private:
  double res_;
  bool any_ = false;
  Vec2 lower_ = Vec2::Zero();
  int width_ = 0;
  int height_ = 0;
  Mask grid_;
};

// Box of a car parked in a known slot (_slot_box).
ObjectBox SlotBox(const Points& points, double floor_z,
                  const SlotOutline& slot, const ObjectsConfig& config,
                  int track_id) {
  const Vec2 entrance = 0.5 * (slot.corners[kEntranceLeftCorner] +
                                slot.corners[kEntranceRightCorner]);
  const Vec2 back =
      0.5 * (slot.corners[kBackRightCorner] + slot.corners[kBackLeftCorner]);
  const double depth = (back - entrance).norm();
  const Vec2 u_axis = (back - entrance) / std::max(depth, kMinNorm);
  const Vec2 v_axis(-u_axis.y(), u_axis.x());
  std::vector<double> u;
  std::vector<double> v;
  std::vector<double> z;
  for (const Vec3& point : points) {
    const Vec2 offset = point.head<2>() - entrance;
    u.push_back(offset.dot(u_axis));
    v.push_back(offset.dot(v_axis));
    z.push_back(point.z());
  }
  double u_min = Percentile(u, kExtentLow);
  double u_max = Percentile(u, kExtentHigh);
  double v_min = Percentile(v, kExtentLow);
  double v_max = Percentile(v, kExtentHigh);
  const double length = config.typical_length;
  const double width = config.typical_width;
  if (u_max - u_min < length - kSizeSlack) {
    u_max = std::min(u_min + length, depth + kBackOverhang);
    u_min = std::max(std::min(u_min, u_max - length), -kMaxEntranceOverhang);
  }
  if (v_max - v_min < width - kSizeSlack) {
    const double limit = std::max(slot.width / 2.0 - width / 2.0, 0.0);
    const double v_center = std::clamp((v_min + v_max) / 2.0, -limit, limit);
    v_min = v_center - width / 2.0;
    v_max = v_center + width / 2.0;
  }
  const Vec2 center = entrance + u_axis * ((u_min + u_max) / 2.0) +
                      v_axis * ((v_min + v_max) / 2.0);
  const double height = Percentile(z, kTopPercentile) - floor_z;
  return MakeBox(track_id, config.vehicle_id,
                 Vec3(center.x(), center.y(), floor_z + height / 2.0),
                 Vec3(u_max - u_min, v_max - v_min, height),
                 std::atan2(u_axis.y(), u_axis.x()), true, Vec3::Zero(),
                 static_cast<int>(points.size()), 1.0);
}

// Box of a parked object aligned with the garage grid (_grid_box);
// `path_grid` is the driven path in the grid frame.
ObjectBox GridBox(const Points& points, double floor_z, double yaw,
                  const std::vector<Vec2>& path_grid,
                  const ObjectsConfig& config, int track_id) {
  std::vector<double> grid_x;
  std::vector<double> grid_y;
  std::vector<double> z;
  for (const Vec3& point : points) {
    const Vec2 grid_point = Rotate(point.head<2>(), -yaw);
    grid_x.push_back(grid_point.x());
    grid_y.push_back(grid_point.y());
    z.push_back(point.z());
  }
  Vec2 lower(Percentile(grid_x, kExtentLow), Percentile(grid_y, kExtentLow));
  Vec2 upper(Percentile(grid_x, kExtentHigh),
             Percentile(grid_y, kExtentHigh));
  Vec2 extent = upper - lower;
  Vec2 center = (lower + upper) / 2.0;
  const auto nearest_index = [&]() {
    size_t best = 0;
    for (size_t k = 1; k < path_grid.size(); ++k) {
      if ((path_grid[k] - center).norm() < (path_grid[best] - center).norm()) {
        best = k;
      }
    }
    return best;
  };
  const Vec2 aisle = path_grid.empty()
                         ? Vec2(center - Vec2(0.0, kNoPathAisleOffset))
                         : path_grid[nearest_index()];
  int length_axis = 0;
  if ((extent.maxCoeff() > kElongatedLength &&
       extent.minCoeff() < kElongatedWidth) ||
      path_grid.size() < kMinPathSamples) {
    length_axis = extent.y() > extent.x() ? 1 : 0;
  } else {
    // Cars park across the aisle.
    const size_t closest = nearest_index();
    const Vec2 tangent =
        path_grid[std::min(path_grid.size() - 1,
                           closest + kTangentHalfWindow)] -
        path_grid[closest >= kTangentHalfWindow ? closest - kTangentHalfWindow
                                                : 0];
    length_axis = std::abs(tangent.x()) >= std::abs(tangent.y()) ? 1 : 0;
  }
  const std::array<std::pair<int, double>, kPlaneDim> targets = {
      std::make_pair(length_axis, config.typical_length),
      std::make_pair(1 - length_axis, config.typical_width)};
  for (const auto& [axis, target] : targets) {
    if (upper[axis] - lower[axis] < target - kSizeSlack) {
      if (aisle[axis] < center[axis]) {
        upper[axis] = lower[axis] + target;
      } else {
        lower[axis] = upper[axis] - target;
      }
    }
  }
  center = (lower + upper) / 2.0;
  extent = upper - lower;
  const double height = Percentile(z, kTopPercentile) - floor_z;
  const Vec2 world_center = Rotate(center, yaw);
  return MakeBox(track_id, config.vehicle_id,
                 Vec3(world_center.x(), world_center.y(),
                      floor_z + height / 2.0),
                 Vec3(extent[length_axis], extent[1 - length_axis], height),
                 yaw + (length_axis == 0 ? 0.0 : kPi / 2.0), true,
                 Vec3::Zero(), static_cast<int>(points.size()), 1.0);
}

// Detections sorted by support, keeping those `radius` away from a better
// one (_nms).
std::vector<Detection> SuppressFragments(std::vector<Detection> detections,
                                         double radius) {
  std::stable_sort(detections.begin(), detections.end(),
                   [](const Detection& a, const Detection& b) {
                     return a.points.size() > b.points.size();
                   });
  std::vector<Detection> kept;
  for (Detection& detection : detections) {
    bool far = true;
    for (const Detection& better : kept) {
      far &= (detection.center.head<2>() - better.center.head<2>()).norm() >=
             radius;
    }
    if (far) kept.push_back(std::move(detection));
  }
  return kept;
}

// Joins fragments of one object: B starts after A ends, within `max_gap`
// seconds, where A's constant-velocity prediction puts it.
std::vector<Track> StitchTracks(std::vector<Track> tracks,
                                const ObjectsConfig& config,
                                double max_gap = kDefaultMaxStitchGap) {
  std::stable_sort(tracks.begin(), tracks.end(),
                   [](const Track& a, const Track& b) {
                     return a.detections.front().stamp <
                            b.detections.front().stamp;
                   });
  bool merged = true;
  while (merged) {
    merged = false;
    for (size_t first = 0; first < tracks.size() && !merged; ++first) {
      for (size_t second = 0; second < tracks.size(); ++second) {
        if (first == second) continue;
        Track& earlier = tracks[first];
        const Track& later = tracks[second];
        if (later.detections.front().stamp <
            earlier.detections.front().stamp) {
          continue;
        }
        const double gap =
            later.detections.front().stamp - earlier.detections.back().stamp;
        if (!(gap >= -kMaxStitchOverlap && gap <= max_gap)) continue;
        const Vec2 predicted = earlier.detections.back().center.head<2>() +
                               earlier.velocity * std::max(gap, 0.0);
        if ((predicted - later.detections.front().center.head<2>()).norm() >
            config.match_gate * kStitchGateFactor) {
          continue;
        }
        // One detection per keyframe, the better supported one; dict order.
        std::vector<int> order;
        std::map<int, Detection> by_keyframe;
        for (const Detection& detection : earlier.detections) {
          if (!by_keyframe.count(detection.keyframe)) {
            order.push_back(detection.keyframe);
          }
          by_keyframe[detection.keyframe] = detection;
        }
        for (const Detection& detection : later.detections) {
          const auto found = by_keyframe.find(detection.keyframe);
          if (found == by_keyframe.end()) {
            order.push_back(detection.keyframe);
            by_keyframe[detection.keyframe] = detection;
          } else if (detection.points.size() > found->second.points.size()) {
            found->second = detection;
          }
        }
        std::vector<Detection> detections;
        for (int keyframe : order) {
          detections.push_back(std::move(by_keyframe[keyframe]));
        }
        std::stable_sort(detections.begin(), detections.end(),
                         [](const Detection& a, const Detection& b) {
                           return a.stamp < b.stamp;
                         });
        earlier.detections = std::move(detections);
        earlier.velocity = later.velocity;
        tracks.erase(tracks.begin() + static_cast<std::ptrdiff_t>(second));
        merged = true;
        break;
      }
    }
  }
  return tracks;
}

}  // namespace

int ClassifyObject(const Vec3& size, bool partial,
                   const ObjectsConfig& config) {
  const double length = size.x();
  const double width = size.y();
  const double height = size.z();
  if (length >= kVehicleMinLength && length <= kVehicleMaxLength &&
      height >= kVehicleMinHeight &&
      (width >= kVehicleMinWidth ||
       (partial && width >= kPartialVehicleMinWidth)) &&
      width <= kVehicleMaxWidth) {
    return config.vehicle_id;
  }
  if (length <= kPedestrianMaxLength && width <= kPedestrianMaxWidth &&
      height >= kPedestrianMinHeight && height <= kPedestrianMaxHeight) {
    return config.pedestrian_id;
  }
  if (length > kPedestrianMaxLength && length <= kVehicleMinLength &&
      width <= kTwoWheelerMaxWidth && height >= kTwoWheelerMinHeight &&
      height <= kTwoWheelerMaxHeight) {
    return config.two_wheeler_id;
  }
  if (height >= kMinObjectSize && length >= kMinObjectSize) {
    return config.other_id;
  }
  return kUnclassified;
}

size_t StructureIndex::KeyHash::operator()(const Key& key) const {
  return static_cast<size_t>(key[0] * internal::kHashPrimeX) ^
         static_cast<size_t>(key[1] * internal::kHashPrimeY) ^
         static_cast<size_t>(key[2] * internal::kHashPrimeZ);
}

StructureIndex::StructureIndex(const Points& structure_points, double voxel)
    : voxel_(voxel) {
  GR_CHECK(voxel > 0.0);
  for (const Vec3& point : structure_points) {
    keys_.insert({static_cast<int64_t>(std::floor(point.x() / voxel)),
                  static_cast<int64_t>(std::floor(point.y() / voxel)),
                  static_cast<int64_t>(std::floor(point.z() / voxel))});
  }
}

std::vector<uint8_t> StructureIndex::IsStructure(const Points& points,
                                                 int dilate) const {
  std::vector<uint8_t> hit(points.size(), 0);
  if (keys_.empty()) return hit;
  for (size_t k = 0; k < points.size(); ++k) {
    const Key base = {
        static_cast<int64_t>(std::floor(points[k].x() / voxel_)),
        static_cast<int64_t>(std::floor(points[k].y() / voxel_)),
        static_cast<int64_t>(std::floor(points[k].z() / voxel_))};
    for (int dx = -dilate; dx <= dilate && !hit[k]; ++dx) {
      for (int dy = -dilate; dy <= dilate && !hit[k]; ++dy) {
        for (int dz = -dilate; dz <= dilate && !hit[k]; ++dz) {
          hit[k] = keys_.count({base[0] + dx, base[1] + dy, base[2] + dz}) > 0;
        }
      }
    }
  }
  return hit;
}

std::vector<Detection> DetectObjects(const Points& world_points, int keyframe,
                                     double stamp, const Vec3& origin,
                                     const LevelModel& levels,
                                     const StructureIndex& structure,
                                     const ObjectsConfig& config) {
  // Far returns carry centimetre-to-decimetre height errors.
  Points near;
  for (const Vec3& p : world_points) {
    if ((p - origin).norm() <= config.detect_max_range) near.push_back(p);
  }
  const LevelAssignment assignment = AssignLevels(levels, near);
  // float32 heights compare with float32 bounds (numpy).
  const float band_min = static_cast<float>(config.height_min);
  const float band_max = static_cast<float>(config.height_max);
  Points band;
  std::vector<float> band_height;
  for (size_t k = 0; k < near.size(); ++k) {
    const float height = assignment.height[k];
    if (height >= band_min && height <= band_max) {
      band.push_back(near[k]);
      band_height.push_back(height);
    }
  }
  const std::vector<uint8_t> is_structure =
      structure.IsStructure(band, config.structure_dilation);
  Points points;
  std::vector<float> height;
  for (size_t k = 0; k < band.size(); ++k) {
    if (is_structure[k]) continue;
    points.push_back(band[k]);
    height.push_back(band_height[k]);
  }
  std::vector<Detection> out;
  if (static_cast<int>(points.size()) < config.min_points) return out;
  for (const std::vector<int>& cluster :
       ClusterPoints(points, config.cluster_res)) {
    if (static_cast<int>(cluster.size()) < config.min_points) continue;
    const RotatedRect rect = MinAreaRect(Xy(points, cluster));
    std::vector<double> floor;
    std::vector<double> z;
    for (int k : cluster) {
      floor.push_back(points[static_cast<size_t>(k)].z() -
                      static_cast<double>(height[static_cast<size_t>(k)]));
      z.push_back(points[static_cast<size_t>(k)].z());
    }
    const double floor_z = Median(floor);
    const double top = Percentile(z, kTopPercentile);
    const Vec3 size(std::max(rect.length, kMinBoxSize),
                    std::max(rect.width, kMinBoxSize),
                    std::max(top - floor_z, kMinBoxSize));
    const int label = ClassifyObject(size, true, config);
    if (label == kUnclassified) continue;
    Detection detection;
    detection.keyframe = keyframe;
    detection.stamp = stamp;
    detection.center =
        Vec3(rect.center.x(), rect.center.y(), floor_z + size.z() / 2.0);
    detection.size = size;
    detection.yaw = rect.yaw;
    detection.label = label;
    detection.score = std::min(
        1.0, static_cast<double>(cluster.size()) / kFullScorePoints);
    for (int k : cluster) {
      detection.points.push_back(points[static_cast<size_t>(k)]);
    }
    out.push_back(std::move(detection));
  }
  return out;
}

std::vector<Track> TrackDetections(
    const std::vector<std::vector<Detection>>& frames,
    const ObjectsConfig& config, int start_id) {
  std::vector<Track> active;
  std::vector<Track> done;
  int next_id = start_id;
  for (const std::vector<Detection>& detections : frames) {
    std::set<size_t> matched_tracks;
    std::set<size_t> matched_detections;
    if (!active.empty() && !detections.empty()) {
      const int rows = static_cast<int>(active.size());
      const int columns = static_cast<int>(detections.size());
      std::vector<double> cost(static_cast<size_t>(rows) * columns, kGated);
      for (int row = 0; row < rows; ++row) {
        const Track& track = active[static_cast<size_t>(row)];
        const Detection& last = track.detections.back();
        const double interval = detections[0].stamp - last.stamp;
        const Vec2 predicted =
            last.center.head<2>() + track.velocity * interval;
        for (int column = 0; column < columns; ++column) {
          const Detection& detection = detections[static_cast<size_t>(column)];
          if ((detection.label == config.pedestrian_id) !=
              (last.label == config.pedestrian_id)) {
            continue;
          }
          const double distance =
              (predicted - detection.center.head<2>()).norm();
          if (distance <= config.match_gate) {
            cost[static_cast<size_t>(row) * columns + column] = distance;
          }
        }
      }
      for (const auto& [row, column] :
           LinearSumAssignment(cost, rows, columns)) {
        if (cost[static_cast<size_t>(row) * columns + column] >= kGated) {
          continue;
        }
        Track& track = active[static_cast<size_t>(row)];
        const Detection& detection = detections[static_cast<size_t>(column)];
        const Detection& last = track.detections.back();
        const double interval =
            std::max(detection.stamp - last.stamp, kMinTrackInterval);
        const Vec2 velocity =
            (detection.center.head<2>() - last.center.head<2>()) / interval;
        track.velocity = track.detections.size() > 1
                             ? Vec2(kPreviousVelocityWeight * track.velocity +
                                    kNewVelocityWeight * velocity)
                             : velocity;
        track.detections.push_back(detection);
        track.missed = 0;
        matched_tracks.insert(static_cast<size_t>(row));
        matched_detections.insert(static_cast<size_t>(column));
      }
    }
    std::vector<Track> still;
    for (size_t row = 0; row < active.size(); ++row) {
      Track& track = active[row];
      if (!matched_tracks.count(row)) ++track.missed;
      (track.missed <= config.max_missed ? still : done)
          .push_back(std::move(track));
    }
    active = std::move(still);
    for (size_t column = 0; column < detections.size(); ++column) {
      if (matched_detections.count(column)) continue;
      Track track;
      track.track_id = next_id++;
      track.detections.push_back(detections[column]);
      active.push_back(std::move(track));
    }
  }
  for (Track& track : active) done.push_back(std::move(track));
  std::vector<Track> kept;
  for (Track& track : done) {
    if (static_cast<int>(track.detections.size()) >= config.min_hits) {
      kept.push_back(std::move(track));
    }
  }
  return StitchTracks(std::move(kept), config);
}

std::map<int, ObjectBox> MovingTrackBoxes(const Track& track,
                                          const ObjectsConfig& config) {
  const std::vector<Detection>& detections = track.detections;
  const size_t count = detections.size();
  double displacement = 0.0;
  for (const Detection& detection : detections) {
    displacement = std::max(
        displacement,
        (detection.center.head<2>() - detections[0].center.head<2>()).norm());
  }
  if (displacement < config.static_displacement) return {};
  // The class: vehicle if any detection says so, else the most frequent
  // (the smallest id among equally frequent ones).
  std::map<int, int> counts;
  for (const Detection& detection : detections) ++counts[detection.label];
  int label = counts.begin()->first;
  if (counts.count(config.vehicle_id)) {
    label = config.vehicle_id;
  } else {
    for (const auto& [id, count] : counts) {
      if (count > counts[label]) label = id;
    }
  }
  // Centres smoothed by a centred moving average (edge padding) of up to
  // kMaxSmoothingWindow, an odd number of detections.
  const int window = std::min<int>(
      kMaxSmoothingWindow,
      static_cast<int>(count) - (1 - static_cast<int>(count) % 2));
  std::vector<Vec3> smooth(count);
  for (size_t k = 0; k < count; ++k) smooth[k] = detections[k].center;
  if (window >= kMinSmoothingWindow) {
    const int pad = window / 2;
    for (size_t k = 0; k < count; ++k) {
      Vec3 sum = Vec3::Zero();
      for (int offset = -pad; offset <= pad; ++offset) {
        const int neighbour = std::clamp(static_cast<int>(k) + offset, 0,
                                         static_cast<int>(count) - 1);
        sum += detections[static_cast<size_t>(neighbour)].center / window;
      }
      smooth[k] = sum;
    }
  }
  std::vector<double> stamps(count);
  std::vector<double> smooth_x(count);
  std::vector<double> smooth_y(count);
  for (size_t k = 0; k < count; ++k) {
    stamps[k] = detections[k].stamp;
    smooth_x[k] = smooth[k].x();
    smooth_y[k] = smooth[k].y();
  }
  std::vector<double> velocity_x(count, 0.0);
  std::vector<double> velocity_y(count, 0.0);
  if (count > 1) {
    velocity_x = Gradient(smooth_x, stamps);
    velocity_y = Gradient(smooth_y, stamps);
  }
  std::vector<double> yaws(count);
  std::vector<double> along;
  std::vector<double> across;
  for (size_t k = 0; k < count; ++k) {
    const Vec2 velocity(velocity_x[k], velocity_y[k]);
    yaws[k] = velocity.norm() > kMinHeadingSpeed
                  ? std::atan2(velocity.y(), velocity.x())
                  : detections[k].yaw;
    for (const Vec3& point : detections[k].points) {
      const Vec2 local =
          Rotate(point.head<2>() - smooth[k].head<2>(), -yaws[k]);
      along.push_back(local.x());
      across.push_back(local.y());
    }
  }
  const double length = Percentile(along, kTrackExtentHigh) -
                        Percentile(along, kTrackExtentLow);
  const double width = Percentile(across, kTrackExtentHigh) -
                       Percentile(across, kTrackExtentLow);
  std::vector<double> heights;
  for (const Detection& detection : detections) {
    heights.push_back(detection.size.z());
  }
  const double height = Median(heights);
  const int moving_label =
      label == config.vehicle_id ? config.vehicle_moving_id : label;
  std::map<int, ObjectBox> out;
  for (size_t k = 0; k < count; ++k) {
    out[detections[k].keyframe] =
        MakeBox(track.track_id, moving_label, smooth[k],
                Vec3(length, width, height), yaws[k], false,
                Vec3(velocity_x[k], velocity_y[k], 0.0),
                static_cast<int>(detections[k].points.size()),
                detections[k].score);
  }
  return out;
}

ParkedObjects ParkedObjectsFromMap(
    const ObjectMapView& map, const LevelModel& levels, double yaw,
    const std::map<int, std::vector<Vec2>>& paths_by_level, int num_trips,
    const std::vector<SlotOutline>& slots, const ObjectsConfig& config,
    const Logger& log) {
  const Points& xyz = *map.xyz;
  const size_t count = xyz.size();
  ParkedObjects out;
  out.boxes.resize(static_cast<size_t>(num_trips));
  out.moving.assign(count, 0);
  const float band_min = static_cast<float>(config.height_min);
  const float band_max = static_cast<float>(config.height_max);
  std::vector<Vec2> floor_xy;
  std::vector<double> floor_z;
  for (size_t i = 0; i < count; ++i) {
    const int label = (*map.labels)[i];
    if (label == map.floor_id || label == map.ramp_id) {
      floor_xy.push_back(xyz[i].head<2>());
      floor_z.push_back(xyz[i].z());
    }
  }
  int next_id = kFirstParkedId;
  for (int level = 0; level < levels.num_levels(); ++level) {
    std::vector<Vec2> near_floor;
    for (size_t k = 0; k < floor_xy.size(); ++k) {
      if (std::abs(floor_z[k] - levels.FloorZ(floor_xy[k], level)) <
          kLevelFloorTolerance) {
        near_floor.push_back(floor_xy[k]);
      }
    }
    const FloorSupport support(near_floor, kSupportRes);
    std::vector<int> map_index;  // Object points of the level in the map.
    for (size_t i = 0; i < count; ++i) {
      const float height = (*map.height)[i];
      if ((*map.labels)[i] == kUnlabeledClass && height >= band_min &&
          height <= band_max && (*map.level)[i] >= 0 &&
          (*map.level)[i] == level) {
        map_index.push_back(static_cast<int>(i));
      }
    }
    if (static_cast<int>(map_index.size()) < config.min_points) continue;
    Points points;
    for (int i : map_index) points.push_back(xyz[static_cast<size_t>(i)]);
    const auto map_point = [&](int k) {
      return static_cast<size_t>(map_index[static_cast<size_t>(k)]);
    };
    const auto height_of = [&](int k) { return (*map.height)[map_point(k)]; };
    const auto kind_of = [&](int k) {
      return (*map.dynamic_kind)[map_point(k)];
    };
    std::vector<Vec2> path_grid;
    const auto path = paths_by_level.find(level);
    if (path != paths_by_level.end()) {
      for (const Vec2& sample : path->second) {
        path_grid.push_back(Rotate(sample, -yaw));
      }
    }
    std::vector<const SlotOutline*> level_slots;
    for (const SlotOutline& slot : slots) {
      if (slot.level == level) level_slots.push_back(&slot);
    }
    std::vector<int> in_slot(points.size(), kNoSlot);
    std::vector<Vec2> centers;
    for (size_t slot = 0; slot < level_slots.size(); ++slot) {
      const std::vector<Vec2> polygon(level_slots[slot]->corners.begin(),
                                      level_slots[slot]->corners.end());
      centers.push_back(SlotCenter(polygon));
      for (size_t k = 0; k < points.size(); ++k) {
        if (in_slot[k] == kNoSlot &&
            PointInPolygon(polygon, points[k].head<2>())) {
          in_slot[k] = static_cast<int>(slot);
        }
      }
    }
    // Parts by (slot or cluster, trips), in first-seen order, so that
    // fragments of one car merge.
    std::vector<std::pair<std::string, std::vector<int>>> pending;
    std::map<std::string, size_t> pending_index;
    std::vector<int> pending_slot;
    std::vector<std::optional<std::vector<int>>> pending_trips;
    const std::vector<std::vector<int>> clusters =
        ClusterPoints(points, config.cluster_res);
    for (size_t cluster = 0; cluster < clusters.size(); ++cluster) {
      const std::vector<int>& members = clusters[cluster];
      if (static_cast<int>(members.size()) < config.min_points) continue;
      // Cars stand in bays: a cluster spanning several slots is split, a
      // car that only spills over a line stays one object.
      std::map<int, int> slot_counts;
      for (int k : members) {
        if (in_slot[static_cast<size_t>(k)] != kNoSlot) {
          ++slot_counts[in_slot[static_cast<size_t>(k)]];
        }
      }
      const double need =
          std::max<double>(config.min_points, kMinSlotShare * members.size());
      std::vector<int> keep;
      for (const auto& [slot, slot_count] : slot_counts) {
        if (slot_count >= need) keep.push_back(slot);
      }
      std::vector<std::pair<std::vector<int>, int>> parts;
      if (keep.empty()) {
        parts.emplace_back(members, kNoSlot);
      } else {
        std::map<int, std::vector<int>> owned;
        for (int k : members) {
          size_t best = 0;
          double best_distance = kInf;
          for (size_t candidate = 0; candidate < keep.size(); ++candidate) {
            const double distance =
                (points[static_cast<size_t>(k)].head<2>() -
                 centers[static_cast<size_t>(keep[candidate])])
                    .norm();
            if (distance < best_distance) {
              best_distance = distance;
              best = candidate;
            }
          }
          owned[keep[best]].push_back(k);
        }
        for (int slot : keep) parts.emplace_back(owned[slot], slot);
      }
      for (const auto& [part, slot] : parts) {
        if (part.empty()) continue;
        size_t moved = 0;
        size_t changed = 0;
        for (int k : part) {
          moved += kind_of(k) == KindCode(DynamicKind::kMoved);
          changed += kind_of(k) == KindCode(DynamicKind::kChanged);
        }
        if (static_cast<double>(moved) / part.size() >=
            config.moving_fraction) {
          for (int k : part) out.moving[map_point(k)] = 1;
          continue;
        }
        std::optional<std::vector<int>> trips;
        if (static_cast<double>(changed) / part.size() >= kMinChangedShare) {
          std::map<int, int> trip_counts;
          for (int k : part) {
            if (kind_of(k) == KindCode(DynamicKind::kChanged)) {
              ++trip_counts[(*map.trip)[map_point(k)]];
            }
          }
          trips.emplace();
          for (const auto& [trip, trip_count] : trip_counts) {
            if (trip_count >= kMinTripShare * static_cast<double>(changed)) {
              trips->push_back(trip);
            }
          }
        }
        std::string key = slot != kNoSlot
                              ? std::to_string(slot)
                              : "c" + std::to_string(cluster + 1);
        key += "|";
        if (trips) {
          for (int trip : *trips) key += std::to_string(trip) + ",";
        } else {
          key += "all";
        }
        const auto found = pending_index.find(key);
        if (found == pending_index.end()) {
          pending_index[key] = pending.size();
          pending.emplace_back(key, part);
          pending_slot.push_back(slot);
          pending_trips.push_back(trips);
        } else {
          std::vector<int>& members_so_far = pending[found->second].second;
          members_so_far.insert(members_so_far.end(), part.begin(), part.end());
        }
      }
    }
    for (size_t entry = 0; entry < pending.size(); ++entry) {
      const std::vector<int>& selected = pending[entry].second;
      if (static_cast<int>(selected.size()) < config.parked_min_points) {
        continue;
      }
      Points object_points;
      Vec2 grid_lower = Vec2::Constant(kInf);
      Vec2 grid_upper = Vec2::Constant(-kInf);
      size_t moved = 0;
      for (int k : selected) {
        object_points.push_back(points[static_cast<size_t>(k)]);
        const Vec2 grid_point =
            Rotate(points[static_cast<size_t>(k)].head<2>(), -yaw);
        grid_lower = grid_lower.cwiseMin(grid_point);
        grid_upper = grid_upper.cwiseMax(grid_point);
        moved += kind_of(k) == KindCode(DynamicKind::kMoved);
      }
      const double span = (grid_upper - grid_lower).maxCoeff();
      if (span > kMaxObjectSpan) {
        // The streak of a moving object holds moved points; without any it
        // is a row of touching parked cars or clutter along a wall.
        if (static_cast<double>(moved) / selected.size() >=
            config.streak_min_moved) {
          for (int k : selected) out.moving[map_point(k)] = 1;
        }
        continue;
      }
      if (span < config.parked_min_extent) continue;
      // Cars have flat surfaces at two heights (bonnet, roof); a sloped
      // surface spread over many heights is structure.
      std::vector<double> flat_heights;
      for (int k : selected) {
        const float normal_z = std::abs((*map.normal_z)[map_point(k)]);
        if (normal_z > kFlatNormalZ) flat_heights.push_back(height_of(k));
      }
      if (static_cast<double>(flat_heights.size()) / selected.size() >
              kMinFlatShare &&
          Percentile(flat_heights, kFlatSpreadHigh) -
                  Percentile(flat_heights, kFlatSpreadLow) >
              config.max_flat_height_span) {
        continue;
      }
      Vec2 mean = Vec2::Zero();
      for (const Vec3& point : object_points) mean += point.head<2>();
      mean /= static_cast<double>(object_points.size());
      if (!support.Inside(mean)) continue;  // Behind walls, outside.
      std::vector<double> floor;
      for (int k : selected) {
        floor.push_back(points[static_cast<size_t>(k)].z() -
                        static_cast<double>(height_of(k)));
      }
      const double floor_level = Median(floor);
      const int slot = pending_slot[entry];
      ObjectBox box =
          slot != kNoSlot
              ? SlotBox(object_points, floor_level,
                        *level_slots[static_cast<size_t>(slot)], config,
                        next_id)
              : GridBox(object_points, floor_level, yaw, path_grid, config,
                        next_id);
      int label = ClassifyObject(box.size, false, config);
      if (label == config.vehicle_id &&
          box.size.z() > config.vehicle_max_height) {
        label = config.other_id;  // Taller than any car: structure.
      }
      if (label == kUnclassified || box.size.z() < kMinObjectSize ||
          (label == config.other_id && slot == kNoSlot)) {
        continue;  // Clutter outside bays is not a parked object.
      }
      box.label = label;
      ++next_id;
      if (pending_trips[entry]) {
        for (int trip : *pending_trips[entry]) {
          if (trip >= 0 && trip < num_trips) {
            out.boxes[static_cast<size_t>(trip)].push_back(box);
          }
        }
      } else {
        for (auto& trip_boxes : out.boxes) trip_boxes.push_back(box);
      }
    }
  }
  if (log) {
    std::string counts;
    for (size_t trip = 0; trip < out.boxes.size(); ++trip) {
      counts +=
          (trip ? ", " : "") + std::to_string(out.boxes[trip].size());
    }
    const size_t moving = static_cast<size_t>(
        std::count(out.moving.begin(), out.moving.end(), uint8_t{1}));
    LogLine(log, StringPrintf("[objects] parked objects per trip: [%s], %zu "
                              "map points re-labelled as moving",
                              counts.c_str(), moving));
  }
  return out;
}

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
                         const ObjectsConfig& config, const Logger& log) {
  const int num_trips = static_cast<int>(trip_names.size());
  ParkedObjects parked = ParkedObjectsFromMap(
      map, levels, yaw, paths_by_level, num_trips, slots, config, log);
  ObjectsResult result;
  result.moving_points = std::move(parked.moving);
  int next_id = 0;
  for (int trip = 0; trip < num_trips; ++trip) {
    const std::vector<ObjectBox>& parked_boxes =
        parked.boxes[static_cast<size_t>(trip)];
    for (const ObjectBox& parked_box : parked_boxes) {
      ObjectTrack track;
      track.trip = trip;
      track.track_id = parked_box.track_id;
      track.is_static = true;
      track.label = parked_box.label;
      track.box = parked_box;
      result.tracks.push_back(track);
    }
    std::vector<int> keyframes;
    for (size_t k = 0; k < keyframe_trip.size(); ++k) {
      if (keyframe_trip[k] == trip) keyframes.push_back(static_cast<int>(k));
    }
    std::vector<std::vector<Detection>> frames;
    size_t unexplained = 0;
    const size_t stride =
        static_cast<size_t>(std::max(kMinFrameStride, config.frame_stride));
    for (size_t index = 0; index < keyframes.size(); index += stride) {
      const int keyframe = keyframes[index];
      const Pose& pose = keyframe_poses[static_cast<size_t>(keyframe)];
      Points world;
      for (const Vec3& point : load(keyframe)) world.push_back(pose * point);
      std::vector<Detection> detections;
      for (Detection& detection :
           DetectObjects(world, keyframe,
                         keyframe_stamp[static_cast<size_t>(keyframe)],
                         pose.translation, levels, structure, config)) {
        // Parked objects are known from the map; clutter is not tracked.
        if (detection.label == config.other_id) continue;
        bool explained = false;
        for (const ObjectBox& parked_box : parked_boxes) {
          explained |= parked_box.Contains(detection.center, kExplainedMargin);
        }
        if (!explained) detections.push_back(std::move(detection));
      }
      frames.push_back(
          SuppressFragments(std::move(detections), config.nms_radius));
      unexplained += frames.back().size();
    }
    const std::vector<Track> tracks = TrackDetections(frames, config, next_id);
    next_id += kTrackIdsPerTrip;
    int moving_tracks = 0;
    for (const Track& track : tracks) {
      const std::map<int, ObjectBox> boxes = MovingTrackBoxes(track, config);
      if (boxes.empty()) continue;
      ++moving_tracks;
      ObjectTrack entry;
      entry.trip = trip;
      entry.track_id = track.track_id;
      entry.label = boxes.begin()->second.label;
      for (const auto& [keyframe, box] : boxes) {
        result.moving[keyframe].push_back(box);
        entry.keyframes.push_back(keyframe);
      }
      result.tracks.push_back(std::move(entry));
    }
    LogLine(log, StringPrintf("[objects] %s: %zu parked, %d moving tracks "
                              "(%zu unexplained detections)",
                              trip_names[static_cast<size_t>(trip)].c_str(),
                              parked_boxes.size(), moving_tracks,
                              unexplained));
  }
  result.parked = std::move(parked.boxes);
  return result;
}

}  // namespace gr
