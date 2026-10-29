// Global point map (sections 4.2 / 4.3): accumulation of the keyframe sweeps,
// oriented normals, and dynamic-point detection by per-session range-image
// visibility voting plus the ego vehicle's swept volume.

#ifndef GARAGE_RECON_GR_MAPPING_STATIC_MAP_H_
#define GARAGE_RECON_GR_MAPPING_STATIC_MAP_H_

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <unordered_map>
#include <vector>

#include "gr/core/geometry.h"
#include "gr/core/pose.h"
#include "gr/core/voxel.h"
#include "gr/mapping/visibility.h"

namespace gr {

// Distinct entries of a symmetric 3 x 3 matrix (xx, xy, xz, yy, yz, zz).
inline constexpr int kSymmetricTerms = 6;

// Neighbours of the map normals (EstimateOrientedNormals defaults).
inline constexpr int kMapNormalMinNeighbors = 6;
inline constexpr int kMapNormalMaxNeighbors = 48;

// How the observations falling into one voxel become a map point.
struct MapFusionConfig {
  // false: the observation from the closest sensor position wins (the v1
  // map). true: the observations taken from comparable ranges are averaged,
  // weighted by 1 / sigma(range)^2, and the point carries a sigma.
  bool fuse = true;
  // Observations from farther than range_ratio * nearest + range_margin are
  // left out: at grazing angles an attitude error moves far points most.
  double range_ratio = 2.0;
  double range_margin = 3.0;      // m
  double range_sigma = 0.01;      // m, ranging noise
  double angle_sigma_deg = 0.02;  // pose and beam angle; grows with range
  double sigma_floor = 0.003;     // m, pose / calibration residual
};

// Voxels with x in [x0, x1) and y in [y0, y1) (voxel indices, any z).
struct VoxelRegion {
  int x0 = 0;
  int x1 = 0;
  int y0 = 0;
  int y1 = 0;

  bool Contains(const VoxelKey& key) const {
    return key.x >= x0 && key.x < x1 && key.y >= y0 && key.y < y1;
  }
};

// Merges sweeps into one voxel-downsampled cloud of the returns within
// `max_range`. Without fusion, per voxel the observation taken from the
// closest sensor position wins: at grazing angles a 0.3 deg attitude error
// already moves a floor point at 25 m by more than 10 cm. With fusion, a
// first pass (AddRanges over all sweeps) finds the nearest range of every
// voxel and Add then averages the observations within the range window.
//
// A voxel only depends on the returns that fall into it, so a map can be
// built region by region (SetRegion) with the same voxels as in one piece.
class MapAccumulator {
 public:
  // `voxel` must be positive.
  MapAccumulator(double voxel, double max_range,
                 const MapFusionConfig& fusion = {});
  MapAccumulator(const MapAccumulator&) = default;
  MapAccumulator& operator=(const MapAccumulator&) = default;
  MapAccumulator(MapAccumulator&&) = default;
  MapAccumulator& operator=(MapAccumulator&&) = default;

  struct Cloud {
    Points points;
    std::vector<float> intensity;
    std::vector<int> keyframe;      // Keyframe of the nearest observation.
    std::vector<int> observations;  // Observations averaged per point.
    // Covariance of the averaged observations (xx, xy, xz, yy, yz, zz), m^2.
    std::vector<std::array<float, kSymmetricTerms>> spread;
    std::vector<float> model_sigma;  // Formal sigma of the weighted mean (m).
    // The order key of Collect: nearest observation range, then the
    // keyframe (high 32 bits) and point index of that observation. The
    // points of separately built regions merge into the order of one map
    // by sorting on (range, sequence).
    std::vector<double> range;
    std::vector<int64_t> sequence;
    std::vector<VoxelKey> voxel;  // The voxel of each point.
  };

  // Only voxels in `region` are kept from now on.
  void SetRegion(const VoxelRegion& region) { region_ = region; }

  // Fusion: first pass, records the nearest observation range per voxel.
  void AddRanges(const Pose& world_from_lidar, const Points& sweep);
  void Add(int keyframe, const Pose& world_from_lidar, const Points& sweep,
           const std::vector<float>& intensity);
  // The merged cloud, ordered by observation range (nearest first).
  Cloud Collect() const;
  size_t size() const { return cells_.size(); }
  const MapFusionConfig& fusion() const { return fusion_; }

 private:
  // A cell that has not seen an observation yet.
  static constexpr int kNoKeyframe = -1;

  struct Cell {
    double range = 0.0;  // Nearest observation.
    int64_t sequence = 0;
    Vec3 point = Vec3::Zero();  // Nearest observation (no fusion).
    float intensity = 0.0f;
    int keyframe = kNoKeyframe;
    // Fusion sums, relative to the voxel centre.
    float weight = 0.0f;
    std::array<float, kSpaceDim> sum{};
    std::array<float, kSymmetricTerms> outer{};
    float weighted_intensity = 0.0f;
    int count = 0;
  };

  double voxel_;
  double inverse_voxel_;
  double max_range_;
  MapFusionConfig fusion_;
  std::optional<VoxelRegion> region_;
  std::unordered_map<VoxelKey, Cell, VoxelKeyHash> cells_;
  std::unordered_map<VoxelKey, float, VoxelKeyHash> nearest_;  // AddRanges.
};

// Sigma (m) of every fused map point along its surface normal: the spread of
// its observations along the normal over sqrt(observations), at least the
// formal sigma of the weighted mean, combined with the floor. Points without
// a normal use the mean variance of the spread.
std::vector<float> PointSigmas(const MapAccumulator::Cloud& cloud,
                               const Points& normals, double sigma_floor);

// Normals of the map points, flipped towards the sensor position of the
// keyframe that observed each point.
NormalsResult EstimateOrientedNormals(
    const Points& points, const std::vector<int>& keyframe,
    const std::vector<Vec3>& keyframe_origins, double radius,
    int min_neighbors = kMapNormalMinNeighbors,
    int max_neighbors = kMapNormalMaxNeighbors);

// Median elevation gap (deg) between LiDAR beams measured on sweeps, so that
// range-image rows match the scanner.
double EstimateBeamSpacing(const std::vector<Points>& sweeps);

struct DynamicConfig {
  bool enabled = true;
  double ray_voxel = 0.1;  // Sweep downsampling for the range images.
  double az_res = 0.8;     // Range-image bins (deg).
  // <= 0: measured beam spacing (SetElevationResolution or the first sweep).
  double el_res = 0.0;
  double el_min = -30.0;
  double el_max = 20.0;
  double max_range = 30.0;
  double margin_abs = 0.3;
  double margin_rel = 0.02;
  double min_cos = 0.25;  // Ignore grazing rays.
  int min_through = 2;
  double min_ratio = 0.6;    // Within one session.
  double min_height = 0.15;  // Never dynamic below this height above floor.
  double ego_radius = 0.9;   // The ego vehicle's path is free (half width).
  double ego_height = 2.0;   // Up to this height above the floor.
  double bucket = 4.0;
};

enum class DynamicKind : uint8_t {
  kStatic = 0,
  kMoved = 1,    // Moved within its own session.
  kChanged = 2,  // Came or went between sessions.
};

// A point is dynamic if the sweeps of some session repeatedly see *through*
// it at a non-grazing angle, or if the ego vehicle drove through it. If its
// own session does, the object moved during that session; if only other
// sessions do, it came or went between sessions (e.g. a car parked during one
// drive).
class DynamicDetector {
 public:
  // One normal and one trip index per map point.
  DynamicDetector(const Points& map, const Points& normals,
                  const std::vector<int>& trip, const DynamicConfig& config);
  DynamicDetector(const DynamicDetector&) = default;
  DynamicDetector& operator=(const DynamicDetector&) = default;
  DynamicDetector(DynamicDetector&&) = default;
  DynamicDetector& operator=(DynamicDetector&&) = default;

  size_t num_points() const { return map_.size(); }
  void SetElevationResolution(double degrees) {
    elevation_resolution_ = degrees;
  }
  double elevation_resolution() const { return elevation_resolution_; }
  // Height of every map point above its floor; lower points are never
  // dynamic (floors are grazed by many rays).
  void SetHeights(const std::vector<double>& height);

  // One keyframe sweep (LiDAR frame) of session `trip`.
  void AddSweep(int trip, const Pose& world_from_lidar, const Points& sweep);
  // Positions of the vehicle origin (on the floor) along a trip.
  void AddEgoPath(int trip, const Points& vehicle_positions);

  std::vector<DynamicKind> Classify() const;
  // Map points inside the volume the ego vehicle of `trip` drove through.
  std::vector<bool> EgoSwept(int trip) const;

 private:
  DynamicConfig config_;
  Points map_;
  std::vector<int> trip_;
  std::vector<double> height_;
  VisibilityVoter voter_;
  double elevation_resolution_ = 0.0;  // deg; <= 0: not measured yet.
  std::map<int, Points> ego_paths_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_MAPPING_STATIC_MAP_H_
