// Levels (B1 / B2 / ...) and ramps of a garage (sections 4.3 / 5).
//
// The vehicle drives on the floor, so its trajectory is the most robust
// source of floor heights: flat stretches give the levels, sloped stretches
// the ramps. Map points then refine a floor height grid per level (drainage
// slopes). The grid takes the upward-facing points near the level's own
// trajectory height at their position, not near one constant: a level may
// rise gently along a long aisle (FusionPortableV2: 4.5 m over 400 m at
// 1.1 %, below the ramp slope).

#ifndef GARAGE_RECON_GR_MAPPING_LEVELS_H_
#define GARAGE_RECON_GR_MAPPING_LEVELS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "gr/core/logging.h"
#include "gr/core/types.h"

namespace gr {

struct LevelConfig {
  double min_separation = 2.2;  // Between level heights (m).
  double flat_slope = 0.03;     // |dz / ds| below this: a level floor.
  int min_samples = 20;         // Flat path samples per level.
  double grid_res = 0.5;        // Floor grid (m).
  double floor_band = 0.3;      // Floor points within this of the path (m).
};

// Floor height raster; cell (col i, row j) covers [x0 + i res, x0 + (i + 1)
// res) x [y0 + j res, ...), row-major.
struct FloorGrid {
  double x0 = 0.0;
  double y0 = 0.0;
  double res = 0.5;
  int width = 0;
  int height = 0;
  std::vector<float> z;

  // Height at `xy` (the nearest cell inside the grid).
  double Sample(const Vec2& xy) const;
};

struct Ramp {
  int level_from = 0;  // Upper level.
  int level_to = 0;    // Lower level.
  Vec3 plane = Vec3::Zero();  // z = a x + b y + c.
  std::vector<Vec2> polygon;  // Outline (convex hull of its floor points).
  double slope = 0.0;

  bool Contains(const Vec2& xy) const;
  double ZAt(const Vec2& xy) const {
    return plane.x() * xy.x() + plane.y() * xy.y() + plane.z();
  }
};

struct LevelModel {
  std::vector<double> z;        // Floor height per level, level 0 on top.
  std::vector<double> ceiling;  // Ceiling height per level.
  std::vector<FloorGrid> grids;
  std::vector<Ramp> ramps;

  int num_levels() const { return static_cast<int>(z.size()); }
  double FloorZ(const Vec2& xy, int level) const;
};

struct LevelAssignment {
  std::vector<int16_t> level;  // -1: below every floor.
  std::vector<float> height;   // Above that floor (NaN without a level).
  std::vector<uint8_t> on_ramp;
};

// A point belongs to the highest floor that is not above it (by more than
// 0.3 m); points over a ramp outline that sit closer to the ramp surface use
// the ramp (and its upper level) instead.
LevelAssignment AssignLevels(const LevelModel& model, const Points& xyz);

// Horizontal bounds of a map's points.
struct MapBounds {
  Vec2 lo = Vec2::Zero();
  Vec2 hi = Vec2::Zero();
};

// Levels and ramps from the vehicle paths (one polyline of vehicle origins,
// on the floor, per trip) and the map points with normals. Only points with
// normal z above 0.8 or below -0.85 (floors, ramps, ceilings) count, so a
// map built in tiles can pass just those, in map order, with the bounds of
// all its points (`bounds`; otherwise the bounds of `xyz`).
LevelModel DetectLevels(const std::vector<Points>& vehicle_paths,
                        const Points& xyz, const Points& normals,
                        const LevelConfig& config = {},
                        const Logger& log = nullptr,
                        const std::optional<MapBounds>& bounds = std::nullopt);

}  // namespace gr

#endif  // GARAGE_RECON_GR_MAPPING_LEVELS_H_
