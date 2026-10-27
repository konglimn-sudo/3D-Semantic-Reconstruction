// Per-level bird's-eye-view rasters of the point map (recon/bev.py): the
// level's grid, the LiDAR intensity and observation count of its floor, the
// highest object standing on it and the floor height (levels and ramps).
//
// The point attributes are those of map/point_map.npz; float32 attributes
// are compared with thresholds in float32, as numpy does with a Python
// float, so that the rasters equal the Python stage's.

#ifndef GARAGE_RECON_GR_MAPPING_BEV_H_
#define GARAGE_RECON_GR_MAPPING_BEV_H_

#include <cstdint>
#include <vector>

#include "gr/core/types.h"
#include "gr/mapping/levels.h"
#include "gr/mapping/raster.h"

namespace gr {

// The static points of one map (point_map.npz fields); `normals` and
// `dynamic` may be empty.
struct BevPoints {
  const Points* xyz = nullptr;
  const std::vector<float>* intensity = nullptr;
  const std::vector<int16_t>* level = nullptr;
  const std::vector<uint8_t>* dynamic = nullptr;
  const std::vector<float>* height = nullptr;  // NaN: no level.
  const Points* normals = nullptr;             // float32 values.
};

// The bounds of the static points of `level` plus `margin` (bev.level_grid);
// an empty grid (0 x 0) without such points.
Grid2D LevelGrid(const BevPoints& points, int level, double res,
                 double margin);

// Floor height per cell (row-major, height x width): the level's floor grid
// (else its height) and, inside the outline of a ramp leaving the level
// downwards, the ramp plane (bev.floor_height_raster). With a floor grid
// the values are float32, as the grid's.
struct FloorHeights {
  std::vector<double> z;
  bool single_precision = false;
};
FloorHeights FloorHeightRaster(const LevelModel& levels, int level,
                               const Grid2D& grid);

struct BevConfig {
  double floor_tolerance = 0.1;    // |height| of floor points (m).
  double object_max_height = 2.2;  // Objects on the floor, below it (m).
};

struct BevRasters {
  Eigen::MatrixXd intensity;    // Mean of floor points, NaN where none.
  Eigen::MatrixXd floor_count;  // Floor points per cell.
  Eigen::MatrixXd height_max;   // Highest object point, NaN where none.
  FloorHeights floor_z;
};

// bev.level_rasters: floor points are within floor_tolerance of the floor
// with an upward normal (z > 0.7); objects stand between floor_tolerance
// and object_max_height (the ceiling and the services below it do not
// count).
BevRasters LevelRasters(const BevPoints& points, const LevelModel& levels,
                        int level, const Grid2D& grid,
                        const BevConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_MAPPING_BEV_H_
