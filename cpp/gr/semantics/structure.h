// Structure by geometric rules (section 5, route C): floor / ramp / ceiling /
// MEP point labels, columns and walls from multi-height slicing, column-grid
// regularisation and axis labels.
//
// Columns are cells that stay occupied across the horizontal slices up to the
// top one (cars stop below it, ducts start above it). Their faces are then
// measured on the map points (median position of the points whose normals
// face that way), which removes the raster quantisation; columns with an
// unseen side are completed with the typical column size away from the seen
// face (or away from the driven path). Grid lines are clustered from the
// centres for the axis labels. Walls are straight runs of occupied cells,
// thin, and either reaching the ceiling or longer than a car; columns
// embedded in a wall show up as thickness bulges.

#ifndef GARAGE_RECON_GR_SEMANTICS_STRUCTURE_H_
#define GARAGE_RECON_GR_SEMANTICS_STRUCTURE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "gr/core/logging.h"
#include "gr/core/types.h"

namespace gr {

struct StructureConfig {
  std::vector<double> slice_heights = {0.5, 1.0, 1.5, 2.0, 2.4};  // m
  double slice_thickness = 0.12;
  double raster_res = 0.05;
  double column_min = 0.2;  // Column size range (m).
  double column_max = 1.6;
  double wall_min_length = 2.0;
  double wall_max_thickness = 0.8;
  double parapet_min_length = 6.0;  // Low walls must be longer than any car.
  double grid_snap_tol = 0.35;
  bool measure_faces = true;  // False: raster boxes snapped to the grid.
  double mep_band_low = 0.05;  // MEP between ceiling - high and ceiling - low.
  double mep_band_high = 1.2;
  double floor_tolerance = 0.06;  // |height| of floor points (m).
  double ceiling_band = 0.15;
  // Class ids written into the labels.
  uint8_t floor_id = 1;
  uint8_t ramp_id = 2;
  uint8_t ceiling_id = 3;
  uint8_t column_id = 4;
  uint8_t wall_id = 5;
  uint8_t mep_id = 6;
};

// The point map, assigned to levels.
struct StructureInput {
  Points xyz;
  Points normals;
  std::vector<float> height;  // Above the level floor (NaN: none).
  std::vector<int16_t> level;
  std::vector<uint8_t> on_ramp;
  std::vector<uint8_t> dynamic;
  std::vector<double> level_z;
  std::vector<double> ceiling;
  std::vector<Vec2> path_xy;  // Vehicle positions of all trips.
};

struct StructureColumn {
  std::string id;  // "L<level>-<axis label>".
  int level = 0;
  Vec2 center = Vec2::Zero();  // World xy.
  Vec2 size = Vec2::Zero();    // Along grid x / y.
  double yaw = 0.0;            // Grid orientation.
  double z0 = 0.0;
  double z1 = 0.0;
  bool completed = false;  // Extent completed from the typical column size.
  std::string axis_label;
};

struct StructureWall {
  int level = 0;
  Vec2 p0 = Vec2::Zero();
  Vec2 p1 = Vec2::Zero();
  double thickness = 0.0;
};

struct StructureResult {
  std::vector<uint8_t> labels;  // Per map point (0: none).
  std::vector<StructureColumn> columns;
  std::vector<StructureWall> walls;
  double yaw = 0.0;  // Dominant grid orientation (rad).
};

// Orientation of the column / wall grid from vertical-surface normals: the
// 90-degree-symmetric circular mean of their azimuths.
double DominantYaw(const Points& normals);

StructureResult ExtractStructure(const StructureInput& input,
                                 const StructureConfig& config = {},
                                 const Logger& log = nullptr);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_STRUCTURE_H_
