// Per-level navigation layers on a grid (nav/layers.py).
//
// Obstacles come from the geometry: every static map point between
// obstacle_min_height and the vehicle height plus top_margin above the floor
// blocks its cell whatever its class, so a mislabelled wall still blocks.
// Unclassified points must add up to min_cell_observations fused
// observations per cell: a static object is seen in many sweeps, remnants
// of moving objects and noise once or twice. The semantic classes then
// split the obstacles into layers: permanent structure, the part of it
// that hangs from above, parked vehicles at capture time and low objects
// the vehicle may cross. float32 point attributes are compared with the
// thresholds in float32, as numpy does, and polygons are filled as
// cv2.fillPoly does, so the layers equal the Python stage's.

#ifndef GARAGE_RECON_GR_NAV_LAYERS_H_
#define GARAGE_RECON_GR_NAV_LAYERS_H_

#include <cstdint>
#include <vector>

#include "gr/core/morphology.h"
#include "gr/core/types.h"
#include "gr/mapping/bev.h"
#include "gr/mapping/levels.h"
#include "gr/mapping/raster.h"

namespace gr {

// Fills the cells of a world polygon (x, y) into `image` (row-major, the
// grid's size) with `value`: cv2.fillPoly of the vertices in cell
// coordinates with 4 fractional bits (layers.fill_polygon).
void FillWorldPolygon(const std::vector<Vec2>& polygon, const Grid2D& grid,
                      int32_t value, std::vector<int32_t>* image);

// `known` plus the enclosed unknown patches (neither known nor obstacle,
// 8-connected, not touching the border) smaller than `max_area` m^2: floor
// hidden behind parked cars or between sparse scan lines
// (layers.fill_holes).
Mask FillHoles(const Mask& known, const Mask& obstacles, int width,
               int height, double res, double max_area);

struct LayerConfig {
  double floor_tolerance = 0.1;
  double obstacle_min_height = 0.1;
  int obstacle_min_points = 3;
  double min_cell_observations = 3.0;
  double max_headroom = 8.0;
  int floor_closing = 2;  // Cells (ellipse radius).
  double max_hole_area = 4.0;
  double vehicle_height = 2.0;
  double vehicle_top_margin = 0.2;
  double vehicle_width = 1.9;
};

// Semantic class ids of the map points (semantics/taxonomy.py).
struct LayerClasses {
  std::vector<int> vehicle;   // Vehicles, two-wheelers, pedestrians.
  std::vector<int> low;       // Wheel stoppers, speed bumps.
  std::vector<int> overhead;  // Ceiling, MEP, beams, signs, lights.
  int unlabeled = 0;
};

// The map points with their class ids and fused observations (empty: every
// point counts as min_cell_observations).
struct LayerPoints {
  BevPoints map;
  const std::vector<uint8_t>* labels = nullptr;
  const std::vector<uint16_t>* observations = nullptr;
};

struct LevelLayers {
  Mask floor;      // Observed floor, minus driven and ramp cells.
  Mask driven;     // Swept by the vehicle during the capture.
  Mask structure;  // Permanent obstacle within the vehicle's height band.
  Mask overhead;   // The part of `structure` hanging from above.
  Mask parked;     // Parked vehicles (and people) at capture time.
  Mask low;        // Speed bumps and wheel stoppers.
  std::vector<int32_t> slot;  // Slot index + 1, 0 outside the slots.
  Mask ramp;
  FloorHeights floor_z;
  std::vector<float> clearance;  // To the nearest blocked cell (m).
  std::vector<float> headroom;   // Lowest static object above (inf: none).
};

// One slot outline (corners) with its index into all slots.
struct LayerSlot {
  int index = 0;
  std::vector<Vec2> corners;
};

// The layers of `level` on `grid` (layers.level_layers): `slots` of this
// level, `path_xy` the vehicle positions on it during the capture.
LevelLayers ComputeLevelLayers(const LayerPoints& points,
                               const LayerClasses& classes,
                               const LevelModel& levels, int level,
                               const std::vector<LayerSlot>& slots,
                               const std::vector<Vec2>& path_xy,
                               const Grid2D& grid, const LayerConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_NAV_LAYERS_H_
