// Vectorised floor markings (section 5, module 3): a binary paint mask of
// one level -> skeleton -> junctions split off -> one line fit per branch
// (curved branches such as arrow heads and text are dropped) -> collinear
// fragments merged -> length / width filters; then the checks that paint
// lies on observed floor and follows the column grid, the sub-pixel
// re-measurement on the floor images (vector_fit.h) and the lane / parking
// line classification. The raster preparation (top-hat, thresholds, the
// line-shaped closing and opening, floor arrows) stays with OpenCV in the
// Python stage.

#ifndef GARAGE_RECON_GR_SEMANTICS_MARKINGS_H_
#define GARAGE_RECON_GR_SEMANTICS_MARKINGS_H_

#include <cstdint>
#include <map>
#include <vector>

#include "gr/core/types.h"
#include "gr/semantics/vector_fit.h"

namespace gr {

struct MarkingConfig {
  double min_length = 0.8;  // m
  double min_width = 0.05;
  double max_width = 0.35;
  double merge_gap = 2.5;     // Collinear fragments this far apart join (m).
  double merge_angle = 4.0;   // deg
  double merge_offset = 0.15;  // m off the line.
  double max_axis_angle = 5.0;  // deg off the grid.
  double lane_min_length = 8.0;  // m
};

struct MarkingSegment {
  int level = 0;
  Vec2 p0 = Vec2::Zero();
  Vec2 p1 = Vec2::Zero();
  double width = 0.0;
  bool lane = false;  // Lane marking, else parking line.
  double confidence = 1.0;
  // p0 / p1 measured on the floor image with observed floor beyond them.
  bool end_ok[2] = {false, false};

  double length() const { return (p1 - p0).norm(); }
};

// Raster frame of a level: cell (col i, row j) covers [x0 + i res, x0 + (i +
// 1) res) x [y0 + j res, ...), row-major (height x width).
struct RasterFrame {
  double x0 = 0.0;
  double y0 = 0.0;
  double res = 0.05;
  int width = 0;
  int height = 0;
};

// Merges segments that are parallel (within `merge_angle`), on one line
// (both ends of the other within `merge_offset` of it) and at most
// `merge_gap` apart along it; each segment absorbs every partner it finds.
std::vector<MarkingSegment> MergeCollinear(std::vector<MarkingSegment> segs,
                                           const MarkingConfig& config);

// Line segments of a paint mask (merged and filtered). `skeleton` is the
// mask's skeleton when the caller has one; empty: Skeletonize (scikit-image's
// thinning, which the slot assembly was tuned with: plain Zhang-Suen splits
// lines differently at diagonal steps).
std::vector<MarkingSegment> VectorizeMarkings(
    const std::vector<uint8_t>& mask, const RasterFrame& frame,
    const MarkingConfig& config, int level,
    const std::vector<uint8_t>& skeleton = {});

// Paint lies on the floor: at least half of each side of the line, offset
// by half its width plus `offset`, must be observed floor.
bool BothSidesObserved(const MarkingSegment& seg,
                       const std::vector<uint8_t>& floor,
                       const RasterFrame& frame, double offset = 0.2,
                       double min_fraction = 0.5);

// Parking and lane lines follow the column grid (angle mod 90 deg).
bool GridAligned(const MarkingSegment& seg, double yaw, double max_angle_deg);

// Re-measures the segments in place on the first image that works; returns
// how many were refined. Others keep the skeleton geometry, ends unreliable.
int RefineSegments(std::vector<MarkingSegment>& segs,
                   const std::vector<const FloorImage*>& images,
                   const LineFitConfig& config = {});

// Long lines along the driven path of their level are lane markings
// (`paths` maps a level to the vehicle positions on it).
void ClassifyMarkings(std::vector<MarkingSegment>& segs,
                      const std::map<int, std::vector<Vec2>>& paths,
                      const MarkingConfig& config);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_MARKINGS_H_
