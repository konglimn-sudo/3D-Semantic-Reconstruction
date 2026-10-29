// Sub-pixel vector fitting of painted lines on the floor orthophoto (plan
// step 4).
//
// The skeleton of a thresholded marking mask finds the lines, but its ends
// are eroded by about half a line width and its centre is quantised to the
// raster cells. Each segment is therefore measured again on a brightness
// image of the floor (the camera orthophoto where it was observed, else the
// LiDAR intensity):
//
//   centre line  at stations along the segment, the brightness profile
//                across it gives the stripe edges (half way between paint
//                and floor); a robust line fit through the stripe centres;
//   ends         along the centre line the end is where the brightness falls
//                half way from the paint to the floor beyond it. An end is
//                reliable only if floor is observed beyond it; otherwise a
//                car, the wall base or the edge of the scan may hide part of
//                the line, and the slot assembly takes the end from the other
//                separators of the row.
//
// The image is sampled bilinearly in world coordinates (1 cm steps), so the
// results are not tied to the raster cells.

#ifndef GARAGE_RECON_GR_SEMANTICS_VECTOR_FIT_H_
#define GARAGE_RECON_GR_SEMANTICS_VECTOR_FIT_H_

#include <optional>
#include <string>
#include <vector>

#include "gr/core/types.h"

namespace gr {

// Brightness (normalised to [0, 1]) and observed fraction of one level's
// floor; cell (col i, row j) is centred at (x0 + (i + 0.5) res, y0 + (j +
// 0.5) res), row-major.
struct FloorImage {
  std::string name;  // "camera" or "lidar".
  std::vector<double> brightness;
  std::vector<double> valid;
  int width = 0;
  int height = 0;
  double x0 = 0.0;
  double y0 = 0.0;
  double res = 0.05;

  // Bilinear; outside the image the edge cells extend.
  double Brightness(const Vec2& xy) const;
  // Bilinear; outside the image counts as unobserved.
  double Observed(const Vec2& xy) const;
};

struct LineFitConfig {
  double station_step = 0.05;  // m between profiles across the line.
  double sample_step = 0.01;   // m along a profile.
  // Profiles across the line stay within 0.25 m: parked cars leave about
  // 0.3 m of floor beside a separator.
  double half_span = 0.25;
  double floor_band_inner = 0.12;  // Floor part of a profile (m) ...
  double floor_band_outer = 0.25;
  double end_search = 0.6;  // m beyond the skeleton end.
  double end_floor_near = 0.12;  // Floor beyond an end (m) ...
  double end_floor_far = 0.6;
  double end_floor_width = 0.1;  // ... and this far to either side (m).
  double min_contrast = 0.08;  // Paint above floor (normalised brightness).
  double min_valid = 0.8;      // Observed fraction of a floor sample.
  double max_end_shift = 0.3;  // m from the skeleton end.
};

// A line segment has two ends.
inline constexpr int kSegmentEnds = 2;

struct LineFit {
  Vec2 p0 = Vec2::Zero();  // Centre line ends.
  Vec2 p1 = Vec2::Zero();
  double width = 0.0;
  bool end_ok[kSegmentEnds] = {false, false};
  int stations = 0;
  double rms = 0.0;  // Of the stripe centres about the fitted line (m).
};

// Re-measures one painted line; nullopt if too few stations are usable.
std::optional<LineFit> FitLine(const Vec2& p0, const Vec2& p1,
                               const FloorImage& image,
                               const LineFitConfig& config = {});

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_VECTOR_FIT_H_
