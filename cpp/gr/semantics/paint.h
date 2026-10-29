// Painted cells of a level's floor rasters (markings stage, before the
// lines are vectorized; the Python stage used scipy / OpenCV, and the
// primitives in core/morphology.h keep their conventions so that the masks
// are the same).
//
//   layers      the LiDAR intensity (sparse samples filled by normalised
//               convolution, sigma 1 cell) and the camera texture's
//               luminance, each normalised by its 2nd / 99.5th percentile
//               of the observed cells; cells under an object (height_max
//               above 0.3 m) are not observed;
//   per layer   white top-hat (ellipse of `tophat_size`) above the largest
//               of Otsu's threshold, median + `mad_k` MAD and `min_contrast`;
//   elongated   along the grid axes (`yaw`, yaw + 90 deg): closed by a line
//               of `line_close_length` (bridges the LiDAR ring samples),
//               opened by one of `line_open_length`; then a 3 x 3 closing
//               and pieces of at least `min_area` m^2.

#ifndef GARAGE_RECON_GR_SEMANTICS_PAINT_H_
#define GARAGE_RECON_GR_SEMANTICS_PAINT_H_

#include <cstdint>
#include <string>
#include <vector>

#include "gr/core/morphology.h"

namespace gr {

struct PaintConfig {
  double tophat_size = 0.6;  // m
  double min_contrast = 0.08;
  double mad_k = 4.0;
  double line_close_length = 0.5;  // m
  double line_open_length = 0.5;   // m
  double min_area = 0.05;          // m^2
};

// Rasters of one level, row-major (height x width); empty when absent.
struct PaintRasters {
  int width = 0;
  int height = 0;
  std::vector<double> intensity;        // NaN where unobserved.
  std::vector<uint8_t> rgb;             // kRgbChannels per cell.
  std::vector<int> texture_nobs;        // Texture observations per cell.
  std::vector<double> height_max;       // NaN where unobserved.
};

struct PaintLayer {
  std::string name;  // lidar_intensity | camera_texture
  std::vector<double> brightness;  // [0, 1]
  Mask valid;
};

// Empty if neither the intensity nor the texture has a sample.
std::vector<PaintLayer> MarkingLayers(const PaintRasters& rasters);

// The floor images the line fit measures on (vector_fit.py): "camera" (the
// texture's luminance) then "lidar" (the filled intensity, observed where
// the fill weight exceeds 0.15), covered cells unobserved, each normalised
// by the percentiles of its observed cells.
std::vector<PaintLayer> FloorImages(const PaintRasters& rasters);

Mask LayerMask(const PaintLayer& layer, int width, int height, double res,
               const PaintConfig& config);

// The union of the layer masks.
Mask PaintMask(const std::vector<PaintLayer>& layers, int width, int height,
               double res, const PaintConfig& config);

Mask Elongated(const Mask& mask, int width, int height, double res,
               double yaw, const PaintConfig& config);

// A line kernel of `length` cells at `angle` through the kernel centre (the
// Python stage's _line_kernel: end points rounded half to even).
Mask LineKernel(int length, double angle);

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_PAINT_H_
