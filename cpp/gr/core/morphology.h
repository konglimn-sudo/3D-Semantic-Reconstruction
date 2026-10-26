// Morphology and connected components on row-major rasters (height x width),
// with the conventions of the Python tooling they replace (scipy.ndimage and
// OpenCV), so that results carry over unchanged:
//
//   DilateCross        scipy binary_dilation, default (4-connected) structure,
//                      outside = 0;
//   MaxFilter3x3       scipy grey_dilation(size=3), edges repeated;
//   OpenLine           cv2.morphologyEx(MORPH_OPEN) with a 1 x L (or L x 1)
//                      kernel anchored at L / 2: erosion counts the outside
//                      as set, dilation as unset;
//   Label8             scipy label with a 3 x 3 structure: components numbered
//                      in raster order of their first cell;
//   DistanceTransform  exact Euclidean distances to the nearest source cell;
//   ThinMask           Zhang-Suen thinning.

#ifndef GARAGE_RECON_GR_CORE_MORPHOLOGY_H_
#define GARAGE_RECON_GR_CORE_MORPHOLOGY_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace gr {

using Mask = std::vector<uint8_t>;  // 0 / 1 per cell.

Mask DilateCross(const Mask& mask, int width, int height);

std::vector<double> MaxFilter3x3(const std::vector<double>& grid, int width,
                                 int height);

// Opening by a line of `length` cells along x (`along_x`) or y.
Mask OpenLine(const Mask& mask, int width, int height, int length,
              bool along_x);

// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher): distance
// in cells from every cell to the nearest nonzero cell of `sources`; +inf
// without any source.
std::vector<float> DistanceTransform(const std::vector<uint8_t>& sources,
                                     int width, int height);

// Zhang-Suen thinning of a binary mask to a one-cell-wide 8-connected
// skeleton.
std::vector<uint8_t> ThinMask(const std::vector<uint8_t>& mask, int width,
                              int height);

// Cells covered by a polygon given in (fractional) cell coordinates (x =
// column, y = row), as cv2.fillPoly with the vertices rounded to cells: the
// cells whose centre lies inside or on the boundary.
Mask FillPolygon(const std::vector<std::pair<int, int>>& vertices, int width,
                 int height);

struct Components {
  std::vector<int> label;               // 0: background, else 1..count.
  std::vector<std::vector<int>> cells;  // Per component, raster order.
};

Components Label8(const Mask& mask, int width, int height);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_MORPHOLOGY_H_
