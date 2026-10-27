// Morphology and connected components on row-major rasters (height x width),
// with the conventions of the Python tooling they replace (scipy.ndimage and
// OpenCV), so that results carry over unchanged:
//
//   DilateCross        scipy binary_dilation, default (4-connected) structure,
//                      outside = 0; ErodeCross likewise (outside = 0, as
//                      binary_erosion's border_value);
//   ErodeKernel /      cv2.erode / cv2.dilate of an 8-bit image by a binary
//   DilateKernel       kernel anchored at its centre, default border (cells
//                      outside are ignored);
//   EllipseKernel      cv2.getStructuringElement(MORPH_ELLIPSE);
//   DrawLine8          cv2.line, 8-connected, thickness 1;
//   FillPoly8          cv2.fillPoly with LINE_8 and fixed-point vertices
//                      (`shift` fractional bits): the clipped outline plus
//                      the edge-table scanline fill;
//   GaussianFilter     scipy gaussian_filter (mode reflect, truncate 4);
//   OtsuThreshold      cv2.threshold with THRESH_OTSU (8-bit);
//   MaxFilter3x3       scipy grey_dilation(size=3), edges repeated;
//   OpenLine           cv2.morphologyEx(MORPH_OPEN) with a 1 x L (or L x 1)
//                      kernel anchored at L / 2: erosion counts the outside
//                      as set, dilation as unset;
//   Label8             scipy label with a 3 x 3 structure: components numbered
//                      in raster order of their first cell;
//   DistanceTransform  exact Euclidean distances to the nearest source cell;
//   ThinMask           Zhang-Suen thinning;
//   Skeletonize        scikit-image's skeletonize (Zhang-Suen with its
//                      lookup table: differs from ThinMask at diagonal
//                      steps).

#ifndef GARAGE_RECON_GR_CORE_MORPHOLOGY_H_
#define GARAGE_RECON_GR_CORE_MORPHOLOGY_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace gr {

using Mask = std::vector<uint8_t>;  // 0 / 1 per cell.

Mask DilateCross(const Mask& mask, int width, int height);
Mask ErodeCross(const Mask& mask, int width, int height);
// scipy binary_closing with the cross structure and `iterations`.
Mask CloseCross(const Mask& mask, int width, int height, int iterations);

std::vector<uint8_t> ErodeKernel(const std::vector<uint8_t>& image, int width,
                                 int height, const Mask& kernel,
                                 int kernel_size);
std::vector<uint8_t> DilateKernel(const std::vector<uint8_t>& image,
                                  int width, int height, const Mask& kernel,
                                  int kernel_size);

// Binary closing / opening with a kernel (cv2.morphologyEx MORPH_CLOSE /
// MORPH_OPEN, default border): 0 / 1 masks.
Mask CloseKernel(const Mask& mask, int width, int height, const Mask& kernel,
                 int size);
Mask OpenKernel(const Mask& mask, int width, int height, const Mask& kernel,
                int size);

// size x size.
Mask EllipseKernel(int size);

// Sets the cells of the 8-connected line from (x0, y0) to (x1, y1) (x =
// column), all inside the raster.
void DrawLine8(int x0, int y0, int x1, int y1, int width, Mask* mask);

// Sets the cells of one polygon (vertices (x = column, y = row) * 2^shift,
// 0 <= shift <= 16) to `value`; parts outside the raster are clipped.
void FillPoly8(const std::vector<std::pair<int64_t, int64_t>>& vertices,
               int shift, int width, int height, int32_t value,
               std::vector<int32_t>* image);

std::vector<double> GaussianFilter(const std::vector<double>& image,
                                   int width, int height, double sigma);

int OtsuThreshold(const std::vector<uint8_t>& values);

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

// skimage.morphology.skeletonize of a 2D mask (identical output): passes
// that remove the cells its lookup table marks 1 or 3, then 2 or 3, from
// the 8-neighbour code of the state at the start of the pass, until none
// is removed.
Mask Skeletonize(const Mask& mask, int width, int height);

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
