// Morphology, filters and connected components on row-major rasters of
// `height` rows by `width` columns (cell (x, y) is element y * width + x;
// x = column, y = row).
//
// Every function follows the Python tooling it replaces (scipy.ndimage,
// OpenCV, scikit-image) closely enough that results carry over unchanged;
// each comment names its reference and the border convention it copies.

#ifndef GARAGE_RECON_GR_CORE_MORPHOLOGY_H_
#define GARAGE_RECON_GR_CORE_MORPHOLOGY_H_

#include <cstdint>
#include <utility>
#include <vector>

namespace gr {

using Mask = std::vector<uint8_t>;  // 0 / 1 per cell.

// scipy binary_dilation with the default (4-connected cross) structure;
// cells outside the raster count as unset.
Mask DilateCross(const Mask& mask, int width, int height);

// scipy binary_erosion with the cross structure; cells outside the raster
// count as unset (binary_erosion's border_value = 0).
Mask ErodeCross(const Mask& mask, int width, int height);

// scipy binary_closing with the cross structure and `iterations`.
Mask CloseCross(const Mask& mask, int width, int height, int iterations);

// scipy binary_fill_holes (cross structure): background not 4-connected to
// the raster border becomes foreground.
Mask FillHolesCross(const Mask& mask, int width, int height);

// cv2.erode of an 8-bit image by a `kernel_size` x `kernel_size` binary
// kernel anchored at its centre, default border (cells outside the raster
// do not take part).
std::vector<uint8_t> ErodeKernel(const std::vector<uint8_t>& image, int width,
                                 int height, const Mask& kernel,
                                 int kernel_size);

// cv2.dilate, as ErodeKernel.
std::vector<uint8_t> DilateKernel(const std::vector<uint8_t>& image, int width,
                                  int height, const Mask& kernel,
                                  int kernel_size);

// Binary closing with a kernel (cv2.morphologyEx MORPH_CLOSE, default
// border) of a 0 / 1 mask.
Mask CloseKernel(const Mask& mask, int width, int height, const Mask& kernel,
                 int kernel_size);

// Binary opening with a kernel (cv2.morphologyEx MORPH_OPEN, default
// border) of a 0 / 1 mask.
Mask OpenKernel(const Mask& mask, int width, int height, const Mask& kernel,
                int kernel_size);

// cv2.getStructuringElement(MORPH_ELLIPSE, (kernel_size, kernel_size)).
Mask EllipseKernel(int kernel_size);

// Sets the cells of the 8-connected line from (start_x, start_y) to
// (end_x, end_y) as cv2.line with thickness 1; both ends must be inside the
// raster.
void DrawLine8(int start_x, int start_y, int end_x, int end_y, int width,
               Mask* mask);

// cv2.fillPoly of one polygon with LINE_8: sets the clipped outline and the
// scanline fill of the polygon to `fill_value`. The vertices are fixed-point
// (x, y) * 2^fraction_bits, 0 <= fraction_bits <= 16; parts outside the
// raster are clipped.
void FillPoly8(const std::vector<std::pair<int64_t, int64_t>>& vertices,
               int fraction_bits, int width, int height, int32_t fill_value,
               std::vector<int32_t>* image);

// scipy gaussian_filter with mode 'reflect' and truncate 4.
std::vector<double> GaussianFilter(const std::vector<double>& image, int width,
                                   int height, double sigma);

// The threshold cv2.threshold with THRESH_OTSU picks for 8-bit `values`;
// 0 without values.
int OtsuThreshold(const std::vector<uint8_t>& values);

// scipy grey_dilation with size 3; outside the raster the edge cells are
// repeated.
std::vector<double> MaxFilter3x3(const std::vector<double>& grid, int width,
                                 int height);

// cv2.morphologyEx(MORPH_OPEN) with a 1 x `line_length` kernel when
// `along_x`, else `line_length` x 1, anchored at line_length / 2: the
// erosion counts cells outside the raster as set, the dilation as unset.
Mask OpenLine(const Mask& mask, int width, int height, int line_length,
              bool along_x);

// Exact Euclidean distance transform (Felzenszwalb & Huttenlocher): the
// distance in cells from every cell to the nearest nonzero cell of
// `sources`; +inf everywhere without any source.
std::vector<float> DistanceTransform(const std::vector<uint8_t>& sources,
                                     int width, int height);

// Zhang-Suen thinning of a binary mask to a one-cell-wide 8-connected
// skeleton.
std::vector<uint8_t> ThinMask(const std::vector<uint8_t>& mask, int width,
                              int height);

// skimage.morphology.skeletonize of a 2D mask (identical output). It runs
// passes that remove the cells its lookup table marks for the first pass,
// then for the second, from the 8-neighbour code of the state at the start
// of each pass, until a pass removes nothing. Differs from ThinMask at
// diagonal steps.
Mask Skeletonize(const Mask& mask, int width, int height);

// Cells covered by a polygon given in cell coordinates, as cv2.fillPoly with
// the vertices rounded to cells: the cells whose centre lies inside or on
// the boundary.
Mask FillPolygon(const std::vector<std::pair<int, int>>& vertices, int width,
                 int height);

// 8-connected components of a mask.
struct Components {
  std::vector<int> label;               // Per cell; 0: background, else
                                        // 1..component count.
  std::vector<std::vector<int>> cells;  // Per component, raster order.
};

// scipy label with a 3 x 3 structure: components numbered in raster order of
// their first cell.
Components Label8(const Mask& mask, int width, int height);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CORE_MORPHOLOGY_H_
