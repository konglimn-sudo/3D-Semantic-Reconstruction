// Depth prior (DA3) aligned to the LiDAR map (plan section 8.2).
//
// A pose-conditioned DA3 depth is already metric, but its scale follows the
// network's own pose estimate and drifts by a few percent between batches;
// near and far surfaces can also be bent. Per image, the prior range is fitted
// to the LiDAR, l = s * p + b, by iteratively reweighted least squares on the
// relative residual (Cauchy). The fit uses the visible LiDAR map points at
// their exact sub-pixel position, not a z-buffer image: on a floor seen at a
// grazing angle a pixel covers a long range interval and the z-buffer keeps
// its nearest point, 1-3 % short. Images whose fit stays poor are rejected,
// and so are cells of the image where the fitted prior still disagrees with
// the LiDAR (wrong depth order at thin structures, reflections, sky). The
// prior then fills only the LiDAR holes, pixels farther than `hole_radius`
// from any rendered LiDAR pixel: the output keeps the rendered LiDAR range,
// takes the aligned prior in the holes, and marks the source of every pixel.

#ifndef GARAGE_RECON_GR_SPLAT_DEPTH_PRIOR_H_
#define GARAGE_RECON_GR_SPLAT_DEPTH_PRIOR_H_

#include <cstdint>
#include <vector>

namespace gr {

// Values of FusedDepth::source.
enum DepthSource : uint8_t {
  kDepthNone = 0,
  kDepthLidar = 1,  // Rendered LiDAR map (measured).
  kDepthPrior = 2,  // Aligned depth prior in a LiDAR hole.
};

// A LiDAR map point seen in the image (pixel coordinates, range in m).
struct DepthSample {
  double u = 0.0;
  double v = 0.0;
  double range = 0.0;
};

struct DepthAlignConfig {
  double min_range = 0.3;  // LiDAR / prior ranges used for the fit (m).
  double max_range = 80.0;
  int min_pixels = 200;  // LiDAR samples needed to fit an image.
  bool fit_shift = true;  // l = s p + b; false: l = s p.
  int iterations = 10;
  double robust_scale = 0.05;  // Cauchy scale of the relative residual.
  // The image is rejected if the median relative residual after the fit is
  // above this, or the scale leaves [min_scale, max_scale].
  double max_residual = 0.10;
  double min_scale = 0.5;
  double max_scale = 2.0;
  // Local check on cells of `cell` x `cell` pixels with at least
  // `min_cell_pixels` LiDAR pixels: the prior is dropped in a cell whose
  // median relative residual is above `max_cell_residual`, and in the cells
  // next to it.
  int cell = 24;
  int min_cell_pixels = 20;
  double max_cell_residual = 0.15;
  // Confidence of a prior pixel: its own confidence (in [0, 1]) times
  // exp(-0.5 (e / tolerance)^2), with e the median residual of its cell, or
  // of the image for cells without LiDAR.
  double tolerance = 0.05;
  double min_confidence = 0.05;  // Prior pixels below this are dropped.
  // Pixels within this many pixels of a rendered LiDAR pixel are not holes:
  // the gaps between the splats of a measured surface stay unknown.
  int hole_radius = 4;
};

struct DepthAlignment {
  bool valid = false;
  double scale = 1.0;
  double shift = 0.0;  // m.
  int pixels = 0;      // LiDAR samples of the fit.
  double residual_median = 0.0;  // Relative, after the fit.
  double residual_p90 = 0.0;
  double inlier_fraction = 0.0;  // Relative residual below robust_scale.
  double raw_ratio = 0.0;        // Median LiDAR / prior before the fit.
  int cells_checked = 0;
  int cells_rejected = 0;
};

struct FusedDepth {
  std::vector<float> range;  // H x W, NaN without a source.
  std::vector<float> confidence;  // 1 for LiDAR pixels.
  std::vector<uint8_t> source;    // DepthSource.
  DepthAlignment alignment;
  int prior_pixels = 0;  // Holes filled by the prior.
};

// Fits `prior` (range along the ray, H x W, NaN where unknown) to the LiDAR
// `samples` and fuses it with `lidar`, the rendered LiDAR range (same layout
// as the prior). Without samples, the rendered LiDAR pixels are used for the
// fit. `prior_confidence` is in [0, 1]; empty means 1 everywhere.
FusedDepth FuseDepthPrior(const std::vector<float>& prior,
                          const std::vector<float>& prior_confidence,
                          const std::vector<float>& lidar,
                          const std::vector<DepthSample>& samples, int height,
                          int width, const DepthAlignConfig& config = {});

}  // namespace gr

#endif  // GARAGE_RECON_GR_SPLAT_DEPTH_PRIOR_H_
