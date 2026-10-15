// Bird's-eye-view rasterisation and multi-view floor texture fusion.

#ifndef GARAGE_RECON_GR_RASTER_H_
#define GARAGE_RECON_GR_RASTER_H_

#include <cstdint>
#include <vector>

#include "gr/camera.h"
#include "gr/exposure.h"
#include "gr/image.h"
#include "gr/pose.h"
#include "gr/projector.h"

namespace gr {

// Cell (col i, row j) covers x in [x0 + i * res, x0 + (i + 1) * res), y
// likewise. Rasters are row-major (height x width), row 0 at y0.
struct Grid2D {
  double x0 = 0.0;
  double y0 = 0.0;
  double res = 0.05;
  int width = 0;
  int height = 0;
};

enum class Reducer { kMean, kMax, kMin, kCount, kSum };

// `values` holds one value per row of `xy`; empty cells get `fill` (kCount
// gives 0).
Eigen::MatrixXd Rasterize(const RowMatX2d& xy, const Eigen::VectorXd& values,
                          const Grid2D& grid, Reducer reducer, double fill);

// Keeps, per floor cell, the top-K camera observations ranked by
// cos(incidence) / range and fuses them with a per-channel median, which
// rejects specular highlights and transient occluders seen in few views.
class FloorTextureFuser {
 public:
  // `floor_z` is height x width (NaN where there is no floor).
  FloorTextureFuser(const Grid2D& grid, const Eigen::MatrixXd& floor_z,
                    int top_k = 5);
  FloorTextureFuser(const FloorTextureFuser&) = default;
  FloorTextureFuser& operator=(const FloorTextureFuser&) = default;
  FloorTextureFuser(FloorTextureFuser&&) = default;
  FloorTextureFuser& operator=(FloorTextureFuser&&) = default;

  // `image` is RGB (3 channels); `depth` is a camera-sized range buffer for
  // the occlusion test, or null.
  void AddFrame(const CameraModel& camera, const CameraExposure& exposure,
                const ImageView& image, const float* depth, double depth_tol,
                double max_range, double min_cos);
  // Same, with the occlusion depth rendered from `occluders` (e.g. the static
  // map: parked cars and columns hide the floor behind them).
  void AddFrameOccluded(const CameraModel& camera,
                        const CameraExposure& exposure, const ImageView& image,
                        const PointCloudIndex& occluders, double point_spacing,
                        double depth_tol, double max_range, double min_cos);

  struct Texture {
    std::vector<uint8_t> rgb;            // height * width * 3
    std::vector<float> score;            // Best score per cell; 0 unobserved.
    std::vector<uint16_t> observations;  // Observations per cell.
  };
  Texture ComputeTexture() const;
  const Grid2D& grid() const { return grid_; }

 private:
  Grid2D grid_;
  Eigen::MatrixXd floor_z_;
  int top_k_;
  std::vector<float> scores_;
  std::vector<uint8_t> colors_;
  std::vector<uint16_t> observations_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_RASTER_H_
