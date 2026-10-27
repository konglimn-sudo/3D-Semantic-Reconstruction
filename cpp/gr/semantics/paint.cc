#include "gr/semantics/paint.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gr/core/numeric.h"
#include "gr/core/types.h"

namespace gr {
namespace {

// Python's int(round(x)): half to even.
int RoundHalfEven(double x) { return static_cast<int>(std::nearbyint(x)); }

// Odd kernel size of `length` metres, at least 3 (int(round(l / res)) | 1).
int OddCells(double length, double res) {
  return std::max(3, RoundHalfEven(length / res) | 1);
}

// Normalised by the 2nd / 99.5th percentile of the valid cells, clipped.
std::vector<double> Normalize(const std::vector<double>& image,
                              const Mask& valid) {
  std::vector<double> values;
  for (size_t k = 0; k < image.size(); ++k) {
    if (valid[k]) values.push_back(image[k]);
  }
  std::vector<double> out(image.size(), 0.0);
  if (values.empty()) return out;
  const double lo = Percentile(values, 2.0);
  const double hi = Percentile(values, 99.5);
  const double span = std::max(hi - lo, 1e-6);
  for (size_t k = 0; k < image.size(); ++k) {
    out[k] = std::clamp((image[k] - lo) / span, 0.0, 1.0);
  }
  return out;
}

}  // namespace

Mask LineKernel(int length, double angle) {
  Mask kernel(static_cast<size_t>(length) * length, 0);
  const double c = (length - 1) / 2.0;
  const double dx = std::cos(angle) * c;
  const double dy = std::sin(angle) * c;
  DrawLine8(RoundHalfEven(c - dx), RoundHalfEven(c - dy),
            RoundHalfEven(c + dx), RoundHalfEven(c + dy), length, &kernel);
  return kernel;
}

std::vector<PaintLayer> MarkingLayers(const PaintRasters& rasters) {
  const size_t n = static_cast<size_t>(rasters.width) * rasters.height;
  std::vector<PaintLayer> layers;
  if (rasters.intensity.size() == n) {
    Mask finite(n, 0);
    std::vector<double> samples(n, 0.0);
    bool any = false;
    for (size_t k = 0; k < n; ++k) {
      finite[k] = std::isfinite(rasters.intensity[k]);
      if (finite[k]) samples[k] = rasters.intensity[k];
      any |= finite[k] != 0;
    }
    if (any) {
      // Normalised convolution of the sparse LiDAR samples.
      std::vector<double> weight(n);
      for (size_t k = 0; k < n; ++k) weight[k] = finite[k];
      const std::vector<double> w =
          GaussianFilter(weight, rasters.width, rasters.height, 1.0);
      const std::vector<double> v =
          GaussianFilter(samples, rasters.width, rasters.height, 1.0);
      std::vector<double> filled(n, 0.0);
      Mask ok(n, 0);
      for (size_t k = 0; k < n; ++k) {
        filled[k] = w[k] > 1e-3 ? v[k] / std::max(w[k], 1e-6) : 0.0;
        ok[k] = w[k] > 0.15;
      }
      layers.push_back({"lidar_intensity", Normalize(filled, ok), ok});
    }
  }
  if (rasters.rgb.size() == 3 * n && rasters.texture_nobs.size() == n) {
    Mask ok(n, 0);
    bool any = false;
    for (size_t k = 0; k < n; ++k) {
      ok[k] = rasters.texture_nobs[k] > 0;
      any |= ok[k] != 0;
    }
    if (any) {
      // cv2.cvtColor(RGB2GRAY): fixed point, coefficients / 2^14.
      std::vector<double> luminance(n);
      for (size_t k = 0; k < n; ++k) {
        const int y = (rasters.rgb[3 * k] * 4899 +
                       rasters.rgb[3 * k + 1] * 9617 +
                       rasters.rgb[3 * k + 2] * 1868 + (1 << 13)) >>
                      14;
        luminance[k] = y / 255.0;
      }
      layers.push_back({"camera_texture", Normalize(luminance, ok), ok});
    }
  }
  if (rasters.height_max.size() == n) {
    for (PaintLayer& layer : layers) {
      for (size_t k = 0; k < n; ++k) {
        const double h = rasters.height_max[k];
        if (std::isfinite(h) && h > 0.3) layer.valid[k] = 0;  // Covered.
      }
    }
  }
  return layers;
}

std::vector<PaintLayer> FloorImages(const PaintRasters& rasters) {
  const size_t n = static_cast<size_t>(rasters.width) * rasters.height;
  Mask covered(n, 0);
  if (rasters.height_max.size() == n) {
    for (size_t k = 0; k < n; ++k) {
      covered[k] =
          std::isfinite(rasters.height_max[k]) && rasters.height_max[k] > 0.3;
    }
  }
  std::vector<PaintLayer> out;
  if (rasters.rgb.size() == 3 * n && rasters.texture_nobs.size() == n) {
    PaintLayer camera;
    camera.name = "camera";
    camera.valid.assign(n, 0);
    camera.brightness.assign(n, 0.0);
    bool any = false;
    for (size_t k = 0; k < n; ++k) {
      camera.valid[k] = rasters.texture_nobs[k] > 0;
      any |= camera.valid[k] != 0;
      camera.brightness[k] =
          ((rasters.rgb[3 * k] * 4899 + rasters.rgb[3 * k + 1] * 9617 +
            rasters.rgb[3 * k + 2] * 1868 + (1 << 13)) >>
           14) /
          255.0;
    }
    if (any) out.push_back(std::move(camera));
  }
  if (rasters.intensity.size() == n) {
    std::vector<double> observed(n, 0.0);
    std::vector<double> samples(n, 0.0);
    bool any = false;
    for (size_t k = 0; k < n; ++k) {
      if (std::isfinite(rasters.intensity[k])) {
        observed[k] = 1.0;
        samples[k] = rasters.intensity[k];
        any = true;
      }
    }
    if (any) {
      const std::vector<double> weight =
          GaussianFilter(observed, rasters.width, rasters.height, 1.0);
      const std::vector<double> filled =
          GaussianFilter(samples, rasters.width, rasters.height, 1.0);
      PaintLayer lidar;
      lidar.name = "lidar";
      lidar.valid.assign(n, 0);
      lidar.brightness.assign(n, 0.0);
      for (size_t k = 0; k < n; ++k) {
        lidar.valid[k] = weight[k] > 0.15;
        if (lidar.valid[k]) {
          lidar.brightness[k] = filled[k] / std::max(weight[k], 1e-6);
        }
      }
      out.push_back(std::move(lidar));
    }
  }
  for (PaintLayer& image : out) {
    for (size_t k = 0; k < n; ++k) image.valid[k] &= !covered[k];
    image.brightness = Normalize(image.brightness, image.valid);
  }
  return out;
}

Mask LayerMask(const PaintLayer& layer, int width, int height, double res,
               const PaintConfig& config) {
  const size_t n = layer.brightness.size();
  std::vector<double> values;
  for (size_t k = 0; k < n; ++k) {
    if (layer.valid[k]) values.push_back(layer.brightness[k]);
  }
  Mask out(n, 0);
  if (values.empty()) return out;
  const double background = Median(values);
  std::vector<uint8_t> image(n);
  for (size_t k = 0; k < n; ++k) {
    const double v = layer.valid[k] ? layer.brightness[k] : background;
    image[k] = static_cast<uint8_t>(v * 255.0);  // numpy: truncation.
  }
  // White top-hat: the image minus its opening by an ellipse.
  const int size = OddCells(config.tophat_size, res);
  const Mask ellipse = EllipseKernel(size);
  const std::vector<uint8_t> opened = DilateKernel(
      ErodeKernel(image, width, height, ellipse, size), width, height,
      ellipse, size);
  std::vector<uint8_t> tophat(n);
  std::vector<uint8_t> valid_tophat;
  std::vector<double> t;
  for (size_t k = 0; k < n; ++k) {
    tophat[k] = static_cast<uint8_t>(image[k] - opened[k]);
    if (layer.valid[k]) {
      valid_tophat.push_back(tophat[k]);
      t.push_back(tophat[k]);
    }
  }
  const double median = Median(t);
  std::vector<double> deviations(t.size());
  for (size_t k = 0; k < t.size(); ++k) {
    deviations[k] = std::abs(t[k] - median);
  }
  const double mad = Median(deviations) * 1.4826;
  const double threshold =
      std::max({static_cast<double>(OtsuThreshold(valid_tophat)),
                median + config.mad_k * mad, config.min_contrast * 255.0});
  for (size_t k = 0; k < n; ++k) {
    out[k] = tophat[k] > threshold && layer.valid[k];
  }
  return out;
}

Mask PaintMask(const std::vector<PaintLayer>& layers, int width, int height,
               double res, const PaintConfig& config) {
  Mask out(static_cast<size_t>(width) * height, 0);
  for (const PaintLayer& layer : layers) {
    const Mask m = LayerMask(layer, width, height, res, config);
    for (size_t k = 0; k < out.size(); ++k) out[k] |= m[k];
  }
  return out;
}

Mask Elongated(const Mask& mask, int width, int height, double res,
               double yaw, const PaintConfig& config) {
  const int open = OddCells(config.line_open_length, res);
  const int close = OddCells(config.line_close_length, res);
  Mask out(mask.size(), 0);
  for (double angle : {yaw, yaw + kPi / 2.0}) {
    // LiDAR samples paint ring by ring: bridge the dots along the line
    // direction first (no smearing across), then require line pieces.
    const Mask closed =
        CloseKernel(mask, width, height, LineKernel(close, angle), close);
    const Mask pieces =
        OpenKernel(closed, width, height, LineKernel(open, angle), open);
    for (size_t k = 0; k < out.size(); ++k) out[k] |= pieces[k];
  }
  out = CloseKernel(out, width, height, Mask(9, 1), 3);
  const Components components = Label8(out, width, height);
  Mask kept(out.size(), 0);
  for (const std::vector<int>& cells : components.cells) {
    if (cells.size() * res * res < config.min_area) continue;
    for (int c : cells) kept[static_cast<size_t>(c)] = 1;
  }
  return kept;
}

}  // namespace gr
