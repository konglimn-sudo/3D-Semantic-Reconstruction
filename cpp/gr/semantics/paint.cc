#include "gr/semantics/paint.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gr/core/image.h"
#include "gr/core/numeric.h"
#include "gr/core/types.h"

namespace gr {
namespace {

// Kernels are odd (the low bit set) and at least kMinKernelCells wide.
constexpr int kOddBit = 1;
constexpr int kMinKernelCells = 3;
// Images are normalised from their kLowPercentile to kHighPercentile (a span
// of at least kMinSpan).
constexpr double kLowPercentile = 2.0;
constexpr double kHighPercentile = 99.5;
constexpr double kMinSpan = 1e-6;
// LiDAR intensity samples are filled by a normalised convolution of this
// sigma (cells); cells with a fill weight above kMinFillWeight get a value
// (divided by at least kMinWeight) and above kObservedFillWeight count as
// observed.
constexpr double kFillSigma = 1.0;
constexpr double kMinFillWeight = 1e-3;
constexpr double kMinWeight = 1e-6;
constexpr double kObservedFillWeight = 0.15;
// cv2.cvtColor(RGB2GRAY): fixed-point coefficients / 2^kLumaShift, rounded.
constexpr int kLumaRed = 4899;
constexpr int kLumaGreen = 9617;
constexpr int kLumaBlue = 1868;
constexpr int kLumaShift = 14;
constexpr int kLumaRounding = 1 << (kLumaShift - 1);
// Brightness of a full 8-bit value.
constexpr double kMaxByte = std::numeric_limits<uint8_t>::max();
// Cells with something higher than this (m) above the floor are covered.
constexpr double kCoveredHeight = 0.3;
// MAD to sigma of a normal distribution.
constexpr double kMadToSigma = 1.4826;
// The final closing: a full square of this side (cells).
constexpr int kClosingSide = 3;

// Python's int(round(x)): half to even.
int RoundHalfEven(double x) { return static_cast<int>(std::nearbyint(x)); }

// Odd kernel size of `length` metres, at least kMinKernelCells
// (int(round(l / res)) | 1).
int OddCells(double length, double res) {
  return std::max(kMinKernelCells, RoundHalfEven(length / res) | kOddBit);
}

// 8-bit luminance of the RGB cell `cell`.
int Luminance(const std::vector<uint8_t>& rgb, size_t cell) {
  const size_t pixel = kRgbChannels * cell;
  return (rgb[pixel] * kLumaRed + rgb[pixel + 1] * kLumaGreen +
          rgb[pixel + 2] * kLumaBlue + kLumaRounding) >>
         kLumaShift;
}

// Normalised by the kLowPercentile / kHighPercentile of the valid cells,
// clipped.
std::vector<double> Normalize(const std::vector<double>& image,
                              const Mask& valid) {
  std::vector<double> values;
  for (size_t cell = 0; cell < image.size(); ++cell) {
    if (valid[cell]) values.push_back(image[cell]);
  }
  std::vector<double> out(image.size(), 0.0);
  if (values.empty()) return out;
  const double low = Percentile(values, kLowPercentile);
  const double high = Percentile(values, kHighPercentile);
  const double span = std::max(high - low, kMinSpan);
  for (size_t cell = 0; cell < image.size(); ++cell) {
    out[cell] = std::clamp((image[cell] - low) / span, 0.0, 1.0);
  }
  return out;
}

}  // namespace

Mask LineKernel(int length, double angle) {
  Mask kernel(static_cast<size_t>(length) * length, 0);
  const double centre = (length - 1) / 2.0;
  const double half_x = std::cos(angle) * centre;
  const double half_y = std::sin(angle) * centre;
  DrawLine8(RoundHalfEven(centre - half_x), RoundHalfEven(centre - half_y),
            RoundHalfEven(centre + half_x), RoundHalfEven(centre + half_y),
            length, &kernel);
  return kernel;
}

std::vector<PaintLayer> MarkingLayers(const PaintRasters& rasters) {
  const size_t cells = static_cast<size_t>(rasters.width) * rasters.height;
  std::vector<PaintLayer> layers;
  if (rasters.intensity.size() == cells) {
    Mask finite(cells, 0);
    std::vector<double> samples(cells, 0.0);
    bool any = false;
    for (size_t cell = 0; cell < cells; ++cell) {
      finite[cell] = std::isfinite(rasters.intensity[cell]);
      if (finite[cell]) samples[cell] = rasters.intensity[cell];
      any |= finite[cell] != 0;
    }
    if (any) {
      // Normalised convolution of the sparse LiDAR samples.
      std::vector<double> sample_weight(cells);
      for (size_t cell = 0; cell < cells; ++cell) {
        sample_weight[cell] = finite[cell];
      }
      const std::vector<double> weight = GaussianFilter(
          sample_weight, rasters.width, rasters.height, kFillSigma);
      const std::vector<double> weighted = GaussianFilter(
          samples, rasters.width, rasters.height, kFillSigma);
      std::vector<double> filled(cells, 0.0);
      Mask observed(cells, 0);
      for (size_t cell = 0; cell < cells; ++cell) {
        filled[cell] = weight[cell] > kMinFillWeight
                           ? weighted[cell] / std::max(weight[cell], kMinWeight)
                           : 0.0;
        observed[cell] = weight[cell] > kObservedFillWeight;
      }
      layers.push_back(
          {"lidar_intensity", Normalize(filled, observed), observed});
    }
  }
  if (rasters.rgb.size() == kRgbChannels * cells &&
      rasters.texture_nobs.size() == cells) {
    Mask observed(cells, 0);
    bool any = false;
    for (size_t cell = 0; cell < cells; ++cell) {
      observed[cell] = rasters.texture_nobs[cell] > 0;
      any |= observed[cell] != 0;
    }
    if (any) {
      std::vector<double> luminance(cells);
      for (size_t cell = 0; cell < cells; ++cell) {
        luminance[cell] = Luminance(rasters.rgb, cell) / kMaxByte;
      }
      layers.push_back(
          {"camera_texture", Normalize(luminance, observed), observed});
    }
  }
  if (rasters.height_max.size() == cells) {
    for (PaintLayer& layer : layers) {
      for (size_t cell = 0; cell < cells; ++cell) {
        const double height = rasters.height_max[cell];
        if (std::isfinite(height) && height > kCoveredHeight) {
          layer.valid[cell] = 0;
        }
      }
    }
  }
  return layers;
}

std::vector<PaintLayer> FloorImages(const PaintRasters& rasters) {
  const size_t cells = static_cast<size_t>(rasters.width) * rasters.height;
  Mask covered(cells, 0);
  if (rasters.height_max.size() == cells) {
    for (size_t cell = 0; cell < cells; ++cell) {
      covered[cell] = std::isfinite(rasters.height_max[cell]) &&
                      rasters.height_max[cell] > kCoveredHeight;
    }
  }
  std::vector<PaintLayer> out;
  if (rasters.rgb.size() == kRgbChannels * cells &&
      rasters.texture_nobs.size() == cells) {
    PaintLayer camera;
    camera.name = "camera";
    camera.valid.assign(cells, 0);
    camera.brightness.assign(cells, 0.0);
    bool any = false;
    for (size_t cell = 0; cell < cells; ++cell) {
      camera.valid[cell] = rasters.texture_nobs[cell] > 0;
      any |= camera.valid[cell] != 0;
      camera.brightness[cell] = Luminance(rasters.rgb, cell) / kMaxByte;
    }
    if (any) out.push_back(std::move(camera));
  }
  if (rasters.intensity.size() == cells) {
    std::vector<double> observed(cells, 0.0);
    std::vector<double> samples(cells, 0.0);
    bool any = false;
    for (size_t cell = 0; cell < cells; ++cell) {
      if (std::isfinite(rasters.intensity[cell])) {
        observed[cell] = 1.0;
        samples[cell] = rasters.intensity[cell];
        any = true;
      }
    }
    if (any) {
      const std::vector<double> weight = GaussianFilter(
          observed, rasters.width, rasters.height, kFillSigma);
      const std::vector<double> filled = GaussianFilter(
          samples, rasters.width, rasters.height, kFillSigma);
      PaintLayer lidar;
      lidar.name = "lidar";
      lidar.valid.assign(cells, 0);
      lidar.brightness.assign(cells, 0.0);
      for (size_t cell = 0; cell < cells; ++cell) {
        lidar.valid[cell] = weight[cell] > kObservedFillWeight;
        if (lidar.valid[cell]) {
          lidar.brightness[cell] =
              filled[cell] / std::max(weight[cell], kMinWeight);
        }
      }
      out.push_back(std::move(lidar));
    }
  }
  for (PaintLayer& image : out) {
    for (size_t cell = 0; cell < cells; ++cell) {
      image.valid[cell] &= !covered[cell];
    }
    image.brightness = Normalize(image.brightness, image.valid);
  }
  return out;
}

Mask LayerMask(const PaintLayer& layer, int width, int height, double res,
               const PaintConfig& config) {
  const size_t cells = layer.brightness.size();
  std::vector<double> values;
  for (size_t cell = 0; cell < cells; ++cell) {
    if (layer.valid[cell]) values.push_back(layer.brightness[cell]);
  }
  Mask out(cells, 0);
  if (values.empty()) return out;
  const double background = Median(values);
  std::vector<uint8_t> image(cells);
  for (size_t cell = 0; cell < cells; ++cell) {
    const double value =
        layer.valid[cell] ? layer.brightness[cell] : background;
    image[cell] = static_cast<uint8_t>(value * kMaxByte);  // numpy: truncation.
  }
  // White top-hat: the image minus its opening by an ellipse.
  const int size = OddCells(config.tophat_size, res);
  const Mask ellipse = EllipseKernel(size);
  const std::vector<uint8_t> opened = DilateKernel(
      ErodeKernel(image, width, height, ellipse, size), width, height,
      ellipse, size);
  std::vector<uint8_t> tophat(cells);
  std::vector<uint8_t> valid_tophat;
  std::vector<double> valid_values;
  for (size_t cell = 0; cell < cells; ++cell) {
    tophat[cell] = static_cast<uint8_t>(image[cell] - opened[cell]);
    if (layer.valid[cell]) {
      valid_tophat.push_back(tophat[cell]);
      valid_values.push_back(tophat[cell]);
    }
  }
  const double median = Median(valid_values);
  std::vector<double> deviations(valid_values.size());
  for (size_t k = 0; k < valid_values.size(); ++k) {
    deviations[k] = std::abs(valid_values[k] - median);
  }
  const double mad = Median(deviations) * kMadToSigma;
  const double threshold =
      std::max({static_cast<double>(OtsuThreshold(valid_tophat)),
                median + config.mad_k * mad, config.min_contrast * kMaxByte});
  for (size_t cell = 0; cell < cells; ++cell) {
    out[cell] = tophat[cell] > threshold && layer.valid[cell];
  }
  return out;
}

Mask PaintMask(const std::vector<PaintLayer>& layers, int width, int height,
               double res, const PaintConfig& config) {
  Mask out(static_cast<size_t>(width) * height, 0);
  for (const PaintLayer& layer : layers) {
    const Mask layer_mask = LayerMask(layer, width, height, res, config);
    for (size_t cell = 0; cell < out.size(); ++cell) {
      out[cell] |= layer_mask[cell];
    }
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
    for (size_t cell = 0; cell < out.size(); ++cell) {
      out[cell] |= pieces[cell];
    }
  }
  out = CloseKernel(out, width, height,
                    Mask(kClosingSide * kClosingSide, 1), kClosingSide);
  const Components components = Label8(out, width, height);
  Mask kept(out.size(), 0);
  for (const std::vector<int>& piece : components.cells) {
    if (piece.size() * res * res < config.min_area) continue;
    for (int cell : piece) kept[static_cast<size_t>(cell)] = 1;
  }
  return kept;
}

}  // namespace gr
