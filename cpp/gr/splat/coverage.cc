#include "gr/splat/coverage.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "gr/core/check.h"

namespace gr {
namespace {

// Map points closer than this (m) to the camera are not rendered.
constexpr double kMinRenderRange = 0.1;
// Points per OpenMP chunk of the per-point loop.
constexpr int kPointsPerChunk = 1024;
// Normals shorter than this are missing.
constexpr double kMinNormalNorm = 0.5;
// Brightness of a full 8-bit value.
constexpr double kMaxByte = 255.0;
// Colour of points no usable view saw (mid grey).
constexpr float kUncolouredGrey = 0.5f;
// Static points occlude without setting any bit.
constexpr int kNoBits = 0;

// Square max filter of the bits of `mask` (H x W), separable.
void DilateBits(std::vector<uint8_t>& mask, int height, int width,
                int radius) {
  if (radius <= 0) return;
  std::vector<uint8_t> horizontal(mask.size(), 0);
  for (int v = 0; v < height; ++v) {
    const size_t row = static_cast<size_t>(v) * static_cast<size_t>(width);
    for (int u = 0; u < width; ++u) {
      uint8_t bits = 0;
      for (int column = std::max(0, u - radius);
           column <= std::min(width - 1, u + radius); ++column) {
        bits |= mask[row + static_cast<size_t>(column)];
      }
      horizontal[row + static_cast<size_t>(u)] = bits;
    }
  }
  for (int u = 0; u < width; ++u) {
    for (int v = 0; v < height; ++v) {
      uint8_t bits = 0;
      for (int row = std::max(0, v - radius);
           row <= std::min(height - 1, v + radius); ++row) {
        bits |= horizontal[static_cast<size_t>(row) *
                               static_cast<size_t>(width) +
                           static_cast<size_t>(u)];
      }
      mask[static_cast<size_t>(v) * static_cast<size_t>(width) +
           static_cast<size_t>(u)] = bits;
    }
  }
}

}  // namespace

CameraCoverage::CameraCoverage(Points points, Points normals,
                               const CoverageConfig& config)
    : index_(std::move(points)), normals_(std::move(normals)),
      config_(config) {
  const size_t count = index_.size();
  GR_CHECK(normals_.empty() || normals_.size() == count);
  views_.assign(count, 0);
  good_views_.assign(count, 0);
  best_gsd_.assign(count, std::numeric_limits<float>::infinity());
  color_sum_.assign(count, Eigen::Vector3d::Zero());
  weight_sum_.assign(count, 0.0);
}

void CameraCoverage::AddFrame(const CameraModel& camera,
                              const CameraExposure& exposure,
                              const ImageView& image, const uint8_t* mask) {
  GR_CHECK(image.empty() || (image.width == camera.width() &&
                             image.height == camera.height() &&
                             image.channels >= kRgbChannels));
  ++frames_;
  const Vec3 center = exposure.center();
  const std::vector<int> near = index_.RadiusSearch(center, config_.max_range);
  ZBuffer buffer(camera, exposure, config_.point_spacing, config_.min_radius,
                 config_.max_radius, kMinRenderRange, config_.max_range);
  for (int point : near) {
    buffer.Splat(index_.points()[static_cast<size_t>(point)],
                 RenderResult::kEmpty, point);
  }
  const RenderResult& render = buffer.result();
  const double focal = camera.focal();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, kPointsPerChunk)
#endif
  for (int64_t near_index = 0;
       near_index < static_cast<int64_t>(near.size()); ++near_index) {
    const size_t point =
        static_cast<size_t>(near[static_cast<size_t>(near_index)]);
    const Vec3& position = index_.points()[point];
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(camera, position);
    if (!projection) continue;
    const double range = projection->point_camera.norm();
    if (range > config_.max_range || !camera.InImage(projection->uv)) continue;
    const int u = static_cast<int>(projection->uv.x());
    const int v = static_cast<int>(projection->uv.y());
    const size_t pixel = static_cast<size_t>(v) *
                             static_cast<size_t>(camera.width()) +
                         static_cast<size_t>(u);
    const float surface_range = render.depth[pixel];
    if (std::isfinite(surface_range) &&
        range > surface_range * (1.0 + config_.depth_rel_tol) +
                    config_.depth_abs_tol) {
      continue;  // Occluded.
    }
    double cosine = 1.0;
    if (!normals_.empty() && normals_[point].norm() > kMinNormalNorm) {
      cosine = std::abs(
          normals_[point].normalized().dot((center - position) / range));
      if (cosine < config_.min_cos) continue;
    }
    ++views_[point];
    if (mask != nullptr && mask[pixel] != 0) continue;
    Eigen::Vector3d rgb = Eigen::Vector3d::Zero();
    if (!image.empty()) {
      const uint8_t* pixel_rgb = image.PixelAt(v, u);
      rgb = Eigen::Vector3d(pixel_rgb[0], pixel_rgb[1], pixel_rgb[2]);
      const double luminance = rgb.mean();
      if (luminance <= config_.dark || luminance >= config_.saturated) continue;
    }
    ++good_views_[point];
    const double gsd = range / focal;
    best_gsd_[point] = std::min(best_gsd_[point], static_cast<float>(gsd));
    if (!image.empty()) {
      const double weight = cosine / (gsd * gsd);
      color_sum_[point] += weight * rgb / kMaxByte;
      weight_sum_[point] += weight;
    }
  }
}

CoverageResult CameraCoverage::Result() const {
  CoverageResult out;
  const size_t count = views_.size();
  out.views = views_;
  out.good_views = good_views_;
  out.best_gsd = best_gsd_;
  out.color.assign(count, Eigen::Vector3f::Constant(kUncolouredGrey));
  out.color_weight.resize(count);
  for (size_t point = 0; point < count; ++point) {
    out.color_weight[point] = static_cast<float>(weight_sum_[point]);
    if (weight_sum_[point] > 0.0) {
      out.color[point] =
          (color_sum_[point] / weight_sum_[point]).cast<float>();
    }
  }
  return out;
}

std::vector<uint8_t> RenderTrainingMask(const CameraModel& camera,
                                        const CameraExposure& exposure,
                                        const PointCloudIndex& static_points,
                                        const Points& moving,
                                        const Points& changed,
                                        const std::vector<ObjectBox>& boxes,
                                        const TrainingMaskConfig& config) {
  ZBuffer buffer(camera, exposure, config.point_spacing, config.min_radius,
                 config.max_radius, kMinRenderRange, config.max_range);
  const Vec3 center = exposure.center();
  for (int point : static_points.RadiusSearch(center, config.max_range)) {
    buffer.Splat(static_points.points()[static_cast<size_t>(point)], kNoBits,
                 RenderResult::kEmpty);
  }
  const auto splat_all = [&](const Points& points, int bit) {
    for (const Vec3& point : points) {
      if ((point - center).norm() <= config.max_range) {
        buffer.Splat(point, bit, RenderResult::kEmpty);
      }
    }
  };
  splat_all(moving, kMaskMoving);
  splat_all(changed, kMaskChanged);
  for (const ObjectBox& box : boxes) {
    if ((box.center - center).norm() > config.max_range + box.size.norm()) {
      continue;
    }
    splat_all(box.SampleSurface(config.box_step), kMaskMoving);
  }
  const RenderResult& render = buffer.result();
  std::vector<uint8_t> mask(render.label.size(), 0);
  for (size_t pixel = 0; pixel < mask.size(); ++pixel) {
    if (render.label[pixel] > kNoBits) {
      mask[pixel] = static_cast<uint8_t>(render.label[pixel]);
    }
  }
  DilateBits(mask, render.height, render.width, config.dilate);
  return mask;
}

}  // namespace gr
