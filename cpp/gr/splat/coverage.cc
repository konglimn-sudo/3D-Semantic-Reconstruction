#include "gr/splat/coverage.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "gr/core/check.h"

namespace gr {
namespace {

// Square max filter of the bits of `mask` (H x W), separable.
void DilateBits(std::vector<uint8_t>& mask, int height, int width,
                int radius) {
  if (radius <= 0) return;
  std::vector<uint8_t> tmp(mask.size(), 0);
  for (int v = 0; v < height; ++v) {
    const size_t row = static_cast<size_t>(v) * static_cast<size_t>(width);
    for (int u = 0; u < width; ++u) {
      uint8_t bits = 0;
      for (int du = std::max(0, u - radius);
           du <= std::min(width - 1, u + radius); ++du) {
        bits |= mask[row + static_cast<size_t>(du)];
      }
      tmp[row + static_cast<size_t>(u)] = bits;
    }
  }
  for (int u = 0; u < width; ++u) {
    for (int v = 0; v < height; ++v) {
      uint8_t bits = 0;
      for (int dv = std::max(0, v - radius);
           dv <= std::min(height - 1, v + radius); ++dv) {
        bits |= tmp[static_cast<size_t>(dv) * static_cast<size_t>(width) +
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
  const size_t n = index_.size();
  GR_CHECK(normals_.empty() || normals_.size() == n);
  views_.assign(n, 0);
  good_views_.assign(n, 0);
  best_gsd_.assign(n, std::numeric_limits<float>::infinity());
  color_sum_.assign(n, Eigen::Vector3d::Zero());
  weight_sum_.assign(n, 0.0);
}

void CameraCoverage::AddFrame(const CameraModel& camera,
                              const CameraExposure& exposure,
                              const ImageView& image, const uint8_t* mask) {
  GR_CHECK(image.empty() || (image.width == camera.width() &&
                             image.height == camera.height() &&
                             image.channels >= 3));
  ++frames_;
  const Vec3 center = exposure.center();
  const std::vector<int> near = index_.RadiusSearch(center, config_.max_range);
  ZBuffer buffer(camera, exposure, config_.point_spacing, config_.min_radius,
                 config_.max_radius, 0.1, config_.max_range);
  for (int i : near) {
    buffer.Splat(index_.points()[static_cast<size_t>(i)], -1, i);
  }
  const RenderResult& z = buffer.result();
  const double focal = camera.focal();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
  for (int64_t mm = 0; mm < static_cast<int64_t>(near.size()); ++mm) {
    const size_t i = static_cast<size_t>(near[static_cast<size_t>(mm)]);
    const Vec3& p = index_.points()[i];
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(camera, p);
    if (!projection) continue;
    const double range = projection->point_camera.norm();
    if (range > config_.max_range || !camera.InImage(projection->uv)) continue;
    const int u = static_cast<int>(projection->uv.x());
    const int v = static_cast<int>(projection->uv.y());
    const size_t pixel = static_cast<size_t>(v) *
                             static_cast<size_t>(camera.width()) +
                         static_cast<size_t>(u);
    const float d = z.depth[pixel];
    if (std::isfinite(d) &&
        range > d * (1.0 + config_.depth_rel_tol) + config_.depth_abs_tol) {
      continue;  // Occluded.
    }
    double cosine = 1.0;
    if (!normals_.empty() && normals_[i].norm() > 0.5) {
      cosine = std::abs(normals_[i].normalized().dot((center - p) / range));
      if (cosine < config_.min_cos) continue;
    }
    ++views_[i];
    if (mask != nullptr && mask[pixel] != 0) continue;
    Eigen::Vector3d rgb = Eigen::Vector3d::Zero();
    if (!image.empty()) {
      const uint8_t* px = image.PixelAt(v, u);
      rgb = Eigen::Vector3d(px[0], px[1], px[2]);
      const double luminance = rgb.mean();
      if (luminance <= config_.dark || luminance >= config_.saturated) continue;
    }
    ++good_views_[i];
    const double gsd = range / focal;
    best_gsd_[i] = std::min(best_gsd_[i], static_cast<float>(gsd));
    if (!image.empty()) {
      const double w = cosine / (gsd * gsd);
      color_sum_[i] += w * rgb / 255.0;
      weight_sum_[i] += w;
    }
  }
}

CoverageResult CameraCoverage::Result() const {
  CoverageResult out;
  const size_t n = views_.size();
  out.views = views_;
  out.good_views = good_views_;
  out.best_gsd = best_gsd_;
  out.color.assign(n, Eigen::Vector3f::Constant(0.5f));
  out.color_weight.resize(n);
  for (size_t i = 0; i < n; ++i) {
    out.color_weight[i] = static_cast<float>(weight_sum_[i]);
    if (weight_sum_[i] > 0.0) {
      out.color[i] = (color_sum_[i] / weight_sum_[i]).cast<float>();
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
                 config.max_radius, 0.1, config.max_range);
  const Vec3 center = exposure.center();
  for (int i : static_points.RadiusSearch(center, config.max_range)) {
    buffer.Splat(static_points.points()[static_cast<size_t>(i)], 0, -1);
  }
  auto splat_all = [&](const Points& points, int bit) {
    for (const Vec3& p : points) {
      if ((p - center).norm() <= config.max_range) buffer.Splat(p, bit, -1);
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
  const RenderResult& z = buffer.result();
  std::vector<uint8_t> mask(z.label.size(), 0);
  for (size_t p = 0; p < mask.size(); ++p) {
    if (z.label[p] > 0) mask[p] = static_cast<uint8_t>(z.label[p]);
  }
  DilateBits(mask, z.height, z.width, config.dilate);
  return mask;
}

}  // namespace gr
