#include "gr/mapping/raster.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "gr/core/check.h"

namespace gr {
namespace {

// Floor cells closer to the camera than this (m) are not textured.
constexpr double kMinRange = 1e-3;
// The occlusion depth splats the occluders as discs of 1 to 4 pixels from
// kOccluderMinRange (m) to kOcclusionRangeMargin beyond the texture range;
// occluders carry no label.
constexpr int kOccluderMinRadius = 1;
constexpr int kOccluderMaxRadius = 4;
constexpr double kOccluderMinRange = 0.1;
constexpr double kOcclusionRangeMargin = 2.0;
constexpr int kNoLabel = -1;

// Bilinear RGB sample at pixel coordinates (u, v), pixel centres at +0.5.
Vec3 SampleRgb(const ImageView& image, double u, double v) {
  u -= 0.5;
  v -= 0.5;
  const int left = static_cast<int>(std::floor(u));
  const int top = static_cast<int>(std::floor(v));
  const double fraction_x = u - left;
  const double fraction_y = v - top;
  Vec3 rgb = Vec3::Zero();
  for (int dy = 0; dy <= 1; ++dy) {
    for (int dx = 0; dx <= 1; ++dx) {
      const int col = std::clamp(left + dx, 0, image.width - 1);
      const int row = std::clamp(top + dy, 0, image.height - 1);
      const double weight = (dx ? fraction_x : 1.0 - fraction_x) *
                            (dy ? fraction_y : 1.0 - fraction_y);
      const uint8_t* pixel = image.PixelAt(row, col);
      for (int channel = 0; channel < kRgbChannels; ++channel) {
        rgb[channel] += weight * pixel[channel];
      }
    }
  }
  return rgb;
}

}  // namespace

Eigen::MatrixXd Rasterize(const RowMatX2d& xy, const Eigen::VectorXd& values,
                          const Grid2D& grid, Reducer reducer, double fill) {
  GR_CHECK(xy.rows() == values.size());
  const int width = grid.width;
  const int height = grid.height;
  Eigen::MatrixXd accumulated = Eigen::MatrixXd::Zero(height, width);
  Eigen::MatrixXi count = Eigen::MatrixXi::Zero(height, width);
  if (reducer == Reducer::kMax) {
    accumulated.setConstant(-std::numeric_limits<double>::infinity());
  }
  if (reducer == Reducer::kMin) {
    accumulated.setConstant(std::numeric_limits<double>::infinity());
  }
  const double inverse_res = 1.0 / grid.res;
  for (Eigen::Index point = 0; point < xy.rows(); ++point) {
    const int col =
        static_cast<int>(std::floor((xy(point, 0) - grid.x0) * inverse_res));
    const int row =
        static_cast<int>(std::floor((xy(point, 1) - grid.y0) * inverse_res));
    if (col < 0 || row < 0 || col >= width || row >= height) continue;
    const double value = values[point];
    switch (reducer) {
      case Reducer::kMean:
      case Reducer::kSum:
        accumulated(row, col) += value;
        break;
      case Reducer::kMax:
        accumulated(row, col) = std::max(accumulated(row, col), value);
        break;
      case Reducer::kMin:
        accumulated(row, col) = std::min(accumulated(row, col), value);
        break;
      case Reducer::kCount:
        break;
    }
    count(row, col) += 1;
  }
  Eigen::MatrixXd raster(height, width);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int cell_count = count(row, col);
      if (reducer == Reducer::kCount) {
        raster(row, col) = cell_count;
      } else if (cell_count == 0) {
        raster(row, col) = fill;
      } else if (reducer == Reducer::kMean) {
        raster(row, col) = accumulated(row, col) / cell_count;
      } else {
        raster(row, col) = accumulated(row, col);
      }
    }
  }
  return raster;
}

FloorTextureFuser::FloorTextureFuser(const Grid2D& grid,
                                     const Eigen::MatrixXd& floor_z, int top_k)
    : grid_(grid), floor_z_(floor_z), top_k_(std::max(1, top_k)) {
  GR_CHECK(floor_z_.rows() == grid.height && floor_z_.cols() == grid.width);
  const size_t cells =
      static_cast<size_t>(grid.width) * static_cast<size_t>(grid.height);
  scores_.assign(cells * static_cast<size_t>(top_k_), kEmptySlot);
  colors_.assign(cells * static_cast<size_t>(top_k_) * kRgbChannels, 0);
  observations_.assign(cells, 0);
}

void FloorTextureFuser::AddFrame(const CameraModel& camera,
                                 const CameraExposure& exposure,
                                 const ImageView& image, const float* depth,
                                 double depth_tol, double max_range,
                                 double min_cos) {
  GR_CHECK(image.channels == kRgbChannels);
  if (image.empty()) return;
  const Vec3& center = exposure.center();
  const double inverse_res = 1.0 / grid_.res;
  const auto cell_of = [inverse_res](double coordinate, double origin) {
    return static_cast<int>(std::floor((coordinate - origin) * inverse_res));
  };
  const int first_col =
      std::max(0, cell_of(center.x() - max_range, grid_.x0));
  const int last_col =
      std::min(grid_.width - 1, cell_of(center.x() + max_range, grid_.x0));
  const int first_row =
      std::max(0, cell_of(center.y() - max_range, grid_.y0));
  const int last_row =
      std::min(grid_.height - 1, cell_of(center.y() + max_range, grid_.y0));
  if (first_col > last_col || first_row > last_row) return;

#ifdef _OPENMP
  constexpr int kRowsPerChunk = 4;
#pragma omp parallel for schedule(dynamic, kRowsPerChunk)
#endif
  for (int row = first_row; row <= last_row; ++row) {
    for (int col = first_col; col <= last_col; ++col) {
      const double z = floor_z_(row, col);
      if (!std::isfinite(z)) continue;
      const Vec3 floor_point(grid_.x0 + (col + 0.5) * grid_.res,
                             grid_.y0 + (row + 0.5) * grid_.res, z);
      const double center_range = (floor_point - center).norm();
      if (center_range > max_range || center_range < kMinRange) continue;
      const std::optional<CameraExposure::Projection> projection =
          exposure.Project(camera, floor_point);
      if (!projection) continue;
      const double range = projection->point_camera.norm();
      if (range > max_range || range < kMinRange) continue;
      const double cos_incidence = (center.z() - z) / range;
      if (cos_incidence < min_cos) continue;
      const Vec2& uv = projection->uv;
      if (!camera.InImage(uv)) continue;
      if (depth != nullptr) {
        const size_t pixel = static_cast<size_t>(static_cast<int>(uv.y())) *
                                 static_cast<size_t>(image.width) +
                             static_cast<size_t>(static_cast<int>(uv.x()));
        const float occluder_depth = depth[pixel];
        if (std::isfinite(occluder_depth) &&
            range > occluder_depth + depth_tol) {
          continue;  // Occluded.
        }
      }
      const Vec3 rgb = SampleRgb(image, uv.x(), uv.y());
      const float score = static_cast<float>(cos_incidence / range);
      const size_t cell =
          static_cast<size_t>(row) * static_cast<size_t>(grid_.width) +
          static_cast<size_t>(col);
      float* cell_scores = &scores_[cell * static_cast<size_t>(top_k_)];
      int worst = 0;
      for (int slot = 1; slot < top_k_; ++slot) {
        if (cell_scores[slot] < cell_scores[worst]) worst = slot;
      }
      if (score > cell_scores[worst]) {
        cell_scores[worst] = score;
        uint8_t* color = &colors_[(cell * static_cast<size_t>(top_k_) +
                                   static_cast<size_t>(worst)) *
                                  kRgbChannels];
        for (int channel = 0; channel < kRgbChannels; ++channel) {
          color[channel] = static_cast<uint8_t>(std::clamp<int64_t>(
              std::llround(rgb[channel]), 0,
              std::numeric_limits<uint8_t>::max()));
        }
      }
      if (observations_[cell] < std::numeric_limits<uint16_t>::max()) {
        ++observations_[cell];
      }
    }
  }
}

void FloorTextureFuser::AddFrameOccluded(const CameraModel& camera,
                                         const CameraExposure& exposure,
                                         const ImageView& image,
                                         const PointCloudIndex& occluders,
                                         double point_spacing, double depth_tol,
                                         double max_range, double min_cos) {
  const std::vector<float> depth =
      OcclusionDepth(camera, exposure, occluders, point_spacing, max_range);
  AddFrame(camera, exposure, image, depth.data(), depth_tol, max_range,
           min_cos);
}

std::vector<float> FloorTextureFuser::OcclusionDepth(
    const CameraModel& camera, const CameraExposure& exposure,
    const PointCloudIndex& occluders, double point_spacing,
    double max_range) {
  const double occlusion_range = max_range + kOcclusionRangeMargin;
  ZBuffer buffer(camera, exposure, point_spacing, kOccluderMinRadius,
                 kOccluderMaxRadius, kOccluderMinRange, occlusion_range);
  for (int index :
       occluders.RadiusSearch(exposure.center(), occlusion_range)) {
    buffer.Splat(occluders.points()[static_cast<size_t>(index)], kNoLabel,
                 index);
  }
  return buffer.result().depth;
}

FloorTextureFuser::Texture FloorTextureFuser::ComputeTexture() const {
  const size_t cells =
      static_cast<size_t>(grid_.width) * static_cast<size_t>(grid_.height);
  const size_t top_k = static_cast<size_t>(top_k_);
  const size_t channels = static_cast<size_t>(kRgbChannels);
  Texture texture;
  texture.rgb.assign(cells * channels, 0);
  texture.score.assign(cells, 0.0f);
  texture.observations = observations_;
  std::vector<uint8_t> values;
  for (size_t cell = 0; cell < cells; ++cell) {
    const float* cell_scores = &scores_[cell * top_k];
    for (size_t channel = 0; channel < channels; ++channel) {
      values.clear();
      for (size_t slot = 0; slot < top_k; ++slot) {
        if (cell_scores[slot] >= 0.0f) {
          values.push_back(colors_[(cell * top_k + slot) * channels + channel]);
        }
      }
      if (values.empty()) continue;
      const auto median =
          values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
      std::nth_element(values.begin(), median, values.end());
      texture.rgb[cell * channels + channel] = *median;
    }
    float best = 0.0f;
    for (size_t slot = 0; slot < top_k; ++slot) {
      best = std::max(best, cell_scores[slot]);
    }
    texture.score[cell] = best;
  }
  return texture;
}

}  // namespace gr
