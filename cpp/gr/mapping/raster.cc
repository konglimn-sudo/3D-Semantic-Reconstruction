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

// Bilinear RGB sample at pixel coordinates (u, v), pixel centres at +0.5.
Vec3 SampleRgb(const ImageView& image, double u, double v) {
  u -= 0.5;
  v -= 0.5;
  const int x0 = static_cast<int>(std::floor(u));
  const int y0 = static_cast<int>(std::floor(v));
  const double ax = u - x0;
  const double ay = v - y0;
  Vec3 rgb = Vec3::Zero();
  for (int dy = 0; dy <= 1; ++dy) {
    for (int dx = 0; dx <= 1; ++dx) {
      const int col = std::clamp(x0 + dx, 0, image.width - 1);
      const int row = std::clamp(y0 + dy, 0, image.height - 1);
      const double w = (dx ? ax : 1.0 - ax) * (dy ? ay : 1.0 - ay);
      const uint8_t* px = image.PixelAt(row, col);
      for (int c = 0; c < 3; ++c) rgb[c] += w * px[c];
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
  const double inv = 1.0 / grid.res;
  for (Eigen::Index n = 0; n < xy.rows(); ++n) {
    const int i = static_cast<int>(std::floor((xy(n, 0) - grid.x0) * inv));
    const int j = static_cast<int>(std::floor((xy(n, 1) - grid.y0) * inv));
    if (i < 0 || j < 0 || i >= width || j >= height) continue;
    const double v = values[n];
    switch (reducer) {
      case Reducer::kMean:
      case Reducer::kSum:
        accumulated(j, i) += v;
        break;
      case Reducer::kMax:
        accumulated(j, i) = std::max(accumulated(j, i), v);
        break;
      case Reducer::kMin:
        accumulated(j, i) = std::min(accumulated(j, i), v);
        break;
      case Reducer::kCount:
        break;
    }
    count(j, i) += 1;
  }
  Eigen::MatrixXd out(height, width);
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      const int c = count(j, i);
      if (reducer == Reducer::kCount) {
        out(j, i) = c;
      } else if (c == 0) {
        out(j, i) = fill;
      } else if (reducer == Reducer::kMean) {
        out(j, i) = accumulated(j, i) / c;
      } else {
        out(j, i) = accumulated(j, i);
      }
    }
  }
  return out;
}

FloorTextureFuser::FloorTextureFuser(const Grid2D& grid,
                                     const Eigen::MatrixXd& floor_z, int top_k)
    : grid_(grid), floor_z_(floor_z), top_k_(std::max(1, top_k)) {
  GR_CHECK(floor_z_.rows() == grid.height && floor_z_.cols() == grid.width);
  const size_t cells =
      static_cast<size_t>(grid.width) * static_cast<size_t>(grid.height);
  scores_.assign(cells * static_cast<size_t>(top_k_), -1.0f);
  colors_.assign(cells * static_cast<size_t>(top_k_) * 3, 0);
  observations_.assign(cells, 0);
}

void FloorTextureFuser::AddFrame(const CameraModel& camera,
                                 const CameraExposure& exposure,
                                 const ImageView& image, const float* depth,
                                 double depth_tol, double max_range,
                                 double min_cos) {
  GR_CHECK(image.channels == 3);
  if (image.empty()) return;
  const Vec3& center = exposure.center();
  const double inv = 1.0 / grid_.res;
  const int i0 = std::max(
      0,
      static_cast<int>(std::floor((center.x() - max_range - grid_.x0) * inv)));
  const int i1 = std::min(
      grid_.width - 1,
      static_cast<int>(std::floor((center.x() + max_range - grid_.x0) * inv)));
  const int j0 = std::max(
      0,
      static_cast<int>(std::floor((center.y() - max_range - grid_.y0) * inv)));
  const int j1 = std::min(
      grid_.height - 1,
      static_cast<int>(std::floor((center.y() + max_range - grid_.y0) * inv)));
  if (i0 > i1 || j0 > j1) return;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
  for (int j = j0; j <= j1; ++j) {
    for (int i = i0; i <= i1; ++i) {
      const double z = floor_z_(j, i);
      if (!std::isfinite(z)) continue;
      const Vec3 floor_point(grid_.x0 + (i + 0.5) * grid_.res,
                             grid_.y0 + (j + 0.5) * grid_.res, z);
      const double center_range = (floor_point - center).norm();
      if (center_range > max_range || center_range < 1e-3) continue;
      const std::optional<CameraExposure::Projection> projection =
          exposure.Project(camera, floor_point);
      if (!projection) continue;
      const double range = projection->point_camera.norm();
      if (range > max_range || range < 1e-3) continue;
      const double cos_incidence = (center.z() - z) / range;
      if (cos_incidence < min_cos) continue;
      const Vec2& uv = projection->uv;
      if (!camera.InImage(uv)) continue;
      if (depth != nullptr) {
        const size_t pixel = static_cast<size_t>(static_cast<int>(uv.y())) *
                                 static_cast<size_t>(image.width) +
                             static_cast<size_t>(static_cast<int>(uv.x()));
        const float d = depth[pixel];
        if (std::isfinite(d) && range > d + depth_tol) continue;  // Occluded.
      }
      const Vec3 rgb = SampleRgb(image, uv.x(), uv.y());
      const float score = static_cast<float>(cos_incidence / range);
      const size_t cell =
          static_cast<size_t>(j) * static_cast<size_t>(grid_.width) +
          static_cast<size_t>(i);
      float* cell_scores = &scores_[cell * static_cast<size_t>(top_k_)];
      int worst = 0;
      for (int k = 1; k < top_k_; ++k) {
        if (cell_scores[k] < cell_scores[worst]) worst = k;
      }
      if (score > cell_scores[worst]) {
        cell_scores[worst] = score;
        uint8_t* color = &colors_[(cell * static_cast<size_t>(top_k_) +
                                   static_cast<size_t>(worst)) *
                                  3];
        for (int c = 0; c < 3; ++c) {
          color[c] = static_cast<uint8_t>(
              std::clamp<int64_t>(std::llround(rgb[c]), 0, 255));
        }
      }
      if (observations_[cell] < 65535) ++observations_[cell];
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
  ZBuffer buffer(camera, exposure, point_spacing, 1, 4, 0.1, max_range + 2.0);
  for (int i : occluders.RadiusSearch(exposure.center(), max_range + 2.0)) {
    buffer.Splat(occluders.points()[static_cast<size_t>(i)], -1, i);
  }
  return buffer.result().depth;
}

FloorTextureFuser::Texture FloorTextureFuser::ComputeTexture() const {
  const size_t cells =
      static_cast<size_t>(grid_.width) * static_cast<size_t>(grid_.height);
  const size_t k = static_cast<size_t>(top_k_);
  Texture texture;
  texture.rgb.assign(cells * 3, 0);
  texture.score.assign(cells, 0.0f);
  texture.observations = observations_;
  std::vector<uint8_t> channel;
  for (size_t cell = 0; cell < cells; ++cell) {
    const float* cell_scores = &scores_[cell * k];
    for (int c = 0; c < 3; ++c) {
      channel.clear();
      for (size_t a = 0; a < k; ++a) {
        if (cell_scores[a] >= 0.0f) {
          channel.push_back(
              colors_[(cell * k + a) * 3 + static_cast<size_t>(c)]);
        }
      }
      if (channel.empty()) continue;
      const auto middle =
          channel.begin() + static_cast<std::ptrdiff_t>(channel.size() / 2);
      std::nth_element(channel.begin(), middle, channel.end());
      texture.rgb[cell * 3 + static_cast<size_t>(c)] = *middle;
    }
    float best = 0.0f;
    for (size_t a = 0; a < k; ++a) best = std::max(best, cell_scores[a]);
    texture.score[cell] = best;
  }
  return texture;
}

}  // namespace gr
