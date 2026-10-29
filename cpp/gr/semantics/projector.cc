#include "gr/semantics/projector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace gr {

ZBuffer::ZBuffer(const CameraModel& camera, const CameraExposure& exposure,
                 double point_spacing, int min_radius, int max_radius,
                 double min_range, double max_range)
    : camera_(camera),
      exposure_(exposure),
      point_spacing_(point_spacing),
      min_range_(min_range),
      max_range_(max_range),
      focal_(camera.focal()),
      min_radius_(min_radius),
      max_radius_(max_radius) {
  result_.width = camera.width();
  result_.height = camera.height();
  const size_t num_pixels =
      static_cast<size_t>(result_.width) * static_cast<size_t>(result_.height);
  result_.depth.assign(num_pixels, std::numeric_limits<float>::infinity());
  result_.label.assign(num_pixels, RenderResult::kEmpty);
  result_.index.assign(num_pixels, RenderResult::kEmpty);
}

void ZBuffer::Splat(const Vec3& point_world, int label, int index) {
  const std::optional<CameraExposure::Projection> projection =
      exposure_.Project(camera_, point_world);
  if (!projection) return;
  const double range = projection->point_camera.norm();
  if (range < min_range_ || range > max_range_) return;
  const int u = static_cast<int>(std::floor(projection->uv.x()));
  const int v = static_cast<int>(std::floor(projection->uv.y()));
  const int radius =
      std::clamp(static_cast<int>(std::lround(focal_ * point_spacing_ / range)),
                 min_radius_, max_radius_);
  if (u + radius < 0 || v + radius < 0 || u - radius >= result_.width ||
      v - radius >= result_.height) {
    return;
  }
  const float range_float = static_cast<float>(range);
  for (int row = v - radius; row <= v + radius; ++row) {
    if (row < 0 || row >= result_.height) continue;
    for (int column = u - radius; column <= u + radius; ++column) {
      if (column < 0 || column >= result_.width) continue;
      const size_t pixel =
          static_cast<size_t>(row) * static_cast<size_t>(result_.width) +
          static_cast<size_t>(column);
      if (range_float < result_.depth[pixel]) {
        result_.depth[pixel] = range_float;
        result_.label[pixel] = label;
        result_.index[pixel] = index;
      }
    }
  }
}

RenderResult RenderPoints(const CameraModel& camera,
                          const CameraExposure& exposure,
                          const Points& points_world,
                          const std::vector<int>& labels, double point_spacing,
                          int min_radius, int max_radius, double min_range,
                          double max_range) {
  ZBuffer buffer(camera, exposure, point_spacing, min_radius, max_radius,
                 min_range, max_range);
  for (size_t point = 0; point < points_world.size(); ++point) {
    buffer.Splat(points_world[point],
                 point < labels.size() ? labels[point] : RenderResult::kEmpty,
                 static_cast<int>(point));
  }
  return buffer.TakeResult();
}

std::vector<float> RenderSurfaceDepth(const CameraModel& camera,
                                      const CameraExposure& exposure,
                                      const Points& points_world,
                                      double point_spacing, int min_radius,
                                      int max_radius, double min_range,
                                      double max_range, double abs_tol,
                                      double rel_tol, int mean_radius) {
  const RenderResult nearest =
      RenderPoints(camera, exposure, points_world, {}, point_spacing,
                   min_radius, max_radius, min_range, max_range);
  const int width = camera.width();
  const int height = camera.height();
  const size_t num_pixels =
      static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<double> sum(num_pixels, 0.0);
  std::vector<int> count(num_pixels, 0);
  const double focal = camera.focal();
  for (const Vec3& point : points_world) {
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(camera, point);
    if (!projection) continue;
    const double range = projection->point_camera.norm();
    if (range < min_range || range > max_range) continue;
    const int u = static_cast<int>(std::floor(projection->uv.x()));
    const int v = static_cast<int>(std::floor(projection->uv.y()));
    const int radius = std::min(
        mean_radius,
        std::clamp(static_cast<int>(std::lround(focal * point_spacing / range)),
                   min_radius, max_radius));
    for (int row = v - radius; row <= v + radius; ++row) {
      if (row < 0 || row >= height) continue;
      for (int column = u - radius; column <= u + radius; ++column) {
        if (column < 0 || column >= width) continue;
        const size_t pixel =
            static_cast<size_t>(row) * static_cast<size_t>(width) +
            static_cast<size_t>(column);
        const double first = nearest.depth[pixel];
        if (!std::isfinite(first) ||
            range > first + abs_tol + rel_tol * first) {
          continue;
        }
        sum[pixel] += range;
        ++count[pixel];
      }
    }
  }
  std::vector<float> out(num_pixels, std::numeric_limits<float>::quiet_NaN());
  for (size_t pixel = 0; pixel < num_pixels; ++pixel) {
    if (count[pixel] > 0) {
      out[pixel] = static_cast<float>(sum[pixel] / count[pixel]);
    }
  }
  return out;
}

VisibilityResult ProjectVisibility(const CameraModel& camera,
                                   const CameraExposure& exposure,
                                   const Points& query_world,
                                   const std::vector<float>& depth,
                                   double abs_tol, double rel_tol) {
  VisibilityResult out;
  const size_t count = query_world.size();
  out.uv = RowMatX2d::Constant(static_cast<Eigen::Index>(count), kPlaneDim,
                               std::numeric_limits<double>::quiet_NaN());
  out.in_image.assign(count, 0);
  out.visible.assign(count, 0);
  out.range.assign(count, std::numeric_limits<double>::quiet_NaN());
  const bool test_depth =
      !depth.empty() &&
      depth.size() == static_cast<size_t>(camera.width()) *
                          static_cast<size_t>(camera.height());
  for (size_t query = 0; query < count; ++query) {
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(camera, query_world[query]);
    if (!projection) {
      out.range[query] = (query_world[query] - exposure.center()).norm();
      continue;
    }
    const double range = projection->point_camera.norm();
    out.range[query] = range;
    const std::optional<Vec2> uv = projection->uv;
    out.uv.row(static_cast<Eigen::Index>(query)) = uv->transpose();
    if (!camera.InImage(*uv)) continue;
    out.in_image[query] = 1;
    if (!test_depth) {
      out.visible[query] = 1;
      continue;
    }
    const size_t pixel = static_cast<size_t>(static_cast<int>(uv->y())) *
                             static_cast<size_t>(camera.width()) +
                         static_cast<size_t>(static_cast<int>(uv->x()));
    const float surface_range = depth[pixel];
    if (!std::isfinite(surface_range) ||
        range <=
            static_cast<double>(surface_range) * (1.0 + rel_tol) + abs_tol) {
      out.visible[query] = 1;
    }
  }
  return out;
}

PointCloudIndex::PointCloudIndex(Points points, double bucket)
    : points_(std::move(points)), bucket_(bucket) {
  for (size_t point = 0; point < points_.size(); ++point) {
    buckets_[CellOf(points_[point], bucket_)].push_back(
        static_cast<int>(point));
  }
}

template <typename Accept>
std::vector<int> PointCloudIndex::Search(const Vec2& center, double radius,
                                         Accept&& accept) const {
  std::vector<int> out;
  const int first_x =
      static_cast<int>(std::floor((center.x() - radius) / bucket_));
  const int last_x =
      static_cast<int>(std::floor((center.x() + radius) / bucket_));
  const int first_y =
      static_cast<int>(std::floor((center.y() - radius) / bucket_));
  const int last_y =
      static_cast<int>(std::floor((center.y() + radius) / bucket_));
  for (int bucket_x = first_x; bucket_x <= last_x; ++bucket_x) {
    for (int bucket_y = first_y; bucket_y <= last_y; ++bucket_y) {
      const auto found = buckets_.find({bucket_x, bucket_y});
      if (found == buckets_.end()) continue;
      for (int point : found->second) {
        if (accept(points_[static_cast<size_t>(point)])) out.push_back(point);
      }
    }
  }
  std::sort(out.begin(), out.end());  // Input order matters for z-buffer ties.
  return out;
}

std::vector<int> PointCloudIndex::RadiusSearch(const Vec3& center,
                                               double radius) const {
  const double squared_radius = radius * radius;
  return Search(center.head<2>(), radius, [&](const Vec3& point) {
    return (point - center).squaredNorm() < squared_radius;
  });
}

std::vector<int> PointCloudIndex::RadiusSearchXy(const Vec2& center,
                                                 double radius) const {
  const double squared_radius = radius * radius;
  return Search(center, radius, [&](const Vec3& point) {
    return (point.head<2>() - center).squaredNorm() < squared_radius;
  });
}

}  // namespace gr
