#include "gr/projector.h"

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
  result_.label.assign(num_pixels, -1);
  result_.index.assign(num_pixels, -1);
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
  const float range_f = static_cast<float>(range);
  for (int dv = -radius; dv <= radius; ++dv) {
    const int vv = v + dv;
    if (vv < 0 || vv >= result_.height) continue;
    for (int du = -radius; du <= radius; ++du) {
      const int uu = u + du;
      if (uu < 0 || uu >= result_.width) continue;
      const size_t pixel =
          static_cast<size_t>(vv) * static_cast<size_t>(result_.width) +
          static_cast<size_t>(uu);
      if (range_f < result_.depth[pixel]) {
        result_.depth[pixel] = range_f;
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
  for (size_t n = 0; n < points_world.size(); ++n) {
    buffer.Splat(points_world[n], n < labels.size() ? labels[n] : -1,
                 static_cast<int>(n));
  }
  return buffer.TakeResult();
}

VisibilityResult ProjectVisibility(const CameraModel& camera,
                                   const CameraExposure& exposure,
                                   const Points& query_world,
                                   const std::vector<float>& depth,
                                   double abs_tol, double rel_tol) {
  VisibilityResult out;
  const size_t n = query_world.size();
  out.uv = RowMatX2d::Constant(static_cast<Eigen::Index>(n), 2,
                               std::numeric_limits<double>::quiet_NaN());
  out.in_image.assign(n, 0);
  out.visible.assign(n, 0);
  out.range.assign(n, std::numeric_limits<double>::quiet_NaN());
  const bool test_depth =
      !depth.empty() &&
      depth.size() == static_cast<size_t>(camera.width()) *
                          static_cast<size_t>(camera.height());
  for (size_t m = 0; m < n; ++m) {
    const std::optional<CameraExposure::Projection> projection =
        exposure.Project(camera, query_world[m]);
    if (!projection) {
      out.range[m] = (query_world[m] - exposure.center()).norm();
      continue;
    }
    const double range = projection->point_camera.norm();
    out.range[m] = range;
    const std::optional<Vec2> uv = projection->uv;
    out.uv.row(static_cast<Eigen::Index>(m)) = uv->transpose();
    if (!camera.InImage(*uv)) continue;
    out.in_image[m] = 1;
    if (!test_depth) {
      out.visible[m] = 1;
      continue;
    }
    const size_t pixel = static_cast<size_t>(static_cast<int>(uv->y())) *
                             static_cast<size_t>(camera.width()) +
                         static_cast<size_t>(static_cast<int>(uv->x()));
    const float d = depth[pixel];
    if (!std::isfinite(d) ||
        range <= static_cast<double>(d) * (1.0 + rel_tol) + abs_tol) {
      out.visible[m] = 1;
    }
  }
  return out;
}

PointCloudIndex::PointCloudIndex(Points points, double bucket)
    : points_(std::move(points)), bucket_(bucket) {
  for (size_t i = 0; i < points_.size(); ++i) {
    buckets_[CellOf(points_[i], bucket_)].push_back(static_cast<int>(i));
  }
}

template <typename Accept>
std::vector<int> PointCloudIndex::Search(const Vec2& center, double radius,
                                         Accept&& accept) const {
  std::vector<int> out;
  const int x0 = static_cast<int>(std::floor((center.x() - radius) / bucket_));
  const int x1 = static_cast<int>(std::floor((center.x() + radius) / bucket_));
  const int y0 = static_cast<int>(std::floor((center.y() - radius) / bucket_));
  const int y1 = static_cast<int>(std::floor((center.y() + radius) / bucket_));
  for (int bx = x0; bx <= x1; ++bx) {
    for (int by = y0; by <= y1; ++by) {
      const auto it = buckets_.find({bx, by});
      if (it == buckets_.end()) continue;
      for (int i : it->second) {
        if (accept(points_[static_cast<size_t>(i)])) out.push_back(i);
      }
    }
  }
  std::sort(out.begin(), out.end());  // Input order matters for z-buffer ties.
  return out;
}

std::vector<int> PointCloudIndex::RadiusSearch(const Vec3& center,
                                               double radius) const {
  const double r2 = radius * radius;
  return Search(center.head<2>(), radius,
                [&](const Vec3& p) { return (p - center).squaredNorm() < r2; });
}

std::vector<int> PointCloudIndex::RadiusSearchXy(const Vec2& center,
                                                 double radius) const {
  const double r2 = radius * radius;
  return Search(center, radius, [&](const Vec3& p) {
    return (p.head<2>() - center).squaredNorm() < r2;
  });
}

}  // namespace gr
