#include "gr/visibility.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "gr/check.h"

namespace gr {
namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();

// Spherical range image of a sweep: the nearest return per bin and the
// minimum over each bin's 3x3 neighbourhood.
class RangeImage {
 public:
  RangeImage(const Points& sweep, const RangeImageSpec& spec, double max_range)
      : spec_(spec),
        width_(
            std::max(1, static_cast<int>(std::ceil(360.0 / spec.az_res_deg)))),
        height_(std::max(
            1, static_cast<int>(std::ceil((spec.el_max_deg - spec.el_min_deg) /
                                          spec.el_res_deg)))),
        nearest_(static_cast<size_t>(width_) * static_cast<size_t>(height_),
                 kInfinity),
        neighbourhood_min_(nearest_.size(), kInfinity) {
    for (const Vec3& p : sweep) {
      const double r = p.norm();
      if (r < 1e-3 || r > max_range * 1.5) continue;
      int u = 0;
      int v = 0;
      if (!Bin(p, r, u, v)) continue;
      float& cell = nearest_[Index(u, v)];
      cell = std::min(cell, static_cast<float>(r));
    }
    // Conservative 3x3 minimum over measured neighbours: "seen through"
    // requires every measured neighbour to be farther, which avoids false
    // votes at object silhouettes. Empty bins (between beams) carry no
    // information.
    for (int v = 0; v < height_; ++v) {
      for (int u = 0; u < width_; ++u) {
        float m = kInfinity;
        for (int dv = -1; dv <= 1; ++dv) {
          const int vv = v + dv;
          if (vv < 0 || vv >= height_) continue;
          for (int du = -1; du <= 1; ++du) {
            m = std::min(m, nearest_[Index((u + du + width_) % width_, vv)]);
          }
        }
        neighbourhood_min_[Index(u, v)] = m;
      }
    }
  }

  // Bin of a point at range r (sensor frame); false outside the elevations.
  bool Bin(const Vec3& p, double r, int& u, int& v) const {
    const double azimuth = std::atan2(p.y(), p.x()) + kPi;
    const double elevation =
        std::asin(std::clamp(p.z() / r, -1.0, 1.0)) / (kPi / 180.0);
    u = std::min(width_ - 1, static_cast<int>(azimuth / (2.0 * kPi) * width_));
    v = static_cast<int>(
        std::floor((elevation - spec_.el_min_deg) / spec_.el_res_deg));
    return v >= 0 && v < height_;
  }
  size_t Index(int u, int v) const {
    return static_cast<size_t>(v) * static_cast<size_t>(width_) +
           static_cast<size_t>(u);
  }
  float nearest(size_t index) const { return nearest_[index]; }
  float neighbourhood_min(size_t index) const {
    return neighbourhood_min_[index];
  }

 private:
  RangeImageSpec spec_;
  int width_;
  int height_;
  std::vector<float> nearest_;
  std::vector<float> neighbourhood_min_;
};

}  // namespace

VisibilityVoter::VisibilityVoter(const Points& map, const Points& normals,
                                 double bucket_size)
    : map_(map), normals_(normals), bucket_size_(bucket_size) {
  GR_CHECK(normals_.size() == map_.size());
  for (size_t i = 0; i < map_.size(); ++i) {
    buckets_[CellOf(map_[i], bucket_size_)].push_back(static_cast<int>(i));
  }
  zeros_.assign(map_.size(), 0);
}

const std::vector<int>& VisibilityVoter::SeenCounts(int session) const {
  const auto it = counts_.find(session);
  return it == counts_.end() ? zeros_ : it->second.seen;
}

const std::vector<int>& VisibilityVoter::ThroughCounts(int session) const {
  const auto it = counts_.find(session);
  return it == counts_.end() ? zeros_ : it->second.through;
}

std::vector<int> VisibilityVoter::Sessions() const {
  std::vector<int> out;
  for (const auto& [session, counts] : counts_) out.push_back(session);
  return out;
}

void VisibilityVoter::AddSweep(const Pose& world_from_lidar,
                               const Points& sweep,
                               const VisibilityVoteOptions& options,
                               int session) {
  Counts& counts = counts_[session];
  if (counts.seen.size() != map_.size()) {
    counts.seen.assign(map_.size(), 0);
    counts.through.assign(map_.size(), 0);
  }
  const RangeImage image(sweep, options.image, options.max_range);
  const Pose lidar_from_world = world_from_lidar.Inverse();
  const Vec3& origin = world_from_lidar.translation;

  // Map buckets within reach of the sensor.
  std::vector<const std::vector<int>*> todo;
  const int reach =
      static_cast<int>(std::ceil(options.max_range / bucket_size_));
  const CellKey center = CellOf(origin, bucket_size_);
  for (int dx = -reach; dx <= reach; ++dx) {
    for (int dy = -reach; dy <= reach; ++dy) {
      const auto it = buckets_.find({center.x + dx, center.y + dy});
      if (it != buckets_.end()) todo.push_back(&it->second);
    }
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t b = 0; b < static_cast<int64_t>(todo.size()); ++b) {
    // Each map point lives in exactly one bucket: no write conflicts.
    for (int i : *todo[static_cast<size_t>(b)]) {
      const size_t ii = static_cast<size_t>(i);
      const Vec3 p = lidar_from_world * map_[ii];
      const double r = p.norm();
      if (r < 1.0 || r > options.max_range) continue;
      int u = 0;
      int v = 0;
      if (!image.Bin(p, r, u, v)) continue;
      const size_t index = image.Index(u, v);
      const float own = image.nearest(index);
      if (!std::isfinite(own)) continue;  // No return there: no evidence.
      const double margin = options.margin_abs + options.margin_rel * r;
      if (std::abs(own - r) <= margin) {
        ++counts.seen[ii];
        continue;
      }
      // Occluded: something at or before the point.
      if (image.neighbourhood_min(index) <= r + margin) continue;
      const Vec3 direction = (map_[ii] - origin) / r;
      if (std::abs(normals_[ii].dot(direction)) < options.min_cos) continue;
      ++counts.through[ii];
    }
  }
}

}  // namespace gr
