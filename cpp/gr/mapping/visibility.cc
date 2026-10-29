#include "gr/mapping/visibility.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr float kInfinity = std::numeric_limits<float>::infinity();
constexpr double kFullTurn = 2.0 * kPi;
constexpr double kDegreesPerTurn = 2.0 * kDegreesPerHalfTurn;
// Returns closer than this (m) are noise; returns beyond this many times the
// voting range still occlude.
constexpr double kMinReturnRange = 1e-3;
constexpr double kOccluderRangeFactor = 1.5;
// Map points closer than this (m) to the sensor do not vote.
constexpr double kMinPointRange = 1.0;
// The minimum is taken over the bins this many steps around a bin.
constexpr int kNeighbourhoodRadius = 1;

// Spherical range image of a sweep: the nearest return per bin and the
// minimum over each bin's neighbourhood.
class RangeImage {
 public:
  RangeImage(const Points& sweep, const RangeImageSpec& spec, double max_range)
      : spec_(spec),
        width_(std::max(1, static_cast<int>(std::ceil(kDegreesPerTurn /
                                                      spec.az_res_deg)))),
        height_(std::max(
            1, static_cast<int>(std::ceil((spec.el_max_deg - spec.el_min_deg) /
                                          spec.el_res_deg)))),
        nearest_(static_cast<size_t>(width_) * static_cast<size_t>(height_),
                 kInfinity),
        neighbourhood_min_(nearest_.size(), kInfinity) {
    for (const Vec3& point : sweep) {
      const double range = point.norm();
      if (range < kMinReturnRange || range > max_range * kOccluderRangeFactor) {
        continue;
      }
      int column = 0;
      int row = 0;
      if (!Bin(point, range, column, row)) continue;
      float& cell = nearest_[Index(column, row)];
      cell = std::min(cell, static_cast<float>(range));
    }
    // Conservative neighbourhood minimum over measured neighbours: "seen
    // through" requires every measured neighbour to be farther, which avoids
    // false votes at object silhouettes. Empty bins (between beams) carry no
    // information.
    for (int row = 0; row < height_; ++row) {
      for (int column = 0; column < width_; ++column) {
        float minimum = kInfinity;
        for (int dv = -kNeighbourhoodRadius; dv <= kNeighbourhoodRadius;
             ++dv) {
          const int neighbour_row = row + dv;
          if (neighbour_row < 0 || neighbour_row >= height_) continue;
          for (int du = -kNeighbourhoodRadius; du <= kNeighbourhoodRadius;
               ++du) {
            // Azimuth wraps around.
            const int neighbour_column = (column + du + width_) % width_;
            minimum = std::min(
                minimum, nearest_[Index(neighbour_column, neighbour_row)]);
          }
        }
        neighbourhood_min_[Index(column, row)] = minimum;
      }
    }
  }

  // Bin (`column` by azimuth, `row` by elevation) of a point at `range`
  // (sensor frame); false outside the elevations.
  bool Bin(const Vec3& point, double range, int& column, int& row) const {
    const double azimuth = std::atan2(point.y(), point.x()) + kPi;
    const double elevation =
        std::asin(std::clamp(point.z() / range, -1.0, 1.0)) /
        (kPi / kDegreesPerHalfTurn);
    column =
        std::min(width_ - 1, static_cast<int>(azimuth / kFullTurn * width_));
    row = static_cast<int>(
        std::floor((elevation - spec_.el_min_deg) / spec_.el_res_deg));
    return row >= 0 && row < height_;
  }
  size_t Index(int column, int row) const {
    return static_cast<size_t>(row) * static_cast<size_t>(width_) +
           static_cast<size_t>(column);
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
  const auto counts = counts_.find(session);
  return counts == counts_.end() ? zeros_ : counts->second.seen;
}

const std::vector<int>& VisibilityVoter::ThroughCounts(int session) const {
  const auto counts = counts_.find(session);
  return counts == counts_.end() ? zeros_ : counts->second.through;
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
  std::vector<const std::vector<int>*> buckets_in_reach;
  const int reach =
      static_cast<int>(std::ceil(options.max_range / bucket_size_));
  const CellKey center = CellOf(origin, bucket_size_);
  for (int dx = -reach; dx <= reach; ++dx) {
    for (int dy = -reach; dy <= reach; ++dy) {
      const auto bucket = buckets_.find({center.x + dx, center.y + dy});
      if (bucket != buckets_.end()) buckets_in_reach.push_back(&bucket->second);
    }
  }
  const int64_t bucket_count = static_cast<int64_t>(buckets_in_reach.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
    // Each map point lives in exactly one bucket: no write conflicts.
    for (int point_index : *buckets_in_reach[static_cast<size_t>(bucket)]) {
      const size_t index = static_cast<size_t>(point_index);
      const Vec3 point = lidar_from_world * map_[index];
      const double range = point.norm();
      if (range < kMinPointRange || range > options.max_range) continue;
      int column = 0;
      int row = 0;
      if (!image.Bin(point, range, column, row)) continue;
      const size_t bin = image.Index(column, row);
      const float measured = image.nearest(bin);
      if (!std::isfinite(measured)) continue;  // No return there: no evidence.
      const double margin = options.margin_abs + options.margin_rel * range;
      if (std::abs(measured - range) <= margin) {
        ++counts.seen[index];
        continue;
      }
      // Occluded: something at or before the point.
      if (image.neighbourhood_min(bin) <= range + margin) continue;
      const Vec3 direction = (map_[index] - origin) / range;
      if (std::abs(normals_[index].dot(direction)) < options.min_cos) continue;
      ++counts.through[index];
    }
  }
}

}  // namespace gr
