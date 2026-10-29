#include "gr/sim/sim.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// Ray direction components below this are parallel to that slab.
constexpr double kParallelDirection = 1e-12;
// Hits closer than this (m) are the ray's own origin.
constexpr double kMinHitDistance = 1e-9;
// Rays per OpenMP chunk.
constexpr int kRaysPerChunk = 512;
// No entry slab yet.
constexpr int kNoAxis = -1;

struct BoxHit {
  double distance;
  Vec3 local_normal;
};

}  // namespace

int BoxScene::AddBox(const Vec3& center, const Vec3& size,
                     const Mat3& rotation) {
  Box box;
  box.center = center;
  box.half_size = 0.5 * size.cwiseAbs();
  box.rotation = rotation;
  box.rotation_t = rotation.transpose();
  box.radius = box.half_size.norm();
  boxes_.push_back(box);
  return static_cast<int>(boxes_.size()) - 1;
}

RaycastHits BoxScene::Raycast(const RowMatX3d& origins,
                              const RowMatX3d& directions,
                              double max_distance) const {
  const Eigen::Index ray_count = directions.rows();
  GR_CHECK(origins.rows() == ray_count || origins.rows() == 1);
  RaycastHits hits;
  hits.distance = Eigen::VectorXd::Constant(ray_count, kInf);
  hits.box_ids = Eigen::VectorXi::Constant(ray_count, RaycastHits::kMiss);
  hits.normals = RowMatX3d::Zero(ray_count, kSpaceDim);
  const bool shared_origin = origins.rows() == 1;
  const int num_boxes = static_cast<int>(boxes_.size());

  // Slab test of one ray against one box in the box frame; nullopt on a miss.
  const auto intersect = [](const Box& box, const Vec3& origin,
                            const Vec3& direction) -> std::optional<BoxHit> {
    const Vec3 local_origin = box.rotation_t * (origin - box.center);
    const Vec3 local_direction = box.rotation_t * direction;
    double t_min = -kInf;
    double t_max = kInf;
    int entry_axis = kNoAxis;
    double sign = 0.0;
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      if (std::abs(local_direction[axis]) < kParallelDirection) {
        if (std::abs(local_origin[axis]) > box.half_size[axis]) {
          return std::nullopt;
        }
        continue;
      }
      const double t_low =
          (-box.half_size[axis] - local_origin[axis]) / local_direction[axis];
      const double t_high =
          (box.half_size[axis] - local_origin[axis]) / local_direction[axis];
      const double t_near = std::min(t_low, t_high);
      const double t_far = std::max(t_low, t_high);
      if (t_near > t_min) {
        t_min = t_near;
        entry_axis = axis;
        sign = t_low < t_high ? -1.0 : 1.0;
      }
      t_max = std::min(t_max, t_far);
      if (t_min > t_max) return std::nullopt;
    }
    if (entry_axis == kNoAxis || t_min <= kMinHitDistance) {
      return std::nullopt;
    }
    Vec3 normal = Vec3::Zero();
    normal[entry_axis] = sign;
    return BoxHit{t_min, normal};
  };

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, kRaysPerChunk)
#endif
  for (Eigen::Index ray = 0; ray < ray_count; ++ray) {
    const Vec3 origin = shared_origin ? Vec3(origins.row(0).transpose())
                                      : Vec3(origins.row(ray).transpose());
    const Vec3 direction = directions.row(ray).transpose();
    double best = max_distance;
    int best_id = RaycastHits::kMiss;
    Vec3 best_normal = Vec3::Zero();
    for (int box_id = 0; box_id < num_boxes; ++box_id) {
      const Box& box = boxes_[static_cast<size_t>(box_id)];
      // Bounding-sphere rejection.
      const Vec3 to_center = box.center - origin;
      const double t_center = to_center.dot(direction);
      if (t_center + box.radius < 0.0) continue;
      if (to_center.squaredNorm() - t_center * t_center >
          box.radius * box.radius) {
        continue;
      }
      if (t_center - box.radius > best) continue;
      const std::optional<BoxHit> hit = intersect(box, origin, direction);
      if (!hit || hit->distance >= best) continue;
      best = hit->distance;
      best_id = box_id;
      best_normal = box.rotation * hit->local_normal;
    }
    if (best_id != RaycastHits::kMiss) {
      hits.distance[ray] = best;
      hits.box_ids[ray] = best_id;
      hits.normals.row(ray) = best_normal.transpose();
    }
  }
  return hits;
}

}  // namespace gr
