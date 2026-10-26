#include "gr/sim/sim.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

#include "gr/core/check.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

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
  const Eigen::Index n = directions.rows();
  GR_CHECK(origins.rows() == n || origins.rows() == 1);
  RaycastHits hits;
  hits.distance = Eigen::VectorXd::Constant(n, kInf);
  hits.box_ids = Eigen::VectorXi::Constant(n, -1);
  hits.normals = RowMatX3d::Zero(n, 3);
  const bool shared_origin = origins.rows() == 1;
  const int num_boxes = static_cast<int>(boxes_.size());

  // Slab test of one ray against one box in the box frame; nullopt on a miss.
  auto intersect = [](const Box& box, const Vec3& origin,
                      const Vec3& direction) -> std::optional<BoxHit> {
    const Vec3 o = box.rotation_t * (origin - box.center);
    const Vec3 d = box.rotation_t * direction;
    double t_min = -kInf;
    double t_max = kInf;
    int axis = -1;
    double sign = 0.0;
    for (int a = 0; a < 3; ++a) {
      if (std::abs(d[a]) < 1e-12) {
        if (std::abs(o[a]) > box.half_size[a]) return std::nullopt;
        continue;
      }
      const double t1 = (-box.half_size[a] - o[a]) / d[a];
      const double t2 = (box.half_size[a] - o[a]) / d[a];
      const double t_near = std::min(t1, t2);
      const double t_far = std::max(t1, t2);
      if (t_near > t_min) {
        t_min = t_near;
        axis = a;
        sign = t1 < t2 ? -1.0 : 1.0;
      }
      t_max = std::min(t_max, t_far);
      if (t_min > t_max) return std::nullopt;
    }
    if (axis < 0 || t_min <= 1e-9) return std::nullopt;
    Vec3 normal = Vec3::Zero();
    normal[axis] = sign;
    return BoxHit{t_min, normal};
  };

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 512)
#endif
  for (Eigen::Index r = 0; r < n; ++r) {
    const Vec3 origin = shared_origin ? Vec3(origins.row(0).transpose())
                                      : Vec3(origins.row(r).transpose());
    const Vec3 direction = directions.row(r).transpose();
    double best = max_distance;
    int best_id = -1;
    Vec3 best_normal = Vec3::Zero();
    for (int b = 0; b < num_boxes; ++b) {
      const Box& box = boxes_[static_cast<size_t>(b)];
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
      best_id = b;
      best_normal = box.rotation * hit->local_normal;
    }
    if (best_id >= 0) {
      hits.distance[r] = best;
      hits.box_ids[r] = best_id;
      hits.normals.row(r) = best_normal.transpose();
    }
  }
  return hits;
}

}  // namespace gr
