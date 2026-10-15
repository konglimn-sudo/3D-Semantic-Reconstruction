#include "gr/exposure.h"

#include <algorithm>
#include <cmath>

#include "gr/check.h"

namespace gr {
namespace {

constexpr int kMaxRowIterations = 4;

bool SamePose(const Pose& a, const Pose& b) {
  return (a.translation - b.translation).squaredNorm() < 1e-24 &&
         (a.rotation - b.rotation).squaredNorm() < 1e-24;
}

}  // namespace

CameraExposure::CameraExposure(const Pose& world_from_camera)
    : camera_from_world_{world_from_camera.Inverse()},
      world_from_center_(world_from_camera) {}

CameraExposure::CameraExposure(const Pose& world_from_first_row,
                               const Pose& world_from_last_row, int height) {
  GR_CHECK(height >= 1);
  if (height == 1 || SamePose(world_from_first_row, world_from_last_row)) {
    camera_from_world_.push_back(world_from_first_row.Inverse());
    world_from_center_ = world_from_first_row;
    return;
  }
  camera_from_world_.reserve(static_cast<size_t>(height));
  const double last = static_cast<double>(height - 1);
  for (int row = 0; row < height; ++row) {
    camera_from_world_.push_back(
        Interpolate(world_from_first_row, world_from_last_row, row / last)
            .Inverse());
  }
  world_from_center_ =
      Interpolate(world_from_first_row, world_from_last_row, 0.5);
}

const Pose& CameraExposure::CameraFromWorld(int row) const {
  const int clamped = std::clamp(row, 0, rows() - 1);
  return camera_from_world_[static_cast<size_t>(clamped)];
}

std::optional<CameraExposure::Projection> CameraExposure::Project(
    const CameraModel& camera, const Vec3& point_world) const {
  int row = rows() / 2;
  Projection projection;
  for (int iteration = 0; iteration < kMaxRowIterations; ++iteration) {
    projection.point_camera = CameraFromWorld(row) * point_world;
    const std::optional<Vec2> uv = camera.Project(projection.point_camera);
    if (!uv) return std::nullopt;
    projection.uv = *uv;
    if (!rolling()) break;
    const int next =
        std::clamp(static_cast<int>(std::floor(uv->y())), 0, rows() - 1);
    if (next == row) break;
    row = next;
  }
  return projection;
}

}  // namespace gr
