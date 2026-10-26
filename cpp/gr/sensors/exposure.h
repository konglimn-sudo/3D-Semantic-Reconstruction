// Camera poses over the exposure of one image.
//
// A rolling-shutter sensor exposes row r at t_first + r * line_time, so every
// row has its own pose. CameraExposure keeps camera_from_world for every row
// (interpolated between the poses of the first and the last row) and projects
// a world point with the pose of the row it lands on. A global-shutter image
// has a single pose. All poses come from the shared trajectory of the drive
// (see trajectory.h): no projection code builds camera poses on its own.

#ifndef GARAGE_RECON_GR_SENSORS_EXPOSURE_H_
#define GARAGE_RECON_GR_SENSORS_EXPOSURE_H_

#include <optional>
#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/sensors/camera.h"

namespace gr {

class CameraExposure {
 public:
  // Global shutter.
  explicit CameraExposure(const Pose& world_from_camera);
  // Rolling shutter: world_from_camera of the first and the last of `height`
  // rows (height >= 1). Identical poses give a global-shutter exposure.
  CameraExposure(const Pose& world_from_first_row,
                 const Pose& world_from_last_row, int height);

  CameraExposure(const CameraExposure&) = default;
  CameraExposure& operator=(const CameraExposure&) = default;
  CameraExposure(CameraExposure&&) = default;
  CameraExposure& operator=(CameraExposure&&) = default;

  bool rolling() const { return camera_from_world_.size() > 1; }
  // Number of rows with their own pose (1 for global shutter).
  int rows() const { return static_cast<int>(camera_from_world_.size()); }
  // camera_from_world of an image row (clamped to the valid rows).
  const Pose& CameraFromWorld(int row) const;
  // world_from_camera of the middle row.
  const Pose& world_from_camera() const { return world_from_center_; }
  const Vec3& center() const { return world_from_center_.translation; }

  struct Projection {
    Vec2 uv;
    Vec3 point_camera;  // In the camera frame of the row the point lands on.
  };
  // Pixel of a world point, or nullopt outside the model's field of view
  // (image bounds are not checked). For rolling shutter the row is found by
  // fixed-point iteration: the camera moves far less than a pixel per row.
  std::optional<Projection> Project(const CameraModel& camera,
                                    const Vec3& point_world) const;

 private:
  std::vector<Pose> camera_from_world_;  // One per row, or a single pose.
  Pose world_from_center_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_SENSORS_EXPOSURE_H_
