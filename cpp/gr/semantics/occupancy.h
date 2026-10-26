// Per-frame semantic occupancy ground truth in the ego (vehicle) frame:
// static labelled map + this frame's object boxes, plus LiDAR and camera
// visibility masks obtained by ray casting (Occ3D-style).

#ifndef GARAGE_RECON_GR_SEMANTICS_OCCUPANCY_H_
#define GARAGE_RECON_GR_SEMANTICS_OCCUPANCY_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "gr/core/pose.h"
#include "gr/semantics/objects.h"
#include "gr/semantics/projector.h"
#include "gr/sensors/camera.h"

namespace gr {

struct OrientedBox {
  Vec3 center = Vec3::Zero();
  Vec3 size = Vec3::Ones();  // Length (x), width (y), height (z), box frame.
  double yaw = 0.0;          // Rotation about z.
  int label = 0;
};

struct CameraView {
  std::shared_ptr<CameraModel> camera;
  Pose ego_from_camera;
};

struct OccupancyConfig {
  Vec3 min_bound = Vec3(-10.0, -10.0, -0.5);
  Vec3 max_bound = Vec3(10.0, 10.0, 3.0);
  double voxel = 0.1;
  int free_label = 0;
  int camera_pixel_stride = 4;
  double camera_max_range = 15.0;
};

struct OccupancyResult {
  int nx = 0;  // Index = (ix * ny + iy) * nz + iz.
  int ny = 0;
  int nz = 0;
  std::vector<uint8_t> semantics;    // free_label or class id.
  std::vector<uint8_t> mask_lidar;   // 1 = observed by the LiDAR sweep.
  std::vector<uint8_t> mask_camera;  // 1 = observed by at least one camera.
};

OccupancyResult BuildOccupancy(
    const OccupancyConfig& config, const Pose& world_from_ego,
    const Points& static_world, const std::vector<int>& static_labels,
    const std::vector<OrientedBox>& boxes_ego, const Vec3& lidar_origin_ego,
    const Points& scan_ego, const std::vector<CameraView>& cameras);

// Occupancy of a whole drive: holds the labelled static map (indexed in xy)
// and the rig, and produces one grid per frame from the vehicle pose, that
// frame's object boxes (world frame) and the LiDAR sweep.
class OccupancyGenerator {
 public:
  // One label per static point; `cameras` carry ego_from_camera (the ego
  // frame is the vehicle frame).
  OccupancyGenerator(const Points& static_world,
                     const std::vector<int>& static_labels,
                     const OccupancyConfig& config,
                     const Pose& vehicle_from_lidar,
                     std::vector<CameraView> cameras);
  OccupancyGenerator(const OccupancyGenerator&) = default;
  OccupancyGenerator& operator=(const OccupancyGenerator&) = default;
  OccupancyGenerator(OccupancyGenerator&&) = default;
  OccupancyGenerator& operator=(OccupancyGenerator&&) = default;

  OccupancyResult Generate(const Pose& world_from_vehicle,
                           const std::vector<ObjectBox>& boxes_world,
                           const Points& sweep_lidar) const;
  const OccupancyConfig& config() const { return config_; }

 private:
  OccupancyConfig config_;
  Pose vehicle_from_lidar_;
  std::vector<CameraView> cameras_;
  PointCloudIndex index_;
  std::vector<int> labels_;
};

}  // namespace gr

#endif  // GARAGE_RECON_GR_SEMANTICS_OCCUPANCY_H_
