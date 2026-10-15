#include "gr/occupancy.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

#include "gr/check.h"
#include "gr/raycast.h"

namespace gr {
namespace {

// Voxel grid geometry of an OccupancyConfig.
class Grid {
 public:
  explicit Grid(const OccupancyConfig& config)
      : min_bound_(config.min_bound), voxel_(config.voxel), inv_(1.0 / voxel_) {
    const Vec3 extent = (config.max_bound - config.min_bound) * inv_;
    nx_ = std::max(1, static_cast<int>(std::round(extent.x())));
    ny_ = std::max(1, static_cast<int>(std::round(extent.y())));
    nz_ = std::max(1, static_cast<int>(std::round(extent.z())));
  }

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }
  size_t size() const {
    return static_cast<size_t>(nx_) * static_cast<size_t>(ny_) *
           static_cast<size_t>(nz_);
  }
  size_t Linear(int i, int j, int k) const {
    return (static_cast<size_t>(i) * static_cast<size_t>(ny_) +
            static_cast<size_t>(j)) *
               static_cast<size_t>(nz_) +
           static_cast<size_t>(k);
  }
  // Continuous grid coordinates of an ego-frame point.
  Vec3 ToGrid(const Vec3& point) const { return (point - min_bound_) * inv_; }
  Vec3 VoxelCenter(int i, int j, int k) const {
    return min_bound_ + (Vec3(i, j, k) + Vec3::Constant(0.5)) * voxel_;
  }
  std::optional<Eigen::Vector3i> IndexOf(const Vec3& point) const {
    const Vec3 g = ToGrid(point);
    const Eigen::Vector3i ijk(static_cast<int>(std::floor(g.x())),
                              static_cast<int>(std::floor(g.y())),
                              static_cast<int>(std::floor(g.z())));
    if (ijk.x() < 0 || ijk.y() < 0 || ijk.z() < 0 || ijk.x() >= nx_ ||
        ijk.y() >= ny_ || ijk.z() >= nz_) {
      return std::nullopt;
    }
    return ijk;
  }

 private:
  Vec3 min_bound_;
  double voxel_;
  double inv_;
  int nx_;
  int ny_;
  int nz_;
};

// Majority label of the static points per voxel (ties: lowest label).
void VoteStaticLabels(const Grid& grid, const OccupancyConfig& config,
                      const Pose& ego_from_world, const Points& static_world,
                      const std::vector<int>& static_labels,
                      std::vector<uint8_t>& semantics) {
  std::vector<std::pair<uint32_t, uint8_t>> votes;
  votes.reserve(static_world.size() / 8 + 16);
  for (size_t m = 0; m < static_world.size(); ++m) {
    const int label = m < static_labels.size() ? static_labels[m] : 0;
    if (label == config.free_label || label < 0 || label > 255) continue;
    const auto ijk = grid.IndexOf(ego_from_world * static_world[m]);
    if (!ijk) continue;
    votes.emplace_back(
        static_cast<uint32_t>(grid.Linear(ijk->x(), ijk->y(), ijk->z())),
        static_cast<uint8_t>(label));
  }
  std::sort(votes.begin(), votes.end());
  for (size_t a = 0; a < votes.size();) {
    size_t b = a;
    uint8_t best_label = votes[a].second;
    size_t best_count = 0;
    while (b < votes.size() && votes[b].first == votes[a].first) {
      size_t c = b;
      while (c < votes.size() && votes[c].first == votes[a].first &&
             votes[c].second == votes[b].second) {
        ++c;
      }
      if (c - b > best_count) {
        best_count = c - b;
        best_label = votes[b].second;
      }
      b = c;
    }
    semantics[votes[a].first] = best_label;
    a = b;
  }
}

// Object boxes override the static labels inside them.
void RasterizeBoxes(const Grid& grid, const std::vector<OrientedBox>& boxes,
                    std::vector<uint8_t>& semantics) {
  for (const OrientedBox& box : boxes) {
    const Mat3 rotation =
        Eigen::AngleAxisd(box.yaw, Vec3::UnitZ()).toRotationMatrix();
    const Vec3 half = 0.5 * box.size;
    Vec3 lo = Vec3::Constant(1e18);
    Vec3 hi = Vec3::Constant(-1e18);
    for (int c = 0; c < 8; ++c) {
      const Vec3 corner(c & 1 ? half.x() : -half.x(),
                        c & 2 ? half.y() : -half.y(),
                        c & 4 ? half.z() : -half.z());
      const Vec3 world = box.center + rotation * corner;
      lo = lo.cwiseMin(world);
      hi = hi.cwiseMax(world);
    }
    const Vec3 glo = grid.ToGrid(lo);
    const Vec3 ghi = grid.ToGrid(hi);
    const auto first = [](double g) {
      return std::max(0, static_cast<int>(std::floor(g)));
    };
    const auto last = [](double g, int n) {
      return std::min(n - 1, static_cast<int>(std::floor(g)));
    };
    for (int i = first(glo.x()); i <= last(ghi.x(), grid.nx()); ++i) {
      for (int j = first(glo.y()); j <= last(ghi.y(), grid.ny()); ++j) {
        for (int k = first(glo.z()); k <= last(ghi.z(), grid.nz()); ++k) {
          const Vec3 local =
              rotation.transpose() * (grid.VoxelCenter(i, j, k) - box.center);
          if (std::abs(local.x()) <= half.x() &&
              std::abs(local.y()) <= half.y() &&
              std::abs(local.z()) <= half.z()) {
            semantics[grid.Linear(i, j, k)] = static_cast<uint8_t>(box.label);
          }
        }
      }
    }
  }
}

// Voxels traversed by or containing a LiDAR return.
void MarkLidarVisibility(const Grid& grid, const Vec3& lidar_origin_ego,
                         const Points& scan_ego, std::vector<uint8_t>& mask) {
  const Vec3 origin = grid.ToGrid(lidar_origin_ego);
  for (const Vec3& p : scan_ego) {
    const Vec3 delta = grid.ToGrid(p) - origin;
    const double length = delta.norm();
    if (length < 1e-9) continue;
    TraverseGrid(origin, delta / length, length, grid.nx(), grid.ny(),
                 grid.nz(), [&](int i, int j, int k) {
                   mask[grid.Linear(i, j, k)] = 1;
                   return true;
                 });
  }
}

// Pixel rays marched until the first occupied voxel.
void MarkCameraVisibility(const Grid& grid, const OccupancyConfig& config,
                          const std::vector<CameraView>& cameras,
                          const std::vector<uint8_t>& semantics,
                          std::vector<uint8_t>& mask) {
  const double t_end = config.camera_max_range * (1.0 / config.voxel);
  const int stride = std::max(1, config.camera_pixel_stride);
  for (const CameraView& view : cameras) {
    if (!view.camera) continue;
    const Vec3 center = grid.ToGrid(view.ego_from_camera.translation);
    for (int v = stride / 2; v < view.camera->height(); v += stride) {
      for (int u = stride / 2; u < view.camera->width(); u += stride) {
        const std::optional<Vec3> ray =
            view.camera->Unproject(Vec2(u + 0.5, v + 0.5));
        if (!ray) continue;
        const Vec3 direction =
            (view.ego_from_camera.rotation * *ray).normalized();
        TraverseGrid(center, direction, t_end, grid.nx(), grid.ny(), grid.nz(),
                     [&](int i, int j, int k) {
                       const size_t index = grid.Linear(i, j, k);
                       mask[index] = 1;
                       return semantics[index] ==
                              static_cast<uint8_t>(config.free_label);
                     });
      }
    }
  }
}

}  // namespace

OccupancyResult BuildOccupancy(
    const OccupancyConfig& config, const Pose& world_from_ego,
    const Points& static_world, const std::vector<int>& static_labels,
    const std::vector<OrientedBox>& boxes_ego, const Vec3& lidar_origin_ego,
    const Points& scan_ego, const std::vector<CameraView>& cameras) {
  const Grid grid(config);
  OccupancyResult result;
  result.nx = grid.nx();
  result.ny = grid.ny();
  result.nz = grid.nz();
  result.semantics.assign(grid.size(), static_cast<uint8_t>(config.free_label));
  result.mask_lidar.assign(grid.size(), 0);
  result.mask_camera.assign(grid.size(), 0);
  VoteStaticLabels(grid, config, world_from_ego.Inverse(), static_world,
                   static_labels, result.semantics);
  RasterizeBoxes(grid, boxes_ego, result.semantics);
  MarkLidarVisibility(grid, lidar_origin_ego, scan_ego, result.mask_lidar);
  MarkCameraVisibility(grid, config, cameras, result.semantics,
                       result.mask_camera);
  return result;
}

OccupancyGenerator::OccupancyGenerator(const Points& static_world,
                                       const std::vector<int>& static_labels,
                                       const OccupancyConfig& config,
                                       const Pose& vehicle_from_lidar,
                                       std::vector<CameraView> cameras)
    : config_(config),
      vehicle_from_lidar_(vehicle_from_lidar),
      cameras_(std::move(cameras)),
      index_(static_world),
      labels_(static_labels) {
  GR_CHECK(labels_.size() == static_world.size());
}

OccupancyResult OccupancyGenerator::Generate(
    const Pose& world_from_vehicle, const std::vector<ObjectBox>& boxes_world,
    const Points& sweep_lidar) const {
  // Static points that can fall into the grid: an xy disc around the vehicle.
  const double extent_x = std::max(std::abs(config_.min_bound.x()),
                                   std::abs(config_.max_bound.x()));
  const double extent_y = std::max(std::abs(config_.min_bound.y()),
                                   std::abs(config_.max_bound.y()));
  const std::vector<int> near =
      index_.RadiusSearchXy(world_from_vehicle.translation.head<2>(),
                            std::hypot(extent_x, extent_y) + 1.0);
  Points points;
  std::vector<int> labels;
  points.reserve(near.size());
  labels.reserve(near.size());
  for (int i : near) {
    points.push_back(index_.points()[static_cast<size_t>(i)]);
    labels.push_back(labels_[static_cast<size_t>(i)]);
  }
  const Pose vehicle_from_world = world_from_vehicle.Inverse();
  const double vehicle_yaw = YawOf(world_from_vehicle.rotation);
  std::vector<OrientedBox> boxes;
  boxes.reserve(boxes_world.size());
  for (const ObjectBox& b : boxes_world) {
    OrientedBox box;
    box.center = vehicle_from_world * b.center;
    box.size = b.size;
    box.yaw = b.yaw - vehicle_yaw;
    box.label = b.label;
    boxes.push_back(box);
  }
  return BuildOccupancy(config_, world_from_vehicle, points, labels, boxes,
                        vehicle_from_lidar_.translation,
                        TransformPoints(sweep_lidar, vehicle_from_lidar_),
                        cameras_);
}

}  // namespace gr
