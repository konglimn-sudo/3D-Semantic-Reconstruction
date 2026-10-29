#include "gr/semantics/occupancy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "gr/core/check.h"
#include "gr/mapping/raycast.h"

namespace gr {
namespace {

// Every grid axis has at least this many voxels.
constexpr int kMinVoxelsPerAxis = 1;
// A voxel's centre, in voxels from its corner.
constexpr double kVoxelCentre = 0.5;
// The vote buffer is reserved for one vote per kPointsPerVoxelEstimate
// points plus kVoteReserveSlack.
constexpr size_t kPointsPerVoxelEstimate = 8;
constexpr size_t kVoteReserveSlack = 16;
// Label of static points beyond the label list.
constexpr int kMissingLabel = 0;
// Labels must fit the uint8 grid.
constexpr int kMaxLabel = std::numeric_limits<uint8_t>::max();
// A box has 2^3 corners; bounds start this far out (m).
constexpr int kBoxCorners = 8;
constexpr double kUnboundedCoordinate = 1e18;
// LiDAR rays shorter than this (voxels) have no direction.
constexpr double kMinRayLength = 1e-9;
// Camera rays go through pixel centres, every `camera_pixel_stride` (at least
// kMinPixelStride) pixels.
constexpr int kMinPixelStride = 1;
constexpr double kPixelCentre = 0.5;
// Static points are fetched this far (m) beyond the grid's xy corners.
constexpr double kStaticSearchMargin = 1.0;

// Voxel grid geometry of an OccupancyConfig.
class Grid {
 public:
  explicit Grid(const OccupancyConfig& config)
      : min_bound_(config.min_bound),
        voxel_(config.voxel),
        inverse_voxel_(1.0 / voxel_) {
    const Vec3 extent = (config.max_bound - config.min_bound) * inverse_voxel_;
    nx_ = std::max(kMinVoxelsPerAxis,
                   static_cast<int>(std::round(extent.x())));
    ny_ = std::max(kMinVoxelsPerAxis,
                   static_cast<int>(std::round(extent.y())));
    nz_ = std::max(kMinVoxelsPerAxis,
                   static_cast<int>(std::round(extent.z())));
  }

  int nx() const { return nx_; }
  int ny() const { return ny_; }
  int nz() const { return nz_; }
  Eigen::Vector3i dims() const { return Eigen::Vector3i(nx_, ny_, nz_); }
  size_t size() const {
    return static_cast<size_t>(nx_) * static_cast<size_t>(ny_) *
           static_cast<size_t>(nz_);
  }
  size_t Linear(int x, int y, int z) const {
    return (static_cast<size_t>(x) * static_cast<size_t>(ny_) +
            static_cast<size_t>(y)) *
               static_cast<size_t>(nz_) +
           static_cast<size_t>(z);
  }
  // Continuous grid coordinates of an ego-frame point.
  Vec3 ToGrid(const Vec3& point) const {
    return (point - min_bound_) * inverse_voxel_;
  }
  Vec3 VoxelCenter(int x, int y, int z) const {
    return min_bound_ +
           (Vec3(x, y, z) + Vec3::Constant(kVoxelCentre)) * voxel_;
  }
  std::optional<Eigen::Vector3i> IndexOf(const Vec3& point) const {
    const Vec3 grid_point = ToGrid(point);
    const Eigen::Vector3i index(static_cast<int>(std::floor(grid_point.x())),
                                static_cast<int>(std::floor(grid_point.y())),
                                static_cast<int>(std::floor(grid_point.z())));
    if (index.x() < 0 || index.y() < 0 || index.z() < 0 ||
        index.x() >= nx_ || index.y() >= ny_ || index.z() >= nz_) {
      return std::nullopt;
    }
    return index;
  }

 private:
  Vec3 min_bound_;
  double voxel_;
  double inverse_voxel_;
  int nx_;
  int ny_;
  int nz_;
};

// Majority label of the static points per voxel (ties: lowest label).
void VoteStaticLabels(const Grid& grid, const OccupancyConfig& config,
                      const Pose& ego_from_world, const Points& static_world,
                      const std::vector<int>& static_labels,
                      std::vector<uint8_t>& semantics) {
  // (voxel, label) per point.
  std::vector<std::pair<uint32_t, uint8_t>> votes;
  votes.reserve(static_world.size() / kPointsPerVoxelEstimate +
                kVoteReserveSlack);
  for (size_t point = 0; point < static_world.size(); ++point) {
    const int label =
        point < static_labels.size() ? static_labels[point] : kMissingLabel;
    if (label == config.free_label || label < 0 || label > kMaxLabel) {
      continue;
    }
    const auto index = grid.IndexOf(ego_from_world * static_world[point]);
    if (!index) continue;
    votes.emplace_back(
        static_cast<uint32_t>(grid.Linear(index->x(), index->y(), index->z())),
        static_cast<uint8_t>(label));
  }
  std::sort(votes.begin(), votes.end());
  for (size_t voxel_start = 0; voxel_start < votes.size();) {
    const uint32_t voxel = votes[voxel_start].first;
    size_t label_start = voxel_start;
    uint8_t best_label = votes[voxel_start].second;
    size_t best_count = 0;
    while (label_start < votes.size() && votes[label_start].first == voxel) {
      size_t label_end = label_start;
      while (label_end < votes.size() && votes[label_end].first == voxel &&
             votes[label_end].second == votes[label_start].second) {
        ++label_end;
      }
      if (label_end - label_start > best_count) {
        best_count = label_end - label_start;
        best_label = votes[label_start].second;
      }
      label_start = label_end;
    }
    semantics[voxel] = best_label;
    voxel_start = label_start;
  }
}

// Object boxes override the static labels inside them.
void RasterizeBoxes(const Grid& grid, const std::vector<OrientedBox>& boxes,
                    std::vector<uint8_t>& semantics) {
  for (const OrientedBox& box : boxes) {
    const Mat3 rotation =
        Eigen::AngleAxisd(box.yaw, Vec3::UnitZ()).toRotationMatrix();
    const Vec3 half = 0.5 * box.size;
    Vec3 lower = Vec3::Constant(kUnboundedCoordinate);
    Vec3 upper = Vec3::Constant(-kUnboundedCoordinate);
    for (int corner_index = 0; corner_index < kBoxCorners; ++corner_index) {
      // Bit `axis` of the corner index: the positive side along that axis.
      Vec3 corner;
      for (int axis = 0; axis < kSpaceDim; ++axis) {
        corner[axis] =
            corner_index & (1 << axis) ? half[axis] : -half[axis];
      }
      const Vec3 world = box.center + rotation * corner;
      lower = lower.cwiseMin(world);
      upper = upper.cwiseMax(world);
    }
    const Vec3 grid_lower = grid.ToGrid(lower);
    const Vec3 grid_upper = grid.ToGrid(upper);
    const auto first = [](double coordinate) {
      return std::max(0, static_cast<int>(std::floor(coordinate)));
    };
    const auto last = [](double coordinate, int count) {
      return std::min(count - 1, static_cast<int>(std::floor(coordinate)));
    };
    for (int x = first(grid_lower.x()); x <= last(grid_upper.x(), grid.nx());
         ++x) {
      for (int y = first(grid_lower.y());
           y <= last(grid_upper.y(), grid.ny()); ++y) {
        for (int z = first(grid_lower.z());
             z <= last(grid_upper.z(), grid.nz()); ++z) {
          const Vec3 local =
              rotation.transpose() * (grid.VoxelCenter(x, y, z) - box.center);
          if (std::abs(local.x()) <= half.x() &&
              std::abs(local.y()) <= half.y() &&
              std::abs(local.z()) <= half.z()) {
            semantics[grid.Linear(x, y, z)] = static_cast<uint8_t>(box.label);
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
  for (const Vec3& point : scan_ego) {
    const Vec3 delta = grid.ToGrid(point) - origin;
    const double length = delta.norm();
    if (length < kMinRayLength) continue;
    TraverseGrid(origin, delta / length, length, grid.dims(),
                 [&](int x, int y, int z) {
                   mask[grid.Linear(x, y, z)] = 1;
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
  const int stride = std::max(kMinPixelStride, config.camera_pixel_stride);
  // The first ray of each stride goes through its middle pixel.
  const int first_pixel = stride / 2;
  for (const CameraView& view : cameras) {
    if (!view.camera) continue;
    const Vec3 center = grid.ToGrid(view.ego_from_camera.translation);
    for (int v = first_pixel; v < view.camera->height(); v += stride) {
      for (int u = first_pixel; u < view.camera->width(); u += stride) {
        const std::optional<Vec3> ray =
            view.camera->Unproject(Vec2(u + kPixelCentre, v + kPixelCentre));
        if (!ray) continue;
        const Vec3 direction =
            (view.ego_from_camera.rotation * *ray).normalized();
        TraverseGrid(center, direction, t_end, grid.dims(),
                     [&](int x, int y, int z) {
                       const size_t index = grid.Linear(x, y, z);
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
                            std::hypot(extent_x, extent_y) +
                                kStaticSearchMargin);
  Points points;
  std::vector<int> labels;
  points.reserve(near.size());
  labels.reserve(near.size());
  for (int point : near) {
    points.push_back(index_.points()[static_cast<size_t>(point)]);
    labels.push_back(labels_[static_cast<size_t>(point)]);
  }
  const Pose vehicle_from_world = world_from_vehicle.Inverse();
  const double vehicle_yaw = YawOf(world_from_vehicle.rotation);
  std::vector<OrientedBox> boxes;
  boxes.reserve(boxes_world.size());
  for (const ObjectBox& world_box : boxes_world) {
    OrientedBox box;
    box.center = vehicle_from_world * world_box.center;
    box.size = world_box.size;
    box.yaw = world_box.yaw - vehicle_yaw;
    box.label = world_box.label;
    boxes.push_back(box);
  }
  return BuildOccupancy(config_, world_from_vehicle, points, labels, boxes,
                        vehicle_from_lidar_.translation,
                        TransformPoints(sweep_lidar, vehicle_from_lidar_),
                        cameras_);
}

}  // namespace gr
