#include "gr/core/geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>

#include "Eigen/Eigenvalues"

#include "gr/core/voxel.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kQuarterTurn = kPi / 2.0;

// Three points span a plane: fewer give no PCA normal or RANSAC sample.
constexpr int kMinPlanePoints = 3;
// A polygon has at least three vertices.
constexpr size_t kMinPolygonVertices = 3;

// Principal components of a point set.
struct Pca {
  Vec3 normal = Vec3::Zero();  // Along the smallest eigenvalue.
  Vec3 centroid = Vec3::Zero();
  double largest_eigenvalue = 0.0;
  double middle_eigenvalue = 0.0;
  double smallest_eigenvalue = 0.0;
  bool valid = false;
};

// (l2 - l3) / l1 of the eigenvalues l1 >= l2 >= l3: near 1 on a plane.
double Planarity(const Pca& pca) {
  return (pca.middle_eigenvalue - pca.smallest_eigenvalue) /
         pca.largest_eigenvalue;
}

// PCA of the `count` points point_at(0 .. count - 1).
template <typename PointAt>
Pca ComputePca(int count, PointAt point_at) {
  // Positions of SelfAdjointEigenSolver's eigenvalues (ascending).
  constexpr int kSmallest = 0;
  constexpr int kMiddle = 1;
  constexpr int kLargest = 2;
  Pca out;
  if (count < kMinPlanePoints) return out;
  Vec3 mean = Vec3::Zero();
  for (int k = 0; k < count; ++k) mean += point_at(k);
  mean /= count;
  Mat3 covariance = Mat3::Zero();
  for (int k = 0; k < count; ++k) {
    const Vec3 offset = point_at(k) - mean;
    covariance.noalias() += offset * offset.transpose();
  }
  covariance /= count;
  Eigen::SelfAdjointEigenSolver<Mat3> solver(covariance);
  if (solver.info() != Eigen::Success) return out;
  const Vec3 eigenvalues = solver.eigenvalues();
  out.smallest_eigenvalue = std::max(eigenvalues[kSmallest], 0.0);
  out.middle_eigenvalue = std::max(eigenvalues[kMiddle], 0.0);
  out.largest_eigenvalue = std::max(eigenvalues[kLargest], 0.0);
  out.normal = solver.eigenvectors().col(kSmallest).normalized();
  out.centroid = mean;
  out.valid = out.largest_eigenvalue > 0.0;
  return out;
}

Pca PcaOfIndices(const Points& points, const std::vector<int>& indices) {
  return ComputePca(static_cast<int>(indices.size()), [&](int position) {
    return points[static_cast<size_t>(indices[static_cast<size_t>(position)])];
  });
}

using VoxelGrid = std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>;

std::vector<Vec2> CounterClockwise(std::vector<Vec2> polygon) {
  if (PolygonArea(polygon) < 0.0) std::reverse(polygon.begin(), polygon.end());
  return polygon;
}

// Keeps the part of `polygon` left of the directed line
// line_start -> line_end shifted left by `shift`.
std::vector<Vec2> ClipHalfPlane(const std::vector<Vec2>& polygon,
                                const Vec2& line_start, const Vec2& line_end,
                                double shift) {
  // Shorter lines have no direction: nothing is clipped.
  constexpr double kMinLineLength = 1e-12;
  const Vec2 direction = line_end - line_start;
  const double length = direction.norm();
  if (length < kMinLineLength) return polygon;
  const Vec2 left(-direction.y() / length, direction.x() / length);
  const auto signed_distance = [&](const Vec2& point) {
    return (point - line_start).dot(left) - shift;
  };
  std::vector<Vec2> out;
  const size_t vertex_count = polygon.size();
  for (size_t i = 0; i < vertex_count; ++i) {
    const Vec2& vertex = polygon[i];
    const Vec2& next = polygon[(i + 1) % vertex_count];
    const double distance = signed_distance(vertex);
    const double next_distance = signed_distance(next);
    if (distance >= 0.0) out.push_back(vertex);
    if ((distance >= 0.0) != (next_distance >= 0.0)) {
      out.push_back(vertex +
                    (next - vertex) * (distance / (distance - next_distance)));
    }
  }
  return out;
}

}  // namespace

NormalsResult EstimateNormals(const Points& points, double radius,
                              int min_neighbors, int max_neighbors) {
  // Voxels as large as the radius: the neighbours of a point lie in the
  // voxels at most this many steps from its own.
  constexpr int kVoxelReach = 1;
  // Room in the grid for a voxel per four points; more voxels grow it.
  constexpr size_t kPointsPerVoxelGuess = 4;
  NormalsResult result;
  const size_t point_count = points.size();
  result.normals.assign(point_count, Vec3::Zero());
  result.planarity.assign(point_count, 0.0);
  result.thickness.assign(point_count, 0.0);
  result.neighbors.assign(point_count, 0);
  result.centroids.assign(point_count, Vec3::Zero());
  if (point_count == 0) return result;
  const double inverse_radius = 1.0 / radius;
  const double squared_radius = radius * radius;
  VoxelGrid grid;
  grid.reserve(point_count / kPointsPerVoxelGuess + 1);
  for (size_t i = 0; i < point_count; ++i) {
    grid[VoxelOf(points[i], inverse_radius)].push_back(static_cast<int>(i));
  }

#ifdef _OPENMP
  constexpr int kChunkSize = 256;  // Points per OpenMP work chunk.
#pragma omp parallel for schedule(dynamic, kChunkSize)
#endif
  for (int64_t signed_index = 0;
       signed_index < static_cast<int64_t>(point_count); ++signed_index) {
    const size_t point_index = static_cast<size_t>(signed_index);
    const VoxelKey center = VoxelOf(points[point_index], inverse_radius);
    // (squared distance, index) of the points within the radius.
    std::vector<std::pair<double, int>> candidates;
    for (int dx = -kVoxelReach; dx <= kVoxelReach; ++dx) {
      for (int dy = -kVoxelReach; dy <= kVoxelReach; ++dy) {
        for (int dz = -kVoxelReach; dz <= kVoxelReach; ++dz) {
          const auto voxel =
              grid.find({center.x + dx, center.y + dy, center.z + dz});
          if (voxel == grid.end()) continue;
          for (int candidate : voxel->second) {
            const double squared_distance =
                (points[static_cast<size_t>(candidate)] - points[point_index])
                    .squaredNorm();
            if (squared_distance <= squared_radius) {
              candidates.emplace_back(squared_distance, candidate);
            }
          }
        }
      }
    }
    // Keep the closest `max_neighbors` so that dense clouds stay unbiased.
    if (static_cast<int>(candidates.size()) > max_neighbors) {
      std::nth_element(candidates.begin(), candidates.begin() + max_neighbors,
                       candidates.end());
      candidates.resize(static_cast<size_t>(max_neighbors));
    }
    std::vector<int> neighbors;
    neighbors.reserve(candidates.size());
    for (const auto& candidate : candidates) {
      neighbors.push_back(candidate.second);
    }
    result.neighbors[point_index] = static_cast<int>(neighbors.size());
    if (static_cast<int>(neighbors.size()) < min_neighbors) continue;
    const Pca pca = PcaOfIndices(points, neighbors);
    if (!pca.valid) continue;
    result.normals[point_index] = pca.normal;
    result.centroids[point_index] = pca.centroid;
    result.planarity[point_index] = Planarity(pca);
    result.thickness[point_index] = std::sqrt(pca.smallest_eigenvalue);
  }
  return result;
}

Points ProjectToLocalPlanes(const Points& points, double radius,
                            int min_neighbors, int max_neighbors,
                            double min_planarity, double max_shift,
                            std::vector<double>* shift) {
  const NormalsResult local =
      EstimateNormals(points, radius, min_neighbors, max_neighbors);
  Points out = points;
  if (shift != nullptr) shift->assign(points.size(), 0.0);
  for (size_t i = 0; i < points.size(); ++i) {
    if (local.normals[i].isZero() || local.planarity[i] < min_planarity) {
      continue;
    }
    const double offset = local.normals[i].dot(points[i] - local.centroids[i]);
    if (std::abs(offset) > max_shift) continue;
    out[i] = points[i] - offset * local.normals[i];
    if (shift != nullptr) (*shift)[i] = offset;
  }
  return out;
}

PlaneFit FitPlaneRansac(const Points& points, double threshold, int iterations,
                        int seed) {
  // Samples with a shorter cross product are collinear: no plane.
  constexpr double kMinSampleCross = 1e-9;
  PlaneFit best;
  const int point_count = static_cast<int>(points.size());
  if (point_count < kMinPlanePoints) return best;
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> pick(0, point_count - 1);
  int best_inlier_count = -1;  // Below any count: the first plane wins.
  Vec3 best_normal = Vec3::UnitZ();
  double best_offset = 0.0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const Vec3& first = points[static_cast<size_t>(pick(rng))];
    const Vec3& second = points[static_cast<size_t>(pick(rng))];
    const Vec3& third = points[static_cast<size_t>(pick(rng))];
    Vec3 normal = (second - first).cross(third - first);
    const double norm = normal.norm();
    if (norm < kMinSampleCross) continue;
    normal /= norm;
    const double offset = -normal.dot(first);
    int inlier_count = 0;
    for (const Vec3& point : points) {
      if (std::abs(normal.dot(point) + offset) < threshold) ++inlier_count;
    }
    if (inlier_count > best_inlier_count) {
      best_inlier_count = inlier_count;
      best_normal = normal;
      best_offset = offset;
    }
  }
  const auto inliers_of = [&](const Vec3& normal, double offset) {
    std::vector<int> inliers;
    for (int i = 0; i < point_count; ++i) {
      if (std::abs(normal.dot(points[static_cast<size_t>(i)]) + offset) <
          threshold) {
        inliers.push_back(i);
      }
    }
    return inliers;
  };
  // Least-squares refinement on the inliers.
  std::vector<int> inliers = inliers_of(best_normal, best_offset);
  const Pca pca = PcaOfIndices(points, inliers);
  if (pca.valid) {
    Vec3 centroid = Vec3::Zero();
    for (int inlier : inliers) centroid += points[static_cast<size_t>(inlier)];
    centroid /= static_cast<double>(inliers.size());
    best_normal = pca.normal.dot(best_normal) < 0 ? -pca.normal : pca.normal;
    best_offset = -best_normal.dot(centroid);
    inliers = inliers_of(best_normal, best_offset);
  }
  best.normal = best_normal;
  best.offset = best_offset;
  best.inliers = std::move(inliers);
  return best;
}

ThicknessStats MeasurePlaneThickness(const Points& points, double voxel_size,
                                     int min_points, double min_planarity) {
  // The 90th percentile is the value this far along the sorted values.
  constexpr double kP90Fraction = 0.9;
  ThicknessStats stats;
  const double inverse_voxel_size = 1.0 / voxel_size;
  VoxelGrid grid;
  for (size_t i = 0; i < points.size(); ++i) {
    grid[VoxelOf(points[i], inverse_voxel_size)].push_back(static_cast<int>(i));
  }
  std::vector<double> thickness;
  for (const auto& [voxel, indices] : grid) {
    if (static_cast<int>(indices.size()) < min_points) continue;
    const Pca pca = PcaOfIndices(points, indices);
    if (!pca.valid) continue;
    if (Planarity(pca) < min_planarity) continue;
    thickness.push_back(std::sqrt(pca.smallest_eigenvalue));
  }
  stats.num_voxels = static_cast<int>(thickness.size());
  if (thickness.empty()) return stats;
  std::sort(thickness.begin(), thickness.end());
  double sum = 0.0;
  for (double value : thickness) sum += value;
  stats.mean = sum / static_cast<double>(thickness.size());
  stats.median = thickness[thickness.size() / 2];
  stats.p90 =
      thickness[std::min(thickness.size() - 1,
                         static_cast<size_t>(kP90Fraction * thickness.size()))];
  return stats;
}

std::vector<Vec2> ConvexHull2D(std::vector<Vec2> points) {
  // Andrew's monotone chain.
  std::sort(points.begin(), points.end(), [](const Vec2& a, const Vec2& b) {
    return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
  });
  points.erase(std::unique(points.begin(), points.end()), points.end());
  if (points.size() < kMinPolygonVertices) return points;
  // Positive when origin -> a -> b turns counter-clockwise.
  const auto turn = [](const Vec2& origin, const Vec2& a, const Vec2& b) {
    return (a.x() - origin.x()) * (b.y() - origin.y()) -
           (a.y() - origin.y()) * (b.x() - origin.x());
  };
  std::vector<Vec2> hull(2 * points.size());
  size_t hull_size = 0;
  for (const Vec2& point : points) {  // Lower hull.
    while (hull_size >= 2 &&
           turn(hull[hull_size - 2], hull[hull_size - 1], point) <= 0.0) {
      --hull_size;
    }
    hull[hull_size++] = point;
  }
  const size_t upper_hull_start = hull_size + 1;
  for (size_t i = points.size() - 1; i-- > 0;) {  // Upper hull.
    const Vec2& point = points[i];
    while (hull_size >= upper_hull_start &&
           turn(hull[hull_size - 2], hull[hull_size - 1], point) <= 0.0) {
      --hull_size;
    }
    hull[hull_size++] = point;
  }
  hull.resize(hull_size - 1);  // The last point is the first one.
  return hull;
}

RotatedRect MinAreaRect(const std::vector<Vec2>& points) {
  RotatedRect rect;
  const std::vector<Vec2> hull = ConvexHull2D(points);
  if (hull.empty()) return rect;
  if (hull.size() == 1) {
    rect.center = hull[0];
    return rect;
  }
  // The best rectangle so far: its side directions and its extent along
  // them.
  double best_area = kInf;
  Vec2 best_along = Vec2::UnitX();
  double best_min_along = 0.0;
  double best_max_along = 0.0;
  double best_min_across = 0.0;
  double best_max_across = 0.0;
  const size_t vertex_count = hull.size();
  // Two points span one edge; a polygon has an edge per vertex.
  const size_t edge_count = vertex_count == 2 ? 1 : vertex_count;
  for (size_t edge = 0; edge < edge_count; ++edge) {
    Vec2 along = hull[(edge + 1) % vertex_count] - hull[edge];
    if (along.norm() <= 0.0) continue;
    along.normalize();
    const Vec2 across(-along.y(), along.x());
    double min_along = kInf;
    double max_along = -kInf;
    double min_across = kInf;
    double max_across = -kInf;
    for (const Vec2& vertex : hull) {
      min_along = std::min(min_along, vertex.dot(along));
      max_along = std::max(max_along, vertex.dot(along));
      min_across = std::min(min_across, vertex.dot(across));
      max_across = std::max(max_across, vertex.dot(across));
    }
    const double area = (max_along - min_along) * (max_across - min_across);
    if (area < best_area) {
      best_area = area;
      best_along = along;
      best_min_along = min_along;
      best_max_along = max_along;
      best_min_across = min_across;
      best_max_across = max_across;
    }
  }
  const Vec2 best_across(-best_along.y(), best_along.x());
  rect.center = best_along * (0.5 * (best_min_along + best_max_along)) +
                best_across * (0.5 * (best_min_across + best_max_across));
  Vec2 long_side = best_along;
  rect.length = best_max_along - best_min_along;
  rect.width = best_max_across - best_min_across;
  if (rect.width > rect.length) {
    std::swap(rect.length, rect.width);
    long_side = best_across;
  }
  rect.yaw = std::atan2(long_side.y(), long_side.x());
  if (rect.yaw <= -kQuarterTurn) rect.yaw += kPi;
  if (rect.yaw > kQuarterTurn) rect.yaw -= kPi;
  return rect;
}

bool PointInPolygon(const std::vector<Vec2>& polygon, const Vec2& point) {
  bool inside = false;
  const size_t vertex_count = polygon.size();
  for (size_t current = 0, previous = vertex_count - 1; current < vertex_count;
       previous = current++) {
    const Vec2& vertex = polygon[current];
    const Vec2& previous_vertex = polygon[previous];
    // Toggle at every edge that crosses the horizontal ray to the right.
    if ((vertex.y() > point.y()) != (previous_vertex.y() > point.y()) &&
        point.x() < (previous_vertex.x() - vertex.x()) *
                            (point.y() - vertex.y()) /
                            (previous_vertex.y() - vertex.y()) +
                        vertex.x()) {
      inside = !inside;
    }
  }
  return inside;
}

double PolygonArea(const std::vector<Vec2>& polygon) {
  // The shoelace formula.
  double twice_area = 0.0;
  const size_t vertex_count = polygon.size();
  for (size_t i = 0; i < vertex_count; ++i) {
    const Vec2& vertex = polygon[i];
    const Vec2& next = polygon[(i + 1) % vertex_count];
    twice_area += vertex.x() * next.y() - next.x() * vertex.y();
  }
  return 0.5 * twice_area;
}

std::vector<Vec2> ClipConvex(const std::vector<Vec2>& subject,
                             const std::vector<Vec2>& clip) {
  std::vector<Vec2> out = CounterClockwise(subject);
  const std::vector<Vec2> clip_ccw = CounterClockwise(clip);
  for (size_t i = 0; i < clip_ccw.size() && !out.empty(); ++i) {
    out = ClipHalfPlane(out, clip_ccw[i], clip_ccw[(i + 1) % clip_ccw.size()],
                        /*shift=*/0.0);
  }
  return out;
}

double ConvexIntersectionArea(const std::vector<Vec2>& a,
                              const std::vector<Vec2>& b) {
  const std::vector<Vec2> clipped = ClipConvex(a, b);
  return clipped.size() < kMinPolygonVertices ? 0.0
                                              : std::abs(PolygonArea(clipped));
}

std::vector<Vec2> InsetConvex(const std::vector<Vec2>& polygon,
                              double distance) {
  // Insets with a smaller area count as empty.
  constexpr double kMinInsetArea = 1e-12;
  const std::vector<Vec2> ccw = CounterClockwise(polygon);
  std::vector<Vec2> out = ccw;
  for (size_t i = 0; i < ccw.size() && !out.empty(); ++i) {
    out = ClipHalfPlane(out, ccw[i], ccw[(i + 1) % ccw.size()], distance);
  }
  if (out.size() < kMinPolygonVertices ||
      std::abs(PolygonArea(out)) < kMinInsetArea) {
    return {};
  }
  return out;
}

}  // namespace gr
