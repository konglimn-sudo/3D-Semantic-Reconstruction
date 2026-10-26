#include "gr/core/geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <utility>

#include "Eigen/Eigenvalues"

#include "gr/core/voxel.h"

namespace gr {
namespace {

struct Pca {
  Vec3 normal = Vec3::Zero();
  Vec3 centroid = Vec3::Zero();
  double l1 = 0.0;  // Eigenvalues, descending.
  double l2 = 0.0;
  double l3 = 0.0;
  bool ok = false;
};

// PCA of n points returned by get(0..n-1).
template <typename Getter>
Pca ComputePca(int n, Getter get) {
  Pca out;
  if (n < 3) return out;
  Vec3 mean = Vec3::Zero();
  for (int k = 0; k < n; ++k) mean += get(k);
  mean /= n;
  Mat3 covariance = Mat3::Zero();
  for (int k = 0; k < n; ++k) {
    const Vec3 d = get(k) - mean;
    covariance.noalias() += d * d.transpose();
  }
  covariance /= n;
  Eigen::SelfAdjointEigenSolver<Mat3> solver(covariance);
  if (solver.info() != Eigen::Success) return out;
  const Vec3 eigenvalues = solver.eigenvalues();  // Ascending.
  out.l3 = std::max(eigenvalues[0], 0.0);
  out.l2 = std::max(eigenvalues[1], 0.0);
  out.l1 = std::max(eigenvalues[2], 0.0);
  out.normal = solver.eigenvectors().col(0).normalized();
  out.centroid = mean;
  out.ok = out.l1 > 0.0;
  return out;
}

Pca PcaOfIndices(const Points& points, const std::vector<int>& indices) {
  return ComputePca(static_cast<int>(indices.size()), [&](int k) {
    return points[static_cast<size_t>(indices[static_cast<size_t>(k)])];
  });
}

using VoxelGrid = std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash>;

}  // namespace

NormalsResult EstimateNormals(const Points& points, double radius,
                              int min_neighbors, int max_neighbors) {
  NormalsResult result;
  const size_t n = points.size();
  result.normals.assign(n, Vec3::Zero());
  result.planarity.assign(n, 0.0);
  result.thickness.assign(n, 0.0);
  result.neighbors.assign(n, 0);
  result.centroids.assign(n, Vec3::Zero());
  if (n == 0) return result;
  const double inv = 1.0 / radius;
  const double r2 = radius * radius;
  VoxelGrid grid;
  grid.reserve(n / 4 + 1);
  for (size_t i = 0; i < n; ++i) {
    grid[VoxelOf(points[i], inv)].push_back(static_cast<int>(i));
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256)
#endif
  for (int64_t ii = 0; ii < static_cast<int64_t>(n); ++ii) {
    const size_t i = static_cast<size_t>(ii);
    const VoxelKey center = VoxelOf(points[i], inv);
    std::vector<std::pair<double, int>> candidates;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          const auto it =
              grid.find({center.x + dx, center.y + dy, center.z + dz});
          if (it == grid.end()) continue;
          for (int k : it->second) {
            const double d2 =
                (points[static_cast<size_t>(k)] - points[i]).squaredNorm();
            if (d2 <= r2) candidates.emplace_back(d2, k);
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
    for (const auto& [d2, k] : candidates) neighbors.push_back(k);
    result.neighbors[i] = static_cast<int>(neighbors.size());
    if (static_cast<int>(neighbors.size()) < min_neighbors) continue;
    const Pca pca = PcaOfIndices(points, neighbors);
    if (!pca.ok) continue;
    result.normals[i] = pca.normal;
    result.centroids[i] = pca.centroid;
    result.planarity[i] = (pca.l2 - pca.l3) / pca.l1;
    result.thickness[i] = std::sqrt(pca.l3);
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
    const double d = local.normals[i].dot(points[i] - local.centroids[i]);
    if (std::abs(d) > max_shift) continue;
    out[i] = points[i] - d * local.normals[i];
    if (shift != nullptr) (*shift)[i] = d;
  }
  return out;
}

PlaneFit FitPlaneRansac(const Points& points, double threshold, int iterations,
                        int seed) {
  PlaneFit best;
  const int n = static_cast<int>(points.size());
  if (n < 3) return best;
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_int_distribution<int> pick(0, n - 1);
  int best_count = -1;
  Vec3 best_normal = Vec3::UnitZ();
  double best_d = 0.0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    const Vec3& a = points[static_cast<size_t>(pick(rng))];
    const Vec3& b = points[static_cast<size_t>(pick(rng))];
    const Vec3& c = points[static_cast<size_t>(pick(rng))];
    Vec3 normal = (b - a).cross(c - a);
    const double norm = normal.norm();
    if (norm < 1e-9) continue;
    normal /= norm;
    const double d = -normal.dot(a);
    int count = 0;
    for (const Vec3& p : points) {
      if (std::abs(normal.dot(p) + d) < threshold) ++count;
    }
    if (count > best_count) {
      best_count = count;
      best_normal = normal;
      best_d = d;
    }
  }
  auto inliers_of = [&](const Vec3& normal, double d) {
    std::vector<int> inliers;
    for (int i = 0; i < n; ++i) {
      if (std::abs(normal.dot(points[static_cast<size_t>(i)]) + d) <
          threshold) {
        inliers.push_back(i);
      }
    }
    return inliers;
  };
  // Least-squares refinement on the inliers.
  std::vector<int> inliers = inliers_of(best_normal, best_d);
  const Pca pca = PcaOfIndices(points, inliers);
  if (pca.ok) {
    Vec3 centroid = Vec3::Zero();
    for (int i : inliers) centroid += points[static_cast<size_t>(i)];
    centroid /= static_cast<double>(inliers.size());
    best_normal = pca.normal.dot(best_normal) < 0 ? -pca.normal : pca.normal;
    best_d = -best_normal.dot(centroid);
    inliers = inliers_of(best_normal, best_d);
  }
  best.normal = best_normal;
  best.d = best_d;
  best.inliers = std::move(inliers);
  return best;
}

ThicknessStats MeasurePlaneThickness(const Points& points, double voxel,
                                     int min_points, double min_planarity) {
  ThicknessStats stats;
  const double inv = 1.0 / voxel;
  VoxelGrid grid;
  for (size_t i = 0; i < points.size(); ++i) {
    grid[VoxelOf(points[i], inv)].push_back(static_cast<int>(i));
  }
  std::vector<double> thickness;
  for (const auto& [key, indices] : grid) {
    if (static_cast<int>(indices.size()) < min_points) continue;
    const Pca pca = PcaOfIndices(points, indices);
    if (!pca.ok) continue;
    if ((pca.l2 - pca.l3) / pca.l1 < min_planarity) continue;
    thickness.push_back(std::sqrt(pca.l3));
  }
  stats.num_voxels = static_cast<int>(thickness.size());
  if (thickness.empty()) return stats;
  std::sort(thickness.begin(), thickness.end());
  double sum = 0.0;
  for (double v : thickness) sum += v;
  stats.mean = sum / static_cast<double>(thickness.size());
  stats.median = thickness[thickness.size() / 2];
  stats.p90 = thickness[std::min(thickness.size() - 1,
                                 static_cast<size_t>(0.9 * thickness.size()))];
  return stats;
}

std::vector<Vec2> ConvexHull2D(std::vector<Vec2> points) {
  std::sort(points.begin(), points.end(), [](const Vec2& a, const Vec2& b) {
    return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
  });
  points.erase(std::unique(points.begin(), points.end()), points.end());
  if (points.size() < 3) return points;
  auto cross = [](const Vec2& o, const Vec2& a, const Vec2& b) {
    return (a.x() - o.x()) * (b.y() - o.y()) -
           (a.y() - o.y()) * (b.x() - o.x());
  };
  std::vector<Vec2> hull(2 * points.size());
  size_t k = 0;
  for (const Vec2& p : points) {  // Lower hull.
    while (k >= 2 && cross(hull[k - 2], hull[k - 1], p) <= 0.0) --k;
    hull[k++] = p;
  }
  const size_t lower = k + 1;
  for (size_t i = points.size() - 1; i-- > 0;) {  // Upper hull.
    const Vec2& p = points[i];
    while (k >= lower && cross(hull[k - 2], hull[k - 1], p) <= 0.0) --k;
    hull[k++] = p;
  }
  hull.resize(k - 1);  // The last point is the first one.
  return hull;
}

bool PointInPolygon(const std::vector<Vec2>& polygon, const Vec2& point) {
  bool inside = false;
  const size_t n = polygon.size();
  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    const Vec2& a = polygon[i];
    const Vec2& b = polygon[j];
    if ((a.y() > point.y()) != (b.y() > point.y()) &&
        point.x() < (b.x() - a.x()) * (point.y() - a.y()) / (b.y() - a.y()) +
                        a.x()) {
      inside = !inside;
    }
  }
  return inside;
}

double PolygonArea(const std::vector<Vec2>& polygon) {
  double area = 0.0;
  const size_t n = polygon.size();
  for (size_t i = 0; i < n; ++i) {
    const Vec2& p = polygon[i];
    const Vec2& q = polygon[(i + 1) % n];
    area += p.x() * q.y() - q.x() * p.y();
  }
  return 0.5 * area;
}

namespace {

std::vector<Vec2> CounterClockwise(std::vector<Vec2> polygon) {
  if (PolygonArea(polygon) < 0.0) std::reverse(polygon.begin(), polygon.end());
  return polygon;
}

// Keeps the part of `polygon` left of the directed line a -> b shifted left
// by `shift`.
std::vector<Vec2> ClipHalfPlane(const std::vector<Vec2>& polygon,
                                const Vec2& a, const Vec2& b, double shift) {
  const Vec2 d = b - a;
  const double length = d.norm();
  if (length < 1e-12) return polygon;
  const Vec2 left(-d.y() / length, d.x() / length);
  auto side = [&](const Vec2& p) { return (p - a).dot(left) - shift; };
  std::vector<Vec2> out;
  const size_t n = polygon.size();
  for (size_t i = 0; i < n; ++i) {
    const Vec2& p = polygon[i];
    const Vec2& q = polygon[(i + 1) % n];
    const double sp = side(p);
    const double sq = side(q);
    if (sp >= 0.0) out.push_back(p);
    if ((sp >= 0.0) != (sq >= 0.0)) {
      out.push_back(p + (q - p) * (sp / (sp - sq)));
    }
  }
  return out;
}

}  // namespace

std::vector<Vec2> ClipConvex(const std::vector<Vec2>& subject,
                             const std::vector<Vec2>& clip) {
  std::vector<Vec2> out = CounterClockwise(subject);
  const std::vector<Vec2> c = CounterClockwise(clip);
  for (size_t i = 0; i < c.size() && !out.empty(); ++i) {
    out = ClipHalfPlane(out, c[i], c[(i + 1) % c.size()], 0.0);
  }
  return out;
}

double ConvexIntersectionArea(const std::vector<Vec2>& a,
                              const std::vector<Vec2>& b) {
  const std::vector<Vec2> clipped = ClipConvex(a, b);
  return clipped.size() < 3 ? 0.0 : std::abs(PolygonArea(clipped));
}

std::vector<Vec2> InsetConvex(const std::vector<Vec2>& polygon,
                              double distance) {
  const std::vector<Vec2> ccw = CounterClockwise(polygon);
  std::vector<Vec2> out = ccw;
  for (size_t i = 0; i < ccw.size() && !out.empty(); ++i) {
    out = ClipHalfPlane(out, ccw[i], ccw[(i + 1) % ccw.size()], distance);
  }
  if (out.size() < 3 || std::abs(PolygonArea(out)) < 1e-12) return {};
  return out;
}

}  // namespace gr
