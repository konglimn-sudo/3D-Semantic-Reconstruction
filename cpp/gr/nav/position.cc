#include "gr/nav/position.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gr/core/geometry.h"

namespace gr {
namespace {

constexpr double kFullTurn = 2.0 * kPi;
// The edge direction at a point is the chord over this far (m) either side.
constexpr double kTangentHalfSpan = 0.5;
// Tangents shorter than this (m) have no direction.
constexpr double kMinTangent = 1e-9;
// Slots are polygons of at least this many corners.
constexpr size_t kMinPolygonCorners = 3;

// Nearest point of a polyline (the first segment of equal distance, as
// shapely's distance / project): distance and arc length.
struct LineHit {
  double distance = std::numeric_limits<double>::infinity();
  double along = 0.0;
};

LineHit Project(const std::vector<Vec2>& line, const Vec2& point) {
  LineHit best;
  double start = 0.0;  // Arc length at the segment start.
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    const Vec2 segment = line[k + 1] - line[k];
    const double length = segment.norm();
    const double fraction =
        length > 0.0 ? std::clamp((point - line[k]).dot(segment) /
                                      (length * length),
                                  0.0, 1.0)
                     : 0.0;
    const double distance = (point - (line[k] + fraction * segment)).norm();
    if (distance < best.distance) {
      best.distance = distance;
      best.along = start + fraction * length;
    }
    start += length;
  }
  if (line.size() == 1) best = {(point - line[0]).norm(), 0.0};
  return best;
}

double Length(const std::vector<Vec2>& line) {
  double total = 0.0;
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    total += (line[k + 1] - line[k]).norm();
  }
  return total;
}

// The point at arc length `along` on the polyline, clamped to its ends.
Vec2 Interpolate(const std::vector<Vec2>& line, double along) {
  if (along <= 0.0) return line.front();
  double start = 0.0;
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    const double length = (line[k + 1] - line[k]).norm();
    if (start + length >= along && length > 0.0) {
      return line[k] + (line[k + 1] - line[k]) * ((along - start) / length);
    }
    start += length;
  }
  return line.back();
}

// numpy-style wrap to [-pi, pi).
double Wrap(double angle) {
  double wrapped = std::fmod(angle + kPi, kFullTurn);
  if (wrapped < 0.0) wrapped += kFullTurn;
  return wrapped - kPi;
}

}  // namespace

double FloorHeights::At(const Vec2& xy) const {
  const int column = static_cast<int>(std::floor((xy.x() - x0) / res));
  const int row = static_cast<int>(std::floor((xy.y() - y0) / res));
  if (column < 0 || row < 0 || column >= width || row >= height) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return z[static_cast<size_t>(row) * width + column];
}

NavPosition Where(const Pose& map_from_vehicle,
                  const std::vector<NavEdge>& edges,
                  const std::vector<NavSlot>& slots,
                  const std::vector<FloorHeights>& floors) {
  NavPosition out;
  out.xyz = map_from_vehicle.translation;
  const Vec2 xy = out.xyz.head<2>();
  const double yaw = std::atan2(map_from_vehicle.rotation(1, 0),
                                map_from_vehicle.rotation(0, 0));
  out.yaw_deg = RadToDeg(yaw);
  double best_gap = std::numeric_limits<double>::infinity();
  for (const FloorHeights& floor : floors) {
    const double z = floor.At(xy);
    if (!std::isfinite(z)) continue;
    const double gap = std::abs(out.xyz.z() - z);
    if (gap < best_gap) {
      out.level = floor.level;
      best_gap = gap;
    }
  }
  if (!out.level) return out;
  for (size_t index = 0; index < edges.size(); ++index) {
    const NavEdge& edge = edges[index];
    if (edge.level != *out.level || edge.kind == "connector" ||
        edge.points.empty()) {
      continue;
    }
    const LineHit hit = Project(edge.points, xy);
    if (out.edge && hit.distance >= out.edge->distance) continue;
    const double length = Length(edge.points);
    const Vec2 ahead = Interpolate(
        edge.points, std::min(hit.along + kTangentHalfSpan, length));
    const Vec2 behind =
        Interpolate(edge.points, std::max(hit.along - kTangentHalfSpan, 0.0));
    const Vec2 tangent = ahead - behind;
    const Vec2 offset = xy - Interpolate(edge.points, hit.along);
    EdgePosition position;
    position.edge = static_cast<int>(index);
    position.kind = edge.kind;
    position.distance = hit.distance;
    position.along = hit.along;
    position.lateral = (tangent.x() * offset.y() - tangent.y() * offset.x()) /
                       std::max(tangent.norm(), kMinTangent);
    position.heading_rel_deg =
        RadToDeg(Wrap(yaw - std::atan2(tangent.y(), tangent.x())));
    out.edge = position;
  }
  for (const NavSlot& slot : slots) {
    if (slot.level == *out.level && slot.corners.size() >= kMinPolygonCorners &&
        PointInPolygon(slot.corners, xy)) {
      out.slot = slot.slot_id;
      break;
    }
  }
  return out;
}

}  // namespace gr
