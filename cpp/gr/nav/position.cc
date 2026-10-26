#include "gr/nav/position.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "gr/core/geometry.h"

namespace gr {
namespace {

// Nearest point of a polyline (the first segment of equal distance, as
// shapely's distance / project): distance and arc length.
struct LineHit {
  double distance = std::numeric_limits<double>::infinity();
  double along = 0.0;
};

LineHit Project(const std::vector<Vec2>& line, const Vec2& p) {
  LineHit best;
  double start = 0.0;
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    const Vec2 d = line[k + 1] - line[k];
    const double length = d.norm();
    const double t =
        length > 0.0
            ? std::clamp((p - line[k]).dot(d) / (length * length), 0.0, 1.0)
            : 0.0;
    const double distance = (p - (line[k] + t * d)).norm();
    if (distance < best.distance) {
      best.distance = distance;
      best.along = start + t * length;
    }
    start += length;
  }
  if (line.size() == 1) best = {(p - line[0]).norm(), 0.0};
  return best;
}

double Length(const std::vector<Vec2>& line) {
  double total = 0.0;
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    total += (line[k + 1] - line[k]).norm();
  }
  return total;
}

// The point `s` along the polyline, clamped to its ends.
Vec2 Interpolate(const std::vector<Vec2>& line, double s) {
  if (s <= 0.0) return line.front();
  double start = 0.0;
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    const double length = (line[k + 1] - line[k]).norm();
    if (start + length >= s && length > 0.0) {
      return line[k] + (line[k + 1] - line[k]) * ((s - start) / length);
    }
    start += length;
  }
  return line.back();
}

// numpy-style wrap to [-pi, pi).
double Wrap(double angle) {
  const double two_pi = 2.0 * kPi;
  double r = std::fmod(angle + kPi, two_pi);
  if (r < 0.0) r += two_pi;
  return r - kPi;
}

}  // namespace

double FloorHeights::At(const Vec2& xy) const {
  const int col = static_cast<int>(std::floor((xy.x() - x0) / res));
  const int row = static_cast<int>(std::floor((xy.y() - y0) / res));
  if (col < 0 || row < 0 || col >= width || row >= height) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return z[static_cast<size_t>(row) * width + col];
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
  for (size_t k = 0; k < edges.size(); ++k) {
    const NavEdge& edge = edges[k];
    if (edge.level != *out.level || edge.kind == "connector" ||
        edge.points.empty()) {
      continue;
    }
    const LineHit hit = Project(edge.points, xy);
    if (out.edge && hit.distance >= out.edge->distance) continue;
    const double length = Length(edge.points);
    const Vec2 ahead = Interpolate(edge.points, std::min(hit.along + 0.5,
                                                         length));
    const Vec2 behind =
        Interpolate(edge.points, std::max(hit.along - 0.5, 0.0));
    const Vec2 tangent = ahead - behind;
    const Vec2 offset = xy - Interpolate(edge.points, hit.along);
    EdgePosition position;
    position.edge = static_cast<int>(k);
    position.kind = edge.kind;
    position.distance = hit.distance;
    position.along = hit.along;
    position.lateral = (tangent.x() * offset.y() - tangent.y() * offset.x()) /
                       std::max(tangent.norm(), 1e-9);
    position.heading_rel_deg =
        RadToDeg(Wrap(yaw - std::atan2(tangent.y(), tangent.x())));
    out.edge = position;
  }
  for (const NavSlot& slot : slots) {
    if (slot.level == *out.level && slot.corners.size() >= 3 &&
        PointInPolygon(slot.corners, xy)) {
      out.slot = slot.slot_id;
      break;
    }
  }
  return out;
}

}  // namespace gr
