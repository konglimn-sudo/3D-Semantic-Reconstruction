#include "gr/nav/slot_state.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include "gr/core/geometry.h"

namespace gr {
namespace {

// Slots farther than this (m) above or below the sensor are on another
// level.
constexpr double kMaxLevelOffset = 4.0;
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;
// A slot's inner outline is a polygon of at least this many corners.
constexpr size_t kMinPolygonCorners = 3;

}  // namespace

std::vector<SlotObservation> ObserveSlots(
    const Points& map_points, const Vec3& sensor,
    const std::vector<SlotGeometry>& slots, const SlotStateConfig& config) {
  struct Candidate {
    int slot;
    std::vector<Vec2> inner;
    Vec2 lo;
    Vec2 hi;
    int cells;  // Coverage cells inside the inner outline.
  };
  std::vector<Candidate> candidates;
  for (size_t index = 0; index < slots.size(); ++index) {
    const SlotGeometry& slot = slots[index];
    Vec2 center = Vec2::Zero();
    for (const Vec2& corner : slot.corners) {
      center += corner / static_cast<double>(kSlotCorners);
    }
    if ((center - sensor.head<2>()).norm() > config.max_range ||
        std::abs(sensor.z() - slot.z) > kMaxLevelOffset) {
      continue;
    }
    Candidate candidate;
    candidate.slot = static_cast<int>(index);
    candidate.inner = InsetConvex(
        std::vector<Vec2>(slot.corners.begin(), slot.corners.end()),
        config.inset);
    if (candidate.inner.size() < kMinPolygonCorners) continue;
    candidate.lo = candidate.hi = candidate.inner[0];
    for (const Vec2& vertex : candidate.inner) {
      candidate.lo = candidate.lo.cwiseMin(vertex);
      candidate.hi = candidate.hi.cwiseMax(vertex);
    }
    candidate.cells = 0;
    for (double x = candidate.lo.x() + kCellCentre * config.cell;
         x < candidate.hi.x(); x += config.cell) {
      for (double y = candidate.lo.y() + kCellCentre * config.cell;
           y < candidate.hi.y(); y += config.cell) {
        candidate.cells += PointInPolygon(candidate.inner, Vec2(x, y));
      }
    }
    candidates.push_back(std::move(candidate));
  }
  std::vector<SlotObservation> out;
  if (candidates.empty()) return out;
  std::vector<int> body(candidates.size(), 0);
  std::vector<std::set<std::pair<int, int>>> floor(candidates.size());
  for (const Vec3& point : map_points) {
    const Vec2 xy = point.head<2>();
    for (size_t k = 0; k < candidates.size(); ++k) {
      const Candidate& candidate = candidates[k];
      if (xy.x() < candidate.lo.x() || xy.y() < candidate.lo.y() ||
          xy.x() > candidate.hi.x() || xy.y() > candidate.hi.y() ||
          !PointInPolygon(candidate.inner, xy)) {
        continue;
      }
      const double height =
          point.z() - slots[static_cast<size_t>(candidate.slot)].z;
      if (std::abs(height) <= config.floor_band) {
        floor[k].insert({static_cast<int>(std::floor(
                             (xy.x() - candidate.lo.x()) / config.cell)),
                         static_cast<int>(std::floor(
                             (xy.y() - candidate.lo.y()) / config.cell))});
      } else if (height >= config.min_height && height <= config.max_height) {
        ++body[k];
      }
    }
  }
  for (size_t k = 0; k < candidates.size(); ++k) {
    SlotObservation observation;
    observation.slot = candidates[k].slot;
    observation.body_points = body[k];
    observation.floor_share = static_cast<double>(floor[k].size()) /
                              std::max(candidates[k].cells, 1);
    if (body[k] >= config.min_body_points) {
      observation.state = SlotState::kOccupied;
    } else if (body[k] <= config.max_free_points &&
               observation.floor_share >= config.min_floor_share) {
      observation.state = SlotState::kFree;
    }
    out.push_back(observation);
  }
  return out;
}

void SlotStateTracker::Update(double stamp,
                              const std::vector<SlotObservation>& observed) {
  for (const SlotObservation& observation : observed) {
    if (observation.state == SlotState::kUnknown || observation.slot < 0 ||
        observation.slot >= static_cast<int>(entries_.size())) {
      continue;
    }
    Entry& entry = entries_[static_cast<size_t>(observation.slot)];
    Pending& pending = pending_[static_cast<size_t>(observation.slot)];
    ++entry.observations;
    if (entry.state == SlotState::kUnknown ||
        entry.state == observation.state) {
      entry.state = observation.state;
      entry.stamp = stamp;
      pending = Pending();
      continue;
    }
    // A change: confirmed by consecutive observations.
    pending.count =
        pending.state == observation.state ? pending.count + 1 : 1;
    pending.state = observation.state;
    if (pending.count >= config_.confirm) {
      entry.state = observation.state;
      entry.stamp = stamp;
      pending = Pending();
    }
  }
}

}  // namespace gr
