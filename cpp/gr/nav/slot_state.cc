#include "gr/nav/slot_state.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include "gr/core/geometry.h"

namespace gr {

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
  for (size_t s = 0; s < slots.size(); ++s) {
    const SlotGeometry& slot = slots[s];
    Vec2 center = Vec2::Zero();
    for (const Vec2& c : slot.corners) center += 0.25 * c;
    if ((center - sensor.head<2>()).norm() > config.max_range ||
        std::abs(sensor.z() - slot.z) > 4.0) {
      continue;
    }
    Candidate c;
    c.slot = static_cast<int>(s);
    c.inner = InsetConvex(
        std::vector<Vec2>(slot.corners.begin(), slot.corners.end()),
        config.inset);
    if (c.inner.size() < 3) continue;
    c.lo = c.hi = c.inner[0];
    for (const Vec2& p : c.inner) {
      c.lo = c.lo.cwiseMin(p);
      c.hi = c.hi.cwiseMax(p);
    }
    c.cells = 0;
    for (double x = c.lo.x() + 0.5 * config.cell; x < c.hi.x();
         x += config.cell) {
      for (double y = c.lo.y() + 0.5 * config.cell; y < c.hi.y();
           y += config.cell) {
        c.cells += PointInPolygon(c.inner, Vec2(x, y));
      }
    }
    candidates.push_back(std::move(c));
  }
  std::vector<SlotObservation> out;
  if (candidates.empty()) return out;
  std::vector<int> body(candidates.size(), 0);
  std::vector<std::set<std::pair<int, int>>> floor(candidates.size());
  for (const Vec3& p : map_points) {
    const Vec2 xy = p.head<2>();
    for (size_t k = 0; k < candidates.size(); ++k) {
      const Candidate& c = candidates[k];
      if (xy.x() < c.lo.x() || xy.y() < c.lo.y() || xy.x() > c.hi.x() ||
          xy.y() > c.hi.y() || !PointInPolygon(c.inner, xy)) {
        continue;
      }
      const double height = p.z() - slots[static_cast<size_t>(c.slot)].z;
      if (std::abs(height) <= config.floor_band) {
        floor[k].insert(
            {static_cast<int>(std::floor((xy.x() - c.lo.x()) / config.cell)),
             static_cast<int>(std::floor((xy.y() - c.lo.y()) / config.cell))});
      } else if (height >= config.min_height && height <= config.max_height) {
        ++body[k];
      }
    }
  }
  for (size_t k = 0; k < candidates.size(); ++k) {
    SlotObservation o;
    o.slot = candidates[k].slot;
    o.body_points = body[k];
    o.floor_share = static_cast<double>(floor[k].size()) /
                    std::max(candidates[k].cells, 1);
    if (body[k] >= config.min_body_points) {
      o.state = SlotState::kOccupied;
    } else if (body[k] <= config.max_free_points &&
               o.floor_share >= config.min_floor_share) {
      o.state = SlotState::kFree;
    }
    out.push_back(o);
  }
  return out;
}

void SlotStateTracker::Update(double stamp,
                              const std::vector<SlotObservation>& observed) {
  for (const SlotObservation& o : observed) {
    if (o.state == SlotState::kUnknown || o.slot < 0 ||
        o.slot >= static_cast<int>(entries_.size())) {
      continue;
    }
    Entry& entry = entries_[static_cast<size_t>(o.slot)];
    Pending& pending = pending_[static_cast<size_t>(o.slot)];
    ++entry.observations;
    if (entry.state == SlotState::kUnknown || entry.state == o.state) {
      entry.state = o.state;
      entry.stamp = stamp;
      pending = Pending();
      continue;
    }
    // A change: confirmed by consecutive observations.
    pending.count = pending.state == o.state ? pending.count + 1 : 1;
    pending.state = o.state;
    if (pending.count >= config_.confirm) {
      entry.state = o.state;
      entry.stamp = stamp;
      pending = Pending();
    }
  }
}

}  // namespace gr
