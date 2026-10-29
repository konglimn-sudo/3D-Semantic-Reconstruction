#include "gr/nav/lanes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>

#include "gr/core/numeric.h"
#include "gr/nav/nav_map.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// Lengths below this (m) have no direction.
constexpr double kMinLength = 1e-9;
// Edges shorter than this in total (m) are covered by nothing.
constexpr double kMinEdgeLength = 1e-6;
// A wall lane needs open ground at every this many metres beyond it, up to
// the wall_open_width.
constexpr double kOpenGroundStep = 2.0;
// Lane pieces are compared with the skeleton every this many metres.
constexpr double kFollowSampleStep = 0.5;

double SegmentDistance(const Vec2& point, const Vec2& start, const Vec2& end) {
  const Vec2 segment = end - start;
  const double squared_length = segment.squaredNorm();
  const double fraction =
      squared_length > 0.0
          ? std::clamp((point - start).dot(segment) / squared_length, 0.0, 1.0)
          : 0.0;
  return (point - (start + fraction * segment)).norm();
}

double PolylineDistance(const Vec2& point, const std::vector<Vec2>& line) {
  double best = kInf;
  if (line.size() == 1) return (point - line[0]).norm();
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    best = std::min(best, SegmentDistance(point, line[k], line[k + 1]));
  }
  return best;
}

// Evenly spaced samples from `start` to `end`, at most `step` apart and at
// least the two ends.
std::vector<Vec2> Samples(const Vec2& start, const Vec2& end, double step) {
  const int count = std::max(
      2, static_cast<int>(std::ceil((end - start).norm() / step)) + 1);
  std::vector<Vec2> out;
  for (int k = 0; k < count; ++k) {
    out.push_back(start +
                  (end - start) * (static_cast<double>(k) / (count - 1)));
  }
  return out;
}

}  // namespace

std::vector<LaneRow> SlotRows(const std::vector<LaneSlot>& slots) {
  std::vector<std::string> order;
  std::map<std::string, std::vector<const LaneSlot*>> groups;
  for (const LaneSlot& slot : slots) {
    if (!groups.count(slot.row)) order.push_back(slot.row);
    groups[slot.row].push_back(&slot);
  }
  std::vector<LaneRow> rows;
  for (const std::string& key : order) {
    const std::vector<const LaneSlot*>& members = groups[key];
    Vec2 into = Vec2::Zero();  // Mean direction into the slots.
    std::vector<Vec2> entrances;
    std::vector<double> widths;
    std::vector<double> heights;
    for (const LaneSlot* slot : members) {
      const Vec2 entrance = 0.5 * (slot->corners[kEntranceLeftCorner] +
                                    slot->corners[kEntranceRightCorner]);
      const Vec2 back = 0.5 * (slot->corners[kBackRightCorner] +
                                slot->corners[kBackLeftCorner]);
      into +=
          (back - entrance) / std::max((back - entrance).norm(), kMinLength);
      entrances.push_back(entrance);
      widths.push_back(slot->width);
      heights.push_back(slot->z);
    }
    into /= static_cast<double>(members.size());
    LaneRow row;
    row.toward_aisle = -into / std::max(into.norm(), kMinLength);
    row.along = Vec2(-row.toward_aisle.y(), row.toward_aisle.x());
    std::vector<double> along;
    std::vector<double> across;
    for (const Vec2& entrance : entrances) {
      along.push_back(entrance.dot(row.along));
      across.push_back(entrance.dot(row.toward_aisle));
    }
    const double half_width = 0.5 * Median(widths);
    row.offset = Median(across);
    row.lo = *std::min_element(along.begin(), along.end()) - half_width;
    row.hi = *std::max_element(along.begin(), along.end()) + half_width;
    row.z = Median(heights);
    row.slots = static_cast<int>(members.size());
    rows.push_back(row);
  }
  return rows;
}

std::vector<LaneLine> LaneLines(const std::vector<LaneRow>& rows,
                                           const LaneConfig& config) {
  const double cos_parallel = std::cos(DegToRad(config.max_row_angle));
  // (gap, row, facing row, lo, hi) of every facing pair.
  std::vector<std::tuple<double, size_t, size_t, double, double>> pairs;
  for (size_t i = 0; i < rows.size(); ++i) {
    for (size_t j = i + 1; j < rows.size(); ++j) {
      const LaneRow& row = rows[i];
      const LaneRow& facing = rows[j];
      if (row.toward_aisle.dot(facing.toward_aisle) > -cos_parallel) continue;
      // The facing entrance line seen from the row: ahead of it by `gap`.
      const Vec2 facing_centre =
          facing.along * 0.5 * (facing.lo + facing.hi) +
          facing.toward_aisle * facing.offset;
      const double gap = facing_centre.dot(row.toward_aisle) - row.offset;
      if (gap < config.min_aisle || gap > config.max_aisle) continue;
      const double end0 =
          (facing.along * facing.lo + facing.toward_aisle * facing.offset)
              .dot(row.along);
      const double end1 =
          (facing.along * facing.hi + facing.toward_aisle * facing.offset)
              .dot(row.along);
      const double lo = std::max(row.lo, std::min(end0, end1));
      const double hi = std::min(row.hi, std::max(end0, end1));
      if (hi - lo < config.min_overlap) continue;
      pairs.emplace_back(gap, i, j, std::min(row.lo, std::min(end0, end1)),
                         std::max(row.hi, std::max(end0, end1)));
    }
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<uint8_t> used(rows.size(), 0);
  std::vector<LaneLine> out;
  for (const auto& [gap, i, j, lo, hi] : pairs) {
    if (used[i] || used[j]) continue;
    used[i] = used[j] = 1;
    const LaneRow& row = rows[i];
    const double lane = row.offset + 0.5 * gap;  // Midway.
    out.push_back({row.along * lo + row.toward_aisle * lane,
                   row.along * hi + row.toward_aisle * lane});
  }
  for (size_t i = 0; i < rows.size(); ++i) {
    if (used[i]) continue;
    const LaneRow& row = rows[i];
    const double lane = row.offset + config.lane_offset;
    out.push_back({row.along * row.lo + row.toward_aisle * lane,
                   row.along * row.hi + row.toward_aisle * lane});
  }
  return out;
}

LanePlan PlanLanes(const std::vector<LaneSlot>& slots,
                   const std::vector<uint8_t>& navigable,
                   const RasterFrame& frame,
                   const std::vector<std::vector<Vec2>>& skeleton,
                   const LaneConfig& config,
                   const std::vector<LaneWall>& walls,
                   const std::vector<std::vector<Vec2>>& drives) {
  LanePlan plan;
  const double step = frame.res;
  const auto navigable_at = [&](const Vec2& point) {
    const int column =
        static_cast<int>(std::floor((point.x() - frame.x0) / step));
    const int row = static_cast<int>(std::floor((point.y() - frame.y0) / step));
    return column >= 0 && row >= 0 && column < frame.width &&
           row < frame.height &&
           navigable[static_cast<size_t>(row) * frame.width + column] != 0;
  };
  // Proposals: slot-row lanes, wall lanes (with the direction away from
  // their wall), drive stretches.
  struct Proposal {
    LaneLine line;
    uint8_t source = kSlotRowLane;
    Vec2 open = Vec2::Zero();
  };
  std::vector<Proposal> proposals;
  if (!slots.empty()) {
    const std::vector<LaneRow> rows = SlotRows(slots);
    plan.rows = static_cast<int>(rows.size());
    for (const auto& line : LaneLines(rows, config)) {
      proposals.push_back({line, kSlotRowLane, Vec2::Zero()});
    }
  }
  for (const LaneWall& wall : walls) {
    const Vec2 direction = wall.p1 - wall.p0;
    if (direction.norm() < config.min_wall_length) continue;
    const Vec2 normal =
        Vec2(-direction.y(), direction.x()) / direction.norm();
    const double offset = 0.5 * wall.thickness + config.wall_offset;
    for (double side : {1.0, -1.0}) {
      proposals.push_back({{wall.p0 + side * offset * normal,
                            wall.p1 + side * offset * normal},
                           kWallLane,
                           side * normal});
    }
  }
  for (const std::vector<Vec2>& drive : drives) {
    if (drive.size() < 2) continue;
    const std::vector<Vec2> simple =
        SimplifyPolyline(drive, config.drive_tolerance);
    for (size_t k = 0; k + 1 < simple.size(); ++k) {
      if ((simple[k + 1] - simple[k]).norm() >= config.drive_min_length) {
        proposals.push_back(
            {{simple[k], simple[k + 1]}, kDriveLane, Vec2::Zero()});
      }
    }
  }
  for (const Proposal& proposal : proposals) {
    const LaneLine& line = proposal.line;
    const bool from_wall = proposal.source == kWallLane;
    const double length = (line[1] - line[0]).norm();
    const int count =
        std::max(2, static_cast<int>(std::ceil(length / step)) + 1);
    std::vector<Vec2> xy(static_cast<size_t>(count));
    std::vector<uint8_t> keep(static_cast<size_t>(count), 0);
    for (int k = 0; k < count; ++k) {
      xy[static_cast<size_t>(k)] =
          line[0] +
          (line[1] - line[0]) * (static_cast<double>(k) / (count - 1));
      const Vec2& point = xy[static_cast<size_t>(k)];
      bool kept = navigable_at(point);
      // A wall lane needs open ground beyond it.
      for (double beyond = kOpenGroundStep;
           kept && from_wall && beyond <= config.wall_open_width;
           beyond += kOpenGroundStep) {
        kept = navigable_at(point + beyond * proposal.open);
      }
      for (const auto& taken : plan.lanes) {
        if (!kept) break;
        kept = SegmentDistance(point, taken[0], taken[1]) >
               config.dedupe_distance;
      }
      keep[static_cast<size_t>(k)] = kept;
    }
    std::vector<LaneLine> pieces;
    for (int k = 0; k < count;) {
      if (!keep[static_cast<size_t>(k)]) {
        ++k;
        continue;
      }
      int end = k;
      while (end + 1 < count && keep[static_cast<size_t>(end + 1)]) ++end;
      if ((end - k) * step >= config.min_length) {
        pieces.push_back(
            {xy[static_cast<size_t>(k)], xy[static_cast<size_t>(end)]});
      }
      k = end + 1;
    }
    bool used = false;
    for (const auto& piece : pieces) {
      // The skeleton already runs along it: not needed.
      int followed = 0;
      const std::vector<Vec2> samples =
          Samples(piece[0], piece[1], kFollowSampleStep);
      for (const Vec2& sample : samples) {
        double distance = kInf;
        for (const std::vector<Vec2>& edge : skeleton) {
          distance = std::min(distance, PolylineDistance(sample, edge));
        }
        followed += distance <= config.follow_distance;
      }
      if (!skeleton.empty() &&
          static_cast<double>(followed) / samples.size() >=
              config.skip_followed) {
        ++plan.followed;
        continue;
      }
      plan.lanes.push_back(piece);
      plan.source.push_back(proposal.source);
      used = true;
    }
    plan.walls += proposal.source == kWallLane && used;
    plan.drives += proposal.source == kDriveLane && used;
  }
  return plan;
}

bool CoveredByLanes(const std::vector<Vec2>& edge,
                    const std::vector<LaneLine>& lanes,
                    const LaneConfig& config) {
  double total = 0.0;
  double covered = 0.0;
  const double cos_max = std::cos(DegToRad(config.replace_angle));
  for (size_t k = 0; k + 1 < edge.size(); ++k) {
    const Vec2 segment = edge[k + 1] - edge[k];
    const double length = segment.norm();
    total += length;
    if (length < kMinLength) continue;
    const Vec2 mid = edge[k] + 0.5 * segment;
    const Vec2 direction = segment / length;
    for (const auto& lane : lanes) {
      Vec2 axis = lane[1] - lane[0];
      const double lane_length = axis.norm();
      axis /= std::max(lane_length, kMinLength);
      if (std::abs(direction.dot(axis)) < cos_max) continue;
      const double along =
          std::clamp((mid - lane[0]).dot(axis), 0.0, lane_length);
      if ((mid - (lane[0] + along * axis)).norm() <= config.replace_distance) {
        covered += length;
        break;
      }
    }
  }
  if (total < kMinEdgeLength) return false;
  return covered >= config.replace_share * total;
}

}  // namespace gr
