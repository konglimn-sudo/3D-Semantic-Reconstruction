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

double SegmentDistance(const Vec2& p, const Vec2& a, const Vec2& b) {
  const Vec2 d = b - a;
  const double length2 = d.squaredNorm();
  const double t =
      length2 > 0.0 ? std::clamp((p - a).dot(d) / length2, 0.0, 1.0) : 0.0;
  return (p - (a + t * d)).norm();
}

double PolylineDistance(const Vec2& p, const std::vector<Vec2>& line) {
  double best = std::numeric_limits<double>::infinity();
  if (line.size() == 1) return (p - line[0]).norm();
  for (size_t k = 0; k + 1 < line.size(); ++k) {
    best = std::min(best, SegmentDistance(p, line[k], line[k + 1]));
  }
  return best;
}

// Samples from a to b: n points with n = max(2, ceil(length / step) + 1).
std::vector<Vec2> Samples(const Vec2& a, const Vec2& b, double step) {
  const int n =
      std::max(2, static_cast<int>(std::ceil((b - a).norm() / step)) + 1);
  std::vector<Vec2> out;
  for (int k = 0; k < n; ++k) {
    out.push_back(a + (b - a) * (static_cast<double>(k) / (n - 1)));
  }
  return out;
}

}  // namespace

std::vector<LaneRow> SlotRows(const std::vector<LaneSlot>& slots) {
  std::vector<std::string> order;
  std::map<std::string, std::vector<const LaneSlot*>> groups;
  for (const LaneSlot& s : slots) {
    if (!groups.count(s.row)) order.push_back(s.row);
    groups[s.row].push_back(&s);
  }
  std::vector<LaneRow> rows;
  for (const std::string& key : order) {
    const std::vector<const LaneSlot*>& members = groups[key];
    Vec2 into = Vec2::Zero();
    std::vector<Vec2> entrances;
    std::vector<double> widths;
    std::vector<double> zs;
    for (const LaneSlot* s : members) {
      const Vec2 entrance = 0.5 * (s->corners[0] + s->corners[1]);
      const Vec2 back = 0.5 * (s->corners[2] + s->corners[3]);
      into += (back - entrance) / std::max((back - entrance).norm(), 1e-9);
      entrances.push_back(entrance);
      widths.push_back(s->width);
      zs.push_back(s->z);
    }
    into /= static_cast<double>(members.size());
    LaneRow row;
    row.n = -into / std::max(into.norm(), 1e-9);
    row.u = Vec2(-row.n.y(), row.n.x());
    std::vector<double> along;
    std::vector<double> across;
    for (const Vec2& e : entrances) {
      along.push_back(e.dot(row.u));
      across.push_back(e.dot(row.n));
    }
    const double half = 0.5 * Median(widths);
    row.offset = Median(across);
    row.lo = *std::min_element(along.begin(), along.end()) - half;
    row.hi = *std::max_element(along.begin(), along.end()) + half;
    row.z = Median(zs);
    row.slots = static_cast<int>(members.size());
    rows.push_back(row);
  }
  return rows;
}

std::vector<std::array<Vec2, 2>> LaneLines(const std::vector<LaneRow>& rows,
                                           const LaneConfig& config) {
  const double cos_parallel = std::cos(DegToRad(config.max_row_angle));
  std::vector<std::tuple<double, size_t, size_t, double, double>> pairs;
  for (size_t i = 0; i < rows.size(); ++i) {
    for (size_t j = i + 1; j < rows.size(); ++j) {
      const LaneRow& a = rows[i];
      const LaneRow& b = rows[j];
      if (a.n.dot(b.n) > -cos_parallel) continue;
      // b's entrance line seen from a: ahead of a by `gap`.
      const Vec2 centre_b = b.u * 0.5 * (b.lo + b.hi) + b.n * b.offset;
      const double gap = centre_b.dot(a.n) - a.offset;
      if (gap < config.min_aisle || gap > config.max_aisle) continue;
      const double e0 = (b.u * b.lo + b.n * b.offset).dot(a.u);
      const double e1 = (b.u * b.hi + b.n * b.offset).dot(a.u);
      const double lo = std::max(a.lo, std::min(e0, e1));
      const double hi = std::min(a.hi, std::max(e0, e1));
      if (hi - lo < config.min_overlap) continue;
      pairs.emplace_back(gap, i, j, std::min(a.lo, std::min(e0, e1)),
                         std::max(a.hi, std::max(e0, e1)));
    }
  }
  std::sort(pairs.begin(), pairs.end());
  std::vector<uint8_t> used(rows.size(), 0);
  std::vector<std::array<Vec2, 2>> out;
  for (const auto& [gap, i, j, lo, hi] : pairs) {
    if (used[i] || used[j]) continue;
    used[i] = used[j] = 1;
    const LaneRow& a = rows[i];
    const double c = a.offset + 0.5 * gap;
    out.push_back({a.u * lo + a.n * c, a.u * hi + a.n * c});
  }
  for (size_t i = 0; i < rows.size(); ++i) {
    if (used[i]) continue;
    const LaneRow& a = rows[i];
    const double c = a.offset + config.lane_offset;
    out.push_back({a.u * a.lo + a.n * c, a.u * a.hi + a.n * c});
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
  const auto navigable_at = [&](const Vec2& p) {
    const int col = static_cast<int>(std::floor((p.x() - frame.x0) / step));
    const int row = static_cast<int>(std::floor((p.y() - frame.y0) / step));
    return col >= 0 && row >= 0 && col < frame.width && row < frame.height &&
           navigable[static_cast<size_t>(row) * frame.width + col] != 0;
  };
  // Proposals: slot-row lanes, wall lanes (with the direction away from
  // their wall), drive stretches.
  struct Proposal {
    std::array<Vec2, 2> line;
    uint8_t source = 0;
    Vec2 open = Vec2::Zero();
  };
  std::vector<Proposal> proposals;
  if (!slots.empty()) {
    const std::vector<LaneRow> rows = SlotRows(slots);
    plan.rows = static_cast<int>(rows.size());
    for (const auto& line : LaneLines(rows, config)) {
      proposals.push_back({line, 0, Vec2::Zero()});
    }
  }
  for (const LaneWall& wall : walls) {
    const Vec2 d = wall.p1 - wall.p0;
    if (d.norm() < config.min_wall_length) continue;
    const Vec2 n = Vec2(-d.y(), d.x()) / d.norm();
    const double offset = 0.5 * wall.thickness + config.wall_offset;
    for (double side : {1.0, -1.0}) {
      proposals.push_back({{wall.p0 + side * offset * n,
                            wall.p1 + side * offset * n},
                           1, side * n});
    }
  }
  for (const std::vector<Vec2>& drive : drives) {
    if (drive.size() < 2) continue;
    const std::vector<Vec2> simple =
        SimplifyPolyline(drive, config.drive_tolerance);
    for (size_t k = 0; k + 1 < simple.size(); ++k) {
      if ((simple[k + 1] - simple[k]).norm() >= config.drive_min_length) {
        proposals.push_back({{simple[k], simple[k + 1]}, 2, Vec2::Zero()});
      }
    }
  }
  for (const Proposal& proposal : proposals) {
    const std::array<Vec2, 2>& line = proposal.line;
    const bool from_wall = proposal.source == 1;
    const double length = (line[1] - line[0]).norm();
    const int n = std::max(2, static_cast<int>(std::ceil(length / step)) + 1);
    std::vector<Vec2> xy(static_cast<size_t>(n));
    std::vector<uint8_t> ok(static_cast<size_t>(n), 0);
    for (int k = 0; k < n; ++k) {
      xy[static_cast<size_t>(k)] =
          line[0] + (line[1] - line[0]) * (static_cast<double>(k) / (n - 1));
      const Vec2& p = xy[static_cast<size_t>(k)];
      bool keep = navigable_at(p);
      // A wall lane needs open ground beyond it.
      for (double d = 2.0; keep && from_wall && d <= config.wall_open_width;
           d += 2.0) {
        keep = navigable_at(p + d * proposal.open);
      }
      for (const auto& taken : plan.lanes) {
        if (!keep) break;
        keep = SegmentDistance(p, taken[0], taken[1]) >
               config.dedupe_distance;
      }
      ok[static_cast<size_t>(k)] = keep;
    }
    std::vector<std::array<Vec2, 2>> pieces;
    for (int k = 0; k < n;) {
      if (!ok[static_cast<size_t>(k)]) {
        ++k;
        continue;
      }
      int end = k;
      while (end + 1 < n && ok[static_cast<size_t>(end + 1)]) ++end;
      if ((end - k) * step >= config.min_length) {
        pieces.push_back({xy[static_cast<size_t>(k)],
                          xy[static_cast<size_t>(end)]});
      }
      k = end + 1;
    }
    bool used = false;
    for (const auto& piece : pieces) {
      // The skeleton already runs along it: not needed.
      int near = 0;
      const std::vector<Vec2> samples = Samples(piece[0], piece[1], 0.5);
      for (const Vec2& p : samples) {
        double d = std::numeric_limits<double>::infinity();
        for (const std::vector<Vec2>& edge : skeleton) {
          d = std::min(d, PolylineDistance(p, edge));
        }
        near += d <= config.follow_distance;
      }
      if (!skeleton.empty() && static_cast<double>(near) / samples.size() >=
                                   config.skip_followed) {
        ++plan.followed;
        continue;
      }
      plan.lanes.push_back(piece);
      plan.source.push_back(proposal.source);
      used = true;
    }
    plan.walls += proposal.source == 1 && used;
    plan.drives += proposal.source == 2 && used;
  }
  return plan;
}

bool CoveredByLanes(const std::vector<Vec2>& edge,
                    const std::vector<std::array<Vec2, 2>>& lanes,
                    const LaneConfig& config) {
  double total = 0.0;
  double covered = 0.0;
  const double cos_max = std::cos(DegToRad(config.replace_angle));
  for (size_t k = 0; k + 1 < edge.size(); ++k) {
    const Vec2 d = edge[k + 1] - edge[k];
    const double length = d.norm();
    total += length;
    if (length < 1e-9) continue;
    const Vec2 mid = edge[k] + 0.5 * d;
    const Vec2 direction = d / length;
    for (const auto& lane : lanes) {
      Vec2 axis = lane[1] - lane[0];
      const double lane_length = axis.norm();
      axis /= std::max(lane_length, 1e-9);
      if (std::abs(direction.dot(axis)) < cos_max) continue;
      const double t = std::clamp((mid - lane[0]).dot(axis), 0.0, lane_length);
      if ((mid - (lane[0] + t * axis)).norm() <= config.replace_distance) {
        covered += length;
        break;
      }
    }
  }
  if (total < 1e-6) return false;
  return covered >= config.replace_share * total;
}

}  // namespace gr
