#include "gr/semantics/slots.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/geometry.h"
#include "gr/core/morphology.h"
#include "gr/core/numeric.h"

namespace gr {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

Eigen::Matrix2d Rot2(double angle) {
  Eigen::Matrix2d r;
  r << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
  return r;
}

// A slot boundary in a row frame: across position, along extent.
struct Separator {
  double c = 0.0;
  double a0 = 0.0;
  double a1 = 0.0;
  bool is_virtual = false;  // Column face, no paint.
  bool ok0 = false;         // a0 / a1 measured with floor observed beyond.
  bool ok1 = false;
  int index = -1;  // Marking segment it comes from.
};

using Row = std::vector<Separator>;

// Separators of the given segments in the frame rotated by row_yaw; `orient`
// 1: separators along the frame's y (slots along x), 0: the other way.
Row Project(const std::vector<MarkingSegment>& segments,
            const std::vector<int>& indices, int orient, double row_yaw,
            const SlotConfig& config) {
  const Eigen::Matrix2d to_g = Rot2(-row_yaw);
  const Vec2 axis = orient == 1 ? Vec2(0.0, 1.0) : Vec2(1.0, 0.0);
  Row out;
  for (int i : indices) {
    const MarkingSegment& s = segments[static_cast<size_t>(i)];
    const Vec2 p0 = to_g * s.p0;
    const Vec2 p1 = to_g * s.p1;
    const Vec2 d = p1 - p0;
    const double length = d.norm();
    if (length < config.min_separator_length ||
        length > 2.0 * config.depth_max + 0.5) {
      continue;
    }
    const double angle =
        RadToDeg(std::acos(std::min(1.0, std::abs(d.dot(axis)) / length)));
    if (angle > config.angle_tol) continue;
    Separator sep;
    sep.c = 0.5 * (p0(1 - orient) + p1(1 - orient));
    sep.index = i;
    if (p0(orient) <= p1(orient)) {
      sep.a0 = p0(orient);
      sep.a1 = p1(orient);
      sep.ok0 = s.end_ok[0];
      sep.ok1 = s.end_ok[1];
    } else {
      sep.a0 = p1(orient);
      sep.a1 = p0(orient);
      sep.ok0 = s.end_ok[1];
      sep.ok1 = s.end_ok[0];
    }
    out.push_back(sep);
  }
  return out;
}

// Groups separators whose along-intervals overlap.
std::vector<Row> Rows(Row seps, double overlap) {
  std::stable_sort(seps.begin(), seps.end(),
                   [](const Separator& a, const Separator& b) {
                     return a.a0 + a.a1 < b.a0 + b.a1;
                   });
  std::vector<Row> rows;
  for (const Separator& s : seps) {
    bool placed = false;
    for (Row& r : rows) {
      double ra0 = r[0].a0;
      double ra1 = r[0].a1;
      for (const Separator& m : r) {
        ra0 = std::min(ra0, m.a0);
        ra1 = std::max(ra1, m.a1);
      }
      const double inter = std::min(ra1, s.a1) - std::max(ra0, s.a0);
      if (inter / std::max(std::min(ra1 - ra0, s.a1 - s.a0), 1e-6) >=
          overlap) {
        r.push_back(s);
        placed = true;
        break;
      }
    }
    if (!placed) rows.push_back({s});
  }
  return rows;
}

// Angle (rad) of a row's entrance / back lines against the frame: the
// Theil-Sen slope of the measured separator ends over their across
// position, kept only if the ends move by at least `min_change` and by four
// robust residual sigmas across the row (end noise over short rows would
// otherwise rotate aligned rows).
double RowTilt(const Row& seps, double min_change = 0.15,
               double min_baseline = 2.0) {
  std::vector<std::vector<std::pair<double, double>>> sets(2);
  for (const Separator& m : seps) {
    if (m.ok0) sets[0].emplace_back(m.c, m.a0);
    if (m.ok1) sets[1].emplace_back(m.c, m.a1);
  }
  std::vector<double> slopes;
  for (const auto& ends : sets) {
    for (size_t i = 0; i < ends.size(); ++i) {
      for (size_t j = i + 1; j < ends.size(); ++j) {
        const double dc = ends[j].first - ends[i].first;
        if (std::abs(dc) >= min_baseline) {
          slopes.push_back((ends[j].second - ends[i].second) / dc);
        }
      }
    }
  }
  if (slopes.size() < 3) return 0.0;
  const double beta = Median(slopes);
  std::vector<double> residuals;
  std::vector<double> cs;
  for (const auto& ends : sets) {
    if (ends.empty()) continue;
    std::vector<double> r;
    for (const auto& [c, a] : ends) {
      r.push_back(a - beta * c);
      cs.push_back(c);
    }
    const double median = Median(r);
    for (double v : r) residuals.push_back(v - median);
  }
  std::vector<double> absolute;
  for (double v : residuals) absolute.push_back(std::abs(v));
  const double sigma = 1.4826 * Median(absolute);
  const double change =
      std::abs(beta) * (*std::max_element(cs.begin(), cs.end()) -
                        *std::min_element(cs.begin(), cs.end()));
  if (change < std::max(min_change, 4.0 * sigma)) return 0.0;
  return std::atan(beta);
}

struct RowEnds {
  double a0 = 0.0;
  double a1 = 0.0;
  bool ok0 = false;
  bool ok1 = false;
};

// Along extent of a row and whether each end was measured: rows share their
// entrance and back lines, so each end is the median of the measured
// separator ends near the row's extreme; without any, the paint envelope.
RowEnds EndsOf(const Row& painted, double near = 0.15) {
  double e0 = painted[0].a0;
  double e1 = painted[0].a1;
  for (const Separator& m : painted) {
    e0 = std::min(e0, m.a0);
    e1 = std::max(e1, m.a1);
  }
  std::vector<double> lo;
  std::vector<double> hi;
  for (const Separator& m : painted) {
    if (m.ok0 && m.a0 < e0 + near) lo.push_back(m.a0);
    if (m.ok1 && m.a1 > e1 - near) hi.push_back(m.a1);
  }
  RowEnds out;
  out.a0 = lo.empty() ? e0 : Median(lo);
  out.a1 = hi.empty() ? e1 : Median(hi);
  out.ok0 = !lo.empty();
  out.ok1 = !hi.empty();
  return out;
}

Vec2 PointAt(double c, double a, int orient) {
  return orient == 1 ? Vec2(c, a) : Vec2(a, c);
}

// numpy.linspace.
std::vector<double> Linspace(double start, double stop, int n) {
  std::vector<double> out(static_cast<size_t>(n));
  const double step = n > 1 ? (stop - start) / (n - 1) : 0.0;
  for (int k = 0; k < n; ++k) out[static_cast<size_t>(k)] = start + k * step;
  if (n > 1) out.back() = stop;
  return out;
}

// Cells of a line at across position c, along a0..a1 (row frame), 0.3 m in
// from both ends, inside the raster; also the number of samples.
std::pair<std::vector<size_t>, int> LineCells(double c, double a0, double a1,
                                              int orient,
                                              const Eigen::Matrix2d& to_w,
                                              const RasterFrame& frame) {
  const int n = std::max(static_cast<int>((a1 - a0) / 0.1), 2);
  std::vector<size_t> cells;
  for (double a : Linspace(a0 + 0.3, a1 - 0.3, n)) {
    const Vec2 w = to_w * PointAt(c, a, orient);
    const int col =
        static_cast<int>(std::floor((w.x() - frame.x0) / frame.res));
    const int row =
        static_cast<int>(std::floor((w.y() - frame.y0) / frame.res));
    if (col < 0 || row < 0 || col >= frame.width || row >= frame.height) {
      continue;
    }
    cells.push_back(static_cast<size_t>(row) * frame.width + col);
  }
  return {cells, n};
}

double NanMean(const std::vector<float>& raster,
               const std::vector<size_t>& cells) {
  double sum = 0.0;
  int count = 0;
  for (size_t c : cells) {
    if (std::isfinite(raster[c])) {
      sum += raster[c];
      ++count;
    }
  }
  return count ? sum / count : kNaN;
}

// (visible share, paint ridge) of a hypothetical separator: its share on
// observed floor with nothing standing on it, and the mean intensity on the
// line minus that 0.35 m to both sides (NaN without intensity).
std::pair<double, double> SeparatorEvidence(double c, double a0, double a1,
                                            int orient,
                                            const Eigen::Matrix2d& to_w,
                                            const SlotRasters& rasters,
                                            double occupied_height) {
  const auto [cells, n] = LineCells(c, a0, a1, orient, to_w, rasters.frame);
  if (cells.empty()) return {0.0, kNaN};
  int visible = 0;
  for (size_t cell : cells) {
    const float h = rasters.height_max[cell];
    const bool empty = !(std::isfinite(h) ? h >= occupied_height
                                          : 0.0 >= occupied_height);
    visible += rasters.floor_count[cell] > 0 && empty;
  }
  const double share = static_cast<double>(visible) / n;
  if (rasters.intensity.empty()) return {share, kNaN};
  double values[3];
  const double offsets[3] = {0.0, -0.35, 0.35};
  for (int k = 0; k < 3; ++k) {
    const auto [side, unused] =
        LineCells(c + offsets[k], a0, a1, orient, to_w, rasters.frame);
    values[k] = side.empty() ? kNaN : NanMean(rasters.intensity, side);
  }
  return {share, values[0] - 0.5 * (values[1] + values[2])};
}

// Whether a hypothetical separator is visibly not there: on observed, empty
// floor and without a paint ridge of `paint_ridge_share` of the row's.
bool Absent(double visible, double ridge, double paint,
            const SlotConfig& config) {
  if (visible < config.visible_separator_share) return false;
  if (!(std::isfinite(ridge) && std::isfinite(paint) && paint > 0.0)) {
    return true;
  }
  return ridge < config.paint_ridge_share * paint;
}

struct Candidate {
  int orient;
  double c0;
  double c1;
  double a0;
  double a1;
  bool inferred;
  std::string row_key;
  double row_yaw;
};

double MinDistance(const std::vector<Vec2>& points, const Vec2& q) {
  double best = std::numeric_limits<double>::infinity();
  for (const Vec2& p : points) best = std::min(best, (p - q).norm());
  return best;
}

// Cells of a world polygon (vertices to cells by `to_cell`).
Mask PolygonCells(const std::vector<Vec2>& polygon, const RasterFrame& frame,
                  bool round_vertices) {
  std::vector<std::pair<int, int>> vertices;
  for (const Vec2& p : polygon) {
    const double x = (p.x() - frame.x0) / frame.res;
    const double y = (p.y() - frame.y0) / frame.res;
    vertices.emplace_back(
        static_cast<int>(round_vertices ? std::nearbyint(x) : std::floor(x)),
        static_cast<int>(round_vertices ? std::nearbyint(y) : std::floor(y)));
  }
  return FillPolygon(vertices, frame.width, frame.height);
}

}  // namespace

std::vector<ParkingSlot> AssembleSlots(
    const std::vector<MarkingSegment>& segments, int level, double yaw,
    const std::vector<Vec2>& path_xy, const std::vector<SlotColumn>& columns,
    const SlotRasters& rasters, const SlotConfig& config, const Logger& log) {
  const RasterFrame& frame = rasters.frame;
  const size_t cells = static_cast<size_t>(frame.width) * frame.height;
  GR_CHECK(rasters.height_max.size() == cells &&
           rasters.floor_z.size() == cells);
  GR_CHECK(rasters.floor_count.empty() || rasters.floor_count.size() == cells);
  GR_CHECK(rasters.intensity.empty() || rasters.intensity.size() == cells);
  const bool have_counts = !rasters.floor_count.empty();
  const double wmin = config.width_min;
  const double wmax = config.width_max;
  const double dmin = config.depth_min;
  const double dmax = config.depth_max;
  std::vector<std::vector<Vec2>> column_boxes;
  for (const SlotColumn& c : columns) {
    const Vec2 h = c.size / 2.0;
    column_boxes.push_back({c.center + Vec2(-h.x(), -h.y()),
                            c.center + Vec2(h.x(), -h.y()),
                            c.center + Vec2(h.x(), h.y()),
                            c.center + Vec2(-h.x(), h.y())});
  }
  std::vector<int> all(segments.size());
  std::iota(all.begin(), all.end(), 0);
  std::vector<Candidate> candidates;
  for (int orient : {1, 0}) {
    std::vector<std::pair<Row, double>> rows;
    const std::vector<Row> groups =
        Rows(Project(segments, all, orient, yaw, config), config.row_overlap);
    // Rows too short or noisy for their own direction take the median of
    // the significant ones.
    std::vector<double> tilts;
    std::vector<double> significant;
    for (const Row& g : groups) {
      tilts.push_back(RowTilt(g));
      if (tilts.back() != 0.0) significant.push_back(tilts.back());
    }
    const double fallback = significant.size() >= 2 ? Median(significant) : 0.0;
    for (size_t gi = 0; gi < groups.size(); ++gi) {
      const double tilt = tilts[gi] != 0.0 ? tilts[gi] : fallback;
      const double row_yaw = yaw + (orient == 1 ? tilt : -tilt);
      Row r;
      if (tilt != 0.0) {
        std::vector<int> indices;
        for (const Separator& m : groups[gi]) indices.push_back(m.index);
        r = Project(segments, indices, orient, row_yaw, config);
      } else {
        r = groups[gi];
      }
      if (r.empty()) continue;
      // Back-to-back rows share painted lines: split the whole group at the
      // middle of its extent. A few long fragments do not make a single row
      // back-to-back.
      std::vector<double> a0s;
      std::vector<double> a1s;
      for (const Separator& m : r) {
        a0s.push_back(m.a0);
        a1s.push_back(m.a1);
      }
      const double q0 = Percentile(a0s, 10.0);
      const double q1 = Percentile(a1s, 90.0);
      if (q1 - q0 <= dmax + 0.5) {
        if (*std::max_element(a1s.begin(), a1s.end()) -
                *std::min_element(a0s.begin(), a0s.end()) >
            dmax + 0.5) {
          for (Separator& m : r) {
            m.a0 = std::max(m.a0, q0 - 0.3);
            m.a1 = std::min(m.a1, q1 + 0.3);
          }
        }
        rows.emplace_back(r, row_yaw);
        continue;
      }
      const RowEnds ends = EndsOf(r);
      const double mid = 0.5 * (ends.a0 + ends.a1);
      const bool measured = ends.ok0 && ends.ok1;
      Row low;
      Row high;
      for (const Separator& m : r) {
        if (m.a0 < mid - 0.5) {
          Separator s = m;
          s.a1 = std::min(m.a1, mid);
          s.ok1 = m.a1 > mid ? measured : m.ok1;
          low.push_back(s);
        }
      }
      for (const Separator& m : r) {
        if (m.a1 > mid + 0.5) {
          Separator s = m;
          s.a0 = std::max(m.a0, mid);
          s.ok0 = m.a0 < mid ? measured : m.ok0;
          high.push_back(s);
        }
      }
      if (!low.empty()) rows.emplace_back(low, row_yaw);
      if (!high.empty()) rows.emplace_back(high, row_yaw);
    }
    // Columns at the row's corners bound bays as reliably as paint: their
    // faces become virtual separators (they make hidden lines inferable).
    for (auto& [r, row_yaw] : rows) {
      const Eigen::Matrix2d to_g = Rot2(-row_yaw);
      double ra0 = r[0].a0;
      double ra1 = r[0].a1;
      for (const Separator& m : r) {
        ra0 = std::min(ra0, m.a0);
        ra1 = std::max(ra1, m.a1);
      }
      for (const SlotColumn& column : columns) {
        const Vec2 cg = to_g * column.center;
        const double half_across = column.size(1 - orient) / 2.0;
        if (cg(orient) < ra0 - 1.0 || cg(orient) > ra1 + 1.0) continue;
        for (double face : {cg(1 - orient) - half_across,
                            cg(1 - orient) + half_across}) {
          bool apart = true;
          for (const Separator& m : r) apart &= std::abs(face - m.c) > 0.1;
          if (apart) {
            Separator v;
            v.c = face;
            v.a0 = ra0;
            v.a1 = ra1;
            v.is_virtual = true;
            r.push_back(v);
          }
        }
      }
    }
    std::vector<double> gaps_all;
    for (const auto& [r, unused] : rows) {
      std::vector<double> cs;
      for (const Separator& m : r) {
        if (!m.is_virtual) cs.push_back(m.c);
      }
      std::sort(cs.begin(), cs.end());
      for (size_t k = 1; k < cs.size(); ++k) {
        const double g = cs[k] - cs[k - 1];
        if (g >= wmin && g <= wmax) gaps_all.push_back(g);
      }
    }
    const double w_global = gaps_all.empty() ? 2.5 : Median(gaps_all);
    // Slot depth from the rows whose two ends were both measured.
    std::vector<double> measured_depths;
    for (const auto& [r, unused] : rows) {
      Row painted;
      for (const Separator& m : r) {
        if (!m.is_virtual) painted.push_back(m);
      }
      if (painted.empty()) continue;
      const RowEnds e = EndsOf(painted);
      if (e.ok0 && e.ok1 && e.a1 - e.a0 >= dmin && e.a1 - e.a0 <= dmax) {
        measured_depths.push_back(e.a1 - e.a0);
      }
    }
    const double typical_depth = measured_depths.empty()
                                     ? config.typical_depth
                                     : Median(measured_depths);
    for (auto& [row, row_yaw] : rows) {
      const Eigen::Matrix2d to_g = Rot2(-row_yaw);
      const Eigen::Matrix2d to_w = Rot2(row_yaw);
      std::vector<Vec2> path_g;
      for (const Vec2& p : path_xy) path_g.push_back(to_g * p);
      Row r = row;
      std::stable_sort(r.begin(), r.end(),
                       [](const Separator& a, const Separator& b) {
                         return a.c < b.c;
                       });
      Row painted;
      for (const Separator& m : r) {
        if (!m.is_virtual) painted.push_back(m);
      }
      if (r.size() < 2 || painted.empty()) continue;
      const RowEnds ends = EndsOf(painted);
      double a0 = ends.a0;
      double a1 = ends.a1;
      if (a1 - a0 > dmax) continue;
      if (ends.ok0 != ends.ok1 && std::abs(a1 - a0 - typical_depth) < 0.4) {
        // One end measured, the other hidden, the paint about as long as a
        // slot: the hidden end at the measured slot depth.
        if (ends.ok0) {
          a1 = a0 + typical_depth;
        } else {
          a0 = a1 - typical_depth;
        }
      } else if (a1 - a0 < typical_depth - 0.25) {
        // One end hidden: keep the measured end and extend the other;
        // without measured ends extend away from the driven aisle.
        bool near_low;
        if (ends.ok0 != ends.ok1) {
          near_low = ends.ok0;
        } else {
          std::vector<double> cs;
          for (const Separator& m : r) cs.push_back(m.c);
          const double c_mid = Median(cs);
          near_low = path_g.empty() ||
                     MinDistance(path_g, PointAt(c_mid, a0, orient)) <=
                         MinDistance(path_g, PointAt(c_mid, a1, orient));
        }
        if (near_low) {
          a1 = a0 + typical_depth;
        } else {
          a0 = a1 - typical_depth;
        }
      }
      if (a1 - a0 < dmin || a1 - a0 > dmax) continue;
      std::vector<double> ok;
      for (size_t k = 1; k < painted.size(); ++k) {
        const double g = painted[k].c - painted[k - 1].c;
        if (g >= wmin && g <= wmax) ok.push_back(g);
      }
      const double w_typ = ok.empty() ? w_global : Median(ok);
      // Paint contrast of this row's separators, for the hidden ones.
      std::vector<double> ridges;
      if (have_counts && !rasters.intensity.empty()) {
        for (const Separator& m : painted) {
          const double ridge =
              SeparatorEvidence(m.c, std::max(m.a0, a0), std::min(m.a1, a1),
                                orient, to_w, rasters, config.occupied_height)
                  .second;
          if (std::isfinite(ridge)) ridges.push_back(ridge);
        }
      }
      const double paint = ridges.empty() ? kNaN : Median(ridges);
      const std::string row_key =
          StringPrintf("%d:%.1f:%.4f", orient, 0.5 * (r[0].a0 + r[0].a1),
                       row_yaw);
      for (size_t k = 0; k + 1 < r.size(); ++k) {
        const double c0 = r[k].c;
        const double g = r[k + 1].c - c0;
        if (r[k].is_virtual && r[k + 1].is_virtual) continue;
        std::vector<std::pair<double, double>> pieces;
        bool inferred = false;
        if (g >= wmin && g <= wmax) {
          pieces.emplace_back(c0, c0 + g);
        } else if (g > wmax) {
          const int m = static_cast<int>(std::nearbyint(g / w_typ));
          if (m < 2 || std::abs(g - m * w_typ) > config.infer_tolerance * m) {
            continue;
          }
          if (have_counts) {
            bool absent = false;
            for (int i = 1; i < m && !absent; ++i) {
              const auto [visible, ridge] = SeparatorEvidence(
                  c0 + i * g / m, a0, a1, orient, to_w, rasters,
                  config.occupied_height);
              absent = Absent(visible, ridge, paint, config);
            }
            if (absent) continue;  // The missing line would have been seen.
          }
          for (int i = 0; i < m; ++i) {
            pieces.emplace_back(c0 + i * g / m, c0 + (i + 1) * g / m);
          }
          inferred = true;
        } else {
          continue;
        }
        for (const auto& [x0, x1] : pieces) {
          candidates.push_back(
              {orient, x0, x1, a0, a1, inferred, row_key, row_yaw});
        }
      }
    }
  }

  Mask seen;
  if (have_counts) {
    seen.assign(cells, 0);
    for (size_t c = 0; c < cells; ++c) {
      const float h = rasters.height_max[c];
      seen[c] = rasters.floor_count[c] > 0 ||
                (std::isfinite(h) ? h >= config.occupied_height
                                  : 0.0 >= config.occupied_height);
    }
  }
  std::vector<ParkingSlot> slots;
  std::vector<std::string> row_keys;
  for (const Candidate& cand : candidates) {
    const Eigen::Matrix2d to_g = Rot2(-cand.row_yaw);
    const Eigen::Matrix2d to_w = Rot2(cand.row_yaw);
    bool entrance_low = true;
    if (!path_xy.empty()) {
      std::vector<Vec2> path_g;
      for (const Vec2& p : path_xy) path_g.push_back(to_g * p);
      const double mid = 0.5 * (cand.c0 + cand.c1);
      entrance_low =
          MinDistance(path_g, PointAt(mid, cand.a0, cand.orient)) <=
          MinDistance(path_g, PointAt(mid, cand.a1, cand.orient));
    }
    const double c0 = cand.c0;
    const double c1 = cand.c1;
    const double a0 = cand.a0;
    const double a1 = cand.a1;
    std::array<Vec2, 4> corners;
    if (cand.orient == 1) {
      corners = entrance_low
                    ? std::array<Vec2, 4>{Vec2(c0, a0), Vec2(c1, a0),
                                          Vec2(c1, a1), Vec2(c0, a1)}
                    : std::array<Vec2, 4>{Vec2(c1, a1), Vec2(c0, a1),
                                          Vec2(c0, a0), Vec2(c1, a0)};
    } else {
      corners = entrance_low
                    ? std::array<Vec2, 4>{Vec2(a0, c1), Vec2(a0, c0),
                                          Vec2(a1, c0), Vec2(a1, c1)}
                    : std::array<Vec2, 4>{Vec2(a1, c0), Vec2(a1, c1),
                                          Vec2(a0, c1), Vec2(a0, c0)};
    }
    for (Vec2& p : corners) p = to_w * p;
    const std::vector<Vec2> poly(corners.begin(), corners.end());
    bool on_column = false;
    for (const std::vector<Vec2>& box : column_boxes) {
      on_column |=
          ConvexIntersectionArea(poly, box) > config.column_overlap_area;
    }
    if (on_column) continue;
    if (cand.inferred && !path_xy.empty()) {
      const std::vector<Vec2> core = InsetConvex(poly, 0.3);
      bool driven = false;
      if (!core.empty()) {
        for (const Vec2& p : path_xy) {
          if (PointInPolygon(core, p)) {
            driven = true;
            break;
          }
        }
      }
      if (driven) continue;  // The vehicle drove through it: an aisle.
    }
    if (cand.inferred && have_counts) {
      // Evidence for the slot itself: observed floor or a parked car.
      const Mask inside = PolygonCells(poly, frame, false);
      int count = 0;
      int observed = 0;
      for (size_t c = 0; c < cells; ++c) {
        if (!inside[c]) continue;
        ++count;
        observed += seen[c];
      }
      if (count == 0 || static_cast<double>(observed) / count <
                            config.inferred_min_evidence) {
        continue;
      }
    }
    ParkingSlot slot;
    slot.level = level;
    slot.corners = corners;
    slot.width = c1 - c0;
    slot.depth = a1 - a0;
    slot.inferred = cand.inferred;
    slot.confidence = cand.inferred ? 0.6 : 1.0;
    slot.row = cand.row_key;
    slots.push_back(slot);
  }
  // Duplicates (both orientations, overlapping rows): higher confidence first.
  std::stable_sort(slots.begin(), slots.end(),
                   [](const ParkingSlot& a, const ParkingSlot& b) {
                     return a.confidence > b.confidence;
                   });
  std::vector<ParkingSlot> kept;
  for (const ParkingSlot& s : slots) {
    const std::vector<Vec2> p(s.corners.begin(), s.corners.end());
    const double area = std::abs(PolygonArea(p));
    const double share = s.inferred ? config.inferred_max_overlap : 0.5;
    bool apart = true;
    for (const ParkingSlot& k : kept) {
      const std::vector<Vec2> q(k.corners.begin(), k.corners.end());
      if (ConvexIntersectionArea(p, q) >= share * area) {
        apart = false;
        break;
      }
    }
    if (apart) kept.push_back(s);
  }
  // Occupancy at capture time, floor height and ids.
  std::vector<std::string> keys;
  for (const ParkingSlot& s : kept) {
    if (std::find(keys.begin(), keys.end(), s.row) == keys.end()) {
      keys.push_back(s.row);
    }
  }
  auto key_order = [](const std::string& k) {
    const size_t first = k.find(':');
    const size_t second = k.find(':', first + 1);
    return std::make_pair(k.substr(0, first),
                          std::stod(k.substr(first + 1, second - first - 1)));
  };
  std::stable_sort(keys.begin(), keys.end(),
                   [&](const std::string& a, const std::string& b) {
                     return key_order(a) < key_order(b);
                   });
  for (ParkingSlot& s : kept) {
    const std::vector<Vec2> inset =
        InsetConvex(std::vector<Vec2>(s.corners.begin(), s.corners.end()), 0.3);
    if (inset.empty()) continue;
    const Mask inside = PolygonCells(inset, frame, true);
    int count = 0;
    int high = 0;
    std::vector<double> floor;
    for (size_t c = 0; c < cells; ++c) {
      if (!inside[c]) continue;
      ++count;
      const float h = rasters.height_max[c];
      high += (std::isfinite(h) ? h : 0.0f) >= config.occupied_height;
      if (std::isfinite(rasters.floor_z[c])) {
        floor.push_back(rasters.floor_z[c]);
      }
    }
    if (count == 0) continue;
    s.occupied = static_cast<double>(high) / count >= config.occupied_fraction;
    s.z = floor.empty() ? kNaN : Median(floor);
  }
  const Eigen::Matrix2d to_g = Rot2(-yaw);
  for (size_t ki = 0; ki < keys.size(); ++ki) {
    const std::string& key = keys[ki];
    const int orient = key[0] - '0';
    std::vector<size_t> members;
    for (size_t i = 0; i < kept.size(); ++i) {
      if (kept[i].row == key) members.push_back(i);
    }
    std::vector<double> along;
    for (size_t i : members) {
      Vec2 c = Vec2::Zero();
      for (const Vec2& p : kept[i].corners) c += to_g * p;
      along.push_back(orient == 1 ? c.x() / 4.0 : c.y() / 4.0);
    }
    std::vector<size_t> order(members.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return along[a] < along[b]; });
    const std::string row = StringPrintf("R%02zu", ki + 1);
    for (size_t k = 0; k < order.size(); ++k) {
      ParkingSlot& m = kept[members[order[k]]];
      m.row = row;
      m.slot_id = StringPrintf("L%d-%s-%03zu", level, row.c_str(), k + 1);
    }
  }
  int inferred = 0;
  int occupied = 0;
  for (const ParkingSlot& s : kept) {
    inferred += s.inferred;
    occupied += s.occupied == 1;
  }
  LogLine(log, StringPrintf("[slots] level %d: %zu slots (%d inferred, %d "
                            "occupied at capture)",
                            level, kept.size(), inferred, occupied));
  return kept;
}

}  // namespace gr
