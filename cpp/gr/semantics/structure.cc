#include "gr/semantics/structure.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

#include "gr/core/check.h"
#include "gr/core/morphology.h"
#include "gr/core/numeric.h"
#include "gr/semantics/projector.h"

namespace gr {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

Eigen::Matrix2d Rot2(double angle) {
  Eigen::Matrix2d r;
  r << std::cos(angle), -std::sin(angle), std::sin(angle), std::cos(angle);
  return r;
}

// [begin, end) runs of true flags.
std::vector<std::pair<int, int>> Runs(const std::vector<uint8_t>& flags) {
  std::vector<std::pair<int, int>> out;
  int start = -1;
  for (int k = 0; k < static_cast<int>(flags.size()); ++k) {
    if (flags[static_cast<size_t>(k)] && start < 0) start = k;
    if (!flags[static_cast<size_t>(k)] && start >= 0) {
      out.emplace_back(start, k);
      start = -1;
    }
  }
  if (start >= 0) out.emplace_back(start, static_cast<int>(flags.size()));
  return out;
}

// 1D clustering (gaps above `tol`): the values snapped to their cluster's
// median and the sorted cluster medians.
std::pair<std::vector<double>, std::vector<double>> Snap(
    const std::vector<double>& values, double tol) {
  std::vector<double> snapped = values;
  std::vector<double> lines;
  if (values.empty()) return {snapped, lines};
  std::vector<int> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return values[static_cast<size_t>(a)] < values[static_cast<size_t>(b)];
  });
  std::vector<std::vector<int>> groups(1, {order[0]});
  for (size_t k = 1; k < order.size(); ++k) {
    if (values[static_cast<size_t>(order[k])] -
            values[static_cast<size_t>(order[k - 1])] >
        tol) {
      groups.emplace_back();
    }
    groups.back().push_back(order[k]);
  }
  for (const std::vector<int>& g : groups) {
    std::vector<double> v;
    for (int i : g) v.push_back(values[static_cast<size_t>(i)]);
    const double line = Median(v);
    lines.push_back(line);
    for (int i : g) snapped[static_cast<size_t>(i)] = line;
  }
  return {snapped, lines};
}

struct Box {
  double x0, x1, y0, y1;
};

// Unions boxes closer than `gap` (fragments of one column seen from
// different sides) while the union stays column-sized.
std::vector<Box> MergeBoxes(std::vector<Box> boxes, double gap,
                            double max_size) {
  bool merged = true;
  while (merged) {
    merged = false;
    for (size_t a = 0; a < boxes.size() && !merged; ++a) {
      for (size_t b = a + 1; b < boxes.size(); ++b) {
        const Box& p = boxes[a];
        const Box& q = boxes[b];
        const double dx = std::max(p.x0, q.x0) - std::min(p.x1, q.x1);
        const double dy = std::max(p.y0, q.y0) - std::min(p.y1, q.y1);
        if (dx > gap || dy > gap) continue;
        const Box u{std::min(p.x0, q.x0), std::max(p.x1, q.x1),
                    std::min(p.y0, q.y0), std::max(p.y1, q.y1)};
        if (std::max(u.x1 - u.x0, u.y1 - u.y0) > max_size) continue;
        boxes[a] = u;
        boxes.erase(boxes.begin() + static_cast<long>(b));
        merged = true;
        break;
      }
    }
  }
  return boxes;
}

// Face positions of a column box (grid frame) from the map points on its
// faces: key (axis, side), side -1 for the low face. A point counts for a
// face within `band` of the raster edge, away from the corners, with its
// normal out of that face (map normals face the sensor).
std::map<std::pair<int, int>, double> MeasureFaces(
    const std::vector<Vec2>& xy, const std::vector<Vec2>& normals,
    const Box& box, double band = 0.15, size_t min_points = 15) {
  std::map<std::pair<int, int>, double> faces;
  for (int axis = 0; axis < 2; ++axis) {
    const double lo = axis == 0 ? box.x0 : box.y0;
    const double hi = axis == 0 ? box.x1 : box.y1;
    const double c0 = axis == 0 ? box.y0 : box.x0;
    const double c1 = axis == 0 ? box.y1 : box.x1;
    for (int side : {-1, 1}) {
      const double edge = side < 0 ? lo : hi;
      std::vector<double> on;
      for (size_t k = 0; k < xy.size(); ++k) {
        const double coord = xy[k](axis);
        const double across = xy[k](1 - axis);
        if (across > c0 + 0.05 && across < c1 - 0.05 &&
            std::abs(coord - edge) < band && side * normals[k](axis) > 0.7) {
          on.push_back(coord);
        }
      }
      if (on.size() >= min_points) faces[{axis, side}] = Median(on);
    }
  }
  return faces;
}

std::string AxisName(int k) {
  std::string s;
  k += 1;
  while (k) {
    const int r = (k - 1) % 26;
    k = (k - 1) / 26;
    s.insert(s.begin(), static_cast<char>('A' + r));
  }
  return s;
}

}  // namespace

double DominantYaw(const Points& normals) {
  double c = 0.0;
  double s = 0.0;
  int count = 0;
  for (const Vec3& n : normals) {
    if (std::abs(n.z()) >= 0.2) continue;
    const double a = std::atan2(n.y(), n.x());
    c += std::cos(4.0 * a);
    s += std::sin(4.0 * a);
    ++count;
  }
  if (count < 10) return 0.0;
  return std::atan2(s / count, c / count) / 4.0;
}

StructureResult ExtractStructure(const StructureInput& in,
                                 const StructureConfig& config,
                                 const Logger& log) {
  const size_t n = in.xyz.size();
  GR_CHECK(in.normals.size() == n && in.height.size() == n &&
           in.level.size() == n && in.on_ramp.size() == n);
  GR_CHECK(in.dynamic.empty() || in.dynamic.size() == n);
  GR_CHECK(in.level_z.size() == in.ceiling.size());
  GR_CHECK(config.raster_res > 0.0);
  StructureResult result;
  result.labels.assign(n, 0);
  std::vector<uint8_t>& labels = result.labels;
  auto is_static = [&](size_t i) {
    return in.dynamic.empty() || !in.dynamic[i];
  };
  for (size_t i = 0; i < n; ++i) {
    const double h = in.height[i];
    if (is_static(i) && std::abs(h) < config.floor_tolerance &&
        in.normals[i].z() > 0.8) {
      labels[i] = in.on_ramp[i] ? config.ramp_id : config.floor_id;
    }
  }
  Points static_normals;
  for (size_t i = 0; i < n; ++i) {
    if (is_static(i)) static_normals.push_back(in.normals[i]);
  }
  const double yaw = DominantYaw(static_normals);
  result.yaw = yaw;
  const Eigen::Matrix2d to_grid = Rot2(-yaw);
  const Eigen::Matrix2d to_world = Rot2(yaw);
  std::vector<Vec2> path_g;
  for (const Vec2& p : in.path_xy) path_g.push_back(to_grid * p);
  const double res = config.raster_res;

  for (int lv = 0; lv < static_cast<int>(in.level_z.size()); ++lv) {
    const double ceil_h = in.ceiling[static_cast<size_t>(lv)] -
                          in.level_z[static_cast<size_t>(lv)];
    std::vector<size_t> sel;
    for (size_t i = 0; i < n; ++i) {
      if (is_static(i) && in.level[i] == lv) sel.push_back(i);
    }
    if (sel.empty()) continue;
    const size_t m = sel.size();
    std::vector<double> hs(m);
    std::vector<Vec2> xy(m);
    Vec2 lo = Vec2::Constant(kInf);
    Vec2 hi = Vec2::Constant(-kInf);
    for (size_t k = 0; k < m; ++k) {
      const size_t i = sel[k];
      hs[k] = in.height[i];
      if (hs[k] > ceil_h - config.ceiling_band && in.normals[i].z() < -0.5) {
        labels[i] = config.ceiling_id;
      }
      xy[k] = to_grid * in.xyz[i].head<2>();
      lo = lo.cwiseMin(xy[k]);
      hi = hi.cwiseMax(xy[k]);
    }
    lo -= Vec2::Ones();
    const int width =
        static_cast<int>(std::ceil((hi.x() + 1.0 - lo.x()) / res));
    const int height =
        static_cast<int>(std::ceil((hi.y() + 1.0 - lo.y()) / res));
    const size_t cells = static_cast<size_t>(width) * height;
    std::vector<int> cell(m);
    for (size_t k = 0; k < m; ++k) {
      const int ci = static_cast<int>((xy[k].x() - lo.x()) / res);
      const int cj = static_cast<int>((xy[k].y() - lo.y()) / res);
      cell[k] = cj * width + ci;
    }
    std::vector<double> slices;
    for (double s : config.slice_heights) {
      if (s < ceil_h - 0.3) slices.push_back(s);
    }
    if (slices.empty()) continue;
    std::vector<Mask> occ;
    for (double s : slices) {
      Mask g(cells, 0);
      for (size_t k = 0; k < m; ++k) {
        if (std::abs(hs[k] - s) < config.slice_thickness / 2.0) {
          g[static_cast<size_t>(cell[k])] = 1;
        }
      }
      occ.push_back(DilateCross(g, width, height));
    }
    // Height reached per cell, the ceiling itself left out (it would make
    // every cell look tall); dilated like the slices.
    std::vector<double> hmax(cells, -kInf);
    for (size_t k = 0; k < m; ++k) {
      if (hs[k] < ceil_h - 0.3 && labels[sel[k]] != config.ceiling_id) {
        double& v = hmax[static_cast<size_t>(cell[k])];
        v = std::max(v, hs[k]);
      }
    }
    hmax = MaxFilter3x3(hmax, width, height);
    Mask persistent(cells, 0);
    Mask any_occ(cells, 0);
    for (size_t c = 0; c < cells; ++c) {
      int count = 0;
      for (const Mask& o : occ) count += o[c];
      persistent[c] =
          count >= static_cast<int>(slices.size()) - 1 && occ.back()[c];
      any_occ[c] = count > 0;
    }

    // Walls: straight runs (>= 1 m along a grid axis), judged per run: thin,
    // and either reaching the ceiling or longer than a car. Columns embedded
    // in a wall are its thickness bulges.
    Mask wall_cells(cells, 0);
    std::vector<std::vector<int>> bulge_columns;
    // An odd length keeps the opening symmetric: with an even one the
    // anchor sits off-centre and the opening drops the last cell of every
    // run, which left a column-sized remnant at free wall ends.
    const int run_len =
        std::max(3, static_cast<int>(std::lround(1.0 / res))) | 1;
    for (int axis = 0; axis < 2; ++axis) {
      const Mask runs = OpenLine(any_occ, width, height, run_len, axis == 0);
      const Components comps = Label8(runs, width, height);
      for (const std::vector<int>& comp : comps.cells) {
        std::vector<int> along(comp.size());
        std::vector<int> across(comp.size());
        for (size_t k = 0; k < comp.size(); ++k) {
          const int ci = comp[k] % width;
          const int cj = comp[k] / width;
          along[k] = axis == 0 ? ci : cj;
          across[k] = axis == 0 ? cj : ci;
        }
        const int amin = *std::min_element(along.begin(), along.end());
        const int amax = *std::max_element(along.begin(), along.end());
        const double length = (amax - amin + 1) * res;
        std::vector<int> prof(static_cast<size_t>(amax - amin + 1), 0);
        for (int a : along) ++prof[static_cast<size_t>(a - amin)];
        std::vector<double> nonzero;
        for (int p : prof) {
          if (p > 0) nonzero.push_back(p);
        }
        const double thick = Median(nonzero) * res;
        if (length < config.wall_min_length ||
            thick > config.wall_max_thickness) {
          continue;
        }
        int tall = 0;
        for (int c : comp) tall += hmax[static_cast<size_t>(c)] >= ceil_h - 0.8;
        if (2 * tall < static_cast<int>(comp.size()) &&
            length < config.parapet_min_length) {
          continue;  // E.g. the side of a parked car.
        }
        for (int c : comp) wall_cells[static_cast<size_t>(c)] = 1;
        std::vector<double> across_d(across.begin(), across.end());
        const double mid = (Median(across_d) + 0.5) * res +
                           (axis == 0 ? lo.y() : lo.x());
        const double offset = axis == 0 ? lo.x() : lo.y();
        const double a0 = amin * res + offset;
        const double a1 = (amax + 1) * res + offset;
        const Vec2 p0 = axis == 0 ? Vec2(a0, mid) : Vec2(mid, a0);
        const Vec2 p1 = axis == 0 ? Vec2(a1, mid) : Vec2(mid, a1);
        result.walls.push_back({lv, to_world * p0, to_world * p1, thick});
        std::vector<int> p_cells;
        std::vector<int> p_along;
        for (size_t k = 0; k < comp.size(); ++k) {
          if (persistent[static_cast<size_t>(comp[k])]) {
            p_cells.push_back(comp[k]);
            p_along.push_back(along[k] - amin);
          }
        }
        if (p_cells.empty()) continue;
        std::vector<int> prof_p(prof.size(), 0);
        for (int a : p_along) ++prof_p[static_cast<size_t>(a)];
        std::vector<uint8_t> bulge(prof.size());
        for (size_t a = 0; a < prof.size(); ++a) {
          bulge[a] = prof_p[a] * res >= thick + 0.2;
        }
        for (const auto& [a, b] : Runs(bulge)) {
          const double span = (b - a) * res;
          if (span < config.column_min || span > config.column_max) continue;
          std::vector<int> column;
          for (size_t k = 0; k < p_cells.size(); ++k) {
            if (p_along[k] >= a && p_along[k] < b) column.push_back(p_cells[k]);
          }
          bulge_columns.push_back(std::move(column));
        }
      }
    }

    // Columns: compact persistent components (+ the bulges of walls).
    Mask free_persistent(cells, 0);
    for (size_t c = 0; c < cells; ++c) {
      free_persistent[c] = persistent[c] && !wall_cells[c];
    }
    std::vector<std::vector<int>> candidates =
        Label8(free_persistent, width, height).cells;
    candidates.insert(candidates.end(), bulge_columns.begin(),
                      bulge_columns.end());
    std::vector<Box> boxes;
    for (const std::vector<int>& cand : candidates) {
      int imin = width;
      int imax = -1;
      int jmin = height;
      int jmax = -1;
      for (int c : cand) {
        imin = std::min(imin, c % width);
        imax = std::max(imax, c % width);
        jmin = std::min(jmin, c / width);
        jmax = std::max(jmax, c / width);
      }
      // The slices were dilated by one cell: shrink the extent back.
      const Box box{lo.x() + (imin + 1) * res, lo.x() + imax * res,
                    lo.y() + (jmin + 1) * res, lo.y() + jmax * res};
      const double w = box.x1 - box.x0;
      const double h = box.y1 - box.y0;
      if (std::max(w, h) > config.column_max ||
          std::min(w, h) < config.column_min * 0.5) {
        continue;
      }
      boxes.push_back(box);
    }
    if (boxes.empty()) continue;
    boxes = MergeBoxes(boxes, 0.3, config.column_max);
    // Faces measured on the map points of the column height band.
    std::vector<Vec2> face_xy;
    std::vector<Vec2> face_normals;
    Points face_points;
    const double face_top = std::min(2.0, ceil_h - 0.5);
    for (size_t k = 0; k < m; ++k) {
      if (hs[k] > 0.3 && hs[k] < face_top) {
        face_xy.push_back(xy[k]);
        face_normals.push_back(to_grid * in.normals[sel[k]].head<2>());
        face_points.emplace_back(xy[k].x(), xy[k].y(), 0.0);
      }
    }
    const PointCloudIndex face_index(face_points);
    std::vector<std::map<std::pair<int, int>, double>> measured;
    for (const Box& box : boxes) {
      if (!config.measure_faces || face_xy.empty()) {
        measured.emplace_back();
        continue;
      }
      const Vec2 c(0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1));
      const double r =
          0.5 * std::hypot(box.x1 - box.x0, box.y1 - box.y0) + 0.3;
      std::vector<Vec2> near_xy;
      std::vector<Vec2> near_normals;
      for (int k : face_index.RadiusSearchXy(c, r)) {
        near_xy.push_back(face_xy[static_cast<size_t>(k)]);
        near_normals.push_back(face_normals[static_cast<size_t>(k)]);
      }
      measured.push_back(MeasureFaces(near_xy, near_normals, box));
    }
    std::vector<double> both;
    for (const auto& faces : measured) {
      for (int axis = 0; axis < 2; ++axis) {
        const auto low = faces.find({axis, -1});
        const auto high = faces.find({axis, 1});
        if (low != faces.end() && high != faces.end()) {
          both.push_back(high->second - low->second);
        }
      }
    }
    std::vector<double> raster_sizes;
    for (const Box& b : boxes) {
      raster_sizes.push_back(std::max(b.x1 - b.x0, b.y1 - b.y0));
    }
    const double typical = both.empty() ? Median(raster_sizes) : Median(both);
    std::vector<double> cx;
    std::vector<double> cy;
    std::vector<Vec2> sizes;
    std::vector<uint8_t> completed;
    for (size_t b = 0; b < boxes.size(); ++b) {
      const Box& box = boxes[b];
      const auto& faces = measured[b];
      const Vec2 c(0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1));
      Vec2 near = c;
      double best = kInf;
      for (const Vec2& p : path_g) {
        const double d = (p - c).norm();
        if (d < best) {
          best = d;
          near = p;
        }
      }
      bool done = false;
      double extent[2][2] = {{box.x0, box.x1}, {box.y0, box.y1}};
      for (int axis = 0; axis < 2; ++axis) {
        const auto low = faces.find({axis, -1});
        const auto high = faces.find({axis, 1});
        if (low != faces.end() && high != faces.end()) {
          extent[axis][0] = low->second;
          extent[axis][1] = high->second;
        } else if (low != faces.end()) {  // Far face unseen: typical size.
          done = true;
          extent[axis][0] = low->second;
          extent[axis][1] = low->second + typical;
        } else if (high != faces.end()) {
          done = true;
          extent[axis][0] = high->second - typical;
          extent[axis][1] = high->second;
        } else if (extent[axis][1] - extent[axis][0] < typical - 0.08) {
          // No face and too small: extend away from the path.
          done = true;
          const double a0 = extent[axis][0];
          const double a1 = extent[axis][1];
          if (near(axis) < c(axis)) {
            extent[axis][1] = a0 + typical;
          } else {
            extent[axis][0] = a1 - typical;
          }
        }
      }
      cx.push_back(0.5 * (extent[0][0] + extent[0][1]));
      cy.push_back(0.5 * (extent[1][0] + extent[1][1]));
      sizes.emplace_back(extent[0][1] - extent[0][0],
                         extent[1][1] - extent[1][0]);
      completed.push_back(done);
    }
    // Grid lines for the axis labels; measured centres stay, raster centres
    // snap to the lines.
    auto [sx, xlines] = Snap(cx, config.grid_snap_tol);
    auto [sy, ylines] = Snap(cy, config.grid_snap_tol);
    if (config.measure_faces) {
      sx = cx;
      sy = cy;
    }
    const size_t first_column = result.columns.size();
    for (size_t k = 0; k < sx.size(); ++k) {
      auto nearest_line = [](const std::vector<double>& lines, double v) {
        size_t best = 0;
        for (size_t l = 1; l < lines.size(); ++l) {
          if (std::abs(lines[l] - v) < std::abs(lines[best] - v)) best = l;
        }
        return static_cast<int>(best);
      };
      const int ax = nearest_line(xlines, sx[k]);
      const int ay = nearest_line(ylines, sy[k]);
      StructureColumn column;
      column.axis_label = AxisName(ay) + std::to_string(ax + 1);
      column.id = "L" + std::to_string(lv) + "-" + column.axis_label;
      column.level = lv;
      column.center = to_world * Vec2(sx[k], sy[k]);
      column.size = sizes[k];
      column.yaw = yaw;
      column.z0 = in.level_z[static_cast<size_t>(lv)];
      column.z1 = in.ceiling[static_cast<size_t>(lv)];
      column.completed = completed[k];
      result.columns.push_back(column);
    }
    // Point labels: columns first, then walls, then MEP below the ceiling.
    for (size_t c = first_column; c < result.columns.size(); ++c) {
      const StructureColumn& column = result.columns[c];
      const Vec2 cg = to_grid * column.center;
      const Vec2 half = column.size / 2.0 + Vec2::Constant(0.05);
      for (size_t k = 0; k < m; ++k) {
        if (labels[sel[k]] != 0 || hs[k] <= 0.02) continue;
        const Vec2 d = (xy[k] - cg).cwiseAbs();
        if (d.x() <= half.x() && d.y() <= half.y()) {
          labels[sel[k]] = config.column_id;
        }
      }
    }
    for (size_t k = 0; k < m; ++k) {
      if (labels[sel[k]] == 0 && wall_cells[static_cast<size_t>(cell[k])] &&
          hs[k] > 0.02 && std::abs(in.normals[sel[k]].z()) < 0.5) {
        labels[sel[k]] = config.wall_id;
      }
    }
    const double mep_lo = std::max(2.2, ceil_h - config.mep_band_high);
    for (size_t k = 0; k < m; ++k) {
      if (labels[sel[k]] == 0 && hs[k] > mep_lo &&
          hs[k] < ceil_h - config.mep_band_low) {
        labels[sel[k]] = config.mep_id;
      }
    }
  }
  LogLine(log, StringPrintf("[structure] grid yaw %.1f deg, %zu columns, %zu "
                            "wall segments",
                            yaw * 180.0 / kPi, result.columns.size(),
                            result.walls.size()));
  return result;
}

}  // namespace gr
