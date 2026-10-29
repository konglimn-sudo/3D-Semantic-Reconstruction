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
// No run started yet.
constexpr int kNone = -1;
// Column axis labels count in letters: A .. Z, AA, AB, ...
constexpr int kLetters = 26;
// A cell's centre, in cells from its corner.
constexpr double kCellCentre = 0.5;

// The grid orientation: wall normals (|normal z| below this) folded by the
// grid's quarter-turn symmetry, from at least kMinWallNormals of them.
constexpr double kWallNormalMaxZ = 0.2;
constexpr double kGridSymmetry = 4.0;
constexpr int kMinWallNormals = 10;

// Point labels: floor normals above this z, ceiling normals below this z,
// wall normals within this |z|.
constexpr double kFloorPointMinNormalZ = 0.8;
constexpr double kCeilingPointMaxNormalZ = -0.5;
constexpr double kWallPointMaxNormalZ = 0.5;
// Points at most this high (m) are floor, never column or wall.
constexpr double kMinLabelHeight = 0.02;
// Columns label their points within this margin (m) of their box.
constexpr double kColumnLabelMargin = 0.05;
// MEP (ducts, pipes) hangs at least this high (m).
constexpr double kMinMepHeight = 2.2;

// The level's raster extends this far (m) beyond its points.
constexpr double kRasterMargin = 1.0;
// Slices and object heights stay this far (m) below the ceiling.
constexpr double kCeilingClearance = 0.3;
// A cell is persistent if occupied in all slices but this many, the top one
// included.
constexpr int kMissingSlices = 1;
// Wall runs: at least kWallRunLength (m) and kMinWallRunCells long (an odd
// cell count); a run is tall where its height reaches within
// kTallBelowCeiling (m) of the ceiling.
constexpr double kWallRunLength = 1.0;
constexpr int kMinWallRunCells = 3;
constexpr double kTallBelowCeiling = 0.8;
// A column embedded in a wall is a bulge at least this much (m) thicker.
constexpr double kMinBulge = 0.2;
// The slices were dilated by this many cells.
constexpr int kDilationCells = 1;
// Column boxes: the short side at least this share of column_min; boxes
// this close (m) merge.
constexpr double kMinColumnSideShare = 0.5;
constexpr double kColumnMergeGap = 0.3;
// Column faces are measured on points in this height band (m), below the
// ceiling by at least kFaceCeilingClearance (m), within the box's half
// diagonal plus kFaceSearchMargin (m).
constexpr double kFaceBandBottom = 0.3;
constexpr double kFaceBandTop = 2.0;
constexpr double kFaceCeilingClearance = 0.5;
constexpr double kFaceSearchMargin = 0.3;
// A face: points within kFaceBand (m) of the raster edge, at least
// kFaceCornerMargin (m) from the corners, with normals out of the face
// (component above kFaceMinNormal), at least kMinFacePoints of them.
constexpr double kFaceBand = 0.15;
constexpr double kFaceCornerMargin = 0.05;
constexpr double kFaceMinNormal = 0.7;
constexpr size_t kMinFacePoints = 15;
// Columns smaller than the typical size by more than this (m) are extended.
constexpr double kColumnSizeSlack = 0.08;

Eigen::Matrix2d Rot2(double angle) {
  Eigen::Matrix2d rotation;
  rotation << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  return rotation;
}

// [begin, end) runs of true flags.
std::vector<std::pair<int, int>> Runs(const std::vector<uint8_t>& flags) {
  std::vector<std::pair<int, int>> out;
  int start = kNone;
  for (int k = 0; k < static_cast<int>(flags.size()); ++k) {
    if (flags[static_cast<size_t>(k)] && start == kNone) start = k;
    if (!flags[static_cast<size_t>(k)] && start != kNone) {
      out.emplace_back(start, k);
      start = kNone;
    }
  }
  if (start != kNone) {
    out.emplace_back(start, static_cast<int>(flags.size()));
  }
  return out;
}

// 1D clustering (gaps above `tolerance`): the values snapped to their
// cluster's median and the sorted cluster medians.
std::pair<std::vector<double>, std::vector<double>> Snap(
    const std::vector<double>& values, double tolerance) {
  std::vector<double> snapped = values;
  std::vector<double> lines;
  if (values.empty()) return {snapped, lines};
  std::vector<int> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
    return values[static_cast<size_t>(left)] <
           values[static_cast<size_t>(right)];
  });
  std::vector<std::vector<int>> groups(1, {order[0]});
  for (size_t k = 1; k < order.size(); ++k) {
    if (values[static_cast<size_t>(order[k])] -
            values[static_cast<size_t>(order[k - 1])] >
        tolerance) {
      groups.emplace_back();
    }
    groups.back().push_back(order[k]);
  }
  for (const std::vector<int>& group : groups) {
    std::vector<double> members;
    for (int index : group) {
      members.push_back(values[static_cast<size_t>(index)]);
    }
    const double line = Median(members);
    lines.push_back(line);
    for (int index : group) snapped[static_cast<size_t>(index)] = line;
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
    for (size_t first = 0; first < boxes.size() && !merged; ++first) {
      for (size_t second = first + 1; second < boxes.size(); ++second) {
        const Box& a = boxes[first];
        const Box& b = boxes[second];
        const double dx = std::max(a.x0, b.x0) - std::min(a.x1, b.x1);
        const double dy = std::max(a.y0, b.y0) - std::min(a.y1, b.y1);
        if (dx > gap || dy > gap) continue;
        const Box united{std::min(a.x0, b.x0), std::max(a.x1, b.x1),
                         std::min(a.y0, b.y0), std::max(a.y1, b.y1)};
        if (std::max(united.x1 - united.x0, united.y1 - united.y0) >
            max_size) {
          continue;
        }
        boxes[first] = united;
        boxes.erase(boxes.begin() + static_cast<long>(second));
        merged = true;
        break;
      }
    }
  }
  return boxes;
}

// Face positions of a column box (grid frame) from the map points on its
// faces: key (axis, side), side -1 for the low face. A point counts for a
// face near the raster edge, away from the corners, with its normal out of
// that face (map normals face the sensor).
std::map<std::pair<int, int>, double> MeasureFaces(
    const std::vector<Vec2>& xy, const std::vector<Vec2>& normals,
    const Box& box) {
  std::map<std::pair<int, int>, double> faces;
  for (int axis = 0; axis < kPlaneDim; ++axis) {
    const double lo = axis == 0 ? box.x0 : box.y0;
    const double hi = axis == 0 ? box.x1 : box.y1;
    const double across_lo = axis == 0 ? box.y0 : box.x0;
    const double across_hi = axis == 0 ? box.y1 : box.x1;
    for (int side : {-1, 1}) {
      const double edge = side < 0 ? lo : hi;
      std::vector<double> on_face;
      for (size_t k = 0; k < xy.size(); ++k) {
        const double coordinate = xy[k](axis);
        const double across = xy[k](1 - axis);
        if (across > across_lo + kFaceCornerMargin &&
            across < across_hi - kFaceCornerMargin &&
            std::abs(coordinate - edge) < kFaceBand &&
            side * normals[k](axis) > kFaceMinNormal) {
          on_face.push_back(coordinate);
        }
      }
      if (on_face.size() >= kMinFacePoints) {
        faces[{axis, side}] = Median(on_face);
      }
    }
  }
  return faces;
}

// "A", "B", ..., "Z", "AA", ... for 0, 1, ...
std::string AxisName(int index) {
  std::string name;
  int remaining = index + 1;
  while (remaining) {
    const int letter = (remaining - 1) % kLetters;
    remaining = (remaining - 1) / kLetters;
    name.insert(name.begin(), static_cast<char>('A' + letter));
  }
  return name;
}

}  // namespace

double DominantYaw(const Points& normals) {
  double cos_sum = 0.0;
  double sin_sum = 0.0;
  int count = 0;
  for (const Vec3& normal : normals) {
    if (std::abs(normal.z()) >= kWallNormalMaxZ) continue;
    const double angle = std::atan2(normal.y(), normal.x());
    cos_sum += std::cos(kGridSymmetry * angle);
    sin_sum += std::sin(kGridSymmetry * angle);
    ++count;
  }
  if (count < kMinWallNormals) return 0.0;
  return std::atan2(sin_sum / count, cos_sum / count) / kGridSymmetry;
}

StructureResult ExtractStructure(const StructureInput& in,
                                 const StructureConfig& config,
                                 const Logger& log) {
  const size_t count = in.xyz.size();
  GR_CHECK(in.normals.size() == count && in.height.size() == count &&
           in.level.size() == count && in.on_ramp.size() == count);
  GR_CHECK(in.dynamic.empty() || in.dynamic.size() == count);
  GR_CHECK(in.level_z.size() == in.ceiling.size());
  GR_CHECK(config.raster_res > 0.0);
  StructureResult result;
  result.labels.assign(count, 0);
  std::vector<uint8_t>& labels = result.labels;
  const auto is_static = [&](size_t i) {
    return in.dynamic.empty() || !in.dynamic[i];
  };
  for (size_t i = 0; i < count; ++i) {
    if (is_static(i) && std::abs(in.height[i]) < config.floor_tolerance &&
        in.normals[i].z() > kFloorPointMinNormalZ) {
      labels[i] = in.on_ramp[i] ? config.ramp_id : config.floor_id;
    }
  }
  Points static_normals;
  for (size_t i = 0; i < count; ++i) {
    if (is_static(i)) static_normals.push_back(in.normals[i]);
  }
  const double yaw = DominantYaw(static_normals);
  result.yaw = yaw;
  const Eigen::Matrix2d to_grid = Rot2(-yaw);
  const Eigen::Matrix2d to_world = Rot2(yaw);
  std::vector<Vec2> path_grid;
  for (const Vec2& sample : in.path_xy) path_grid.push_back(to_grid * sample);
  const double res = config.raster_res;

  for (int level = 0; level < static_cast<int>(in.level_z.size()); ++level) {
    const double ceiling_height = in.ceiling[static_cast<size_t>(level)] -
                                  in.level_z[static_cast<size_t>(level)];
    std::vector<size_t> selected;
    for (size_t i = 0; i < count; ++i) {
      if (is_static(i) && in.level[i] == level) selected.push_back(i);
    }
    if (selected.empty()) continue;
    const size_t selected_count = selected.size();
    std::vector<double> heights(selected_count);
    std::vector<Vec2> xy(selected_count);
    Vec2 lo = Vec2::Constant(kInf);
    Vec2 hi = Vec2::Constant(-kInf);
    for (size_t k = 0; k < selected_count; ++k) {
      const size_t i = selected[k];
      heights[k] = in.height[i];
      if (heights[k] > ceiling_height - config.ceiling_band &&
          in.normals[i].z() < kCeilingPointMaxNormalZ) {
        labels[i] = config.ceiling_id;
      }
      xy[k] = to_grid * in.xyz[i].head<2>();
      lo = lo.cwiseMin(xy[k]);
      hi = hi.cwiseMax(xy[k]);
    }
    lo -= Vec2::Constant(kRasterMargin);
    const int width = static_cast<int>(
        std::ceil((hi.x() + kRasterMargin - lo.x()) / res));
    const int height = static_cast<int>(
        std::ceil((hi.y() + kRasterMargin - lo.y()) / res));
    const size_t cells = static_cast<size_t>(width) * height;
    std::vector<int> cell_of(selected_count);
    for (size_t k = 0; k < selected_count; ++k) {
      const int column = static_cast<int>((xy[k].x() - lo.x()) / res);
      const int row = static_cast<int>((xy[k].y() - lo.y()) / res);
      cell_of[k] = row * width + column;
    }
    std::vector<double> slices;
    for (double slice : config.slice_heights) {
      if (slice < ceiling_height - kCeilingClearance) slices.push_back(slice);
    }
    if (slices.empty()) continue;
    std::vector<Mask> occupancy;
    for (double slice : slices) {
      Mask grid(cells, 0);
      for (size_t k = 0; k < selected_count; ++k) {
        if (std::abs(heights[k] - slice) < config.slice_thickness / 2.0) {
          grid[static_cast<size_t>(cell_of[k])] = 1;
        }
      }
      occupancy.push_back(DilateCross(grid, width, height));
    }
    // Height reached per cell, the ceiling itself left out (it would make
    // every cell look tall); dilated like the slices.
    std::vector<double> max_height(cells, -kInf);
    for (size_t k = 0; k < selected_count; ++k) {
      if (heights[k] < ceiling_height - kCeilingClearance &&
          labels[selected[k]] != config.ceiling_id) {
        double& cell_height = max_height[static_cast<size_t>(cell_of[k])];
        cell_height = std::max(cell_height, heights[k]);
      }
    }
    max_height = MaxFilter3x3(max_height, width, height);
    Mask persistent(cells, 0);
    Mask any_occupied(cells, 0);
    for (size_t cell = 0; cell < cells; ++cell) {
      int occupied_slices = 0;
      for (const Mask& slice : occupancy) occupied_slices += slice[cell];
      persistent[cell] =
          occupied_slices >= static_cast<int>(slices.size()) - kMissingSlices &&
          occupancy.back()[cell];
      any_occupied[cell] = occupied_slices > 0;
    }

    // Walls: straight runs (along a grid axis), judged per run: thin, and
    // either reaching the ceiling or longer than a car. Columns embedded in
    // a wall are its thickness bulges.
    Mask wall_cells(cells, 0);
    std::vector<std::vector<int>> bulge_columns;
    // An odd length keeps the opening symmetric: with an even one the
    // anchor sits off-centre and the opening drops the last cell of every
    // run, which left a column-sized remnant at free wall ends.
    const int run_length =
        std::max(kMinWallRunCells,
                 static_cast<int>(std::lround(kWallRunLength / res))) |
        1;
    for (int axis = 0; axis < kPlaneDim; ++axis) {
      const Mask runs =
          OpenLine(any_occupied, width, height, run_length, axis == 0);
      const Components components = Label8(runs, width, height);
      for (const std::vector<int>& component : components.cells) {
        std::vector<int> along(component.size());
        std::vector<int> across(component.size());
        for (size_t k = 0; k < component.size(); ++k) {
          const int column = component[k] % width;
          const int row = component[k] / width;
          along[k] = axis == 0 ? column : row;
          across[k] = axis == 0 ? row : column;
        }
        const int along_min = *std::min_element(along.begin(), along.end());
        const int along_max = *std::max_element(along.begin(), along.end());
        const double length = (along_max - along_min + 1) * res;
        // Cells across the run at each position along it.
        std::vector<int> profile(
            static_cast<size_t>(along_max - along_min + 1), 0);
        for (int position : along) {
          ++profile[static_cast<size_t>(position - along_min)];
        }
        std::vector<double> nonzero;
        for (int cells_across : profile) {
          if (cells_across > 0) nonzero.push_back(cells_across);
        }
        const double thickness = Median(nonzero) * res;
        if (length < config.wall_min_length ||
            thickness > config.wall_max_thickness) {
          continue;
        }
        int tall = 0;
        for (int cell : component) {
          tall += max_height[static_cast<size_t>(cell)] >=
                  ceiling_height - kTallBelowCeiling;
        }
        // Mostly low and short: e.g. the side of a parked car.
        if (2 * tall < static_cast<int>(component.size()) &&
            length < config.parapet_min_length) {
          continue;
        }
        for (int cell : component) wall_cells[static_cast<size_t>(cell)] = 1;
        std::vector<double> across_cells(across.begin(), across.end());
        const double middle = (Median(across_cells) + kCellCentre) * res +
                              (axis == 0 ? lo.y() : lo.x());
        const double offset = axis == 0 ? lo.x() : lo.y();
        const double start = along_min * res + offset;
        const double end = (along_max + 1) * res + offset;
        const Vec2 p0 = axis == 0 ? Vec2(start, middle) : Vec2(middle, start);
        const Vec2 p1 = axis == 0 ? Vec2(end, middle) : Vec2(middle, end);
        result.walls.push_back(
            {level, to_world * p0, to_world * p1, thickness});
        std::vector<int> persistent_cells;
        std::vector<int> persistent_along;
        for (size_t k = 0; k < component.size(); ++k) {
          if (persistent[static_cast<size_t>(component[k])]) {
            persistent_cells.push_back(component[k]);
            persistent_along.push_back(along[k] - along_min);
          }
        }
        if (persistent_cells.empty()) continue;
        std::vector<int> persistent_profile(profile.size(), 0);
        for (int position : persistent_along) {
          ++persistent_profile[static_cast<size_t>(position)];
        }
        std::vector<uint8_t> bulge(profile.size());
        for (size_t position = 0; position < profile.size(); ++position) {
          bulge[position] =
              persistent_profile[position] * res >= thickness + kMinBulge;
        }
        for (const auto& [begin, run_end] : Runs(bulge)) {
          const double span = (run_end - begin) * res;
          if (span < config.column_min || span > config.column_max) continue;
          std::vector<int> column;
          for (size_t k = 0; k < persistent_cells.size(); ++k) {
            if (persistent_along[k] >= begin && persistent_along[k] < run_end) {
              column.push_back(persistent_cells[k]);
            }
          }
          bulge_columns.push_back(std::move(column));
        }
      }
    }

    // Columns: compact persistent components (+ the bulges of walls).
    Mask free_persistent(cells, 0);
    for (size_t cell = 0; cell < cells; ++cell) {
      free_persistent[cell] = persistent[cell] && !wall_cells[cell];
    }
    std::vector<std::vector<int>> candidates =
        Label8(free_persistent, width, height).cells;
    candidates.insert(candidates.end(), bulge_columns.begin(),
                      bulge_columns.end());
    std::vector<Box> boxes;
    for (const std::vector<int>& candidate : candidates) {
      int column_min = width;
      int column_max = kNone;
      int row_min = height;
      int row_max = kNone;
      for (int cell : candidate) {
        column_min = std::min(column_min, cell % width);
        column_max = std::max(column_max, cell % width);
        row_min = std::min(row_min, cell / width);
        row_max = std::max(row_max, cell / width);
      }
      // The slices were dilated: shrink the extent back.
      const Box box{lo.x() + (column_min + kDilationCells) * res,
                    lo.x() + (column_max + 1 - kDilationCells) * res,
                    lo.y() + (row_min + kDilationCells) * res,
                    lo.y() + (row_max + 1 - kDilationCells) * res};
      const double box_width = box.x1 - box.x0;
      const double box_height = box.y1 - box.y0;
      if (std::max(box_width, box_height) > config.column_max ||
          std::min(box_width, box_height) <
              config.column_min * kMinColumnSideShare) {
        continue;
      }
      boxes.push_back(box);
    }
    if (boxes.empty()) continue;
    boxes = MergeBoxes(boxes, kColumnMergeGap, config.column_max);
    // Faces measured on the map points of the column height band.
    std::vector<Vec2> face_xy;
    std::vector<Vec2> face_normals;
    Points face_points;
    const double face_top =
        std::min(kFaceBandTop, ceiling_height - kFaceCeilingClearance);
    for (size_t k = 0; k < selected_count; ++k) {
      if (heights[k] > kFaceBandBottom && heights[k] < face_top) {
        face_xy.push_back(xy[k]);
        face_normals.push_back(to_grid * in.normals[selected[k]].head<2>());
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
      const Vec2 center(0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1));
      const double radius =
          0.5 * std::hypot(box.x1 - box.x0, box.y1 - box.y0) +
          kFaceSearchMargin;
      std::vector<Vec2> near_xy;
      std::vector<Vec2> near_normals;
      for (int k : face_index.RadiusSearchXy(center, radius)) {
        near_xy.push_back(face_xy[static_cast<size_t>(k)]);
        near_normals.push_back(face_normals[static_cast<size_t>(k)]);
      }
      measured.push_back(MeasureFaces(near_xy, near_normals, box));
    }
    std::vector<double> both_faces;
    for (const auto& faces : measured) {
      for (int axis = 0; axis < kPlaneDim; ++axis) {
        const auto low = faces.find({axis, -1});
        const auto high = faces.find({axis, 1});
        if (low != faces.end() && high != faces.end()) {
          both_faces.push_back(high->second - low->second);
        }
      }
    }
    std::vector<double> raster_sizes;
    for (const Box& box : boxes) {
      raster_sizes.push_back(std::max(box.x1 - box.x0, box.y1 - box.y0));
    }
    const double typical =
        both_faces.empty() ? Median(raster_sizes) : Median(both_faces);
    std::vector<double> center_x;
    std::vector<double> center_y;
    std::vector<Vec2> sizes;
    std::vector<uint8_t> completed;
    for (size_t index = 0; index < boxes.size(); ++index) {
      const Box& box = boxes[index];
      const auto& faces = measured[index];
      const Vec2 center(0.5 * (box.x0 + box.x1), 0.5 * (box.y0 + box.y1));
      Vec2 nearest_path = center;
      double best = kInf;
      for (const Vec2& sample : path_grid) {
        const double distance = (sample - center).norm();
        if (distance < best) {
          best = distance;
          nearest_path = sample;
        }
      }
      bool extended = false;
      double extent[kPlaneDim][2] = {{box.x0, box.x1}, {box.y0, box.y1}};
      for (int axis = 0; axis < kPlaneDim; ++axis) {
        const auto low = faces.find({axis, -1});
        const auto high = faces.find({axis, 1});
        if (low != faces.end() && high != faces.end()) {
          extent[axis][0] = low->second;
          extent[axis][1] = high->second;
        } else if (low != faces.end()) {  // Far face unseen: typical size.
          extended = true;
          extent[axis][0] = low->second;
          extent[axis][1] = low->second + typical;
        } else if (high != faces.end()) {
          extended = true;
          extent[axis][0] = high->second - typical;
          extent[axis][1] = high->second;
        } else if (extent[axis][1] - extent[axis][0] <
                   typical - kColumnSizeSlack) {
          // No face and too small: extend away from the path.
          extended = true;
          const double start = extent[axis][0];
          const double end = extent[axis][1];
          if (nearest_path(axis) < center(axis)) {
            extent[axis][1] = start + typical;
          } else {
            extent[axis][0] = end - typical;
          }
        }
      }
      center_x.push_back(0.5 * (extent[0][0] + extent[0][1]));
      center_y.push_back(0.5 * (extent[1][0] + extent[1][1]));
      sizes.emplace_back(extent[0][1] - extent[0][0],
                         extent[1][1] - extent[1][0]);
      completed.push_back(extended);
    }
    // Grid lines for the axis labels; measured centres stay, raster centres
    // snap to the lines.
    auto [snapped_x, x_lines] = Snap(center_x, config.grid_snap_tol);
    auto [snapped_y, y_lines] = Snap(center_y, config.grid_snap_tol);
    if (config.measure_faces) {
      snapped_x = center_x;
      snapped_y = center_y;
    }
    const size_t first_column = result.columns.size();
    const auto nearest_line = [](const std::vector<double>& lines,
                                 double value) {
      size_t best = 0;
      for (size_t line = 1; line < lines.size(); ++line) {
        if (std::abs(lines[line] - value) < std::abs(lines[best] - value)) {
          best = line;
        }
      }
      return static_cast<int>(best);
    };
    for (size_t k = 0; k < snapped_x.size(); ++k) {
      const int x_line = nearest_line(x_lines, snapped_x[k]);
      const int y_line = nearest_line(y_lines, snapped_y[k]);
      StructureColumn column;
      column.axis_label = AxisName(y_line) + std::to_string(x_line + 1);
      column.id = "L" + std::to_string(level) + "-" + column.axis_label;
      column.level = level;
      column.center = to_world * Vec2(snapped_x[k], snapped_y[k]);
      column.size = sizes[k];
      column.yaw = yaw;
      column.z0 = in.level_z[static_cast<size_t>(level)];
      column.z1 = in.ceiling[static_cast<size_t>(level)];
      column.completed = completed[k];
      result.columns.push_back(column);
    }
    // Point labels: columns first, then walls, then MEP below the ceiling.
    for (size_t index = first_column; index < result.columns.size();
         ++index) {
      const StructureColumn& column = result.columns[index];
      const Vec2 center_grid = to_grid * column.center;
      const Vec2 half = column.size / 2.0 + Vec2::Constant(kColumnLabelMargin);
      for (size_t k = 0; k < selected_count; ++k) {
        if (labels[selected[k]] != 0 || heights[k] <= kMinLabelHeight) {
          continue;
        }
        const Vec2 offset = (xy[k] - center_grid).cwiseAbs();
        if (offset.x() <= half.x() && offset.y() <= half.y()) {
          labels[selected[k]] = config.column_id;
        }
      }
    }
    for (size_t k = 0; k < selected_count; ++k) {
      if (labels[selected[k]] == 0 &&
          wall_cells[static_cast<size_t>(cell_of[k])] &&
          heights[k] > kMinLabelHeight &&
          std::abs(in.normals[selected[k]].z()) < kWallPointMaxNormalZ) {
        labels[selected[k]] = config.wall_id;
      }
    }
    const double mep_low =
        std::max(kMinMepHeight, ceiling_height - config.mep_band_high);
    for (size_t k = 0; k < selected_count; ++k) {
      if (labels[selected[k]] == 0 && heights[k] > mep_low &&
          heights[k] < ceiling_height - config.mep_band_low) {
        labels[selected[k]] = config.mep_id;
      }
    }
  }
  LogLine(log, StringPrintf("[structure] grid yaw %.1f deg, %zu columns, %zu "
                            "wall segments",
                            RadToDeg(yaw), result.columns.size(),
                            result.walls.size()));
  return result;
}

}  // namespace gr
