// Bindings of the navigation map support: distance transform, skeleton
// graph, polyline simplification, route graph and grid A*.

#include <array>
#include <string>
#include <vector>

#include "bind_util.h"
#include "gr/nav/arrows.h"
#include "gr/nav/features.h"
#include "gr/nav/lanes.h"
#include "gr/nav/layers.h"
#include "gr/nav/nav_map.h"
#include "gr/nav/traffic.h"

namespace gr::bindings {
namespace {

namespace py = ::pybind11;
using ::pybind11::literals::operator""_a;

using U8Image = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using F32Image = py::array_t<float, py::array::c_style | py::array::forcecast>;

// An (H, W) array as a flat vector; sets `width` and `height`.
template <typename T, typename Array>
std::vector<T> Flat(const py::object& image, int& width, int& height,
                    const char* name) {
  const Array array = Array::ensure(image);
  Require(array && array.ndim() == 2, std::string(name) + " must be HxW");
  height = static_cast<int>(array.shape(0));
  width = static_cast<int>(array.shape(1));
  return std::vector<T>(array.data(), array.data() + array.size());
}

// Fields of the Python tuples the bindings take.
constexpr size_t kTrafficEdgeFields = 5;  // (a, b, level, drivable, points)
constexpr size_t kArrowFields = 3;        // (level, center, direction)
constexpr size_t kPathFields = 2;         // (level, points)
constexpr size_t kSignFields = 4;         // (level, xy, kind, facing)
constexpr size_t kLaneSlotFields = 4;     // (row, corners, z, width)
constexpr size_t kWallFields = 3;         // (p0, p1, thickness)
constexpr size_t kLayerSlotFields = 2;    // (index, corners)
// Sign facings shorter than this have no direction.
constexpr double kMinFacingNorm = 1e-9;
// cv2.fillPoly takes at most this many fractional bits (XY_SHIFT).
constexpr int kMaxFractionBits = 16;

py::array_t<double> Vec2Array(const std::vector<Vec2>& points) {
  py::array_t<double> out({static_cast<py::ssize_t>(points.size()),
                           py::ssize_t{kPlaneDim}});
  auto rows = out.mutable_unchecked<2>();
  for (size_t i = 0; i < points.size(); ++i) {
    rows(static_cast<py::ssize_t>(i), 0) = points[i].x();
    rows(static_cast<py::ssize_t>(i), 1) = points[i].y();
  }
  return out;
}

}  // namespace

namespace {

std::vector<Vec2> PolylineFrom(const py::handle& points) {
  const RowMatX2d xy =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          points)
          .cast<RowMatX2d>();
  std::vector<Vec2> out;
  for (Eigen::Index k = 0; k < xy.rows(); ++k) {
    out.push_back(xy.row(k).transpose());
  }
  return out;
}

}  // namespace

void BindNav(py::module_& m) {
  py::class_<TrafficConfig>(m, "TrafficConfig")
      .def(py::init<>())
      .def_readwrite("arrow_max_distance", &TrafficConfig::arrow_max_distance)
      .def_readwrite("arrow_max_angle", &TrafficConfig::arrow_max_angle)
      .def_readwrite("center_tolerance", &TrafficConfig::center_tolerance)
      .def_readwrite("min_arrows", &TrafficConfig::min_arrows)
      .def_readwrite("opposite_radius", &TrafficConfig::opposite_radius)
      .def_readwrite("max_turn", &TrafficConfig::max_turn)
      .def_readwrite("pass_max_distance", &TrafficConfig::pass_max_distance)
      .def_readwrite("pass_max_angle", &TrafficConfig::pass_max_angle)
      .def_readwrite("pass_min_cover", &TrafficConfig::pass_min_cover)
      .def_readwrite("pass_min_length", &TrafficConfig::pass_min_length)
      .def_readwrite("min_passes", &TrafficConfig::min_passes)
      .def_readwrite("sign_distance", &TrafficConfig::sign_distance)
      .def_readwrite("sign_lateral", &TrafficConfig::sign_lateral);

  m.def(
      "decide_traffic",
      [](const py::list& edges, const py::list& arrows,
         const TrafficConfig& config, const py::list& paths,
         const py::list& signs) {
        std::vector<TrafficEdge> traffic_edges;
        for (const py::handle& item : edges) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kTrafficEdgeFields,
                  "edges are (a, b, level, drivable, (M, 2) points)");
          traffic_edges.push_back({tuple[0].cast<int>(), tuple[1].cast<int>(),
                                   tuple[2].cast<int>(), tuple[3].cast<bool>(),
                                   PolylineFrom(tuple[4])});
        }
        std::vector<TrafficArrow> traffic_arrows;
        for (const py::handle& item : arrows) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kArrowFields,
                  "arrows are (level, center, direction)");
          traffic_arrows.push_back({tuple[0].cast<int>(), tuple[1].cast<Vec2>(),
                       tuple[2].cast<Vec2>().normalized()});
        }
        std::vector<DrivenPath> driven_paths;
        for (const py::handle& item : paths) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kPathFields,
                  "paths are (level, (N, 2) points)");
          driven_paths.push_back(
              {tuple[0].cast<int>(), PolylineFrom(tuple[1])});
        }
        std::vector<TrafficSign> traffic_signs;
        for (const py::handle& item : signs) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kSignFields,
                  "signs are (level, xy, 'no_entry' | 'one_way', facing)");
          TrafficSign sign;
          sign.level = tuple[0].cast<int>();
          sign.xy = tuple[1].cast<Vec2>();
          const std::string kind = tuple[2].cast<std::string>();
          Require(kind == "no_entry" || kind == "one_way",
                  "sign kinds are no_entry and one_way");
          sign.kind = kind == "no_entry" ? TrafficSign::Kind::kNoEntry
                                         : TrafficSign::Kind::kOneWay;
          if (!tuple[3].is_none()) {
            const Vec2 facing = tuple[3].cast<Vec2>();
            if (facing.norm() > kMinFacingNorm) {
              sign.facing = facing.normalized();
            }
          }
          traffic_signs.push_back(sign);
        }
        const std::vector<ArrowAssignment> assigned =
            AssignArrows(traffic_edges, traffic_arrows, config);
        int count = 0;
        for (const ArrowAssignment& assignment : assigned) {
          count += assignment.edge != ArrowAssignment::kNoEdge;
        }
        static const char* const kVerdict[] = {
            "two-way", "two-way (one lane seen)", "unknown (one arrow)",
            "one-way", "one-way (driven)", "one-way (sign)",
            "two-way (conflicting evidence)"};
        py::list corridors;
        for (const Corridor& corridor :
             DecideTraffic(traffic_edges, traffic_arrows, config, driven_paths,
                           traffic_signs)) {
          py::list steps;
          for (const CorridorStep& step : corridor.steps) {
            steps.append(py::make_tuple(step.edge, step.forward));
          }
          corridors.append(py::dict(
              "steps"_a = steps, "level"_a = corridor.level,
              "length"_a = corridor.length, "arrows"_a = corridor.arrows,
              "passes_along"_a = corridor.passes_along,
              "passes_against"_a = corridor.passes_against,
              "no_entry_signs"_a = corridor.no_entry_signs,
              "one_way_signs"_a = corridor.one_way_signs,
              "verdict"_a = kVerdict[static_cast<int>(corridor.verdict)],
              "travel"_a = corridor.travel));
        }
        return py::make_tuple(count, corridors);
      },
      "edges"_a, "arrows"_a, "config"_a = TrafficConfig(),
      "paths"_a = py::list(), "signs"_a = py::list(),
      "Traffic direction from the floor arrows, the driven paths ([(level, "
      "(N, 2))], time order) and the signs ([(level, xy, kind, facing or "
      "None)]): (arrows on drivable edges, corridors with arrows or signs: "
      "steps (edge, forward), level, length, arrows, passes_along / "
      "_against, no_entry_signs, one_way_signs, verdict, travel).");

  py::class_<LaneConfig>(m, "LaneConfig")
      .def(py::init<>())
      .def_readwrite("max_row_angle", &LaneConfig::max_row_angle)
      .def_readwrite("min_aisle", &LaneConfig::min_aisle)
      .def_readwrite("max_aisle", &LaneConfig::max_aisle)
      .def_readwrite("min_overlap", &LaneConfig::min_overlap)
      .def_readwrite("lane_offset", &LaneConfig::lane_offset)
      .def_readwrite("min_length", &LaneConfig::min_length)
      .def_readwrite("dedupe_distance", &LaneConfig::dedupe_distance)
      .def_readwrite("follow_distance", &LaneConfig::follow_distance)
      .def_readwrite("skip_followed", &LaneConfig::skip_followed)
      .def_readwrite("replace_distance", &LaneConfig::replace_distance)
      .def_readwrite("replace_angle", &LaneConfig::replace_angle)
      .def_readwrite("replace_share", &LaneConfig::replace_share)
      .def_readwrite("wall_offset", &LaneConfig::wall_offset)
      .def_readwrite("min_wall_length", &LaneConfig::min_wall_length)
      .def_readwrite("wall_open_width", &LaneConfig::wall_open_width)
      .def_readwrite("drive_tolerance", &LaneConfig::drive_tolerance)
      .def_readwrite("drive_min_length", &LaneConfig::drive_min_length);

  m.def(
      "plan_lanes",
      [](const py::list& slots, const py::array& navigable, double x0,
         double y0, double res, const py::list& skeleton,
         const LaneConfig& config, const py::list& walls,
         const py::list& drives) {
        Require(navigable.ndim() == 2, "navigable is an (H, W) mask");
        Require(res > 0.0, "res must be positive");
        std::vector<LaneSlot> lane_slots;
        for (const py::handle& item : slots) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kLaneSlotFields,
                  "slots are (row, (4, 2) corners, z, width)");
          LaneSlot slot;
          slot.row = tuple[0].cast<std::string>();
          const std::vector<Vec2> corners = PolylineFrom(tuple[1]);
          Require(corners.size() == kParkingSlotCorners,
                  "a slot has four corners");
          std::copy(corners.begin(), corners.end(), slot.corners.begin());
          slot.z = tuple[2].cast<double>();
          slot.width = tuple[3].cast<double>();
          lane_slots.push_back(slot);
        }
        const auto mask =
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(
                navigable);
        RasterFrame frame{x0, y0, res, static_cast<int>(mask.shape(1)),
                          static_cast<int>(mask.shape(0))};
        std::vector<uint8_t> cells(mask.data(), mask.data() + mask.size());
        std::vector<std::vector<Vec2>> lines;
        for (const py::handle& item : skeleton) {
          lines.push_back(PolylineFrom(item));
        }
        std::vector<LaneWall> lane_walls;
        for (const py::handle& item : walls) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kWallFields, "walls are (p0, p1, thickness)");
          lane_walls.push_back({tuple[0].cast<Vec2>(), tuple[1].cast<Vec2>(),
                       tuple[2].cast<double>()});
        }
        std::vector<std::vector<Vec2>> drive_paths;
        for (const py::handle& item : drives)
          drive_paths.push_back(PolylineFrom(item));
        LanePlan plan;
        {
          py::gil_scoped_release release;
          plan = PlanLanes(lane_slots, cells, frame, lines, config, lane_walls,
                           drive_paths);
        }
        py::list lanes;
        for (const auto& lane : plan.lanes) {
          RowMatX2d ends(kLineEnds, kPlaneDim);
          for (int end = 0; end < kLineEnds; ++end) {
            ends.row(end) = lane[end].transpose();
          }
          lanes.append(ends);
        }
        return py::dict("rows"_a = plan.rows, "walls"_a = plan.walls,
                        "drives"_a = plan.drives,
                        "followed"_a = plan.followed, "lanes"_a = lanes,
                        "source"_a = plan.source);
      },
      "slots"_a, "navigable"_a, "x0"_a, "y0"_a, "res"_a, "skeleton"_a,
      "config"_a = LaneConfig(), "walls"_a = py::list(),
      "drives"_a = py::list(),
      "Lane pieces of one level from its slots (row, corners, z, width), "
      "walls (p0, p1, thickness) and drives ((N, 2) in time order): dict of "
      "rows, walls and drives (that proposed a lane), followed (pieces the "
      "skeleton already runs along), lanes ((2, 2) end points) and source "
      "per lane (0 row, 1 wall, 2 drive).");

  m.def(
      "covered_by_lanes",
      [](const py::list& edges, const py::list& lanes,
         const LaneConfig& config) {
        std::vector<LaneLine> lane_ends;
        for (const py::handle& item : lanes) {
          const std::vector<Vec2> ends = PolylineFrom(item);
          Require(ends.size() == kLineEnds, "a lane is (2, 2)");
          lane_ends.push_back({ends[0], ends[1]});
        }
        std::vector<bool> out;
        for (const py::handle& item : edges) {
          out.push_back(CoveredByLanes(PolylineFrom(item), lane_ends, config));
        }
        return out;
      },
      "edges"_a, "lanes"_a, "config"_a = LaneConfig(),
      "Whether each edge ((M, 2) points) runs along one of the lanes.");

  py::class_<ArrowConfig>(m, "ArrowConfig")
      .def(py::init<>())
      .def_readwrite("min_length", &ArrowConfig::min_length)
      .def_readwrite("max_length", &ArrowConfig::max_length)
      .def_readwrite("min_area", &ArrowConfig::min_area)
      .def_readwrite("max_area", &ArrowConfig::max_area)
      .def_readwrite("max_shaft_width", &ArrowConfig::max_shaft_width)
      .def_readwrite("min_head_width", &ArrowConfig::min_head_width)
      .def_readwrite("min_head_ratio", &ArrowConfig::min_head_ratio)
      .def_readwrite("min_head_offset", &ArrowConfig::min_head_offset)
      .def_readwrite("min_tip_length", &ArrowConfig::min_tip_length)
      .def_readwrite("bin", &ArrowConfig::bin);

  m.def(
      "detect_arrows",
      [](const py::array& mask, double res, double x0, double y0,
         const ArrowConfig& config) {
        Require(mask.ndim() == 2, "mask must be (height, width)");
        Require(res > 0.0 && config.bin > 0.0, "res and bin must be positive");
        const auto flags =
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(
                mask);
        const int height = static_cast<int>(flags.shape(0));
        const int width = static_cast<int>(flags.shape(1));
        std::vector<uint8_t> cells(flags.data(),
                                   flags.data() + flags.size());
        for (uint8_t& cell : cells) cell = cell != 0;
        std::vector<FloorArrow> arrows;
        {
          py::gil_scoped_release release;
          arrows = DetectArrows(cells, width, height, res, x0, y0, config);
        }
        py::list out;
        for (const FloorArrow& arrow : arrows) {
          out.append(py::dict(
              "center"_a = arrow.center, "direction"_a = arrow.direction,
              "tail"_a = arrow.tail, "tip"_a = arrow.tip,
              "length"_a = arrow.length, "shaft_width"_a = arrow.shaft_width,
              "head_width"_a = arrow.head_width, "area"_a = arrow.area,
              "cells"_a = ToArray(arrow.cells)));
        }
        return out;
      },
      "mask"_a, "res"_a, "x0"_a, "y0"_a, "config"_a = ArrowConfig(),
      "Floor arrows in a plan-view paint mask (row j at y0 + (j + 0.5) res): "
      "list of dicts of center, direction, tail, tip, length, shaft_width, "
      "head_width, area and cells (flat mask indices).");

  m.def(
      "distance_transform",
      [](const py::object& sources) {
        int width = 0;
        int height = 0;
        const std::vector<uint8_t> source_cells =
            Flat<uint8_t, U8Image>(sources, width, height, "sources");
        std::vector<float> distance;
        {
          py::gil_scoped_release release;
          distance = DistanceTransform(source_cells, width, height);
        }
        return ToImage(distance, height, width);
      },
      "sources"_a,
      "(H, W) float32: Euclidean distance in cells to the nearest nonzero "
      "cell (+inf without any).");

  m.def(
      "thin_mask",
      [](const py::object& mask) {
        int width = 0;
        int height = 0;
        const std::vector<uint8_t> mask_cells =
            Flat<uint8_t, U8Image>(mask, width, height, "mask");
        std::vector<uint8_t> out;
        {
          py::gil_scoped_release release;
          out = ThinMask(mask_cells, width, height);
        }
        return ToImage(out, height, width);
      },
      "mask"_a, "Zhang-Suen skeleton (H, W) uint8.");

  m.def(
      "skeletonize",
      [](const py::object& mask) {
        int width = 0;
        int height = 0;
        const std::vector<uint8_t> mask_cells =
            Flat<uint8_t, U8Image>(mask, width, height, "mask");
        std::vector<uint8_t> out;
        {
          py::gil_scoped_release release;
          out = Skeletonize(mask_cells, width, height);
        }
        return ToImage(out, height, width);
      },
      "mask"_a,
      "skimage.morphology.skeletonize of a 2D mask (identical): (H, W) "
      "uint8.");

  m.def(
      "fill_polygon",
      [](py::array_t<int32_t, py::array::c_style> image,
         const py::object& polygon, double x0, double y0, double res,
         int value) {
        Require(image.ndim() == 2, "image must be (H, W) int32");
        Require(res > 0.0, "res must be positive");
        const Eigen::MatrixXd vertices = polygon.cast<Eigen::MatrixXd>();
        Require(vertices.cols() >= kPlaneDim, "polygon must be (N, 2+)");
        std::vector<Vec2> corners;
        for (Eigen::Index vertex = 0; vertex < vertices.rows(); ++vertex) {
          corners.emplace_back(vertices(vertex, 0), vertices(vertex, 1));
        }
        Grid2D grid;
        grid.x0 = x0;
        grid.y0 = y0;
        grid.res = res;
        grid.height = static_cast<int>(image.shape(0));
        grid.width = static_cast<int>(image.shape(1));
        std::vector<int32_t> pixels(image.data(), image.data() + image.size());
        FillWorldPolygon(corners, grid, value, &pixels);
        std::copy(pixels.begin(), pixels.end(), image.mutable_data());
      },
      "image"_a, "polygon"_a, "x0"_a, "y0"_a, "res"_a, "value"_a,
      "layers.fill_polygon in place: the cells of a world polygon, as "
      "cv2.fillPoly of its cell coordinates with 4 fractional bits.");

  m.def(
      "fill_holes",
      [](const py::object& known, const py::object& obstacles, double res,
         double max_area) {
        int width = 0;
        int height = 0;
        const std::vector<uint8_t> known_cells =
            Flat<uint8_t, U8Image>(known, width, height, "known");
        int obstacles_width = 0;
        int obstacles_height = 0;
        const std::vector<uint8_t> obstacle_cells = Flat<uint8_t, U8Image>(
            obstacles, obstacles_width, obstacles_height, "obstacles");
        Require(obstacles_width == width && obstacles_height == height,
                "known and obstacles must have the same shape");
        const Mask out = FillHoles(known_cells, obstacle_cells, width, height,
                                   res, max_area);
        py::array_t<bool> image({height, width});
        std::copy(out.begin(), out.end(), image.mutable_data());
        return image;
      },
      "known"_a, "obstacles"_a, "res"_a, "max_area"_a,
      "layers.fill_holes: `known` plus the enclosed unknown patches smaller "
      "than max_area (m^2).");

  py::class_<LayerConfig>(m, "LayerConfig")
      .def(py::init<>())
      .def_readwrite("floor_tolerance", &LayerConfig::floor_tolerance)
      .def_readwrite("obstacle_min_height", &LayerConfig::obstacle_min_height)
      .def_readwrite("obstacle_min_points", &LayerConfig::obstacle_min_points)
      .def_readwrite("min_cell_observations",
                     &LayerConfig::min_cell_observations)
      .def_readwrite("max_headroom", &LayerConfig::max_headroom)
      .def_readwrite("floor_closing", &LayerConfig::floor_closing)
      .def_readwrite("max_hole_area", &LayerConfig::max_hole_area)
      .def_readwrite("vehicle_height", &LayerConfig::vehicle_height)
      .def_readwrite("vehicle_top_margin", &LayerConfig::vehicle_top_margin)
      .def_readwrite("vehicle_width", &LayerConfig::vehicle_width);

  m.def(
      "level_layers",
      [](const py::object& xyz, const py::object& level,
         const py::object& dynamic, const py::object& height,
         const py::object& normals, const py::object& labels,
         const py::object& observations, const std::vector<int>& vehicle,
         const std::vector<int>& low, const std::vector<int>& overhead,
         int unlabeled, const std::vector<double>& z, const py::list& grids,
         const py::list& ramps, int selected_level, const py::list& slots,
         const py::object& path_xy, double x0, double y0, double res,
         int width, int height_cells, const LayerConfig& config) {
        const Points points = PointsFrom(xyz);
        const std::vector<int16_t> levels = VectorOf<int16_t>(level);
        const std::vector<uint8_t> moving = VectorOf<uint8_t>(dynamic);
        const std::vector<float> heights = VectorOf<float>(height);
        const Points normal = PointsFrom(normals);
        const std::vector<uint8_t> label = VectorOf<uint8_t>(labels);
        const std::vector<uint16_t> seen = VectorOf<uint16_t>(observations);
        const size_t count = points.size();
        Require(levels.size() == count && heights.size() == count &&
                    label.size() == count &&
                    (moving.empty() || moving.size() == count) &&
                    (normal.empty() || normal.size() == count) &&
                    (seen.empty() || seen.size() == count),
                "one level, height and label (and dynamic flag, normal, "
                "observation count) per point");
        const LevelModel model = LevelModelFrom(z, grids, ramps);
        RequireIndex(selected_level >= 0 && selected_level < model.num_levels(),
                     "level out of range");
        Require(res > 0.0 && width >= 0 && height_cells >= 0, "invalid grid");
        std::vector<LayerSlot> slot_list;
        for (const py::handle& item : slots) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kLayerSlotFields,
                  "slots are (index, corners) tuples");
          LayerSlot slot;
          slot.index = tuple[0].cast<int>();
          const Eigen::MatrixXd corner_rows = tuple[1].cast<Eigen::MatrixXd>();
          Require(corner_rows.cols() >= kPlaneDim,
                  "slot corners must be (N, 2+)");
          for (Eigen::Index row = 0; row < corner_rows.rows(); ++row) {
            slot.corners.emplace_back(corner_rows(row, 0), corner_rows(row, 1));
          }
          slot_list.push_back(std::move(slot));
        }
        std::vector<Vec2> path;
        if (!path_xy.is_none()) {
          const Eigen::MatrixXd path_rows = path_xy.cast<Eigen::MatrixXd>();
          Require(path_rows.rows() == 0 || path_rows.cols() >= kPlaneDim,
                  "path_xy must be (N, 2+)");
          for (Eigen::Index row = 0; row < path_rows.rows(); ++row) {
            path.emplace_back(path_rows(row, 0), path_rows(row, 1));
          }
        }
        LayerPoints input;
        input.map.xyz = &points;
        input.map.level = &levels;
        input.map.dynamic = &moving;
        input.map.height = &heights;
        input.map.normals = &normal;
        input.labels = &label;
        input.observations = &seen;
        LayerClasses classes;
        classes.vehicle = vehicle;
        classes.low = low;
        classes.overhead = overhead;
        classes.unlabeled = unlabeled;
        Grid2D grid;
        grid.x0 = x0;
        grid.y0 = y0;
        grid.res = res;
        grid.width = width;
        grid.height = height_cells;
        LevelLayers layers;
        {
          py::gil_scoped_release release;
          layers = ComputeLevelLayers(input, classes, model, selected_level,
                                      slot_list, path, grid, config);
        }
        const auto mask = [&](const Mask& layer_mask) {
          py::array_t<bool> image({height_cells, width});
          std::copy(layer_mask.begin(), layer_mask.end(), image.mutable_data());
          return image;
        };
        py::array floor_z =
            layers.floor_z.single_precision
                ? py::array(ToImage(std::vector<float>(layers.floor_z.z.begin(),
                                                       layers.floor_z.z.end()),
                                    height_cells, width))
                : py::array(ToImage(layers.floor_z.z, height_cells, width));
        return py::dict(
            "floor"_a = mask(layers.floor), "driven"_a = mask(layers.driven),
            "structure"_a = mask(layers.structure),
            "overhead"_a = mask(layers.overhead),
            "parked"_a = mask(layers.parked), "low"_a = mask(layers.low),
            "slot"_a = ToImage(layers.slot, height_cells, width),
            "ramp"_a = mask(layers.ramp), "floor_z"_a = floor_z,
            "clearance"_a = ToImage(layers.clearance, height_cells, width),
            "headroom"_a = ToImage(layers.headroom, height_cells, width));
      },
      "xyz"_a, "level"_a, "dynamic"_a, "height"_a, "normals"_a, "labels"_a,
      "observations"_a, "vehicle"_a, "low"_a, "overhead"_a, "unlabeled"_a,
      "z"_a, "grids"_a, "ramps"_a, "lv"_a, "slots"_a, "path_xy"_a, "x0"_a,
      "y0"_a, "res"_a, "width"_a, "height_cells"_a, "config"_a,
      "layers.level_layers: dict of the layer rasters (height, width).");

  m.def(
      "fill_poly",
      [](py::array_t<int32_t, py::array::c_style> image,
         const py::object& vertices, int shift, int value) {
        Require(image.ndim() == 2, "image must be (H, W) int32");
        Require(shift >= 0 && shift <= kMaxFractionBits,
                "shift must be in [0, 16]");
        const auto vertex_rows =
            py::array_t<int64_t, py::array::c_style |
                                     py::array::forcecast>::ensure(vertices);
        Require(vertex_rows && vertex_rows.ndim() == 2 &&
                    vertex_rows.shape(1) == kPlaneDim,
                "vertices must be (N, 2) integers");
        std::vector<std::pair<int64_t, int64_t>> points;
        for (py::ssize_t row = 0; row < vertex_rows.shape(0); ++row) {
          points.emplace_back(vertex_rows.at(row, 0), vertex_rows.at(row, 1));
        }
        const int height = static_cast<int>(image.shape(0));
        const int width = static_cast<int>(image.shape(1));
        std::vector<int32_t> pixels(image.data(), image.data() + image.size());
        FillPoly8(points, shift, width, height, value, &pixels);
        std::copy(pixels.begin(), pixels.end(), image.mutable_data());
      },
      "image"_a, "vertices"_a, "shift"_a, "value"_a,
      "cv2.fillPoly(image, [vertices], value, LINE_8, shift) in place "
      "(identical).");

  m.def(
      "skeleton_graph",
      [](const py::object& skeleton, double min_spur) {
        int width = 0;
        int height = 0;
        const std::vector<uint8_t> skeleton_cells =
            Flat<uint8_t, U8Image>(skeleton, width, height, "skeleton");
        SkeletonGraph graph;
        {
          py::gil_scoped_release release;
          graph = ExtractSkeletonGraph(skeleton_cells, width, height, min_spur);
        }
        py::list edges;
        for (const SkeletonEdge& skeleton_edge : graph.edges) {
          py::dict edge;
          edge["a"] = skeleton_edge.start_node;
          edge["b"] = skeleton_edge.end_node;
          edge["cells"] = Vec2Array(skeleton_edge.cells);
          edge["length"] = skeleton_edge.length;
          edges.append(edge);
        }
        py::dict out;
        out["nodes"] = Vec2Array(graph.nodes);
        out["edges"] = edges;
        return out;
      },
      "skeleton"_a, "min_spur"_a = 0.0,
      "Graph of a skeleton: {nodes (N, 2) (col, row), edges [{a, b, cells "
      "(M, 2), length (cells)}]}.");

  m.def(
      "simplify_polyline",
      [](const Eigen::Ref<const RowMatX2d>& points, double tolerance) {
        std::vector<Vec2> polyline(static_cast<size_t>(points.rows()));
        for (Eigen::Index i = 0; i < points.rows(); ++i) {
          polyline[static_cast<size_t>(i)] = points.row(i).transpose();
        }
        return Vec2Array(SimplifyPolyline(polyline, tolerance));
      },
      "points"_a, "tolerance"_a, "Douglas-Peucker simplification.");

  py::class_<RouteGraph>(m, "RouteGraph")
      .def(py::init([](int num_nodes) {
             Require(num_nodes >= 0, "num_nodes must be non-negative");
             return RouteGraph(num_nodes);
           }),
           "num_nodes"_a)
      .def(
          "add_edge",
          [](RouteGraph& graph, int from, int to, double cost, bool one_way) {
            RequireIndex(from >= 0 && from < graph.num_nodes() && to >= 0 &&
                             to < graph.num_nodes(),
                         "node out of range");
            Require(cost >= 0.0, "cost must be non-negative");
            return graph.AddEdge(from, to, cost, one_way);
          },
          "a"_a, "b"_a, "cost"_a, "one_way"_a = false)
      .def(
          "distances",
          [](const RouteGraph& graph, int source) {
            RequireIndex(source >= 0 && source < graph.num_nodes(),
                         "source out of range");
            return ToArray(graph.Distances(source));
          },
          "source"_a)
      .def(
          "shortest_path",
          [](const RouteGraph& graph, int source, int target) {
            RequireIndex(source >= 0 && source < graph.num_nodes() &&
                             target >= 0 && target < graph.num_nodes(),
                         "node out of range");
            const GraphPath path = graph.ShortestPath(source, target);
            py::dict out;
            out["found"] = path.found;
            out["cost"] = path.cost;
            out["nodes"] = path.nodes;
            out["edges"] = path.edges;
            return out;
          },
          "source"_a, "target"_a)
      .def_property_readonly("num_nodes", &RouteGraph::num_nodes)
      .def_property_readonly("num_edges", &RouteGraph::num_edges);

  m.def(
      "grid_astar",
      [](const py::object& cost, const Eigen::Vector2i& start,
         const Eigen::Vector2i& goal) {
        int width = 0;
        int height = 0;
        const std::vector<float> cell_costs =
            Flat<float, F32Image>(cost, width, height, "cost");
        std::vector<Eigen::Vector2i> path;
        {
          py::gil_scoped_release release;
          path = GridAStar(cell_costs, width, height, start, goal);
        }
        py::array_t<int> out({static_cast<py::ssize_t>(path.size()),
                              py::ssize_t{kPlaneDim}});
        auto rows = out.mutable_unchecked<2>();
        for (size_t i = 0; i < path.size(); ++i) {
          rows(static_cast<py::ssize_t>(i), 0) = path[i].x();
          rows(static_cast<py::ssize_t>(i), 1) = path[i].y();
        }
        return out;
      },
      "cost"_a, "start"_a, "goal"_a,
      "8-connected A* on an (H, W) cost grid (negative or non-finite = "
      "blocked); start / goal as (col, row); returns (K, 2) cells.");

  m.def(
      "object_footprints",
      [](const Eigen::Ref<const Eigen::MatrixXd>& xy, double cell,
         int min_points) {
        Require(xy.rows() == 0 || xy.cols() >= kPlaneDim,
                "expected (N, 2) points");
        Require(cell > 0.0, "cell must be positive");
        std::vector<Vec2> points;
        for (Eigen::Index row = 0; row < xy.rows(); ++row) {
          points.emplace_back(xy(row, 0), xy(row, 1));
        }
        py::list out;
        for (const Footprint& footprint :
             ObjectFootprints(points, cell, min_points)) {
          out.append(py::dict("min"_a = footprint.min, "max"_a = footprint.max,
                              "center"_a = footprint.center,
                              "points"_a = footprint.points));
        }
        return out;
      },
      "xy"_a, "cell"_a = 0.2, "min_points"_a = 10,
      "Footprints (8-connected cells) of labelled object points: dicts of "
      "min, max, center, points; largest first.");

  m.def(
      "edges_at_footprints",
      [](const py::list& footprints, const py::list& edges, double margin) {
        std::vector<Footprint> footprint_boxes;
        for (const py::handle& item : footprints) {
          const auto [min_corner, max_corner] =
              item.cast<std::tuple<Vec2, Vec2>>();
          Footprint footprint;
          footprint.min = min_corner;
          footprint.max = max_corner;
          footprint_boxes.push_back(footprint);
        }
        std::vector<std::vector<Vec2>> edge_lines;
        for (const py::handle& item : edges)
          edge_lines.push_back(PolylineFrom(item));
        py::list out;
        for (const FootprintEdge& hit :
             EdgesAtFootprints(footprint_boxes, edge_lines, margin)) {
          out.append(py::make_tuple(hit.footprint, hit.edge, hit.along,
                                    hit.distance));
        }
        return out;
      },
      "footprints"_a, "edges"_a, "margin"_a,
      "Nearest edge within `margin` of each footprint box ([(min, max)]; "
      "edges as (M, 2+) polylines): [(footprint, edge, along, distance)].");
}

}  // namespace gr::bindings
