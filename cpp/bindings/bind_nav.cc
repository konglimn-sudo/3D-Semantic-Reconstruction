// Bindings of the navigation map support: distance transform, skeleton
// graph, polyline simplification, route graph and grid A*.

#include <array>
#include <string>
#include <vector>

#include "bind_util.h"
#include "gr/nav/arrows.h"
#include "gr/nav/lanes.h"
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

py::array_t<double> Vec2Array(const std::vector<Vec2>& points) {
  py::array_t<double> out({static_cast<py::ssize_t>(points.size()),
                           py::ssize_t{2}});
  auto o = out.mutable_unchecked<2>();
  for (size_t i = 0; i < points.size(); ++i) {
    o(static_cast<py::ssize_t>(i), 0) = points[i].x();
    o(static_cast<py::ssize_t>(i), 1) = points[i].y();
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
      .def_readwrite("max_turn", &TrafficConfig::max_turn);

  m.def(
      "decide_traffic",
      [](const py::list& edges, const py::list& arrows,
         const TrafficConfig& config) {
        std::vector<TrafficEdge> e;
        for (const py::handle& item : edges) {
          const py::tuple t = item.cast<py::tuple>();
          Require(t.size() == 5,
                  "edges are (a, b, level, drivable, (M, 2) points)");
          e.push_back({t[0].cast<int>(), t[1].cast<int>(), t[2].cast<int>(),
                       t[3].cast<bool>(), PolylineFrom(t[4])});
        }
        std::vector<TrafficArrow> a;
        for (const py::handle& item : arrows) {
          const py::tuple t = item.cast<py::tuple>();
          Require(t.size() == 3, "arrows are (level, center, direction)");
          a.push_back({t[0].cast<int>(), t[1].cast<Vec2>(),
                       t[2].cast<Vec2>().normalized()});
        }
        const std::vector<ArrowAssignment> assigned =
            AssignArrows(e, a, config);
        int count = 0;
        for (const ArrowAssignment& x : assigned) count += x.edge >= 0;
        static const char* const kVerdict[] = {
            "two-way", "two-way (one lane seen)", "unknown (one arrow)",
            "one-way"};
        py::list corridors;
        for (const Corridor& c : DecideTraffic(e, a, config)) {
          py::list steps;
          for (const CorridorStep& s : c.steps) {
            steps.append(py::make_tuple(s.edge, s.forward));
          }
          corridors.append(py::dict(
              "steps"_a = steps, "level"_a = c.level, "length"_a = c.length,
              "arrows"_a = c.arrows,
              "verdict"_a = kVerdict[static_cast<int>(c.verdict)],
              "travel"_a = c.travel));
        }
        return py::make_tuple(count, corridors);
      },
      "edges"_a, "arrows"_a, "config"_a = TrafficConfig(),
      "Traffic direction from the floor arrows: (arrows on drivable edges, "
      "corridors with arrows: steps (edge, forward), level, length, arrows, "
      "verdict, travel).");

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
      .def_readwrite("replace_share", &LaneConfig::replace_share);

  m.def(
      "plan_lanes",
      [](const py::list& slots, const py::array& navigable, double x0,
         double y0, double res, const py::list& skeleton,
         const LaneConfig& config) {
        Require(navigable.ndim() == 2, "navigable is an (H, W) mask");
        Require(res > 0.0, "res must be positive");
        std::vector<LaneSlot> s;
        for (const py::handle& item : slots) {
          const py::tuple t = item.cast<py::tuple>();
          Require(t.size() == 4, "slots are (row, (4, 2) corners, z, width)");
          LaneSlot slot;
          slot.row = t[0].cast<std::string>();
          const std::vector<Vec2> corners = PolylineFrom(t[1]);
          Require(corners.size() == 4, "a slot has four corners");
          for (int k = 0; k < 4; ++k) slot.corners[k] = corners[k];
          slot.z = t[2].cast<double>();
          slot.width = t[3].cast<double>();
          s.push_back(slot);
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
        LanePlan plan;
        {
          py::gil_scoped_release release;
          plan = PlanLanes(s, cells, frame, lines, config);
        }
        py::list lanes;
        for (const auto& lane : plan.lanes) {
          RowMatX2d xy(2, 2);
          xy.row(0) = lane[0].transpose();
          xy.row(1) = lane[1].transpose();
          lanes.append(xy);
        }
        return py::dict("rows"_a = plan.rows, "followed"_a = plan.followed,
                        "lanes"_a = lanes);
      },
      "slots"_a, "navigable"_a, "x0"_a, "y0"_a, "res"_a, "skeleton"_a,
      "config"_a = LaneConfig(),
      "Lane pieces of one level from its slots (row, corners, z, width): "
      "dict of rows, followed (pieces the skeleton already runs along) and "
      "lanes ((2, 2) end points).");

  m.def(
      "covered_by_lanes",
      [](const py::list& edges, const py::list& lanes,
         const LaneConfig& config) {
        std::vector<std::array<Vec2, 2>> l;
        for (const py::handle& item : lanes) {
          const std::vector<Vec2> p = PolylineFrom(item);
          Require(p.size() == 2, "a lane is (2, 2)");
          l.push_back({p[0], p[1]});
        }
        std::vector<bool> out;
        for (const py::handle& item : edges) {
          out.push_back(CoveredByLanes(PolylineFrom(item), l, config));
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
        for (uint8_t& c : cells) c = c != 0;
        std::vector<FloorArrow> arrows;
        {
          py::gil_scoped_release release;
          arrows = DetectArrows(cells, width, height, res, x0, y0, config);
        }
        py::list out;
        for (const FloorArrow& a : arrows) {
          out.append(py::dict(
              "center"_a = a.center, "direction"_a = a.direction,
              "tail"_a = a.tail, "tip"_a = a.tip, "length"_a = a.length,
              "shaft_width"_a = a.shaft_width, "head_width"_a = a.head_width,
              "area"_a = a.area, "cells"_a = ToArray(a.cells)));
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
        const std::vector<uint8_t> s =
            Flat<uint8_t, U8Image>(sources, width, height, "sources");
        std::vector<float> d;
        {
          py::gil_scoped_release release;
          d = DistanceTransform(s, width, height);
        }
        return ToImage(d, height, width);
      },
      "sources"_a,
      "(H, W) float32: Euclidean distance in cells to the nearest nonzero "
      "cell (+inf without any).");

  m.def(
      "thin_mask",
      [](const py::object& mask) {
        int width = 0;
        int height = 0;
        const std::vector<uint8_t> s =
            Flat<uint8_t, U8Image>(mask, width, height, "mask");
        std::vector<uint8_t> out;
        {
          py::gil_scoped_release release;
          out = ThinMask(s, width, height);
        }
        return ToImage(out, height, width);
      },
      "mask"_a, "Zhang-Suen skeleton (H, W) uint8.");

  m.def(
      "skeleton_graph",
      [](const py::object& skeleton, double min_spur) {
        int width = 0;
        int height = 0;
        const std::vector<uint8_t> s =
            Flat<uint8_t, U8Image>(skeleton, width, height, "skeleton");
        SkeletonGraph graph;
        {
          py::gil_scoped_release release;
          graph = ExtractSkeletonGraph(s, width, height, min_spur);
        }
        py::list edges;
        for (const SkeletonEdge& e : graph.edges) {
          py::dict edge;
          edge["a"] = e.a;
          edge["b"] = e.b;
          edge["cells"] = Vec2Array(e.cells);
          edge["length"] = e.length;
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
        std::vector<Vec2> p(static_cast<size_t>(points.rows()));
        for (Eigen::Index i = 0; i < points.rows(); ++i) {
          p[static_cast<size_t>(i)] = points.row(i).transpose();
        }
        return Vec2Array(SimplifyPolyline(p, tolerance));
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
          [](RouteGraph& graph, int a, int b, double cost, bool one_way) {
            RequireIndex(a >= 0 && a < graph.num_nodes() && b >= 0 &&
                             b < graph.num_nodes(),
                         "node out of range");
            Require(cost >= 0.0, "cost must be non-negative");
            return graph.AddEdge(a, b, cost, one_way);
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
        const std::vector<float> c =
            Flat<float, F32Image>(cost, width, height, "cost");
        std::vector<Eigen::Vector2i> path;
        {
          py::gil_scoped_release release;
          path = GridAStar(c, width, height, start, goal);
        }
        py::array_t<int> out({static_cast<py::ssize_t>(path.size()),
                              py::ssize_t{2}});
        auto o = out.mutable_unchecked<2>();
        for (size_t i = 0; i < path.size(); ++i) {
          o(static_cast<py::ssize_t>(i), 0) = path[i].x();
          o(static_cast<py::ssize_t>(i), 1) = path[i].y();
        }
        return out;
      },
      "cost"_a, "start"_a, "goal"_a,
      "8-connected A* on an (H, W) cost grid (negative or non-finite = "
      "blocked); start / goal as (col, row); returns (K, 2) cells.");
}

}  // namespace gr::bindings
