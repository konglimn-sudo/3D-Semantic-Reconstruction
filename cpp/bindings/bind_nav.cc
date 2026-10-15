// Bindings of the navigation map support: distance transform, skeleton
// graph, polyline simplification, route graph and grid A*.

#include <vector>

#include "bind_util.h"
#include "gr/nav_map.h"

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

void BindNav(py::module_& m) {
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
