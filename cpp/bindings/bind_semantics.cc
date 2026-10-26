// Bindings of the semantic rules ported from Python: structure (columns,
// walls, point labels; gr/semantics/structure.h), floor markings and their
// sub-pixel fit (gr/semantics/markings.h, vector_fit.h), parking slots
// (gr/semantics/slots.h).

#include <exception>
#include <map>
#include <string>
#include <vector>

#include "bind_util.h"
#include "gr/semantics/markings.h"
#include "gr/semantics/slots.h"
#include "gr/semantics/structure.h"
#include "gr/semantics/vector_fit.h"

namespace py = pybind11;
using namespace pybind11::literals;  // NOLINT(build/namespaces)

namespace gr::bindings {
namespace {

using DoubleImage =
    py::array_t<double, py::array::c_style | py::array::forcecast>;
using MaskImage =
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;

// Marking segments as dicts (the Python MarkingSegment fields).
MarkingSegment SegmentFrom(const py::handle& item) {
  const py::dict d = item.cast<py::dict>();
  MarkingSegment seg;
  seg.level = d["level"].cast<int>();
  seg.p0 = d["p0"].cast<Vec2>();
  seg.p1 = d["p1"].cast<Vec2>();
  seg.width = d["width"].cast<double>();
  seg.lane = d.contains("cls") &&
             d["cls"].cast<std::string>() == "lane_marking";
  seg.confidence = d.contains("confidence") ? d["confidence"].cast<double>()
                                            : 1.0;
  if (d.contains("end_ok")) {
    const std::vector<bool> ok = d["end_ok"].cast<std::vector<bool>>();
    Require(ok.size() == 2, "end_ok has two flags");
    seg.end_ok[0] = ok[0];
    seg.end_ok[1] = ok[1];
  }
  return seg;
}

std::vector<MarkingSegment> SegmentsFrom(const py::list& items) {
  std::vector<MarkingSegment> out;
  for (const py::handle& item : items) out.push_back(SegmentFrom(item));
  return out;
}

py::list SegmentsTo(const std::vector<MarkingSegment>& segs) {
  py::list out;
  for (const MarkingSegment& s : segs) {
    out.append(py::dict(
        "level"_a = s.level, "p0"_a = s.p0, "p1"_a = s.p1,
        "width"_a = s.width,
        "cls"_a = s.lane ? "lane_marking" : "parking_line",
        "confidence"_a = s.confidence,
        "end_ok"_a = std::vector<bool>{s.end_ok[0], s.end_ok[1]}));
  }
  return out;
}

std::vector<uint8_t> MaskFrom(const py::object& mask, RasterFrame& frame) {
  const MaskImage array = MaskImage::ensure(mask);
  Require(array && array.ndim() == 2, "a mask is an (H, W) array");
  frame.height = static_cast<int>(array.shape(0));
  frame.width = static_cast<int>(array.shape(1));
  std::vector<uint8_t> out(array.data(), array.data() + array.size());
  for (uint8_t& v : out) v = v != 0;
  return out;
}

// (name, brightness (H, W), valid (H, W), x0, y0, res).
FloorImage ImageFrom(const py::handle& item) {
  const py::tuple t = item.cast<py::tuple>();
  Require(t.size() == 6, "images are (name, brightness, valid, x0, y0, res)");
  FloorImage image;
  image.name = t[0].cast<std::string>();
  const DoubleImage brightness = DoubleImage::ensure(t[1]);
  const DoubleImage valid = DoubleImage::ensure(t[2]);
  Require(brightness && valid && brightness.ndim() == 2 &&
              valid.ndim() == 2 && brightness.shape(0) == valid.shape(0) &&
              brightness.shape(1) == valid.shape(1),
          "brightness and valid are (H, W) arrays of one shape");
  image.height = static_cast<int>(brightness.shape(0));
  image.width = static_cast<int>(brightness.shape(1));
  image.brightness.assign(brightness.data(),
                          brightness.data() + brightness.size());
  image.valid.assign(valid.data(), valid.data() + valid.size());
  image.x0 = t[3].cast<double>();
  image.y0 = t[4].cast<double>();
  image.res = t[5].cast<double>();
  Require(image.res > 0.0, "res must be positive");
  return image;
}

using FloatImage =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

// A float raster of the frame's shape (None: empty).
std::vector<float> RasterFrom(const py::object& raster,
                              const RasterFrame& frame, bool optional) {
  if (raster.is_none()) {
    Require(optional, "a raster is required");
    return {};
  }
  const FloatImage array = FloatImage::ensure(raster);
  Require(array && array.ndim() == 2 && array.shape(0) == frame.height &&
              array.shape(1) == frame.width,
          "rasters must be (height, width) of the frame");
  return std::vector<float>(array.data(), array.data() + array.size());
}

}  // namespace

void BindSemantics(py::module_& m) {
  py::class_<SlotConfig>(m, "SlotConfig")
      .def(py::init<>())
      .def_readwrite("width_min", &SlotConfig::width_min)
      .def_readwrite("width_max", &SlotConfig::width_max)
      .def_readwrite("depth_min", &SlotConfig::depth_min)
      .def_readwrite("depth_max", &SlotConfig::depth_max)
      .def_readwrite("min_separator_length",
                     &SlotConfig::min_separator_length)
      .def_readwrite("typical_depth", &SlotConfig::typical_depth)
      .def_readwrite("angle_tol", &SlotConfig::angle_tol)
      .def_readwrite("row_overlap", &SlotConfig::row_overlap)
      .def_readwrite("infer_tolerance", &SlotConfig::infer_tolerance)
      .def_readwrite("occupied_height", &SlotConfig::occupied_height)
      .def_readwrite("occupied_fraction", &SlotConfig::occupied_fraction)
      .def_readwrite("visible_separator_share",
                     &SlotConfig::visible_separator_share)
      .def_readwrite("paint_ridge_share", &SlotConfig::paint_ridge_share)
      .def_readwrite("inferred_max_overlap",
                     &SlotConfig::inferred_max_overlap)
      .def_readwrite("inferred_min_evidence",
                     &SlotConfig::inferred_min_evidence)
      .def_readwrite("column_overlap_area", &SlotConfig::column_overlap_area);

  m.def(
      "assemble_slots",
      [](const py::list& segments, int level, double yaw,
         const py::object& path_xy, const py::list& columns, double x0,
         double y0, double res, const py::object& height_max,
         const py::object& floor_z, const py::object& floor_count,
         const py::object& intensity, const SlotConfig& config,
         const py::object& log) {
        Require(res > 0.0, "res must be positive");
        SlotRasters rasters;
        const FloatImage hm = FloatImage::ensure(height_max);
        Require(hm && hm.ndim() == 2, "height_max is an (H, W) array");
        rasters.frame = RasterFrame{x0, y0, res, static_cast<int>(hm.shape(1)),
                                    static_cast<int>(hm.shape(0))};
        rasters.height_max = RasterFrom(height_max, rasters.frame, false);
        rasters.floor_z = RasterFrom(floor_z, rasters.frame, false);
        rasters.floor_count = RasterFrom(floor_count, rasters.frame, true);
        rasters.intensity = RasterFrom(intensity, rasters.frame, true);
        std::vector<Vec2> path;
        for (const Vec3& p : PointsFrom(path_xy)) path.push_back(p.head<2>());
        std::vector<SlotColumn> cols;
        for (const py::handle& item : columns) {
          const py::tuple t = item.cast<py::tuple>();
          Require(t.size() == 4, "columns are (cx, cy, sx, sy) tuples");
          cols.push_back({Vec2(t[0].cast<double>(), t[1].cast<double>()),
                          Vec2(t[2].cast<double>(), t[3].cast<double>())});
        }
        const std::vector<MarkingSegment> segs = SegmentsFrom(segments);
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        std::vector<ParkingSlot> slots;
        {
          py::gil_scoped_release release;
          slots = AssembleSlots(segs, level, yaw, path, cols, rasters, config,
                                logger);
        }
        RethrowIfSet(log_error);
        py::list out;
        for (const ParkingSlot& s : slots) {
          py::list corners;
          for (const Vec2& c : s.corners) {
            corners.append(py::make_tuple(c.x(), c.y()));
          }
          out.append(py::dict(
              "slot_id"_a = s.slot_id, "row"_a = s.row, "level"_a = s.level,
              "corners"_a = corners, "z"_a = s.z, "width"_a = s.width,
              "depth"_a = s.depth,
              "occupied"_a = s.occupied < 0 ? py::object(py::none())
                                            : py::object(py::bool_(s.occupied)),
              "inferred"_a = s.inferred, "confidence"_a = s.confidence));
        }
        return out;
      },
      "segments"_a, "level"_a, "yaw"_a, "path_xy"_a, "columns"_a, "x0"_a,
      "y0"_a, "res"_a, "height_max"_a, "floor_z"_a, "floor_count"_a,
      "intensity"_a, "config"_a = SlotConfig(), "log"_a = py::none(),
      "Parking slots of one level from its marking segments (dicts), the "
      "driven path ((N, 3)), the columns ((cx, cy, sx, sy)) and the level "
      "rasters (row j at y0 + (j + 0.5) res; floor_count and intensity may "
      "be None).");

  py::class_<MarkingConfig>(m, "MarkingConfig")
      .def(py::init<>())
      .def_readwrite("min_length", &MarkingConfig::min_length)
      .def_readwrite("min_width", &MarkingConfig::min_width)
      .def_readwrite("max_width", &MarkingConfig::max_width)
      .def_readwrite("merge_gap", &MarkingConfig::merge_gap)
      .def_readwrite("merge_angle", &MarkingConfig::merge_angle)
      .def_readwrite("merge_offset", &MarkingConfig::merge_offset)
      .def_readwrite("max_axis_angle", &MarkingConfig::max_axis_angle)
      .def_readwrite("lane_min_length", &MarkingConfig::lane_min_length);

  py::class_<LineFitConfig>(m, "LineFitConfig")
      .def(py::init<>())
      .def_readwrite("station_step", &LineFitConfig::station_step)
      .def_readwrite("sample_step", &LineFitConfig::sample_step)
      .def_readwrite("half_span", &LineFitConfig::half_span)
      .def_readwrite("floor_band_inner", &LineFitConfig::floor_band_inner)
      .def_readwrite("floor_band_outer", &LineFitConfig::floor_band_outer)
      .def_readwrite("end_search", &LineFitConfig::end_search)
      .def_readwrite("end_floor_near", &LineFitConfig::end_floor_near)
      .def_readwrite("end_floor_far", &LineFitConfig::end_floor_far)
      .def_readwrite("end_floor_width", &LineFitConfig::end_floor_width)
      .def_readwrite("min_contrast", &LineFitConfig::min_contrast)
      .def_readwrite("min_valid", &LineFitConfig::min_valid)
      .def_readwrite("max_end_shift", &LineFitConfig::max_end_shift);

  m.def(
      "vectorize_markings",
      [](const py::object& mask, double x0, double y0, double res,
         const MarkingConfig& config, int level,
         const py::object& skeleton) {
        Require(res > 0.0, "res must be positive");
        RasterFrame frame{x0, y0, res, 0, 0};
        const std::vector<uint8_t> cells = MaskFrom(mask, frame);
        std::vector<uint8_t> skel;
        if (!skeleton.is_none()) {
          RasterFrame skeleton_frame = frame;
          skel = MaskFrom(skeleton, skeleton_frame);
          Require(skeleton_frame.width == frame.width &&
                      skeleton_frame.height == frame.height,
                  "skeleton and mask must have the same shape");
        }
        std::vector<MarkingSegment> segs;
        {
          py::gil_scoped_release release;
          segs = VectorizeMarkings(cells, frame, config, level, skel);
        }
        return SegmentsTo(segs);
      },
      "mask"_a, "x0"_a, "y0"_a, "res"_a, "config"_a = MarkingConfig(),
      "level"_a = 0, "skeleton"_a = py::none(),
      "Line segments of a paint mask ((H, W), row j at y0 + (j + 0.5) res): "
      "skeleton branches fitted, collinear pieces merged, filtered. "
      "Without `skeleton` the mask is thinned by Zhang-Suen.");

  m.def(
      "filter_markings",
      [](const py::list& segments, const py::object& floor, double x0,
         double y0, double res, double yaw, double max_axis_angle) {
        RasterFrame frame{x0, y0, res, 0, 0};
        const std::vector<uint8_t> cells = MaskFrom(floor, frame);
        std::vector<MarkingSegment> kept;
        for (const MarkingSegment& s : SegmentsFrom(segments)) {
          if (BothSidesObserved(s, cells, frame) &&
              GridAligned(s, yaw, max_axis_angle)) {
            kept.push_back(s);
          }
        }
        return SegmentsTo(kept);
      },
      "segments"_a, "floor"_a, "x0"_a, "y0"_a, "res"_a, "yaw"_a,
      "max_axis_angle"_a,
      "The segments on observed floor (both sides) that follow the grid.");

  m.def(
      "refine_markings",
      [](const py::list& segments, const py::list& images,
         const LineFitConfig& config) {
        std::vector<MarkingSegment> segs = SegmentsFrom(segments);
        std::vector<FloorImage> floors;
        for (const py::handle& item : images) floors.push_back(ImageFrom(item));
        std::vector<const FloorImage*> order;
        for (const FloorImage& f : floors) order.push_back(&f);
        int refined = 0;
        {
          py::gil_scoped_release release;
          refined = RefineSegments(segs, order, config);
        }
        return py::make_tuple(SegmentsTo(segs), refined);
      },
      "segments"_a, "images"_a, "config"_a = LineFitConfig(),
      "Re-measures the segments on the first image that works: (segments, "
      "refined count). Images are (name, brightness, valid, x0, y0, res).");

  m.def(
      "fit_line",
      [](const Vec2& p0, const Vec2& p1, const py::tuple& image,
         const LineFitConfig& config) -> py::object {
        const FloorImage floor = ImageFrom(image);
        const std::optional<LineFit> fit = FitLine(p0, p1, floor, config);
        if (!fit) return py::none();
        return py::dict("p0"_a = fit->p0, "p1"_a = fit->p1,
                        "width"_a = fit->width,
                        "end_ok"_a = py::make_tuple(fit->end_ok[0],
                                                    fit->end_ok[1]),
                        "stations"_a = fit->stations, "rms"_a = fit->rms);
      },
      "p0"_a, "p1"_a, "image"_a, "config"_a = LineFitConfig(),
      "One painted line re-measured on a floor image (or None).");

  m.def(
      "classify_markings",
      [](const py::list& segments, const py::dict& paths,
         const MarkingConfig& config) {
        std::vector<MarkingSegment> segs = SegmentsFrom(segments);
        std::map<int, std::vector<Vec2>> by_level;
        for (const auto& [key, value] : paths) {
          std::vector<Vec2>& path = by_level[key.cast<int>()];
          for (const Vec3& p : PointsFrom(
                   py::reinterpret_borrow<py::object>(value))) {
            path.push_back(p.head<2>());
          }
        }
        ClassifyMarkings(segs, by_level, config);
        return SegmentsTo(segs);
      },
      "segments"_a, "paths"_a, "config"_a = MarkingConfig(),
      "Lane markings (long, along the driven path of their level) and "
      "parking lines; paths map a level to (N, 3) positions.");

  py::class_<StructureConfig>(m, "StructureConfig")
      .def(py::init<>())
      .def_readwrite("slice_heights", &StructureConfig::slice_heights)
      .def_readwrite("slice_thickness", &StructureConfig::slice_thickness)
      .def_readwrite("raster_res", &StructureConfig::raster_res)
      .def_readwrite("column_min", &StructureConfig::column_min)
      .def_readwrite("column_max", &StructureConfig::column_max)
      .def_readwrite("wall_min_length", &StructureConfig::wall_min_length)
      .def_readwrite("wall_max_thickness",
                     &StructureConfig::wall_max_thickness)
      .def_readwrite("parapet_min_length",
                     &StructureConfig::parapet_min_length)
      .def_readwrite("grid_snap_tol", &StructureConfig::grid_snap_tol)
      .def_readwrite("measure_faces", &StructureConfig::measure_faces)
      .def_readwrite("mep_band_low", &StructureConfig::mep_band_low)
      .def_readwrite("mep_band_high", &StructureConfig::mep_band_high)
      .def_readwrite("floor_tolerance", &StructureConfig::floor_tolerance)
      .def_readwrite("ceiling_band", &StructureConfig::ceiling_band)
      .def_readwrite("floor_id", &StructureConfig::floor_id)
      .def_readwrite("ramp_id", &StructureConfig::ramp_id)
      .def_readwrite("ceiling_id", &StructureConfig::ceiling_id)
      .def_readwrite("column_id", &StructureConfig::column_id)
      .def_readwrite("wall_id", &StructureConfig::wall_id)
      .def_readwrite("mep_id", &StructureConfig::mep_id);

  m.def(
      "extract_structure",
      [](const PointsIn& xyz, const PointsIn& normals, const py::object& height,
         const py::object& level, const py::object& on_ramp,
         const py::object& dynamic, const std::vector<double>& level_z,
         const std::vector<double>& ceiling, const py::object& path_xy,
         const StructureConfig& config, const py::object& log) {
        StructureInput in;
        in.xyz = ToPoints(xyz);
        in.normals = ToPoints(normals);
        const size_t n = in.xyz.size();
        in.height = ToFloatVector(height);
        const std::vector<int> levels = ToIntVector(level);
        in.level.assign(levels.begin(), levels.end());
        for (const int v : ToIntVector(on_ramp)) in.on_ramp.push_back(v != 0);
        if (!dynamic.is_none()) {
          for (const int v : ToIntVector(dynamic)) in.dynamic.push_back(v != 0);
        }
        Require(in.normals.size() == n && in.height.size() == n &&
                    in.level.size() == n && in.on_ramp.size() == n &&
                    (in.dynamic.empty() || in.dynamic.size() == n),
                "per-point arrays must match xyz");
        Require(level_z.size() == ceiling.size(),
                "level_z and ceiling must match");
        Require(config.raster_res > 0.0, "raster_res must be positive");
        for (const Vec3& p : PointsFrom(path_xy)) {
          in.path_xy.push_back(p.head<2>());
        }
        in.level_z = level_z;
        in.ceiling = ceiling;
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        StructureResult result;
        {
          py::gil_scoped_release release;
          result = ExtractStructure(in, config, logger);
        }
        RethrowIfSet(log_error);
        py::list columns;
        for (const StructureColumn& c : result.columns) {
          columns.append(py::dict(
              "id"_a = c.id, "level"_a = c.level, "center"_a = c.center,
              "size"_a = c.size, "yaw"_a = c.yaw, "z0"_a = c.z0, "z1"_a = c.z1,
              "completed"_a = c.completed, "axis_label"_a = c.axis_label));
        }
        py::list walls;
        for (const StructureWall& w : result.walls) {
          walls.append(py::dict("level"_a = w.level, "p0"_a = w.p0,
                                "p1"_a = w.p1, "thickness"_a = w.thickness));
        }
        return py::dict("labels"_a = ToArray(result.labels),
                        "columns"_a = columns, "walls"_a = walls,
                        "yaw"_a = result.yaw);
      },
      "xyz"_a, "normals"_a, "height"_a, "level"_a, "on_ramp"_a, "dynamic"_a,
      "level_z"_a, "ceiling"_a, "path_xy"_a, "config"_a = StructureConfig(),
      "log"_a = py::none(),
      "Columns, walls and structure labels of the point map: dict of labels "
      "(uint8 per point), columns, walls, yaw. `path_xy` is (N, 3) "
      "(z unused).");
}

}  // namespace gr::bindings
