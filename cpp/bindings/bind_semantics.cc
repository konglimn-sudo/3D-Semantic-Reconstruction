// Bindings of the semantic rules ported from Python: structure (columns,
// walls, point labels; gr/semantics/structure.h), floor markings and their
// sub-pixel fit (gr/semantics/markings.h, vector_fit.h), parking slots
// (gr/semantics/slots.h).

#include <exception>
#include <map>
#include <type_traits>
#include <string>
#include <vector>

#include "bind_util.h"
#include "gr/core/image.h"
#include "gr/semantics/markings.h"
#include "gr/semantics/objects4d.h"
#include "gr/semantics/paint.h"
#include "gr/semantics/slot_paint.h"
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

// Fields of the Python tuples the bindings take.
constexpr size_t kFloorImageFields = 6;  // (name, brightness, valid, x0, y0,
                                         //  res)
constexpr size_t kColumnFields = 4;      // (cx, cy, sx, sy)
// Rasters without a channel axis have one channel.
constexpr int kSingleChannel = 1;

// Marking segments as dicts (the Python MarkingSegment fields).
MarkingSegment SegmentFrom(const py::handle& item) {
  const py::dict fields = item.cast<py::dict>();
  MarkingSegment segment;
  segment.level = fields["level"].cast<int>();
  segment.p0 = fields["p0"].cast<Vec2>();
  segment.p1 = fields["p1"].cast<Vec2>();
  segment.width = fields["width"].cast<double>();
  segment.lane = fields.contains("cls") &&
                 fields["cls"].cast<std::string>() == "lane_marking";
  if (fields.contains("confidence")) {
    segment.confidence = fields["confidence"].cast<double>();
  }
  if (fields.contains("end_ok")) {
    const std::vector<bool> end_ok =
        fields["end_ok"].cast<std::vector<bool>>();
    Require(end_ok.size() == kSegmentEnds, "end_ok has two flags");
    for (int end = 0; end < kSegmentEnds; ++end) {
      segment.end_ok[end] = end_ok[end];
    }
  }
  return segment;
}

std::vector<MarkingSegment> SegmentsFrom(const py::list& items) {
  std::vector<MarkingSegment> out;
  for (const py::handle& item : items) out.push_back(SegmentFrom(item));
  return out;
}

py::list SegmentsTo(const std::vector<MarkingSegment>& segments) {
  py::list out;
  for (const MarkingSegment& segment : segments) {
    out.append(py::dict(
        "level"_a = segment.level, "p0"_a = segment.p0, "p1"_a = segment.p1,
        "width"_a = segment.width,
        "cls"_a = segment.lane ? "lane_marking" : "parking_line",
        "confidence"_a = segment.confidence,
        "end_ok"_a = std::vector<bool>(std::begin(segment.end_ok),
                                       std::end(segment.end_ok))));
  }
  return out;
}

std::vector<uint8_t> MaskFrom(const py::object& mask, RasterFrame& frame) {
  const MaskImage array = MaskImage::ensure(mask);
  Require(array && array.ndim() == 2, "a mask is an (H, W) array");
  frame.height = static_cast<int>(array.shape(0));
  frame.width = static_cast<int>(array.shape(1));
  std::vector<uint8_t> out(array.data(), array.data() + array.size());
  for (uint8_t& value : out) value = value != 0;
  return out;
}

// (name, brightness (H, W), valid (H, W), x0, y0, res).
FloorImage ImageFrom(const py::handle& item) {
  const py::tuple tuple = item.cast<py::tuple>();
  Require(tuple.size() == kFloorImageFields,
          "images are (name, brightness, valid, x0, y0, res)");
  FloorImage image;
  image.name = tuple[0].cast<std::string>();
  const DoubleImage brightness = DoubleImage::ensure(tuple[1]);
  const DoubleImage valid = DoubleImage::ensure(tuple[2]);
  Require(brightness && valid && brightness.ndim() == 2 &&
              valid.ndim() == 2 && brightness.shape(0) == valid.shape(0) &&
              brightness.shape(1) == valid.shape(1),
          "brightness and valid are (H, W) arrays of one shape");
  image.height = static_cast<int>(brightness.shape(0));
  image.width = static_cast<int>(brightness.shape(1));
  image.brightness.assign(brightness.data(),
                          brightness.data() + brightness.size());
  image.valid.assign(valid.data(), valid.data() + valid.size());
  image.x0 = tuple[3].cast<double>();
  image.y0 = tuple[4].cast<double>();
  image.res = tuple[5].cast<double>();
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

// (K, 2) points (any array-like) as Vec2s.
std::vector<Vec2> CornersFrom(const py::handle& points) {
  const RowMatX2d xy =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          points)
          .cast<RowMatX2d>();
  std::vector<Vec2> out;
  for (Eigen::Index row = 0; row < xy.rows(); ++row) {
    out.push_back(xy.row(row).transpose());
  }
  return out;
}

}  // namespace

void BindSemantics(py::module_& m) {
  py::class_<ObjectsConfig>(m, "ObjectsConfig")
      .def(py::init<>())
      .def_readwrite("frame_stride", &ObjectsConfig::frame_stride)
      .def_readwrite("cluster_res", &ObjectsConfig::cluster_res)
      .def_readwrite("min_points", &ObjectsConfig::min_points)
      .def_readwrite("height_min", &ObjectsConfig::height_min)
      .def_readwrite("height_max", &ObjectsConfig::height_max)
      .def_readwrite("detect_max_range", &ObjectsConfig::detect_max_range)
      .def_readwrite("match_gate", &ObjectsConfig::match_gate)
      .def_readwrite("max_missed", &ObjectsConfig::max_missed)
      .def_readwrite("min_hits", &ObjectsConfig::min_hits)
      .def_readwrite("static_displacement", &ObjectsConfig::static_displacement)
      .def_readwrite("typical_length", &ObjectsConfig::typical_length)
      .def_readwrite("typical_width", &ObjectsConfig::typical_width)
      .def_readwrite("moving_fraction", &ObjectsConfig::moving_fraction)
      .def_readwrite("streak_min_moved", &ObjectsConfig::streak_min_moved)
      .def_readwrite("parked_min_points", &ObjectsConfig::parked_min_points)
      .def_readwrite("parked_min_extent", &ObjectsConfig::parked_min_extent)
      .def_readwrite("vehicle_max_height", &ObjectsConfig::vehicle_max_height)
      .def_readwrite("max_flat_height_span",
                     &ObjectsConfig::max_flat_height_span)
      .def_readwrite("nms_radius", &ObjectsConfig::nms_radius)
      .def_readwrite("structure_dilation", &ObjectsConfig::structure_dilation)
      .def_readwrite("structure_voxel", &ObjectsConfig::structure_voxel)
      .def_readwrite("vehicle_id", &ObjectsConfig::vehicle_id)
      .def_readwrite("vehicle_moving_id", &ObjectsConfig::vehicle_moving_id)
      .def_readwrite("pedestrian_id", &ObjectsConfig::pedestrian_id)
      .def_readwrite("two_wheeler_id", &ObjectsConfig::two_wheeler_id)
      .def_readwrite("other_id", &ObjectsConfig::other_id);

  m.def(
      "run_objects",
      [](const py::dict& map, const std::vector<double>& z,
         const py::list& grids, const py::list& ramps,
         const py::object& structure_points, double yaw,
         const py::dict& paths_by_level, const py::list& slots,
         const py::object& keyframe_poses, const py::object& keyframe_trip,
         const py::object& keyframe_stamp,
         const std::vector<std::string>& trip_names, const py::object& load,
         const ObjectsConfig& config, const py::object& log) {
        const LevelModel levels = LevelModelFrom(z, grids, ramps);
        const Points xyz = PointsFrom(map["xyz"]);
        const std::vector<float> height = ToFloatVector(map["height"]);
        const std::vector<int16_t> level = VectorOf<int16_t>(map["level"]);
        const std::vector<uint8_t> kind =
            VectorOf<uint8_t>(map["dynamic_kind"]);
        const std::vector<int16_t> trip = VectorOf<int16_t>(map["trip"]);
        const std::vector<float> normal_z = ToFloatVector(map["normal_z"]);
        const std::vector<uint8_t> labels = VectorOf<uint8_t>(map["labels"]);
        const size_t count = xyz.size();
        Require(height.size() == count && level.size() == count &&
                    kind.size() == count && trip.size() == count &&
                    normal_z.size() == count && labels.size() == count,
                "run_objects: one height, level, dynamic_kind, trip, "
                "normal_z and label per map point");
        ObjectMapView view;
        view.xyz = &xyz;
        view.height = &height;
        view.level = &level;
        view.dynamic_kind = &kind;
        view.trip = &trip;
        view.normal_z = &normal_z;
        view.labels = &labels;
        view.floor_id = map["floor_id"].cast<int>();
        view.ramp_id = map["ramp_id"].cast<int>();
        const StructureIndex structure(PointsFrom(structure_points),
                                       config.structure_voxel);
        std::map<int, std::vector<Vec2>> paths;
        for (const auto& [key, value] : paths_by_level) {
          const Points path = PointsFrom(
              py::reinterpret_borrow<py::object>(value));
          std::vector<Vec2>& xy = paths[key.cast<int>()];
          for (const Vec3& point : path) xy.push_back(point.head<2>());
        }
        std::vector<SlotOutline> outlines;
        for (const py::handle& slot : slots) {
          const py::tuple tuple = py::reinterpret_borrow<py::tuple>(slot);
          SlotOutline outline;
          outline.level = tuple[0].cast<int>();
          const Points corners =
              PointsFrom(py::reinterpret_borrow<py::object>(tuple[1]));
          Require(corners.size() == kParkingSlotCorners,
                  "a slot has 4 corners");
          for (int corner = 0; corner < kParkingSlotCorners; ++corner) {
            outline.corners[corner] = corners[corner].head<kPlaneDim>();
          }
          outline.width = tuple[2].cast<double>();
          outlines.push_back(outline);
        }
        const std::vector<Pose> poses = PosesFrom(keyframe_poses);
        const std::vector<int> trips = ToIntVector(keyframe_trip);
        const std::vector<double> stamps = ToDoubleVector(keyframe_stamp);
        Require(trips.size() == poses.size() && stamps.size() == poses.size(),
                "one trip and stamp per keyframe pose");
        std::exception_ptr error;
        const KeyframeSweep sweep = [&](int keyframe) {
          py::gil_scoped_acquire acquire;
          if (error) return Points();
          try {
            return PointsFrom(load(keyframe));
          } catch (...) {
            error = std::current_exception();
            return Points();
          }
        };
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        ObjectsResult result;
        {
          py::gil_scoped_release release;
          result = RunObjects(view, levels, structure, yaw, paths, outlines,
                              poses, trips, stamps, trip_names, sweep, config,
                              logger);
        }
        RethrowIfSet(error);
        RethrowIfSet(log_error);
        py::list tracks;
        for (const ObjectTrack& track : result.tracks) {
          tracks.append(py::dict(
              "trip"_a = track.trip, "track_id"_a = track.track_id,
              "static"_a = track.is_static, "label"_a = track.label,
              "frames"_a = track.keyframes,
              "box"_a = track.is_static ? py::cast(track.box) : py::none()));
        }
        py::dict moving;
        for (const auto& [keyframe, boxes] : result.moving) {
          moving[py::int_(keyframe)] = boxes;
        }
        return py::dict("tracks"_a = tracks, "parked"_a = result.parked,
                        "moving"_a = moving,
                        "moving_points"_a = ToBoolArray(result.moving_points));
      },
      "map"_a, "z"_a, "grids"_a, "ramps"_a, "structure_points"_a, "yaw"_a,
      "paths_by_level"_a, "slots"_a, "keyframe_poses"_a, "keyframe_trip"_a,
      "keyframe_stamp"_a, "trip_names"_a, "load"_a, "config"_a,
      "log"_a = py::none(),
      "The objects stage (gr/semantics/objects4d.h). map: dict of xyz, "
      "height, level, dynamic_kind, trip, normal_z, labels, floor_id, "
      "ramp_id; slots: (level, corners (4, 2), width) tuples; load(k) -> the "
      "keyframe's sweep (LiDAR frame). Returns tracks (dicts), parked (boxes "
      "per trip), moving (boxes per keyframe) and moving_points.");

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
        const FloatImage height_image = FloatImage::ensure(height_max);
        Require(height_image && height_image.ndim() == 2,
                "height_max is an (H, W) array");
        rasters.frame =
            RasterFrame{x0, y0, res, static_cast<int>(height_image.shape(1)),
                        static_cast<int>(height_image.shape(0))};
        rasters.height_max = RasterFrom(height_max, rasters.frame, false);
        rasters.floor_z = RasterFrom(floor_z, rasters.frame, false);
        rasters.floor_count = RasterFrom(floor_count, rasters.frame, true);
        rasters.intensity = RasterFrom(intensity, rasters.frame, true);
        std::vector<Vec2> path;
        for (const Vec3& point : PointsFrom(path_xy))
          path.push_back(point.head<2>());
        std::vector<SlotColumn> slot_columns;
        for (const py::handle& item : columns) {
          const py::tuple tuple = item.cast<py::tuple>();
          Require(tuple.size() == kColumnFields,
                  "columns are (cx, cy, sx, sy) tuples");
          slot_columns.push_back(
              {Vec2(tuple[0].cast<double>(), tuple[1].cast<double>()),
               Vec2(tuple[2].cast<double>(), tuple[3].cast<double>())});
        }
        const std::vector<MarkingSegment> marking_segments =
            SegmentsFrom(segments);
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        std::vector<ParkingSlot> slots;
        {
          py::gil_scoped_release release;
          slots = AssembleSlots(marking_segments, level, yaw, path,
                                slot_columns, rasters, config, logger);
        }
        RethrowIfSet(log_error);
        py::list out;
        for (const ParkingSlot& slot : slots) {
          py::list corners;
          for (const Vec2& corner : slot.corners) {
            corners.append(py::make_tuple(corner.x(), corner.y()));
          }
          out.append(py::dict(
              "slot_id"_a = slot.slot_id, "row"_a = slot.row,
              "level"_a = slot.level, "corners"_a = corners, "z"_a = slot.z,
              "width"_a = slot.width, "depth"_a = slot.depth,
              "occupied"_a = slot.occupied == ParkingSlot::kOccupancyUnknown
                                 ? py::object(py::none())
                                 : py::object(py::bool_(slot.occupied)),
              "inferred"_a = slot.inferred, "confidence"_a = slot.confidence));
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
        for (const MarkingSegment& segment : SegmentsFrom(segments)) {
          if (BothSidesObserved(segment, cells, frame) &&
              GridAligned(segment, yaw, max_axis_angle)) {
            kept.push_back(segment);
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
        std::vector<MarkingSegment> marking_segments = SegmentsFrom(segments);
        std::vector<FloorImage> floors;
        for (const py::handle& item : images) floors.push_back(ImageFrom(item));
        std::vector<const FloorImage*> order;
        for (const FloorImage& floor_image : floors)
          order.push_back(&floor_image);
        int refined = 0;
        {
          py::gil_scoped_release release;
          refined = RefineSegments(marking_segments, order, config);
        }
        return py::make_tuple(SegmentsTo(marking_segments), refined);
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
        std::vector<MarkingSegment> marking_segments = SegmentsFrom(segments);
        std::map<int, std::vector<Vec2>> by_level;
        for (const auto& [key, value] : paths) {
          std::vector<Vec2>& path = by_level[key.cast<int>()];
          for (const Vec3& point : PointsFrom(
                   py::reinterpret_borrow<py::object>(value))) {
            path.push_back(point.head<2>());
          }
        }
        ClassifyMarkings(marking_segments, by_level, config);
        return SegmentsTo(marking_segments);
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
        StructureInput input;
        input.xyz = ToPoints(xyz);
        input.normals = ToPoints(normals);
        const size_t count = input.xyz.size();
        input.height = ToFloatVector(height);
        const std::vector<int> levels = ToIntVector(level);
        input.level.assign(levels.begin(), levels.end());
        for (const int flag : ToIntVector(on_ramp))
          input.on_ramp.push_back(flag != 0);
        if (!dynamic.is_none()) {
          for (const int flag : ToIntVector(dynamic))
            input.dynamic.push_back(flag != 0);
        }
        Require(input.normals.size() == count && input.height.size() == count &&
                    input.level.size() == count &&
                    input.on_ramp.size() == count &&
                    (input.dynamic.empty() || input.dynamic.size() == count),
                "per-point arrays must match xyz");
        Require(level_z.size() == ceiling.size(),
                "level_z and ceiling must match");
        Require(config.raster_res > 0.0, "raster_res must be positive");
        for (const Vec3& point : PointsFrom(path_xy)) {
          input.path_xy.push_back(point.head<2>());
        }
        input.level_z = level_z;
        input.ceiling = ceiling;
        std::exception_ptr log_error;
        const Logger logger = MakeLogger(log, &log_error);
        StructureResult result;
        {
          py::gil_scoped_release release;
          result = ExtractStructure(input, config, logger);
        }
        RethrowIfSet(log_error);
        py::list columns;
        for (const StructureColumn& column : result.columns) {
          columns.append(py::dict("id"_a = column.id, "level"_a = column.level,
                                  "center"_a = column.center,
                                  "size"_a = column.size, "yaw"_a = column.yaw,
                                  "z0"_a = column.z0, "z1"_a = column.z1,
                                  "completed"_a = column.completed,
                                  "axis_label"_a = column.axis_label));
        }
        py::list walls;
        for (const StructureWall& wall : result.walls) {
          walls.append(py::dict("level"_a = wall.level, "p0"_a = wall.p0,
                                "p1"_a = wall.p1,
                                "thickness"_a = wall.thickness));
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

  py::class_<PaintConfig>(m, "PaintConfig")
      .def(py::init<>())
      .def_readwrite("tophat_size", &PaintConfig::tophat_size)
      .def_readwrite("min_contrast", &PaintConfig::min_contrast)
      .def_readwrite("mad_k", &PaintConfig::mad_k)
      .def_readwrite("line_close_length", &PaintConfig::line_close_length)
      .def_readwrite("line_open_length", &PaintConfig::line_open_length)
      .def_readwrite("min_area", &PaintConfig::min_area);

  // The rasters of a level (each (H, W[, 3]) or None).
  const auto rasters_from = [](const py::object& intensity,
                               const py::object& rgb,
                               const py::object& texture_nobs,
                               const py::object& height_max) {
    PaintRasters rasters;
    const auto take = [&](const py::object& raster, auto* out, int channels) {
      if (raster.is_none()) return;
      using T = typename std::remove_pointer_t<decltype(out)>::value_type;
      const auto array =
          py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
              raster);
      Require(array && array.ndim() >= 2, "rasters are (H, W) arrays");
      const int height = static_cast<int>(array.shape(0));
      const int width = static_cast<int>(array.shape(1));
      Require(
          array.size() == static_cast<py::ssize_t>(height) * width * channels,
          "raster shape");
      Require((rasters.width == 0 && rasters.height == 0) ||
                  (rasters.width == width && rasters.height == height),
              "rasters of different sizes");
      rasters.width = width;
      rasters.height = height;
      out->assign(array.data(), array.data() + array.size());
    };
    take(intensity, &rasters.intensity, kSingleChannel);
    take(rgb, &rasters.rgb, kRgbChannels);
    take(texture_nobs, &rasters.texture_nobs, kSingleChannel);
    take(height_max, &rasters.height_max, kSingleChannel);
    return rasters;
  };

  m.def(
      "marking_layers",
      [rasters_from](const py::object& intensity, const py::object& rgb,
                     const py::object& texture_nobs,
                     const py::object& height_max) {
        const PaintRasters rasters =
            rasters_from(intensity, rgb, texture_nobs, height_max);
        std::vector<PaintLayer> layers;
        {
          py::gil_scoped_release release;
          layers = MarkingLayers(rasters);
        }
        py::list out;
        for (const PaintLayer& layer : layers) {
          out.append(py::make_tuple(
              layer.name,
              ToImage(layer.brightness, rasters.height, rasters.width),
              ToImage(layer.valid, rasters.height, rasters.width)));
        }
        return out;
      },
      "intensity"_a, "rgb"_a, "texture_nobs"_a, "height_max"_a,
      "Brightness layers (C++ MarkingLayers): [(name, (H, W) brightness, "
      "(H, W) valid uint8)].");

  m.def(
      "floor_images",
      [rasters_from](const py::object& intensity, const py::object& rgb,
                     const py::object& texture_nobs,
                     const py::object& height_max) {
        const PaintRasters rasters =
            rasters_from(intensity, rgb, texture_nobs, height_max);
        std::vector<PaintLayer> images;
        {
          py::gil_scoped_release release;
          images = FloorImages(rasters);
        }
        py::list out;
        for (const PaintLayer& image : images) {
          out.append(py::make_tuple(
              image.name,
              ToImage(image.brightness, rasters.height, rasters.width),
              ToImage(image.valid, rasters.height, rasters.width)));
        }
        return out;
      },
      "intensity"_a, "rgb"_a, "texture_nobs"_a, "height_max"_a,
      "Floor images of the line fit (C++ FloorImages): [(camera | lidar, "
      "(H, W) normalised brightness, (H, W) valid uint8)].");

  m.def(
      "paint_mask",
      [rasters_from](const py::object& intensity, const py::object& rgb,
                     const py::object& texture_nobs,
                     const py::object& height_max, double res,
                     const PaintConfig& config) {
        const PaintRasters rasters =
            rasters_from(intensity, rgb, texture_nobs, height_max);
        Mask mask;
        {
          py::gil_scoped_release release;
          mask = PaintMask(MarkingLayers(rasters), rasters.width,
                           rasters.height, res, config);
        }
        return ToImage(mask, rasters.height, rasters.width);
      },
      "intensity"_a, "rgb"_a, "texture_nobs"_a, "height_max"_a, "res"_a,
      "config"_a = PaintConfig(),
      "Painted cells of all layers (C++ PaintMask): (H, W) uint8.");

  m.def(
      "elongated",
      [](const py::array_t<uint8_t, py::array::c_style |
                                        py::array::forcecast>& mask,
         double res, double yaw, const PaintConfig& config) {
        Require(mask.ndim() == 2, "mask is (H, W)");
        const int height = static_cast<int>(mask.shape(0));
        const int width = static_cast<int>(mask.shape(1));
        Mask cells(mask.data(), mask.data() + mask.size());
        Mask out;
        {
          py::gil_scoped_release release;
          out = Elongated(cells, width, height, res, yaw, config);
        }
        return ToImage(out, height, width);
      },
      "mask"_a, "res"_a, "yaw"_a, "config"_a = PaintConfig(),
      "Paint along the grid axes (C++ Elongated): (H, W) uint8.");

  m.def(
      "close_mask",
      [](const py::array_t<uint8_t, py::array::c_style |
                                        py::array::forcecast>& mask,
         const std::string& shape, int size, int iterations) {
        Require(mask.ndim() == 2, "mask is (H, W)");
        Require(shape == "ellipse" || shape == "cross",
                "shape is ellipse or cross");
        const int height = static_cast<int>(mask.shape(0));
        const int width = static_cast<int>(mask.shape(1));
        Mask cells(mask.data(), mask.data() + mask.size());
        for (uint8_t& v : cells) v = v != 0;
        Mask out;
        {
          py::gil_scoped_release release;
          out = shape == "cross" ? CloseCross(cells, width, height, iterations)
                                 : CloseKernel(cells, width, height,
                                               EllipseKernel(size), size);
        }
        return ToImage(out, height, width);
      },
      "mask"_a, "shape"_a, "size"_a = 3, "iterations"_a = 1,
      "Binary closing: 'ellipse' (cv2.morphologyEx with an ellipse of "
      "`size`) or 'cross' (scipy binary_closing, `iterations`).");

  m.def(
      "dilate_cross",
      [](const py::array_t<uint8_t, py::array::c_style |
                                        py::array::forcecast>& mask,
         int iterations) {
        Require(mask.ndim() == 2, "mask is (H, W)");
        const int height = static_cast<int>(mask.shape(0));
        const int width = static_cast<int>(mask.shape(1));
        Mask cells(mask.data(), mask.data() + mask.size());
        for (uint8_t& v : cells) v = v != 0;
        for (int iteration = 0; iteration < iterations; ++iteration)
          cells = DilateCross(cells, width, height);
        return ToImage(cells, height, width);
      },
      "mask"_a, "iterations"_a = 1,
      "scipy binary_dilation with the cross structure, `iterations`.");

  py::class_<SlotPaintConfig>(m, "SlotPaintConfig")
      .def(py::init<>())
      .def_readwrite("inset", &SlotPaintConfig::inset)
      .def_readwrite("min_saturation", &SlotPaintConfig::min_saturation)
      .def_readwrite("hue_min", &SlotPaintConfig::hue_min)
      .def_readwrite("hue_max", &SlotPaintConfig::hue_max);

  m.def(
      "measure_slot_paint",
      [](const py::list& slots,
         const py::array_t<uint8_t, py::array::c_style |
                                        py::array::forcecast>& rgb,
         const py::array_t<uint8_t, py::array::c_style |
                                        py::array::forcecast>& observed,
         double x0, double y0, double res, const SlotPaintConfig& config) {
        Require(rgb.ndim() == 3 && rgb.shape(2) == kRgbChannels,
                "rgb must be (H, W, 3)");
        Require(observed.ndim() == 2 && observed.shape(0) == rgb.shape(0) &&
                    observed.shape(1) == rgb.shape(1),
                "observed must be (H, W) like rgb");
        std::vector<std::array<Vec2, kParkingSlotCorners>> corners;
        for (const py::handle& item : slots) {
          const std::vector<Vec2> slot_corners = CornersFrom(item);
          Require(slot_corners.size() == kParkingSlotCorners,
                  "a slot needs 4 corners");
          std::array<Vec2, kParkingSlotCorners> slot;
          std::copy(slot_corners.begin(), slot_corners.end(), slot.begin());
          corners.push_back(slot);
        }
        const RasterFrame frame{x0, y0, res, static_cast<int>(rgb.shape(1)),
                                static_cast<int>(rgb.shape(0))};
        const std::vector<uint8_t> colours(rgb.data(),
                                           rgb.data() + rgb.size());
        const std::vector<uint8_t> seen(observed.data(),
                                        observed.data() + observed.size());
        std::vector<SlotPaint> paint;
        {
          py::gil_scoped_release release;
          paint = MeasureSlotPaint(corners, colours, seen, frame, config);
        }
        py::list out;
        for (const SlotPaint& slot_paint : paint) {
          out.append(py::dict("cells"_a = slot_paint.cells,
                              "observed"_a = slot_paint.observed,
                              "in_band"_a = slot_paint.in_band,
                              "mean_rgb"_a = slot_paint.mean_rgb));
        }
        return out;
      },
      "slots"_a, "rgb"_a, "observed"_a, "x0"_a, "y0"_a, "res"_a,
      "config"_a = SlotPaintConfig(),
      "Floor colour inside each slot ((4, 2) corners) on the orthophoto: "
      "dicts of cells, observed, in_band (share of observed cells in the "
      "hue band), mean_rgb.");
}

}  // namespace gr::bindings
