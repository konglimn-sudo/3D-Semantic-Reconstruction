// Bindings of section 5: object boxes and their timeline, occupancy ground
// truth, per-frame labels by projection, 2D -> 3D label lifting and the
// cross-view identity association of 2D masks.

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "bind_util.h"
#include "gr/core/image.h"
#include "gr/semantics/identity.h"
#include "gr/semantics/labeling.h"
#include "gr/semantics/objects.h"
#include "gr/semantics/occupancy.h"
#include "gr/sensors/exposure.h"

namespace gr::bindings {
namespace {

namespace py = ::pybind11;
using ::pybind11::literals::operator""_a;

using ImageArray =
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
// Slot corners in image or world coordinates: one (u, v) / (x, y) row each.
using CornerMatrix =
    Eigen::Matrix<double, kParkingSlotCorners, kPlaneDim, Eigen::RowMajor>;
// Images without a channel axis have one channel.
constexpr int kSingleChannel = 1;
// Cameras are (CameraModel, T_vehicle_cam) pairs.
constexpr size_t kCameraPairSize = 2;

py::array_t<double> PolylineArray(const Polyline2& polyline) {
  py::array_t<double> array({static_cast<py::ssize_t>(polyline.size()),
                             static_cast<py::ssize_t>(kPlaneDim)});
  double* data = array.mutable_data();
  for (size_t i = 0; i < polyline.size(); ++i) {
    data[kPlaneDim * i] = polyline[i].x();
    data[kPlaneDim * i + 1] = polyline[i].y();
  }
  return array;
}

py::list PolylineList(const std::vector<Polyline2>& polylines) {
  py::list out;
  for (const Polyline2& polyline : polylines)
    out.append(PolylineArray(polyline));
  return out;
}

// An (H, W) or (H, W, C) uint8 image (None -> empty view); `holder` keeps the
// converted array alive.
ImageView ImageViewOf(const py::object& image, ImageArray& holder) {
  if (image.is_none()) return ImageView();
  holder = ImageArray::ensure(image);
  Require(holder && (holder.ndim() == 2 || holder.ndim() == 3),
          "image must be HxW[xC] uint8");
  return ImageView{holder.data(), static_cast<int>(holder.shape(0)),
                   static_cast<int>(holder.shape(1)),
                   holder.ndim() == 3 ? static_cast<int>(holder.shape(2))
                                      : kSingleChannel};
}

py::dict FrameRecord(const FrameLabels& labels) {
  py::list slots;
  for (const SlotProjection& slot : labels.slots) {
    CornerMatrix corners;
    py::list corner_visible;
    for (int corner = 0; corner < kParkingSlotCorners; ++corner) {
      corners.row(corner) = slot.corners_uv[corner].transpose();
      corner_visible.append(slot.corner_visible[corner]);
    }
    slots.append(py::dict("index"_a = slot.index, "corners_uv"_a = corners,
                          "corner_visible"_a = corner_visible,
                          "entrance_visible"_a = slot.entrance_visible,
                          "visible_fraction"_a = slot.visible_fraction,
                          "polylines"_a = PolylineList(slot.polylines)));
  }
  py::list markings;
  for (const MarkingProjection& marking : labels.markings) {
    markings.append(py::dict("index"_a = marking.index,
                             "polylines"_a = PolylineList(marking.polylines)));
  }
  py::list boxes;
  for (const BoxProjection& box : labels.boxes) {
    boxes.append(
        py::dict("index"_a = box.index,
                 "bbox"_a = std::vector<double>{box.bbox[0], box.bbox[1],
                                                box.bbox[2], box.bbox[3]},
                 "visible_fraction"_a = box.visible_fraction));
  }
  const py::object score =
      std::isfinite(labels.consistency_score)
          ? py::object(py::float_(labels.consistency_score))
          : py::object(py::none());
  return py::dict("slots"_a = slots, "markings"_a = markings, "boxes"_a = boxes,
                  "consistency_score"_a = score);
}

void BindObjects(py::module_& m) {
  py::class_<ObjectBox>(m, "ObjectBox")
      .def(py::init([](int track_id, int label, const Vec3& center,
                       const Vec3& size, double yaw, bool is_static,
                       const Vec3& velocity, int num_points, double score) {
             ObjectBox box;
             box.track_id = track_id;
             box.label = label;
             box.center = center;
             box.size = size;
             box.yaw = yaw;
             box.is_static = is_static;
             box.velocity = velocity;
             box.num_points = num_points;
             box.score = score;
             return box;
           }),
           "track_id"_a = -1, "label"_a = 0, "center"_a = Vec3::Zero(),
           "size"_a = Vec3::Ones(), "yaw"_a = 0.0, "is_static"_a = true,
           "velocity"_a = Vec3::Zero(), "num_points"_a = 0, "score"_a = 1.0)
      .def_readwrite("track_id", &ObjectBox::track_id)
      .def_readwrite("label", &ObjectBox::label)
      .def_readwrite("center", &ObjectBox::center)
      .def_readwrite("size", &ObjectBox::size)
      .def_readwrite("yaw", &ObjectBox::yaw)
      .def_readwrite("is_static", &ObjectBox::is_static)
      .def_readwrite("velocity", &ObjectBox::velocity)
      .def_readwrite("num_points", &ObjectBox::num_points)
      .def_readwrite("score", &ObjectBox::score)
      .def(
          "contains",
          [](const ObjectBox& box, const PointsIn& points, double margin) {
            std::vector<uint8_t> inside(static_cast<size_t>(points.rows()));
            for (Eigen::Index i = 0; i < points.rows(); ++i) {
              inside[static_cast<size_t>(i)] =
                  box.Contains(points.row(i).transpose(), margin);
            }
            return ToBoolArray(inside);
          },
          "points"_a, "margin"_a = 0.0)
      .def(
          "surface",
          [](const ObjectBox& box, double step) {
            Require(step > 0.0, "step must be positive");
            return ToMatrix(box.SampleSurface(step));
          },
          "step"_a = 0.05);

  py::class_<BoxTimeline>(m, "BoxTimeline")
      .def(py::init<>())
      .def("add_static", &BoxTimeline::AddStatic, "trip"_a, "box"_a)
      .def("add_moving", &BoxTimeline::AddMoving, "trip"_a, "stamp"_a, "box"_a)
      .def("at", &BoxTimeline::At, "trip"_a, "stamp"_a, "slack"_a = 0.5,
           "Parked boxes of the trip plus moving tracks interpolated at "
           "`stamp`.");
}

void BindOccupancyGenerator(py::module_& m) {
  py::class_<OccupancyConfig>(m, "OccupancyConfig")
      .def(py::init<>())
      .def_readwrite("min_bound", &OccupancyConfig::min_bound)
      .def_readwrite("max_bound", &OccupancyConfig::max_bound)
      .def_readwrite("voxel", &OccupancyConfig::voxel)
      .def_readwrite("free_label", &OccupancyConfig::free_label)
      .def_readwrite("camera_pixel_stride",
                     &OccupancyConfig::camera_pixel_stride)
      .def_readwrite("camera_max_range", &OccupancyConfig::camera_max_range);

  py::class_<OccupancyGenerator>(m, "OccupancyGenerator")
      .def(py::init([](const PointsIn& static_world, const py::object& labels,
                       const OccupancyConfig& config,
                       const Mat4& vehicle_from_lidar,
                       const std::vector<py::tuple>& cameras) {
             const std::vector<int> static_labels = ToIntVector(labels);
             Require(static_cast<Eigen::Index>(static_labels.size()) ==
                         static_world.rows(),
                     "one label per static point expected");
             Require(config.voxel > 0.0, "voxel must be positive");
             std::vector<CameraView> views;
             for (const py::tuple& camera : cameras) {
               Require(camera.size() == kCameraPairSize,
                       "cameras: (CameraModel, T_vehicle_cam) pairs");
               CameraView view;
               view.camera = camera[0].cast<std::shared_ptr<CameraModel>>();
               view.ego_from_camera = Pose::FromMatrix(camera[1].cast<Mat4>());
               views.push_back(view);
             }
             return OccupancyGenerator(
                 ToPoints(static_world), static_labels, config,
                 Pose::FromMatrix(vehicle_from_lidar), std::move(views));
           }),
           "static_world"_a, "static_labels"_a, "config"_a, "T_vehicle_lidar"_a,
           "cameras"_a, "cameras: list of (CameraModel, T_vehicle_cam).")
      .def(
          "generate",
          [](const OccupancyGenerator& generator,
             const Mat4& world_from_vehicle,
             const std::vector<ObjectBox>& boxes, const py::object& sweep) {
            const Points points = PointsFrom(sweep);
            OccupancyResult result;
            {
              py::gil_scoped_release release;
              result = generator.Generate(Pose::FromMatrix(world_from_vehicle),
                                          boxes, points);
            }
            const std::vector<py::ssize_t> shape = {result.nx, result.ny,
                                                    result.nz};
            const auto grid = [&](const std::vector<uint8_t>& values) {
              py::array_t<uint8_t> array(shape);
              std::memcpy(array.mutable_data(), values.data(), values.size());
              return array;
            };
            return py::make_tuple(grid(result.semantics),
                                  grid(result.mask_lidar),
                                  grid(result.mask_camera));
          },
          "T_world_vehicle"_a, "boxes"_a, "sweep_lidar"_a,
          "Returns (semantics, mask_lidar, mask_camera) as (nx, ny, nz) uint8 "
          "in the vehicle frame.");
}

void BindFrameLabels(py::module_& m) {
  py::class_<FrameLabelConfig>(m, "FrameLabelConfig")
      .def(py::init<>())
      .def_readwrite("max_range", &FrameLabelConfig::max_range)
      .def_readwrite("sample_step", &FrameLabelConfig::sample_step)
      .def_readwrite("depth_abs_tol", &FrameLabelConfig::depth_abs_tol)
      .def_readwrite("depth_rel_tol", &FrameLabelConfig::depth_rel_tol)
      .def_readwrite("render_spacing", &FrameLabelConfig::render_spacing)
      .def_readwrite("box_render_step", &FrameLabelConfig::box_render_step)
      .def_readwrite("box_sample_step", &FrameLabelConfig::box_sample_step);

  using SlotTuple = std::pair<CornerMatrix, double>;
  using MarkingTuple = std::tuple<Vec2, Vec2, double>;
  py::class_<FrameLabeler>(m, "FrameLabeler")
      .def(py::init([](const PointsIn& static_world, const py::object& labels,
                       const std::vector<SlotTuple>& slots,
                       const std::vector<MarkingTuple>& markings,
                       const FrameLabelConfig& config) {
             const std::vector<int> static_labels = ToIntVector(labels);
             Require(static_cast<Eigen::Index>(static_labels.size()) ==
                         static_world.rows(),
                     "one label per static point expected");
             Require(config.sample_step > 0.0 && config.box_render_step > 0.0 &&
                         config.box_sample_step > 0.0,
                     "sample steps must be positive");
             std::vector<SlotGeometry> slot_geometry;
             for (const auto& [corners, z] : slots) {
               SlotGeometry geometry;
               for (int corner = 0; corner < kParkingSlotCorners; ++corner) {
                 geometry.corners[corner] = corners.row(corner).transpose();
               }
               geometry.z = z;
               slot_geometry.push_back(geometry);
             }
             std::vector<MarkingGeometry> marking_geometry;
             for (const auto& [p0, p1, width] : markings) {
               MarkingGeometry geometry;
               geometry.p0 = p0;
               geometry.p1 = p1;
               geometry.width = width;
               marking_geometry.push_back(geometry);
             }
             return FrameLabeler(ToPoints(static_world), static_labels,
                                 std::move(slot_geometry),
                                 std::move(marking_geometry), config);
           }),
           "static_world"_a, "static_labels"_a, "slots"_a, "markings"_a,
           "config"_a,
           "slots: list of (corners (4, 2), floor z), entrance edge first; "
           "markings: list of (p0, p1, width).")
      .def(
          "label",
          [](const FrameLabeler& labeler, const CameraModel& camera,
             const py::object& world_from_camera,
             const std::vector<ObjectBox>& boxes, const py::object& image) {
            const CameraExposure exposure =
                ExposureFrom(world_from_camera, camera.height(), false);
            ImageArray holder;
            const ImageView view = ImageViewOf(image, holder);
            FrameLabels labels;
            {
              py::gil_scoped_release release;
              labels = labeler.Label(camera, exposure, boxes, view);
            }
            return py::make_tuple(
                FrameRecord(labels),
                ToImage(labels.label_image, labels.height, labels.width));
          },
          "camera"_a, "T_world_cam"_a, "boxes"_a, "image"_a = py::none(),
          "Returns (record, label image HxW uint8); record entries refer to "
          "slots / markings / boxes by index. T_world_cam is (4, 4), or "
          "(2, 4, 4) for the first and last row of a rolling-shutter image.");

  m.def(
      "edge_strength",
      [](const py::object& image) {
        ImageArray holder;
        const ImageView view = ImageViewOf(image, holder);
        Require(!view.empty(), "image must not be empty");
        return ToImage(EdgeStrength(view), view.height, view.width);
      },
      "image"_a,
      "Sobel gradient magnitude of the channel mean, normalised by its 95th "
      "percentile.");
}

void BindLifting(py::module_& m) {
  py::class_<LabelLifter>(m, "LabelLifter")
      .def(py::init([](const PointsIn& points, int num_classes, double spacing,
                       double max_range, int boundary, int min_views) {
             Require(num_classes > 0, "num_classes must be positive");
             return LabelLifter(ToPoints(points), num_classes, spacing,
                                max_range, boundary, min_views);
           }),
           "points"_a, "num_classes"_a,
           "spacing"_a = LabelLifter::kDefaultSpacing,
           "max_range"_a = LabelLifter::kDefaultMaxRange,
           "boundary"_a = LabelLifter::kDefaultBoundary,
           "min_views"_a = LabelLifter::kDefaultMinViews)
      .def(
          "add_view",
          [](LabelLifter& lifter, const CameraModel& camera,
             const py::object& camera_from_world,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast>
                 labels,
             const py::object& weights) {
            const CameraExposure exposure =
                ExposureFrom(camera_from_world, camera.height(), true);
            Require(labels.ndim() == 2 && labels.shape(0) == camera.height() &&
                        labels.shape(1) == camera.width(),
                    "labels must be a (height, width) integer image of the "
                    "camera");
            const std::vector<float> w =
                ToDepth(weights, camera.height(), camera.width());
            const int32_t* data = labels.data();
            py::gil_scoped_release release;
            lifter.AddView(camera, exposure, data,
                           w.empty() ? nullptr : w.data());
          },
          "camera"_a, "T_cam_world"_a, "labels"_a, "weights"_a = py::none())
      .def(
          "result",
          [](const LabelLifter& lifter) {
            const LabelLifter::Lifted lifted = lifter.Lift();
            const std::vector<float>& votes = lifter.votes();
            py::array_t<float> vote_array(
                {static_cast<py::ssize_t>(lifted.labels.size()),
                 static_cast<py::ssize_t>(lifter.num_classes())});
            if (!votes.empty()) {
              std::memcpy(vote_array.mutable_data(), votes.data(),
                          votes.size() * sizeof(float));
            }
            return py::make_tuple(ToArray(lifted.labels),
                                  ToArray(lifted.confidence), vote_array);
          },
          "Returns (labels (-1 = none), confidence, votes (N, C)).");
}

py::dict MaskRecordArrays(const std::vector<MaskRecord>& records) {
  std::vector<int> view, mask, identity, points, pixels;
  std::vector<double> mean_range;
  std::vector<uint8_t> touches_border;
  for (const MaskRecord& record : records) {
    view.push_back(record.view);
    mask.push_back(record.mask);
    identity.push_back(record.identity);
    points.push_back(record.points);
    pixels.push_back(record.pixels);
    mean_range.push_back(record.mean_range);
    touches_border.push_back(record.touches_border ? 1 : 0);
  }
  return py::dict("view"_a = ToArray(view), "mask"_a = ToArray(mask),
                  "identity"_a = ToArray(identity),
                  "points"_a = ToArray(points), "pixels"_a = ToArray(pixels),
                  "mean_range"_a = ToArray(mean_range),
                  "touches_border"_a = ToBoolArray(touches_border));
}

void BindIdentity(py::module_& m) {
  py::class_<IdentityConfig>(m, "IdentityConfig")
      .def(py::init<>())
      .def_readwrite("max_range", &IdentityConfig::max_range)
      .def_readwrite("point_spacing", &IdentityConfig::point_spacing)
      .def_readwrite("depth_abs_tol", &IdentityConfig::depth_abs_tol)
      .def_readwrite("depth_rel_tol", &IdentityConfig::depth_rel_tol)
      .def_readwrite("grazing_pixels", &IdentityConfig::grazing_pixels)
      .def_readwrite("merge_share", &IdentityConfig::merge_share)
      .def_readwrite("merge_coverage", &IdentityConfig::merge_coverage)
      .def_readwrite("boundary", &IdentityConfig::boundary)
      .def_readwrite("min_mask_points", &IdentityConfig::min_mask_points)
      .def_readwrite("min_common_points", &IdentityConfig::min_common_points)
      .def_readwrite("min_share", &IdentityConfig::min_share)
      .def_readwrite("min_known_fraction", &IdentityConfig::min_known_fraction);

  py::class_<IdentityAssociator>(m, "IdentityAssociator")
      .def(py::init([](const PointsIn& points, const py::object& normals,
                       const IdentityConfig& config) {
             Require(config.max_range > 0.0 && config.point_spacing > 0.0,
                     "max_range and point_spacing must be positive");
             Points carrier_normals = PointsFrom(normals);
             Require(carrier_normals.empty() ||
                         carrier_normals.size() ==
                             static_cast<size_t>(points.rows()),
                     "one normal per point expected");
             return IdentityAssociator(ToPoints(points),
                                       std::move(carrier_normals), config);
           }),
           "points"_a, "normals"_a = py::none(), "config"_a = IdentityConfig(),
           "Carrier points (e.g. the static LiDAR map) and optional normals "
           "oriented towards the sensors (back-face culling).")
      .def(
          "add_view",
          [](IdentityAssociator& associator, const CameraModel& camera,
             const py::object& world_from_camera,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast>
                 masks) {
            Require(masks.ndim() == 2 && masks.shape(0) == camera.height() &&
                        masks.shape(1) == camera.width(),
                    "masks must be a (height, width) integer image of the "
                    "camera");
            const CameraExposure exposure =
                ExposureFrom(world_from_camera, camera.height(), false);
            const int32_t* data = masks.data();
            std::vector<MaskRecord> records;
            {
              py::gil_scoped_release release;
              records = associator.AddView(camera, exposure, data);
            }
            return MaskRecordArrays(records);
          },
          "camera"_a, "T_world_cam"_a, "masks"_a,
          "Adds a view: masks (H, W) int, -1 = none. T_world_cam is (4, 4) or "
          "(2, 4, 4) (first / last row). Returns the view's mask records as "
          "arrays: view, mask, identity, points, pixels, mean_range, "
          "touches_border.")
      .def(
          "finish",
          [](const IdentityAssociator& associator) {
            const IdentityAssociator::Result result = associator.Finish();
            return py::dict(
                "point_identity"_a = ToArray(result.point_identity),
                "point_confidence"_a = ToArray(result.point_confidence),
                "masks"_a = MaskRecordArrays(result.masks),
                "num_identities"_a = result.num_identities);
          },
          "Returns point_identity (-1 = none), point_confidence, masks (all "
          "views) and num_identities.")
      .def_property_readonly("num_views", &IdentityAssociator::num_views);
}

}  // namespace

void BindLabels(py::module_& m) {
  BindObjects(m);
  BindOccupancyGenerator(m);
  BindFrameLabels(m);
  BindLifting(m);
  BindIdentity(m);
}

}  // namespace gr::bindings
