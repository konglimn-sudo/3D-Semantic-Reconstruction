// Bindings of the 3DGS support code: LiDAR-anchored Gaussians, the ray
// renderer, camera coverage, training masks and the depth prior fusion.

#include <memory>
#include <utility>
#include <vector>

#include "bind_util.h"
#include "gr/core/image.h"
#include "gr/mapping/raycast.h"
#include "gr/semantics/objects.h"
#include "gr/semantics/projector.h"
#include "gr/sensors/exposure.h"
#include "gr/splat/coverage.h"
#include "gr/splat/depth_prior.h"
#include "gr/splat/gaussians.h"

namespace gr::bindings {
namespace {

namespace py = ::pybind11;
using ::pybind11::literals::operator""_a;

using U8Array = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Quaternion columns of the Gaussian rotations: (w, x, y, z).
constexpr int kQuaternionDim = 4;
constexpr int kQuaternionW = 0;
constexpr int kQuaternionX = 1;
constexpr int kQuaternionY = 2;
constexpr int kQuaternionZ = 3;
// Depth samples are (u, v, range) rows.
constexpr int kDepthSampleColumns = 3;

// (N, k) float64 rows of an array-like with `columns` columns.
F64Array Rows(const py::object& values, py::ssize_t rows, int columns,
              const char* name) {
  F64Array array = F64Array::ensure(values);
  Require(array && array.ndim() == 2 && array.shape(0) == rows &&
              array.shape(1) == columns,
          std::string(name) + " must have shape (N, " +
              std::to_string(columns) + ")");
  return array;
}

GaussianSet GaussianSetFrom(const PointsIn& means, const py::object& quats,
                            const py::object& scales, const py::object& opacity,
                            const py::object& intensity,
                            const py::object& trip) {
  const py::ssize_t count = means.rows();
  GaussianSet set;
  set.means = ToPoints(means);
  const F64Array quat_rows = Rows(quats, count, kQuaternionDim, "quats");
  const F64Array scale_rows = Rows(scales, count, kSpaceDim, "scales");
  auto quat = quat_rows.unchecked<2>();
  auto scale = scale_rows.unchecked<2>();
  set.rotations.resize(static_cast<size_t>(count));
  set.scales.resize(static_cast<size_t>(count));
  for (py::ssize_t i = 0; i < count; ++i) {
    set.rotations[static_cast<size_t>(i)] = Eigen::Quaterniond(
        quat(i, kQuaternionW), quat(i, kQuaternionX), quat(i, kQuaternionY),
        quat(i, kQuaternionZ));
    set.scales[static_cast<size_t>(i)] =
        Vec3(scale(i, 0), scale(i, 1), scale(i, 2));
  }
  set.opacity = ToFloatVector(opacity);
  Require(set.opacity.size() == static_cast<size_t>(count),
          "opacity must have N values");
  set.intensity = ToFloatVector(intensity);
  Require(set.intensity.empty() ||
              set.intensity.size() == static_cast<size_t>(count),
          "intensity must have N values");
  set.trip = ToIntVector(trip);
  Require(set.trip.empty() || set.trip.size() == static_cast<size_t>(count),
          "trip must have N values");
  set.FillDefaults();
  return set;
}

U8Array MaskView(const py::object& mask, int height, int width) {
  if (mask.is_none()) return U8Array();
  U8Array array = U8Array::ensure(mask);
  Require(array && array.ndim() == 2 && array.shape(0) == height &&
              array.shape(1) == width,
          "mask must be HxW uint8 like the camera");
  return array;
}

}  // namespace

void BindSplat(py::module_& m) {
  py::class_<GaussianInitConfig>(m, "GaussianInitConfig")
      .def(py::init<>())
      .def_readwrite("neighbors", &GaussianInitConfig::neighbors)
      .def_readwrite("search_radius", &GaussianInitConfig::search_radius)
      .def_readwrite("default_spacing", &GaussianInitConfig::default_spacing)
      .def_readwrite("tangent_factor", &GaussianInitConfig::tangent_factor)
      .def_readwrite("min_tangent", &GaussianInitConfig::min_tangent)
      .def_readwrite("max_tangent", &GaussianInitConfig::max_tangent)
      .def_readwrite("normal_floor", &GaussianInitConfig::normal_floor)
      .def_readwrite("opacity", &GaussianInitConfig::opacity);

  m.def(
      "init_gaussians",
      [](const PointsIn& points, const py::object& normals,
         const py::object& sigma, const GaussianInitConfig& config) {
        Require(config.search_radius > 0.0 && config.neighbors > 0,
                "search_radius and neighbors must be positive");
        const Points cloud = ToPoints(points);
        const Points point_normals = PointsFrom(normals);
        Require(point_normals.empty() || point_normals.size() == cloud.size(),
                "normals must match the points");
        const std::vector<double> point_sigma = ToDoubleVector(sigma);
        Require(point_sigma.empty() || point_sigma.size() == cloud.size(),
                "sigma must match the points");
        GaussianSet set;
        {
          py::gil_scoped_release release;
          set = InitGaussians(cloud, point_normals, point_sigma, config);
        }
        const py::ssize_t size = static_cast<py::ssize_t>(set.size());
        py::array_t<double> quats({size, py::ssize_t{kQuaternionDim}});
        py::array_t<double> scales({size, py::ssize_t{kSpaceDim}});
        auto quat = quats.mutable_unchecked<2>();
        auto scale = scales.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < size; ++i) {
          const Eigen::Quaterniond& rotation =
              set.rotations[static_cast<size_t>(i)];
          quat(i, kQuaternionW) = rotation.w();
          quat(i, kQuaternionX) = rotation.x();
          quat(i, kQuaternionY) = rotation.y();
          quat(i, kQuaternionZ) = rotation.z();
          for (int axis = 0; axis < kSpaceDim; ++axis) {
            scale(i, axis) = set.scales[static_cast<size_t>(i)][axis];
          }
        }
        py::dict out;
        out["quats"] = quats;
        out["scales"] = scales;
        out["opacity"] = ToArray(set.opacity);
        return out;
      },
      "points"_a, "normals"_a = py::none(), "sigma"_a = py::none(),
      "config"_a = GaussianInitConfig(),
      "Flat Gaussians on the points; returns {quats (N, 4) wxyz world <- "
      "Gaussian, scales (N, 3) sigma per axis (m), opacity (N,)}.");

  py::class_<RayRenderConfig>(m, "RayRenderConfig")
      .def(py::init<>())
      .def_readwrite("cutoff", &RayRenderConfig::cutoff)
      .def_readwrite("cell", &RayRenderConfig::cell)
      .def_readwrite("max_cells", &RayRenderConfig::max_cells)
      .def_readwrite("min_alpha", &RayRenderConfig::min_alpha)
      .def_readwrite("max_alpha", &RayRenderConfig::max_alpha)
      .def_readwrite("min_transmittance", &RayRenderConfig::min_transmittance)
      .def_readwrite("max_range", &RayRenderConfig::max_range);

  py::class_<GaussianRayRenderer>(m, "GaussianRayRenderer")
      .def(py::init([](const PointsIn& means, const py::object& quats,
                       const py::object& scales, const py::object& opacity,
                       const py::object& intensity, const py::object& trip,
                       const RayRenderConfig& config) {
             Require(config.cell > 0.0 && config.cutoff > 0.0,
                     "cell and cutoff must be positive");
             GaussianSet set = GaussianSetFrom(means, quats, scales, opacity,
                                               intensity, trip);
             py::gil_scoped_release release;
             return std::make_unique<GaussianRayRenderer>(std::move(set),
                                                          config);
           }),
           "means"_a, "quats"_a, "scales"_a, "opacity"_a,
           "intensity"_a = py::none(), "trip"_a = py::none(),
           "config"_a = RayRenderConfig())
      .def(
          "render",
          [](const GaussianRayRenderer& renderer, const PointsIn& origins,
             const PointsIn& directions, int trip) {
            Require(origins.rows() == directions.rows(),
                    "origins and directions must have the same length");
            const Points ray_origins = ToPoints(origins);
            const Points ray_directions = ToPoints(directions);
            RayRender render;
            {
              py::gil_scoped_release release;
              render = renderer.Render(ray_origins, ray_directions, trip);
            }
            py::dict out;
            out["depth"] = ToArray(render.depth);
            out["mean_depth"] = ToArray(render.mean_depth);
            out["intensity"] = ToArray(render.intensity);
            out["opacity"] = ToArray(render.opacity);
            out["dominant"] = ToArray(render.dominant);
            return out;
          },
          "origins"_a, "directions"_a, "trip"_a = -1,
          "Renders rays; returns {depth (median), mean_depth, intensity, "
          "opacity, dominant}. Trip-specific Gaussians only join rays of "
          "their trip.")
      .def_property_readonly("cell", &GaussianRayRenderer::cell);

  py::class_<CoverageConfig>(m, "CoverageConfig")
      .def(py::init<>())
      .def_readwrite("max_range", &CoverageConfig::max_range)
      .def_readwrite("point_spacing", &CoverageConfig::point_spacing)
      .def_readwrite("min_radius", &CoverageConfig::min_radius)
      .def_readwrite("max_radius", &CoverageConfig::max_radius)
      .def_readwrite("depth_abs_tol", &CoverageConfig::depth_abs_tol)
      .def_readwrite("depth_rel_tol", &CoverageConfig::depth_rel_tol)
      .def_readwrite("dark", &CoverageConfig::dark)
      .def_readwrite("saturated", &CoverageConfig::saturated)
      .def_readwrite("min_cos", &CoverageConfig::min_cos);

  py::class_<CameraCoverage>(m, "CameraCoverage")
      .def(py::init([](const PointsIn& points, const py::object& normals,
                       const CoverageConfig& config) {
             Points cloud = ToPoints(points);
             Points point_normals = PointsFrom(normals);
             Require(point_normals.empty() ||
                         point_normals.size() == cloud.size(),
                     "normals must match the points");
             return std::make_unique<CameraCoverage>(
                 std::move(cloud), std::move(point_normals), config);
           }),
           "points"_a, "normals"_a = py::none(),
           "config"_a = CoverageConfig())
      .def(
          "add_frame",
          [](CameraCoverage& coverage, const CameraModel& camera,
             const py::object& world_from_camera, const py::object& image,
             const py::object& mask) {
            const CameraExposure exposure =
                ExposureFrom(world_from_camera, camera.height(), false);
            U8Array holder;
            ImageView view;
            if (!image.is_none()) {
              holder = U8Array::ensure(image);
              Require(holder && holder.ndim() == 3 &&
                          holder.shape(2) == kRgbChannels &&
                          holder.shape(0) == camera.height() &&
                          holder.shape(1) == camera.width(),
                      "image must be HxWx3 uint8 like the camera");
              view = ImageView{holder.data(), camera.height(), camera.width(),
                               kRgbChannels};
            }
            const U8Array mask_array =
                MaskView(mask, camera.height(), camera.width());
            const uint8_t* mask_data =
                mask.is_none() ? nullptr : mask_array.data();
            py::gil_scoped_release release;
            coverage.AddFrame(camera, exposure, view, mask_data);
          },
          "camera"_a, "T_world_cam"_a, "image"_a = py::none(),
          "mask"_a = py::none(),
          "Adds a frame; T_world_cam is (4, 4) or (2, 4, 4) for the first and "
          "last row. Without an image every visible point counts.")
      .def("result",
           [](const CameraCoverage& coverage) {
             const CoverageResult result = coverage.Result();
             py::array_t<float> color(
                 {static_cast<py::ssize_t>(result.color.size()),
                  py::ssize_t{kRgbChannels}});
             auto color_rows = color.mutable_unchecked<2>();
             for (size_t i = 0; i < result.color.size(); ++i) {
               for (int channel = 0; channel < kRgbChannels; ++channel) {
                 color_rows(static_cast<py::ssize_t>(i), channel) =
                     result.color[i][channel];
               }
             }
             py::dict out;
             out["views"] = ToArray(result.views);
             out["good_views"] = ToArray(result.good_views);
             out["best_gsd"] = ToArray(result.best_gsd);
             out["color"] = color;
             out["color_weight"] = ToArray(result.color_weight);
             return out;
           })
      .def_property_readonly("frames", &CameraCoverage::frames);

  py::class_<TrainingMaskConfig>(m, "TrainingMaskConfig")
      .def(py::init<>())
      .def_readwrite("max_range", &TrainingMaskConfig::max_range)
      .def_readwrite("point_spacing", &TrainingMaskConfig::point_spacing)
      .def_readwrite("min_radius", &TrainingMaskConfig::min_radius)
      .def_readwrite("max_radius", &TrainingMaskConfig::max_radius)
      .def_readwrite("box_step", &TrainingMaskConfig::box_step)
      .def_readwrite("dilate", &TrainingMaskConfig::dilate);
  py::class_<DepthAlignConfig>(m, "DepthAlignConfig")
      .def(py::init<>())
      .def_readwrite("min_range", &DepthAlignConfig::min_range)
      .def_readwrite("max_range", &DepthAlignConfig::max_range)
      .def_readwrite("min_pixels", &DepthAlignConfig::min_pixels)
      .def_readwrite("fit_shift", &DepthAlignConfig::fit_shift)
      .def_readwrite("iterations", &DepthAlignConfig::iterations)
      .def_readwrite("robust_scale", &DepthAlignConfig::robust_scale)
      .def_readwrite("max_residual", &DepthAlignConfig::max_residual)
      .def_readwrite("min_scale", &DepthAlignConfig::min_scale)
      .def_readwrite("max_scale", &DepthAlignConfig::max_scale)
      .def_readwrite("cell", &DepthAlignConfig::cell)
      .def_readwrite("min_cell_pixels", &DepthAlignConfig::min_cell_pixels)
      .def_readwrite("max_cell_residual", &DepthAlignConfig::max_cell_residual)
      .def_readwrite("tolerance", &DepthAlignConfig::tolerance)
      .def_readwrite("min_confidence", &DepthAlignConfig::min_confidence)
      .def_readwrite("hole_radius", &DepthAlignConfig::hole_radius);
  m.attr("DEPTH_NONE") = static_cast<int>(kDepthNone);
  m.attr("DEPTH_LIDAR") = static_cast<int>(kDepthLidar);
  m.attr("DEPTH_PRIOR") = static_cast<int>(kDepthPrior);

  m.def(
      "fuse_depth_prior",
      [](const py::object& prior, const py::object& confidence,
         const py::object& lidar, const py::object& samples,
         const DepthAlignConfig& config) {
        auto shape = F64Array::ensure(lidar);
        Require(shape && shape.ndim() == 2, "lidar must be (height, width)");
        const int height = static_cast<int>(shape.shape(0));
        const int width = static_cast<int>(shape.shape(1));
        Require(config.cell > 0, "cell must be positive");
        const std::vector<float> prior_range = ToDepth(prior, height, width);
        const std::vector<float> prior_confidence =
            ToDepth(confidence, height, width);
        const std::vector<float> lidar_range = ToDepth(lidar, height, width);
        Require(!prior_range.empty(), "prior must be (height, width)");
        std::vector<DepthSample> depth_samples;
        if (!samples.is_none()) {
          F64Array array = F64Array::ensure(samples);
          Require(array && array.ndim() == 2 &&
                      array.shape(1) == kDepthSampleColumns,
                  "samples must be (N, 3) rows of u, v, range");
          const double* data = array.data();
          depth_samples.resize(static_cast<size_t>(array.shape(0)));
          for (size_t i = 0; i < depth_samples.size(); ++i) {
            const double* row = data + kDepthSampleColumns * i;
            depth_samples[i] = {row[0], row[1], row[2]};
          }
        }
        FusedDepth fused;
        {
          py::gil_scoped_release release;
          fused = FuseDepthPrior(prior_range, prior_confidence, lidar_range,
                                 depth_samples, height, width, config);
        }
        const DepthAlignment& fit = fused.alignment;
        py::dict alignment;
        alignment["valid"] = fit.valid;
        alignment["scale"] = fit.scale;
        alignment["shift"] = fit.shift;
        alignment["raw_ratio"] = fit.raw_ratio;
        alignment["pixels"] = fit.pixels;
        alignment["residual_median"] = fit.residual_median;
        alignment["residual_p90"] = fit.residual_p90;
        alignment["inlier_fraction"] = fit.inlier_fraction;
        alignment["cells_checked"] = fit.cells_checked;
        alignment["cells_rejected"] = fit.cells_rejected;
        alignment["prior_pixels"] = fused.prior_pixels;
        return py::make_tuple(ToImage(fused.range, height, width),
                              ToImage(fused.confidence, height, width),
                              ToImage(fused.source, height, width), alignment);
      },
      "prior"_a, "confidence"_a, "lidar"_a, "samples"_a = py::none(),
      "config"_a = DepthAlignConfig(),
      "Fits the prior range to the LiDAR (l = s p + b, robust) at the "
      "samples ((N, 3) u, v, range of visible map points; None: the lidar "
      "pixels) and fills the LiDAR holes with it; returns (range, "
      "confidence, source HxW uint8 DEPTH_*, alignment dict). Ranges along "
      "the ray, NaN where unknown; confidence in [0, 1] or None.");
  m.attr("MASK_MOVING") = static_cast<int>(kMaskMoving);
  m.attr("MASK_CHANGED") = static_cast<int>(kMaskChanged);

  m.def(
      "render_training_mask",
      [](const CameraModel& camera, const py::object& world_from_camera,
         const PointCloudIndex& static_points, const py::object& moving,
         const py::object& changed, const std::vector<ObjectBox>& boxes,
         const TrainingMaskConfig& config) {
        const CameraExposure exposure =
            ExposureFrom(world_from_camera, camera.height(), false);
        const Points moving_points = PointsFrom(moving);
        const Points changed_points = PointsFrom(changed);
        std::vector<uint8_t> mask;
        {
          py::gil_scoped_release release;
          mask = RenderTrainingMask(camera, exposure, static_points,
                                    moving_points, changed_points, boxes,
                                    config);
        }
        return ToImage(mask, camera.height(), camera.width());
      },
      "camera"_a, "T_world_cam"_a, "static_points"_a, "moving"_a = py::none(),
      "changed"_a = py::none(), "boxes"_a = std::vector<ObjectBox>(),
      "config"_a = TrainingMaskConfig(),
      "HxW uint8 mask: MASK_MOVING where moving points or boxes are the "
      "nearest surface, MASK_CHANGED for changed points; dilated.");
}

}  // namespace gr::bindings
