// Bindings of the library primitives: geometry, cameras, registration,
// odometry, Scan Context, pose graph, visibility, projection, rasters,
// occupancy and the box-scene simulator.

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bind_util.h"
#include "gr/camera.h"
#include "gr/geometry.h"
#include "gr/image.h"
#include "gr/occupancy.h"
#include "gr/odometry.h"
#include "gr/pose_graph.h"
#include "gr/projector.h"
#include "gr/raster.h"
#include "gr/raycast.h"
#include "gr/registration.h"
#include "gr/scan_context.h"
#include "gr/sim.h"
#include "gr/visibility.h"
#include "gr/voxel.h"

namespace gr::bindings {
namespace {

namespace py = ::pybind11;
using ::pybind11::literals::operator""_a;

py::dict IcpDict(const IcpResult& result) {
  return py::dict("T"_a = result.pose.Matrix(), "rmse"_a = result.rmse,
                  "fitness"_a = result.fitness, "inliers"_a = result.inliers,
                  "iterations"_a = result.iterations,
                  "converged"_a = result.converged,
                  "hessian"_a = result.hessian);
}

std::vector<Mat4> PoseMatrices(const std::vector<Pose>& poses) {
  std::vector<Mat4> out;
  out.reserve(poses.size());
  for (const Pose& pose : poses) out.push_back(pose.Matrix());
  return out;
}

Reducer ParseReducer(const std::string& name) {
  if (name == "mean") return Reducer::kMean;
  if (name == "max") return Reducer::kMax;
  if (name == "min") return Reducer::kMin;
  if (name == "count") return Reducer::kCount;
  Require(name == "sum", "unknown reducer: " + name);
  return Reducer::kSum;
}

// An (H, W, 3) uint8 image as a view; `holder` keeps the converted array.
ImageView RgbView(
    const py::object& image,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& holder) {
  holder =
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast>::ensure(
          image);
  Require(holder && holder.ndim() == 3 && holder.shape(2) == 3,
          "image must be HxWx3 uint8");
  return ImageView{holder.data(), static_cast<int>(holder.shape(0)),
                   static_cast<int>(holder.shape(1)), 3};
}

void BindRegistration(py::module_& m) {
  py::class_<VoxelHashMap>(m, "VoxelHashMap")
      .def(
          py::init([](double voxel_size, double max_distance,
                      int max_points_per_voxel) {
            Require(voxel_size > 0.0, "voxel_size must be positive");
            return VoxelHashMap(voxel_size, max_distance, max_points_per_voxel);
          }),
          "voxel_size"_a, "max_distance"_a = 1e9, "max_points_per_voxel"_a = 20)
      .def(
          "add_points",
          [](VoxelHashMap& map, const PointsIn& points, int frame_id) {
            map.AddPoints(ToPoints(points), frame_id);
          },
          "points"_a, "frame_id"_a = -1)
      .def("remove_far", &VoxelHashMap::RemoveFar, "origin"_a)
      .def("point_cloud",
           [](const VoxelHashMap& map) { return ToMatrix(map.PointCloud()); })
      .def("clear", &VoxelHashMap::Clear)
      .def_property_readonly("num_voxels", &VoxelHashMap::num_voxels)
      .def_property_readonly("num_points", &VoxelHashMap::CountPoints)
      .def_property_readonly("voxel_size", &VoxelHashMap::voxel_size);

  m.def(
      "icp",
      [](const PointsIn& source, const VoxelHashMap& map, const Mat4& init,
         double max_corr_dist, double kernel, int max_iters, double conv_eps,
         int exclude_frame, bool point_to_plane, double plane_radius,
         double max_plane_radius, double point_weight) {
        IcpOptions options;
        options.max_corr_dist = max_corr_dist;
        options.kernel = kernel;
        options.max_iters = max_iters;
        options.conv_eps = conv_eps;
        options.exclude_frame = exclude_frame;
        options.point_to_plane = point_to_plane;
        options.plane_radius = plane_radius;
        options.max_plane_radius = max_plane_radius;
        options.point_weight = point_weight;
        const Points points = ToPoints(source);
        IcpResult result;
        {
          py::gil_scoped_release release;
          result = AlignToMap(points, map, Pose::FromMatrix(init), options);
        }
        return IcpDict(result);
      },
      "source"_a, "map"_a, "init"_a, "max_corr_dist"_a = 1.0, "kernel"_a = 0.3,
      "max_iters"_a = 50, "conv_eps"_a = 1e-4, "exclude_frame"_a = -1,
      "point_to_plane"_a = true, "plane_radius"_a = 0.0,
      "max_plane_radius"_a = 1.2, "point_weight"_a = 0.05,
      "Robust ICP (point-to-plane where the map is planar) of `source` "
      "(sensor frame) against a VoxelHashMap; returns a dict with T "
      "(world<-source), rmse, fitness, inliers, iterations, converged, "
      "hessian.");

  m.def(
      "icp_multiscale",
      [](const PointsIn& source, const PointsIn& target, const Mat4& init,
         const std::vector<double>& schedule, double base_voxel,
         int max_iters) {
        Require(base_voxel > 0.0, "base_voxel must be positive");
        const Points s = ToPoints(source);
        const Points t = ToPoints(target);
        IcpResult result;
        {
          py::gil_scoped_release release;
          result = AlignMultiScale(s, t, Pose::FromMatrix(init), schedule,
                                   base_voxel, max_iters);
        }
        return IcpDict(result);
      },
      "source"_a, "target"_a, "init"_a, "schedule"_a, "base_voxel"_a,
      "max_iters"_a = 40,
      "Coarse-to-fine ICP of a source cloud against a target cloud; "
      "statistics at the finest gate.");
}

void BindOdometry(py::module_& m) {
  py::class_<OdometryConfig>(m, "OdometryConfig")
      .def(py::init<>())
      .def_readwrite("max_range", &OdometryConfig::max_range)
      .def_readwrite("min_range", &OdometryConfig::min_range)
      .def_readwrite("voxel_size", &OdometryConfig::voxel_size)
      .def_readwrite("max_points_per_voxel",
                     &OdometryConfig::max_points_per_voxel)
      .def_readwrite("initial_threshold", &OdometryConfig::initial_threshold)
      .def_readwrite("min_motion_th", &OdometryConfig::min_motion_th)
      .def_readwrite("max_icp_iters", &OdometryConfig::max_icp_iters)
      .def_readwrite("icp_conv", &OdometryConfig::icp_conv)
      .def_readwrite("deskew", &OdometryConfig::deskew)
      .def_readwrite("use_wheel_prior", &OdometryConfig::use_wheel_prior)
      .def_readwrite("use_gyro_prior", &OdometryConfig::use_gyro_prior);

  py::class_<Odometry>(m, "Odometry")
      .def(py::init([](const OdometryConfig& config, const Mat4& initial) {
             Require(config.voxel_size > 0.0, "voxel_size must be positive");
             return Odometry(config, Pose::FromMatrix(initial));
           }),
           "config"_a, "initial_pose"_a = Mat4::Identity())
      .def(
          "register_frame",
          [](Odometry& odometry, const PointsIn& points,
             const py::object& times, double stamp, const py::object& prior) {
            const Points p = ToPoints(points);
            const std::vector<double> t = ToDoubleVector(times);
            std::optional<Pose> predicted;
            if (!prior.is_none()) {
              predicted = Pose::FromMatrix(prior.cast<Mat4>());
            }
            Pose pose;
            {
              py::gil_scoped_release release;
              pose = odometry.RegisterFrame(p, t, stamp, predicted);
            }
            return pose.Matrix();
          },
          "points"_a, "times"_a = py::none(), "stamp"_a = 0.0,
          "prior_delta"_a = py::none(),
          "Registers one sweep (sensor frame). `times` are seconds relative "
          "to `stamp` (<= 0).")
      .def("poses",
           [](const Odometry& odometry) {
             return PoseMatrices(odometry.poses());
           })
      .def("local_map",
           [](const Odometry& odometry) {
             return ToMatrix(odometry.LocalMap());
           })
      .def("last_deskewed",
           [](const Odometry& odometry) {
             return ToMatrix(odometry.last_deskewed());
           })
      .def(
          "last_icp",
          [](const Odometry& odometry) { return IcpDict(odometry.last_icp()); })
      .def_property_readonly("last_sigma", &Odometry::last_sigma);

  m.def(
      "deskew",
      [](const PointsIn& points, const Eigen::VectorXd& times,
         const Vec3& omega, const Vec3& velocity) {
        const std::vector<double> t(times.data(), times.data() + times.size());
        return ToMatrix(
            DeskewConstantVelocity(ToPoints(points), t, omega, velocity));
      },
      "points"_a, "times"_a, "omega"_a, "velocity"_a);

  m.def(
      "scan_context",
      [](const PointsIn& points, int rings, int sectors, double max_radius,
         double lidar_height) {
        Require(rings > 0 && sectors > 0 && max_radius > 0.0,
                "rings, sectors and max_radius must be positive");
        ScanContextConfig config;
        config.rings = rings;
        config.sectors = sectors;
        config.max_radius = max_radius;
        config.lidar_height = lidar_height;
        return ComputeScanContext(ToPoints(points), config);
      },
      "points"_a, "rings"_a = 20, "sectors"_a = 60, "max_radius"_a = 30.0,
      "lidar_height"_a = 2.0);
  m.def("scan_context_ring_key", &ScanContextRingKey, "sc"_a);
  m.def(
      "scan_context_distance",
      [](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
        const ScanContextMatch match = MatchScanContexts(a, b);
        return py::make_tuple(match.distance, match.shift, match.yaw);
      },
      "a"_a, "b"_a,
      "Returns (distance, column shift, yaw of b relative to a).");
}

void BindPoseGraph(py::module_& m) {
  py::class_<PoseGraph>(m, "PoseGraph")
      .def(py::init<>())
      .def(
          "add_node",
          [](PoseGraph& graph, const Mat4& pose, bool fixed) {
            return graph.AddNode(Pose::FromMatrix(pose), fixed);
          },
          "pose"_a, "fixed"_a = false)
      .def(
          "set_fixed",
          [](PoseGraph& graph, int i, bool fixed) {
            RequireIndex(i >= 0 && i < static_cast<int>(graph.num_nodes()),
                         "invalid node index");
            graph.SetFixed(i, fixed);
          },
          "i"_a, "fixed"_a)
      .def(
          "set_pose",
          [](PoseGraph& graph, int i, const Mat4& pose) {
            RequireIndex(i >= 0 && i < static_cast<int>(graph.num_nodes()),
                         "invalid node index");
            graph.SetPose(i, Pose::FromMatrix(pose));
          },
          "i"_a, "pose"_a)
      .def(
          "add_between",
          [](PoseGraph& graph, int i, int j, const Mat4& measurement,
             const Mat6& information, double robust_scale) {
            const int n = static_cast<int>(graph.num_nodes());
            RequireIndex(i >= 0 && i < n && j >= 0 && j < n,
                         "add_between: invalid node index");
            graph.AddBetween(i, j, Pose::FromMatrix(measurement), information,
                             robust_scale);
          },
          "i"_a, "j"_a, "measurement"_a, "information"_a,
          "robust_scale"_a = 0.0)
      .def(
          "add_prior",
          [](PoseGraph& graph, int i, const Mat4& measurement,
             const Mat6& information) {
            RequireIndex(i >= 0 && i < static_cast<int>(graph.num_nodes()),
                         "add_prior: invalid node index");
            graph.AddPrior(i, Pose::FromMatrix(measurement), information);
          },
          "i"_a, "measurement"_a, "information"_a)
      .def(
          "add_position",
          [](PoseGraph& graph, int i, const Vec3& measurement,
             const Mat3& information, const Vec3& lever) {
            RequireIndex(i >= 0 && i < static_cast<int>(graph.num_nodes()),
                         "add_position: invalid node index");
            graph.AddPosition(i, measurement, information, lever);
          },
          "i"_a, "measurement"_a, "information"_a, "lever"_a = Vec3::Zero())
      .def(
          "add_gravity",
          [](PoseGraph& graph, int i, const Vec3& up_body, double sigma_rad) {
            RequireIndex(i >= 0 && i < static_cast<int>(graph.num_nodes()),
                         "add_gravity: invalid node index");
            Require(sigma_rad > 0.0, "sigma_rad must be positive");
            graph.AddGravity(i, up_body, sigma_rad);
          },
          "i"_a, "up_body"_a, "sigma_rad"_a,
          "Roll/pitch constraint: world +z measured in the node frame (e.g. "
          "from the accelerometer).")
      .def(
          "optimize",
          [](PoseGraph& graph, int max_iterations, double lambda_init,
             double rel_tol) {
            PoseGraphOptions options;
            options.max_iterations = max_iterations;
            options.lambda_init = lambda_init;
            options.rel_tol = rel_tol;
            PoseGraphSummary summary;
            {
              py::gil_scoped_release release;
              summary = graph.Optimize(options);
            }
            return py::dict("iterations"_a = summary.iterations,
                            "initial_cost"_a = summary.initial_cost,
                            "final_cost"_a = summary.final_cost,
                            "converged"_a = summary.converged);
          },
          "max_iterations"_a = 50, "lambda_init"_a = 1e-4, "rel_tol"_a = 1e-10)
      .def("poses",
           [](const PoseGraph& graph) { return PoseMatrices(graph.poses()); })
      .def("between_chi2",
           [](const PoseGraph& graph) { return ToArray(graph.BetweenChi2()); })
      .def_property_readonly("num_nodes", &PoseGraph::num_nodes);

  m.def(
      "between_residual",
      [](const Mat4& pose_i, const Mat4& pose_j, const Mat4& measurement) {
        Mat6 jacobian_i, jacobian_j;
        const Vec6 e = PoseGraph::BetweenResidual(
            Pose::FromMatrix(pose_i), Pose::FromMatrix(pose_j),
            Pose::FromMatrix(measurement), &jacobian_i, &jacobian_j);
        return py::make_tuple(e, jacobian_i, jacobian_j);
      },
      "Ti"_a, "Tj"_a, "Z"_a);
}

void BindVisibility(py::module_& m) {
  py::class_<FreeSpaceCarver>(m, "FreeSpaceCarver")
      .def(py::init([](const Vec3& min_bound, const Vec3& max_bound,
                       double voxel, double endpoint_margin) {
             Require(voxel > 0.0, "voxel must be positive");
             Require(FreeSpaceCarver::GridSize(min_bound, max_bound, voxel) <=
                         FreeSpaceCarver::kMaxVoxels,
                     "FreeSpaceCarver: grid too large, increase voxel size");
             return FreeSpaceCarver(min_bound, max_bound, voxel,
                                    endpoint_margin);
           }),
           "min_bound"_a, "max_bound"_a, "voxel"_a, "endpoint_margin"_a = 0.2)
      .def(
          "integrate",
          [](FreeSpaceCarver& carver, const Vec3& origin,
             const PointsIn& endpoints) {
            const Points p = ToPoints(endpoints);
            py::gil_scoped_release release;
            carver.Integrate(origin, p);
          },
          "origin"_a, "endpoints"_a)
      .def(
          "query",
          [](const FreeSpaceCarver& carver, const PointsIn& points) {
            const FreeSpaceCarver::Counts counts =
                carver.Query(ToPoints(points));
            return py::make_tuple(ToArray(counts.hits), ToArray(counts.misses));
          },
          "points"_a,
          "Returns (hits, misses) of the voxel containing each point.")
      .def(
          "free_mask",
          [](const FreeSpaceCarver& carver, int min_misses, int max_hits) {
            const std::vector<uint8_t> free =
                carver.FreeMask(min_misses, max_hits);
            const Eigen::Vector3i dims = carver.dims();
            py::array_t<uint8_t> out({static_cast<py::ssize_t>(dims.x()),
                                      static_cast<py::ssize_t>(dims.y()),
                                      static_cast<py::ssize_t>(dims.z())});
            std::memcpy(out.mutable_data(), free.data(), free.size());
            return out;
          },
          "min_misses"_a = 2, "max_hits"_a = 0,
          "(nx, ny, nz) uint8: 1 where at least min_misses rays passed and "
          "at most max_hits endpoints fell.")
      .def_property_readonly("min_bound", &FreeSpaceCarver::min_bound)
      .def_property_readonly("dims", &FreeSpaceCarver::dims)
      .def_property_readonly("voxel", &FreeSpaceCarver::voxel);

  py::class_<VisibilityVoter>(m, "VisibilityVoter")
      .def(py::init([](const PointsIn& map, const PointsIn& normals,
                       double bucket) {
             Require(map.rows() == normals.rows(),
                     "map and normals must have the same length");
             Require(bucket > 0.0, "bucket_size must be positive");
             return VisibilityVoter(ToPoints(map), ToPoints(normals), bucket);
           }),
           "map"_a, "normals"_a, "bucket_size"_a = 4.0)
      .def(
          "add_sweep",
          [](VisibilityVoter& voter, const Mat4& world_from_lidar,
             const PointsIn& sweep, double az_res, double el_res, double el_min,
             double el_max, double max_range, double margin_abs,
             double margin_rel, double min_cos, int session) {
            Require(az_res > 0.0 && el_res > 0.0 && el_max > el_min,
                    "invalid range-image resolution");
            VisibilityVoteOptions options;
            options.image.az_res_deg = az_res;
            options.image.el_res_deg = el_res;
            options.image.el_min_deg = el_min;
            options.image.el_max_deg = el_max;
            options.max_range = max_range;
            options.margin_abs = margin_abs;
            options.margin_rel = margin_rel;
            options.min_cos = min_cos;
            const Points p = ToPoints(sweep);
            py::gil_scoped_release release;
            voter.AddSweep(Pose::FromMatrix(world_from_lidar), p, options,
                           session);
          },
          "T_world_lidar"_a, "sweep"_a, "az_res_deg"_a = 1.0,
          "el_res_deg"_a = 1.0, "el_min_deg"_a = -30.0, "el_max_deg"_a = 30.0,
          "max_range"_a = 30.0, "margin_abs"_a = 0.3, "margin_rel"_a = 0.02,
          "min_cos"_a = 0.25, "session"_a = 0)
      .def("reset_counts", &VisibilityVoter::ResetCounts)
      .def("sessions", &VisibilityVoter::Sessions)
      .def(
          "counts",
          [](const VisibilityVoter& voter, int session) {
            return py::make_tuple(ToArray(voter.SeenCounts(session)),
                                  ToArray(voter.ThroughCounts(session)));
          },
          "session"_a = 0,
          "Returns (seen, seen_through) votes per map point of one session.");
}

void BindCameras(py::module_& m) {
  py::class_<CameraModel, std::shared_ptr<CameraModel>>(m, "CameraModel")
      .def_property_readonly("width", &CameraModel::width)
      .def_property_readonly("height", &CameraModel::height)
      .def_property_readonly("type", &CameraModel::type)
      .def_property_readonly("params", &CameraModel::params)
      .def_property_readonly("focal", &CameraModel::focal)
      .def(
          "project",
          [](const CameraModel& camera, const PointsIn& points) {
            const Eigen::Index n = points.rows();
            RowMatX2d uv = RowMatX2d::Constant(
                n, 2, std::numeric_limits<double>::quiet_NaN());
            std::vector<uint8_t> valid(static_cast<size_t>(n), 0);
            std::vector<uint8_t> inside(static_cast<size_t>(n), 0);
            for (Eigen::Index i = 0; i < n; ++i) {
              const std::optional<Vec2> p =
                  camera.Project(points.row(i).transpose());
              if (!p) continue;
              uv.row(i) = p->transpose();
              valid[static_cast<size_t>(i)] = 1;
              inside[static_cast<size_t>(i)] = camera.InImage(*p) ? 1 : 0;
            }
            return py::make_tuple(uv, ToBoolArray(valid), ToBoolArray(inside));
          },
          "points"_a, "Returns (uv, valid, in_image) for camera-frame points.")
      .def(
          "unproject",
          [](const CameraModel& camera, const UvIn& uv) {
            const Eigen::Index n = uv.rows();
            RowMatX3d rays = RowMatX3d::Zero(n, 3);
            std::vector<uint8_t> valid(static_cast<size_t>(n), 0);
            for (Eigen::Index i = 0; i < n; ++i) {
              const std::optional<Vec3> ray =
                  camera.Unproject(uv.row(i).transpose());
              if (!ray) continue;
              rays.row(i) = ray->transpose();
              valid[static_cast<size_t>(i)] = 1;
            }
            return py::make_tuple(rays, ToBoolArray(valid));
          },
          "uv"_a, "Returns (unit rays, valid).");

  m.def(
      "make_camera",
      [](const std::string& type, int width, int height,
         const std::vector<double>& params) {
        Require(width > 0 && height > 0, "camera size must be positive");
        std::shared_ptr<CameraModel> camera =
            MakeCamera(type, width, height, params);
        Require(camera != nullptr,
                "unknown camera model or too few parameters: " + type);
        return camera;
      },
      "type"_a, "width"_a, "height"_a, "params"_a);
}

void BindProjection(py::module_& m) {
  m.def(
      "render_points",
      [](const CameraModel& camera, const py::object& camera_from_world,
         const PointsIn& points, const py::object& labels, double point_spacing,
         int min_radius, int max_radius, double min_range, double max_range) {
        const CameraExposure exposure =
            ExposureFrom(camera_from_world, camera.height(), true);
        const Points p = ToPoints(points);
        const std::vector<int> l = ToIntVector(labels);
        RenderResult render;
        {
          py::gil_scoped_release release;
          render = RenderPoints(camera, exposure, p, l, point_spacing,
                                min_radius, max_radius, min_range, max_range);
        }
        return py::make_tuple(
            ToImage(render.depth, render.height, render.width),
            ToImage(render.label, render.height, render.width),
            ToImage(render.index, render.height, render.width));
      },
      "camera"_a, "T_cam_world"_a, "points"_a, "labels"_a = py::none(),
      "point_spacing"_a = 0.05, "min_radius"_a = 0, "max_radius"_a = 4,
      "min_range"_a = 0.1, "max_range"_a = 100.0,
      "Z-buffer splatting; returns (range HxW float32, label HxW int32, "
      "index HxW int32). T_cam_world is (4, 4), or (2, 4, 4) for the first "
      "and last row of a rolling-shutter image.");

  m.def(
      "project_visibility",
      [](const CameraModel& camera, const py::object& camera_from_world,
         const PointsIn& points, const py::object& depth, double abs_tol,
         double rel_tol) {
        const CameraExposure exposure =
            ExposureFrom(camera_from_world, camera.height(), true);
        const Points p = ToPoints(points);
        const std::vector<float> d =
            ToDepth(depth, camera.height(), camera.width());
        VisibilityResult visibility;
        {
          py::gil_scoped_release release;
          visibility =
              ProjectVisibility(camera, exposure, p, d, abs_tol, rel_tol);
        }
        return py::make_tuple(visibility.uv, ToBoolArray(visibility.in_image),
                              ToBoolArray(visibility.visible),
                              ToArray(visibility.range));
      },
      "camera"_a, "T_cam_world"_a, "points"_a, "depth"_a = py::none(),
      "abs_tol"_a = 0.1, "rel_tol"_a = 0.03,
      "Returns (uv, in_image, visible, range); T_cam_world as for "
      "render_points.");

  py::class_<PointCloudIndex>(m, "PointCloudIndex")
      .def(py::init([](const py::object& points, double bucket) {
             Require(bucket > 0.0, "bucket must be positive");
             return PointCloudIndex(PointsFrom(points), bucket);
           }),
           "points"_a, "bucket"_a = 4.0)
      .def("__len__", &PointCloudIndex::size)
      .def(
          "radius_search",
          [](const PointCloudIndex& index, const Vec3& center, double radius) {
            return ToArray(index.RadiusSearch(center, radius));
          },
          "center"_a, "radius"_a)
      .def(
          "radius_search_xy",
          [](const PointCloudIndex& index, const Vec2& center, double radius) {
            return ToArray(index.RadiusSearchXy(center, radius));
          },
          "center"_a, "radius"_a);
}

void BindRasters(py::module_& m) {
  m.def(
      "rasterize",
      [](const UvIn& xy, const Eigen::VectorXd& values, double x0, double y0,
         double res, int width, int height, const std::string& reducer,
         double fill) {
        Require(xy.rows() == values.size(), "xy and values size mismatch");
        Require(res > 0.0 && width >= 0 && height >= 0, "invalid raster grid");
        Grid2D grid;
        grid.x0 = x0;
        grid.y0 = y0;
        grid.res = res;
        grid.width = width;
        grid.height = height;
        const RowMatX2d points = xy;
        const Reducer r = ParseReducer(reducer);
        py::gil_scoped_release release;
        return Rasterize(points, values, grid, r, fill);
      },
      "xy"_a, "values"_a, "x0"_a, "y0"_a, "res"_a, "width"_a, "height"_a,
      "reducer"_a = "mean", "fill"_a = std::numeric_limits<double>::quiet_NaN(),
      "Returns a (height, width) raster; row j <-> y0 + j*res.");

  using RgbArray =
      py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
  py::class_<FloorTextureFuser>(m, "FloorTextureFuser")
      .def(py::init([](double x0, double y0, double res, int width, int height,
                       const Eigen::MatrixXd& floor_z, int top_k) {
             Require(floor_z.rows() == height && floor_z.cols() == width,
                     "floor_z must be (height, width)");
             Grid2D grid;
             grid.x0 = x0;
             grid.y0 = y0;
             grid.res = res;
             grid.width = width;
             grid.height = height;
             return FloorTextureFuser(grid, floor_z, top_k);
           }),
           "x0"_a, "y0"_a, "res"_a, "width"_a, "height"_a, "floor_z"_a,
           "top_k"_a = 5)
      .def(
          "add_frame",
          [](FloorTextureFuser& fuser, const CameraModel& camera,
             const py::object& camera_from_world, const py::object& image,
             const py::object& depth, double depth_tol, double max_range,
             double min_cos) {
            const CameraExposure exposure =
                ExposureFrom(camera_from_world, camera.height(), true);
            RgbArray holder;
            const ImageView view = RgbView(image, holder);
            const std::vector<float> d =
                ToDepth(depth, view.height, view.width);
            py::gil_scoped_release release;
            fuser.AddFrame(camera, exposure, view,
                           d.empty() ? nullptr : d.data(), depth_tol, max_range,
                           min_cos);
          },
          "camera"_a, "T_cam_world"_a, "image"_a, "depth"_a = py::none(),
          "depth_tol"_a = 0.15, "max_range"_a = 8.0, "min_cos"_a = 0.2)
      .def(
          "add_frame_occluded",
          [](FloorTextureFuser& fuser, const CameraModel& camera,
             const py::object& camera_from_world, const py::object& image,
             const PointCloudIndex& occluders, double point_spacing,
             double depth_tol, double max_range, double min_cos) {
            const CameraExposure exposure =
                ExposureFrom(camera_from_world, camera.height(), true);
            RgbArray holder;
            const ImageView view = RgbView(image, holder);
            Require(
                view.height == camera.height() && view.width == camera.width(),
                "image must have the camera's size");
            py::gil_scoped_release release;
            fuser.AddFrameOccluded(camera, exposure, view, occluders,
                                   point_spacing, depth_tol, max_range,
                                   min_cos);
          },
          "camera"_a, "T_cam_world"_a, "image"_a, "occluders"_a,
          "point_spacing"_a = 0.05, "depth_tol"_a = 0.15, "max_range"_a = 8.0,
          "min_cos"_a = 0.2,
          "Adds an image; the occlusion depth is rendered from the occluder "
          "points near the camera.")
      .def("result", [](const FloorTextureFuser& fuser) {
        const FloorTextureFuser::Texture texture = fuser.ComputeTexture();
        const int height = fuser.grid().height;
        const int width = fuser.grid().width;
        return py::make_tuple(ToImage(texture.rgb, height, width, 3),
                              ToImage(texture.score, height, width),
                              ToImage(texture.observations, height, width));
      });
}

void BindOccupancy(py::module_& m) {
  m.def(
      "build_occupancy",
      [](const Vec3& min_bound, const Vec3& max_bound, double voxel,
         int free_label, int camera_pixel_stride, double camera_max_range,
         const Mat4& world_from_ego, const PointsIn& static_world,
         const py::object& static_labels,
         const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 8,
                                              Eigen::RowMajor>>& boxes,
         const Vec3& lidar_origin_ego, const PointsIn& scan_ego,
         const std::vector<py::tuple>& cameras) {
        Require(voxel > 0.0, "voxel must be positive");
        OccupancyConfig config;
        config.min_bound = min_bound;
        config.max_bound = max_bound;
        config.voxel = voxel;
        config.free_label = free_label;
        config.camera_pixel_stride = camera_pixel_stride;
        config.camera_max_range = camera_max_range;
        std::vector<OrientedBox> oriented;
        for (Eigen::Index i = 0; i < boxes.rows(); ++i) {
          OrientedBox box;
          box.center = boxes.row(i).segment<3>(0).transpose();
          box.size = boxes.row(i).segment<3>(3).transpose();
          box.yaw = boxes(i, 6);
          box.label = static_cast<int>(boxes(i, 7));
          oriented.push_back(box);
        }
        std::vector<CameraView> views;
        for (const py::tuple& camera : cameras) {
          Require(camera.size() == 2,
                  "cameras: (CameraModel, T_ego_cam) pairs");
          CameraView view;
          view.camera = camera[0].cast<std::shared_ptr<CameraModel>>();
          view.ego_from_camera = Pose::FromMatrix(camera[1].cast<Mat4>());
          views.push_back(view);
        }
        const Points points = ToPoints(static_world);
        const Points scan = ToPoints(scan_ego);
        const std::vector<int> labels = ToIntVector(static_labels);
        OccupancyResult result;
        {
          py::gil_scoped_release release;
          result =
              BuildOccupancy(config, Pose::FromMatrix(world_from_ego), points,
                             labels, oriented, lidar_origin_ego, scan, views);
        }
        const std::vector<py::ssize_t> shape = {result.nx, result.ny,
                                                result.nz};
        const auto grid = [&](const std::vector<uint8_t>& values) {
          py::array_t<uint8_t> array(shape);
          std::memcpy(array.mutable_data(), values.data(), values.size());
          return array;
        };
        return py::make_tuple(grid(result.semantics), grid(result.mask_lidar),
                              grid(result.mask_camera));
      },
      "min_bound"_a, "max_bound"_a, "voxel"_a, "free_label"_a,
      "camera_pixel_stride"_a, "camera_max_range"_a, "T_world_ego"_a,
      "static_world"_a, "static_labels"_a, "boxes"_a, "lidar_origin_ego"_a,
      "scan_ego"_a, "cameras"_a,
      "boxes: (M, 8) [cx, cy, cz, length, width, height, yaw, label] in the "
      "ego frame; cameras: list of (CameraModel, T_ego_cam). Returns "
      "(semantics, mask_lidar, mask_camera) as (nx, ny, nz) uint8.");
}

void BindGeometry(py::module_& m) {
  m.def("so3_exp", &So3Exp, "w"_a);
  m.def("so3_log", &So3Log, "R"_a);
  m.def(
      "voxel_downsample_indices",
      [](const PointsIn& points, double voxel) {
        return ToArray(VoxelDownsampleIndices(ToPoints(points), voxel));
      },
      "points"_a, "voxel"_a,
      "Index of the first point in every occupied voxel.");
  m.def(
      "interpolate_poses",
      [](const std::vector<double>& times, const py::object& poses,
         const std::vector<double>& query) {
        const std::vector<Pose> p = PosesFrom(poses);
        Require(!times.empty() && times.size() == p.size(),
                "times and poses must be non-empty and of equal length");
        return PosesToArray(InterpolatePoses(times, p, query));
      },
      "times"_a, "poses"_a, "query"_a,
      "Slerp / lerp of an (N, 4, 4) pose sequence at query times (clamped).");
  m.def(
      "estimate_normals",
      [](const PointsIn& points, double radius, int min_neighbors,
         int max_neighbors) {
        Require(radius > 0.0, "radius must be positive");
        const Points p = ToPoints(points);
        NormalsResult normals;
        {
          py::gil_scoped_release release;
          normals = EstimateNormals(p, radius, min_neighbors, max_neighbors);
        }
        return py::make_tuple(
            ToMatrix(normals.normals), ToArray(normals.planarity),
            ToArray(normals.thickness), ToArray(normals.neighbors));
      },
      "points"_a, "radius"_a, "min_neighbors"_a = 5, "max_neighbors"_a = 64,
      "Returns (normals, planarity, thickness, neighbor_count).");
  m.def(
      "ransac_plane",
      [](const PointsIn& points, double threshold, int iterations, int seed) {
        const PlaneFit fit =
            FitPlaneRansac(ToPoints(points), threshold, iterations, seed);
        return py::make_tuple(fit.normal, fit.d, ToArray(fit.inliers));
      },
      "points"_a, "threshold"_a, "iterations"_a = 200, "seed"_a = 1);
  m.def(
      "plane_thickness",
      [](const PointsIn& points, double voxel, int min_points,
         double min_planarity) {
        Require(voxel > 0.0, "voxel must be positive");
        const ThicknessStats stats = MeasurePlaneThickness(
            ToPoints(points), voxel, min_points, min_planarity);
        return py::dict("num_voxels"_a = stats.num_voxels,
                        "mean"_a = stats.mean, "median"_a = stats.median,
                        "p90"_a = stats.p90);
      },
      "points"_a, "voxel"_a = 0.5, "min_points"_a = 20,
      "min_planarity"_a = 0.7);
}

void BindSim(py::module_& m) {
  py::class_<BoxScene>(m, "BoxScene")
      .def(py::init<>())
      .def("add_box", &BoxScene::AddBox, "center"_a, "size"_a,
           "R"_a = Mat3::Identity())
      .def("clear", &BoxScene::Clear)
      .def("__len__", &BoxScene::size)
      .def(
          "raycast",
          [](const BoxScene& scene, const PointsIn& origins,
             const PointsIn& directions, double max_t) {
            Require(origins.rows() == directions.rows() || origins.rows() == 1,
                    "raycast: origins must be N x 3 or 1 x 3");
            const RowMatX3d o = origins;
            const RowMatX3d d = directions;
            RaycastHits hits;
            {
              py::gil_scoped_release release;
              hits = scene.Raycast(o, d, max_t);
            }
            return py::make_tuple(hits.distance, hits.box_ids, hits.normals);
          },
          "origins"_a, "dirs"_a, "max_t"_a = 100.0,
          "Returns (t, box_id, normal); t=inf / id=-1 for misses.");
}

}  // namespace

void BindCore(py::module_& m) {
  BindGeometry(m);
  BindRegistration(m);
  BindOdometry(m);
  BindPoseGraph(m);
  BindVisibility(m);
  BindCameras(m);
  BindProjection(m);
  BindRasters(m);
  BindOccupancy(m);
  BindSim(m);
}

}  // namespace gr::bindings
