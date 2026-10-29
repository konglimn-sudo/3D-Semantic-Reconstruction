#include "bind_util.h"

#include <utility>

namespace gr::bindings {

namespace py = ::pybind11;

namespace {

// Elements of one 4 x 4 pose matrix.
constexpr size_t kPoseMatrixSize =
    static_cast<size_t>(kHomogeneousDim) * kHomogeneousDim;
// A rolling-shutter exposure has a pose for the first and the last row.
constexpr py::ssize_t kRollingShutterPoses = 2;
// Sizes of the (x0, y0, res, z) grid and the (level_from, level_to, plane,
// polygon, slope) ramp tuples of levels.py.
constexpr size_t kGridTupleSize = 4;
constexpr size_t kRampTupleSize = 5;

// The `pose`-th 4 x 4 row-major matrix of `data`.
Mat4 MatrixAt(const double* data, size_t pose) {
  Mat4 matrix;
  for (int row = 0; row < kHomogeneousDim; ++row) {
    for (int column = 0; column < kHomogeneousDim; ++column) {
      matrix(row, column) =
          data[kPoseMatrixSize * pose +
               static_cast<size_t>(kHomogeneousDim * row + column)];
    }
  }
  return matrix;
}

}  // namespace

void Require(bool condition, const std::string& message) {
  if (!condition) throw py::value_error(message);
}

void RequireIndex(bool condition, const std::string& message) {
  if (!condition) throw py::index_error(message);
}

py::array_t<bool> ToBoolArray(const std::vector<uint8_t>& flags) {
  py::array_t<bool> array(static_cast<py::ssize_t>(flags.size()));
  bool* data = array.mutable_data();
  for (size_t i = 0; i < flags.size(); ++i) data[i] = flags[i] != 0;
  return array;
}

py::array_t<bool> ToBoolArray(const std::vector<bool>& flags) {
  py::array_t<bool> array(static_cast<py::ssize_t>(flags.size()));
  bool* data = array.mutable_data();
  for (size_t i = 0; i < flags.size(); ++i) data[i] = flags[i];
  return array;
}

std::vector<double> ToDoubleVector(const py::object& values) {
  if (values.is_none()) return {};
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          values);
  Require(static_cast<bool>(array), "expected a float array");
  return std::vector<double>(array.data(), array.data() + array.size());
}

std::vector<float> ToFloatVector(const py::object& values) {
  if (values.is_none()) return {};
  auto array =
      py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
          values);
  Require(static_cast<bool>(array), "expected a float array");
  return std::vector<float>(array.data(), array.data() + array.size());
}

std::vector<int> ToIntVector(const py::object& values) {
  if (values.is_none()) return {};
  auto array =
      py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(
          values);
  Require(static_cast<bool>(array), "expected an integer array");
  return std::vector<int>(array.data(), array.data() + array.size());
}

std::vector<float> ToDepth(const py::object& depth, int height, int width) {
  if (depth.is_none()) return {};
  auto array =
      py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(
          depth);
  Require(array && array.ndim() == 2 && array.shape(0) == height &&
              array.shape(1) == width,
          "depth must be a (height, width) float array");
  return std::vector<float>(array.data(), array.data() + array.size());
}

Points PointsFrom(const py::object& points) {
  if (points.is_none()) return {};
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          points);
  Require(static_cast<bool>(array), "expected an (N, 3) float array");
  if (array.size() == 0) return {};
  Require(array.ndim() == 2 && array.shape(1) == kSpaceDim,
          "expected an (N, 3) float array");
  const double* data = array.data();
  Points out(static_cast<size_t>(array.shape(0)));
  for (size_t i = 0; i < out.size(); ++i) {
    const double* point = data + kSpaceDim * i;
    out[i] = Vec3(point[0], point[1], point[2]);
  }
  return out;
}

py::array_t<double> PointsToArray(const Points& points) {
  py::array_t<double> array({static_cast<py::ssize_t>(points.size()),
                             static_cast<py::ssize_t>(kSpaceDim)});
  double* data = array.mutable_data();
  for (size_t i = 0; i < points.size(); ++i) {
    for (int axis = 0; axis < kSpaceDim; ++axis) {
      data[kSpaceDim * i + static_cast<size_t>(axis)] = points[i][axis];
    }
  }
  return array;
}

std::vector<Pose> PosesFrom(const py::object& poses) {
  if (poses.is_none()) return {};
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          poses);
  Require(static_cast<bool>(array), "expected an (N, 4, 4) pose array");
  if (array.size() == 0) return {};
  Require(array.ndim() == 3 && array.shape(1) == kHomogeneousDim &&
              array.shape(2) == kHomogeneousDim,
          "expected an (N, 4, 4) pose array");
  const double* data = array.data();
  std::vector<Pose> out(static_cast<size_t>(array.shape(0)));
  for (size_t pose = 0; pose < out.size(); ++pose) {
    out[pose] = Pose::FromMatrix(MatrixAt(data, pose));
  }
  return out;
}

py::array_t<double> PosesToArray(const std::vector<Pose>& poses) {
  py::array_t<double> array({static_cast<py::ssize_t>(poses.size()),
                             static_cast<py::ssize_t>(kHomogeneousDim),
                             static_cast<py::ssize_t>(kHomogeneousDim)});
  double* data = array.mutable_data();
  for (size_t pose = 0; pose < poses.size(); ++pose) {
    const Mat4 matrix = poses[pose].Matrix();
    for (int row = 0; row < kHomogeneousDim; ++row) {
      for (int column = 0; column < kHomogeneousDim; ++column) {
        data[kPoseMatrixSize * pose +
             static_cast<size_t>(kHomogeneousDim * row + column)] =
            matrix(row, column);
      }
    }
  }
  return array;
}

CameraExposure ExposureFrom(const py::object& poses, int height,
                            bool camera_from_world) {
  auto array =
      py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
          poses);
  const bool single = array && array.ndim() == 2 &&
                      array.shape(0) == kHomogeneousDim &&
                      array.shape(1) == kHomogeneousDim;
  const bool pair = array && array.ndim() == 3 &&
                    array.shape(0) == kRollingShutterPoses &&
                    array.shape(1) == kHomogeneousDim &&
                    array.shape(2) == kHomogeneousDim;
  Require(single || pair,
          "camera pose must be (4, 4) or (2, 4, 4) for the first and last row");
  Require(height >= 1, "camera height must be positive");
  const double* data = array.data();
  const auto pose_at = [&](size_t index) {
    const Pose pose = Pose::FromMatrix(MatrixAt(data, index));
    return camera_from_world ? pose.Inverse() : pose;
  };
  if (single) return CameraExposure(pose_at(0));
  return CameraExposure(pose_at(0), pose_at(1), height);
}

Logger MakeLogger(const py::object& callable, std::exception_ptr* error) {
  if (callable.is_none()) return nullptr;
  const py::object* function = &callable;
  return [function, error](const std::string& message) {
    if (*error) return;
    py::gil_scoped_acquire gil;
    try {
      (*function)(message);
    } catch (...) {
      *error = std::current_exception();
    }
  };
}

void RethrowIfSet(const std::exception_ptr& error) {
  if (error) std::rethrow_exception(error);
}

// The level model from the Python LevelModel's parts.
LevelModel LevelModelFrom(const std::vector<double>& z,
                          const py::list& grids, const py::list& ramps) {
  LevelModel model;
  model.z = z;
  for (const py::handle& item : grids) {
    const py::tuple tuple = item.cast<py::tuple>();
    Require(tuple.size() == kGridTupleSize,
            "grids are (x0, y0, res, z) tuples");
    FloorGrid grid;
    grid.x0 = tuple[0].cast<double>();
    grid.y0 = tuple[1].cast<double>();
    grid.res = tuple[2].cast<double>();
    const auto values =
        py::array_t<float, py::array::c_style |
                               py::array::forcecast>::ensure(tuple[3]);
    Require(values && values.ndim() == 2, "a grid is an (H, W) array");
    Require(grid.res > 0.0, "grid res must be positive");
    grid.height = static_cast<int>(values.shape(0));
    grid.width = static_cast<int>(values.shape(1));
    grid.z.assign(values.data(), values.data() + values.size());
    model.grids.push_back(std::move(grid));
  }
  for (const py::handle& item : ramps) {
    const py::tuple tuple = item.cast<py::tuple>();
    Require(tuple.size() == kRampTupleSize,
            "ramps are (level_from, level_to, plane, polygon, slope) tuples");
    Ramp ramp;
    ramp.level_from = tuple[0].cast<int>();
    ramp.level_to = tuple[1].cast<int>();
    ramp.plane = tuple[2].cast<Vec3>();
    const RowMatX2d polygon = tuple[3].cast<RowMatX2d>();
    for (Eigen::Index vertex = 0; vertex < polygon.rows(); ++vertex) {
      ramp.polygon.push_back(polygon.row(vertex).transpose());
    }
    ramp.slope = tuple[4].cast<double>();
    model.ramps.push_back(std::move(ramp));
  }
  return model;
}

}  // namespace gr::bindings
