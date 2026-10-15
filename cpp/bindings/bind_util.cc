#include "bind_util.h"

#include <utility>

namespace gr::bindings {

namespace py = ::pybind11;

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
  Require(array.ndim() == 2 && array.shape(1) == 3,
          "expected an (N, 3) float array");
  const double* data = array.data();
  Points out(static_cast<size_t>(array.shape(0)));
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = Vec3(data[3 * i], data[3 * i + 1], data[3 * i + 2]);
  }
  return out;
}

py::array_t<double> PointsToArray(const Points& points) {
  py::array_t<double> array(
      {static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(3)});
  double* data = array.mutable_data();
  for (size_t i = 0; i < points.size(); ++i) {
    for (int k = 0; k < 3; ++k) {
      data[3 * i + static_cast<size_t>(k)] = points[i][k];
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
  Require(array.ndim() == 3 && array.shape(1) == 4 && array.shape(2) == 4,
          "expected an (N, 4, 4) pose array");
  const double* data = array.data();
  std::vector<Pose> out(static_cast<size_t>(array.shape(0)));
  for (size_t n = 0; n < out.size(); ++n) {
    Mat4 matrix;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        matrix(r, c) = data[16 * n + static_cast<size_t>(4 * r + c)];
      }
    }
    out[n] = Pose::FromMatrix(matrix);
  }
  return out;
}

py::array_t<double> PosesToArray(const std::vector<Pose>& poses) {
  py::array_t<double> array({static_cast<py::ssize_t>(poses.size()),
                             static_cast<py::ssize_t>(4),
                             static_cast<py::ssize_t>(4)});
  double* data = array.mutable_data();
  for (size_t n = 0; n < poses.size(); ++n) {
    const Mat4 matrix = poses[n].Matrix();
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        data[16 * n + static_cast<size_t>(4 * r + c)] = matrix(r, c);
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
  const bool single =
      array && array.ndim() == 2 && array.shape(0) == 4 && array.shape(1) == 4;
  const bool pair = array && array.ndim() == 3 && array.shape(0) == 2 &&
                    array.shape(1) == 4 && array.shape(2) == 4;
  Require(single || pair,
          "camera pose must be (4, 4) or (2, 4, 4) for the first and last row");
  Require(height >= 1, "camera height must be positive");
  const double* data = array.data();
  const auto pose_at = [&](size_t n) {
    Mat4 matrix;
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 4; ++c) {
        matrix(r, c) = data[16 * n + static_cast<size_t>(4 * r + c)];
      }
    }
    const Pose pose = Pose::FromMatrix(matrix);
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

}  // namespace gr::bindings
