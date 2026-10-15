// Shared helpers of the garage_recon._core bindings: numpy <-> Eigen / std
// conversions, input validation and the Python logger adapter.
//
// The gr library does not use exceptions. pybind11, however, reports errors
// to Python through C++ exceptions, so the bindings validate every input that
// the library would CHECK and raise the matching Python exception instead.

#ifndef GARAGE_RECON_BINDINGS_BIND_UTIL_H_
#define GARAGE_RECON_BINDINGS_BIND_UTIL_H_

#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "pybind11/eigen.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "gr/exposure.h"
#include "gr/logging.h"
#include "gr/pose.h"
#include "gr/types.h"

namespace gr::bindings {

using PointsIn = Eigen::Ref<const RowMatX3d>;
using UvIn = Eigen::Ref<const RowMatX2d>;

// Raises ValueError / IndexError with `message` unless `condition` holds.
void Require(bool condition, const std::string& message);
void RequireIndex(bool condition, const std::string& message);

template <typename T>
pybind11::array_t<T> ToArray(const std::vector<T>& values) {
  pybind11::array_t<T> array(static_cast<pybind11::ssize_t>(values.size()));
  if (!values.empty()) {
    std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(T));
  }
  return array;
}

template <typename T>
pybind11::array_t<T> ToImage(const std::vector<T>& values, int height,
                             int width, int channels = 1) {
  std::vector<pybind11::ssize_t> shape = {height, width};
  if (channels > 1) shape.push_back(channels);
  pybind11::array_t<T> array(shape);
  if (!values.empty()) {
    std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(T));
  }
  return array;
}

pybind11::array_t<bool> ToBoolArray(const std::vector<uint8_t>& flags);
pybind11::array_t<bool> ToBoolArray(const std::vector<bool>& flags);

// Python sequences / arrays (None -> empty) as std vectors.
std::vector<double> ToDoubleVector(const pybind11::object& values);
std::vector<float> ToFloatVector(const pybind11::object& values);
std::vector<int> ToIntVector(const pybind11::object& values);
// A (height, width) float image, or empty for None.
std::vector<float> ToDepth(const pybind11::object& depth, int height,
                           int width);

// (N, 3) arrays of any real dtype <-> points (None -> empty).
Points PointsFrom(const pybind11::object& points);
pybind11::array_t<double> PointsToArray(const Points& points);

// (N, 4, 4) arrays <-> poses (None -> empty).
std::vector<Pose> PosesFrom(const pybind11::object& poses);
pybind11::array_t<double> PosesToArray(const std::vector<Pose>& poses);

// A camera exposure from a (4, 4) pose (global shutter) or a (2, 4, 4) pair
// for the first and the last image row (rolling shutter). The matrices are
// world_from_camera, or camera_from_world when `camera_from_world` is set.
CameraExposure ExposureFrom(const pybind11::object& poses, int height,
                            bool camera_from_world);

// Adapts a Python callable (or None) to a gr::Logger. Each call takes the GIL,
// so stages may run with the GIL released. Python exceptions are kept out of
// the exception-free library: the first one is stored in `*error` and later
// calls are skipped; RethrowIfSet(*error) raises it once the stage returns.
// `callable` must outlive the logger.
Logger MakeLogger(const pybind11::object& callable, std::exception_ptr* error);
void RethrowIfSet(const std::exception_ptr& error);

void BindCore(pybind11::module_& m);
void BindRecon(pybind11::module_& m);
void BindLabels(pybind11::module_& m);
void BindSplat(pybind11::module_& m);
void BindNav(pybind11::module_& m);

}  // namespace gr::bindings

#endif  // GARAGE_RECON_BINDINGS_BIND_UTIL_H_
