// pybind11 module garage_recon._core.

#include "bind_util.h"

#ifdef _OPENMP
#include <omp.h>
#endif

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "garage_recon C++ core: registration, odometry, loop closure, pose graph "
      "(with gravity), map and dynamic points, occupancy, projection, "
      "LiDAR-anchored Gaussians and simulation";
#ifdef _OPENMP
  m.attr("has_openmp") = true;
  m.def("set_num_threads", [](int n) { omp_set_num_threads(n); });
#else
  m.attr("has_openmp") = false;
  m.def("set_num_threads", [](int) {});
#endif
  gr::bindings::BindCore(m);
  gr::bindings::BindRecon(m);
  gr::bindings::BindLabels(m);
  gr::bindings::BindSplat(m);
  gr::bindings::BindNav(m);
}
