cmake_minimum_required(VERSION 3.18)
project(garage_recon VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(GR_BUILD_PYTHON "Build the garage_recon._core Python extension" ON)
option(GR_BUILD_TESTS "Build C++ unit tests" ON)
option(GR_USE_OPENMP "Parallelise hot loops with OpenMP" ON)
option(GR_INPLACE "Write the extension into python/garage_recon (development layout)" ON)

# Some Command Line Tools installs ship libc++ headers only inside the SDK and
# clang then fails on `#include <vector>`; point it at the SDK copy.
if(APPLE AND CMAKE_OSX_SYSROOT)
  get_filename_component(_gr_cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  if(NOT EXISTS "${_gr_cxx_bin}/../include/c++/v1/vector" AND EXISTS "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1/vector")
    message(STATUS "garage_recon: using libc++ headers from ${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
    include_directories(SYSTEM "${CMAKE_OSX_SYSROOT}/usr/include/c++/v1")
  endif()
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

# --------------------------------------------------------------------- OpenMP
set(GR_OPENMP_TARGET "")
if(GR_USE_OPENMP)
  if(APPLE AND NOT OpenMP_CXX_FLAGS)
    # Apple clang needs Homebrew's libomp.
    foreach(prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
      if(EXISTS "${prefix}/include/omp.h")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${prefix}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${prefix}/lib/libomp.dylib")
        break()
      endif()
    endforeach()
  endif()
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    set(GR_OPENMP_TARGET OpenMP::OpenMP_CXX)
    message(STATUS "garage_recon: OpenMP enabled")
  else()
    message(STATUS "garage_recon: OpenMP not found, building single-threaded")
  endif()
endif()

# ------------------------------------------------------------------- gr_core
# C++ sources follow the Google C++ style guide: headers and sources live side
# by side under cpp/gr and are included as "gr/<name>.h".
set(GR_CORE_SOURCES
  cpp/gr/types.cc
  cpp/gr/pose.cc
  cpp/gr/voxel.cc
  cpp/gr/numeric.cc
  cpp/gr/logging.cc
  cpp/gr/camera.cc
  cpp/gr/camera_calibration.cc
  cpp/gr/clock_sync.cc
  cpp/gr/cloud_compare.cc
  cpp/gr/exposure.cc
  cpp/gr/geometry.cc
  # Section 4: registration, odometry, loops, back-end, map.
  cpp/gr/registration.cc
  cpp/gr/imu.cc
  cpp/gr/imu_preintegration.cc
  cpp/gr/odometry.cc
  cpp/gr/lio.cc
  cpp/gr/keyframes.cc
  cpp/gr/lidar_ba.cc
  cpp/gr/scan_context.cc
  cpp/gr/loop_closure.cc
  cpp/gr/plane_factor.cc
  cpp/gr/pose_graph.cc
  cpp/gr/backend.cc
  cpp/gr/trajectory.cc
  cpp/gr/visibility.cc
  cpp/gr/static_map.cc
  cpp/gr/raycast.cc
  cpp/gr/raster.cc
  # Section 5: objects, occupancy, projection.
  cpp/gr/objects.cc
  cpp/gr/occupancy.cc
  cpp/gr/projector.cc
  cpp/gr/labeling.cc
  cpp/gr/identity.cc
  # 3DGS: LiDAR-anchored Gaussians, camera coverage, training masks,
  # the depth prior aligned to the LiDAR.
  cpp/gr/gaussians.cc
  cpp/gr/depth_prior.cc
  cpp/gr/coverage.cc
  # Navigation map: drivable layers, aisle route graph, planning.
  cpp/gr/nav_map.cc
  # Synthetic scenes.
  cpp/gr/sim.cc
)
add_library(gr_core STATIC ${GR_CORE_SOURCES})
target_include_directories(gr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(gr_core PUBLIC Eigen3::Eigen ${GR_OPENMP_TARGET})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # The library does not use exceptions (preconditions are checked with
  # GR_CHECK); -fno-exceptions enforces it.
  target_compile_options(gr_core PRIVATE -Wall -Wextra -Wno-unused-parameter -fno-exceptions)
endif()

# ------------------------------------------------------------ Python module
if(GR_BUILD_PYTHON)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  if(NOT pybind11_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_pybind11_dir)
      set(pybind11_DIR "${_pybind11_dir}")
    endif()
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  # pybind11 reports errors to Python through C++ exceptions, so only the
  # bindings are built with exceptions; they validate input before calling
  # into gr_core.
  pybind11_add_module(_core
    cpp/bindings/module.cc
    cpp/bindings/bind_util.cc
    cpp/bindings/bind_core.cc
    cpp/bindings/bind_recon.cc
    cpp/bindings/bind_labels.cc
    cpp/bindings/bind_splat.cc
    cpp/bindings/bind_nav.cc
  )
  target_link_libraries(_core PRIVATE gr_core)
  if(GR_INPLACE)
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/garage_recon")
  endif()
  install(TARGETS _core LIBRARY DESTINATION garage_recon)
endif()

# --------------------------------------------------------------------- tests
if(GR_BUILD_TESTS)
  enable_testing()
  add_executable(gr_tests
    cpp/gr/testing/test.cc
    cpp/gr/testing/test_main.cc
    cpp/gr/testing/sim_garage.cc
    cpp/gr/pose_test.cc
    cpp/gr/voxel_test.cc
    cpp/gr/numeric_test.cc
    cpp/gr/geometry_test.cc
    cpp/gr/camera_test.cc
    cpp/gr/camera_calibration_test.cc
    cpp/gr/clock_sync_test.cc
    cpp/gr/cloud_compare_test.cc
    cpp/gr/exposure_test.cc
    cpp/gr/registration_test.cc
    cpp/gr/odometry_test.cc
    cpp/gr/lio_test.cc
    cpp/gr/imu_test.cc
    cpp/gr/imu_preintegration_test.cc
    cpp/gr/scan_context_test.cc
    cpp/gr/plane_factor_test.cc
    cpp/gr/pose_graph_test.cc
    cpp/gr/keyframes_test.cc
    cpp/gr/lidar_ba_test.cc
    cpp/gr/loop_closure_test.cc
    cpp/gr/backend_test.cc
    cpp/gr/trajectory_test.cc
    cpp/gr/raycast_test.cc
    cpp/gr/visibility_test.cc
    cpp/gr/static_map_test.cc
    cpp/gr/projector_test.cc
    cpp/gr/raster_test.cc
    cpp/gr/objects_test.cc
    cpp/gr/occupancy_test.cc
    cpp/gr/labeling_test.cc
    cpp/gr/identity_test.cc
    cpp/gr/gaussians_test.cc
    cpp/gr/depth_prior_test.cc
    cpp/gr/coverage_test.cc
    cpp/gr/nav_map_test.cc
    cpp/gr/e2e_test.cc
  )
  target_link_libraries(gr_tests PRIVATE gr_core)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gr_tests PRIVATE -Wall -Wextra -Wno-unused-parameter -fno-exceptions)
  endif()
  add_test(NAME gr_tests COMMAND gr_tests)
endif()
