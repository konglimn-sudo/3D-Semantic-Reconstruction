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
if(APPLE AND CMAKE_OSX_SYSROOT AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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
  if(APPLE AND NOT OpenMP_CXX_FLAGS AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
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
# by side under cpp/gr/<folder> (one folder per function) and are included as
# "gr/<folder>/<name>.h"; tests sit next to them as <name>_test.cc.
set(GR_CORE_SOURCES
  # core/: Base types, SE(3), voxels, numerics, logging, checks, local geometry.
  cpp/gr/core/geometry.cc
  cpp/gr/core/kdtree.cc
  cpp/gr/core/morphology.cc
  cpp/gr/core/logging.cc
  cpp/gr/core/numeric.cc
  cpp/gr/core/pose.cc
  cpp/gr/core/types.cc
  cpp/gr/core/voxel.cc
  # sensors/: Camera models, exposure (rolling shutter), IMU, preintegration.
  cpp/gr/sensors/camera.cc
  cpp/gr/sensors/exposure.cc
  cpp/gr/sensors/imu.cc
  cpp/gr/sensors/imu_preintegration.cc
  # calibration/: Camera time offset / extrinsics, clock sync between sensors.
  cpp/gr/calibration/camera_calibration.cc
  cpp/gr/calibration/clock_sync.cc
  cpp/gr/calibration/lidar_imu.cc
  # registration/: Voxel-hash ICP, coarse alignment, placement, cloud comparison.
  cpp/gr/registration/cloud_compare.cc
  cpp/gr/registration/registration.cc
  # odometry/: LiDAR / LiDAR-inertial odometry, continuous-time trajectory.
  cpp/gr/odometry/lio.cc
  cpp/gr/odometry/odometry.cc
  cpp/gr/odometry/trajectory.cc
  # backend/: Keyframes, loops, pose graph, LiDAR bundle adjustment.
  cpp/gr/backend/backend.cc
  cpp/gr/backend/keyframes.cc
  cpp/gr/backend/lidar_ba.cc
  cpp/gr/backend/loop_closure.cc
  cpp/gr/backend/plane_factor.cc
  cpp/gr/backend/pose_graph.cc
  cpp/gr/backend/scan_context.cc
  # mapping/: Survey map, dynamic points, free space, BEV rasters.
  cpp/gr/mapping/bev.cc
  cpp/gr/mapping/change_detection.cc
  cpp/gr/mapping/levels.cc
  cpp/gr/mapping/raster.cc
  cpp/gr/mapping/raycast.cc
  cpp/gr/mapping/static_map.cc
  cpp/gr/mapping/tiled_map.cc
  cpp/gr/mapping/visibility.cc
  # semantics/: Object timelines, occupancy, projection, labels, identities.
  cpp/gr/semantics/identity.cc
  cpp/gr/semantics/labeling.cc
  cpp/gr/semantics/markings.cc
  cpp/gr/semantics/objects.cc
  cpp/gr/semantics/objects4d.cc
  cpp/gr/semantics/occupancy.cc
  cpp/gr/semantics/paint.cc
  cpp/gr/semantics/projector.cc
  cpp/gr/semantics/slot_paint.cc
  cpp/gr/semantics/slots.cc
  cpp/gr/semantics/structure.cc
  cpp/gr/semantics/vector_fit.cc
  # splat/: 3DGS support: anchored Gaussians, coverage, depth prior.
  cpp/gr/splat/coverage.cc
  cpp/gr/splat/depth_prior.cc
  cpp/gr/splat/gaussians.cc
  # nav/: Navigation map, floor arrows, traffic, lanes, localization in
  # the map (per sweep and per trip), position on the route graph.
  cpp/gr/nav/arrows.cc
  cpp/gr/nav/features.cc
  cpp/gr/nav/lanes.cc
  cpp/gr/nav/layers.cc
  cpp/gr/nav/map_update.cc
  cpp/gr/nav/localization.cc
  cpp/gr/nav/nav_map.cc
  cpp/gr/nav/position.cc
  cpp/gr/nav/slot_state.cc
  cpp/gr/nav/traffic.cc
  cpp/gr/nav/trip_localizer.cc
  # sim/: Box-scene ray caster of the synthetic garage.
  cpp/gr/sim/sim.cc
)
add_library(gr_core STATIC ${GR_CORE_SOURCES})
target_include_directories(gr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp)
target_link_libraries(gr_core PUBLIC Eigen3::Eigen ${GR_OPENMP_TARGET})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  # The library does not use exceptions (preconditions are checked with
  # GR_CHECK); -fno-exceptions enforces it.
  target_compile_options(gr_core PRIVATE -Wall -Wextra -Wno-unused-parameter -fno-exceptions)
endif()

# --------------------------------------------------------------------- gr_io
# cpp/gr/io: the file formats of the pipeline (YAML / JSON, npy / npz, CSV,
# TUM, trip directories, navigation map files, configuration sections) for
# programs that run without Python. Malformed files are reported as
# std::runtime_error from deep inside the parsers, so gr_io is built with
# exceptions, as the bindings are; gr_core stays without.
find_package(ZLIB REQUIRED)
add_library(gr_io STATIC
  cpp/gr/io/config.cc
  cpp/gr/io/digest.cc
  cpp/gr/io/nav_files.cc
  cpp/gr/io/npy.cc
  cpp/gr/io/recon_files.cc
  cpp/gr/io/tables.cc
  cpp/gr/io/trip.cc
  cpp/gr/io/value.cc
)
target_link_libraries(gr_io PUBLIC gr_core ZLIB::ZLIB)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gr_io PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# ---------------------------------------------------------------- gr_pipeline
# cpp/gr/pipeline: the stages of the offline reconstruction on trip and work
# directories (gr_core algorithms, gr_io files), with exceptions as gr_io.
add_library(gr_pipeline STATIC
  cpp/gr/pipeline/recon_pipeline.cc
)
target_link_libraries(gr_pipeline PUBLIC gr_io)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(gr_pipeline PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# ------------------------------------------------------------------ programs
# cpp/apps: command-line programs on gr_core + gr_io.
option(GR_BUILD_APPS "Build the command-line programs (gr_localize, gr_recon)" ON)
if(GR_BUILD_APPS)
  add_executable(gr_localize cpp/apps/gr_localize.cc)
  target_link_libraries(gr_localize PRIVATE gr_io)
  add_executable(gr_recon cpp/apps/gr_recon.cc)
  target_link_libraries(gr_recon PRIVATE gr_pipeline)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gr_localize PRIVATE -Wall -Wextra -Wno-unused-parameter)
    target_compile_options(gr_recon PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  # `cmake --install`: the programs and the default configuration they take
  # as their first --config.
  include(GNUInstallDirs)
  install(TARGETS gr_localize gr_recon
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES python/garage_recon/default_config.yaml
          DESTINATION ${CMAKE_INSTALL_DATADIR}/garage_recon)

  # `cpack`: a tarball (and a Debian package on Linux) of the programs.
  set(CPACK_PACKAGE_NAME garage-recon)
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
      "Garage reconstruction and map localization without Python")
  set(CPACK_PACKAGE_CONTACT "garage_recon maintainers")
  set(CPACK_GENERATOR TGZ)
  if(UNIX AND NOT APPLE)
    list(APPEND CPACK_GENERATOR DEB)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libgomp1, zlib1g")
  endif()
  include(CPack)
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
    cpp/bindings/bind_localization.cc
    cpp/bindings/bind_mapping.cc
    cpp/bindings/bind_semantics.cc
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
    cpp/gr/core/geometry_test.cc
    cpp/gr/core/kdtree_test.cc
    cpp/gr/core/numeric_test.cc
    cpp/gr/core/pose_test.cc
    cpp/gr/core/voxel_test.cc
    cpp/gr/sensors/camera_test.cc
    cpp/gr/sensors/exposure_test.cc
    cpp/gr/sensors/imu_preintegration_test.cc
    cpp/gr/sensors/imu_test.cc
    cpp/gr/calibration/camera_calibration_test.cc
    cpp/gr/calibration/clock_sync_test.cc
    cpp/gr/calibration/lidar_imu_test.cc
    cpp/gr/registration/cloud_compare_test.cc
    cpp/gr/registration/registration_test.cc
    cpp/gr/odometry/lio_test.cc
    cpp/gr/odometry/odometry_test.cc
    cpp/gr/odometry/trajectory_test.cc
    cpp/gr/backend/backend_test.cc
    cpp/gr/backend/keyframes_test.cc
    cpp/gr/backend/lidar_ba_test.cc
    cpp/gr/backend/loop_closure_test.cc
    cpp/gr/backend/plane_factor_test.cc
    cpp/gr/backend/pose_graph_test.cc
    cpp/gr/backend/scan_context_test.cc
    cpp/gr/mapping/change_detection_test.cc
    cpp/gr/mapping/levels_test.cc
    cpp/gr/mapping/raster_test.cc
    cpp/gr/mapping/raycast_test.cc
    cpp/gr/mapping/static_map_test.cc
    cpp/gr/mapping/tiled_map_test.cc
    cpp/gr/mapping/visibility_test.cc
    cpp/gr/semantics/identity_test.cc
    cpp/gr/semantics/labeling_test.cc
    cpp/gr/semantics/markings_test.cc
    cpp/gr/semantics/objects_test.cc
    cpp/gr/semantics/objects4d_test.cc
    cpp/gr/semantics/occupancy_test.cc
    cpp/gr/semantics/projector_test.cc
    cpp/gr/semantics/slot_paint_test.cc
    cpp/gr/semantics/slots_test.cc
    cpp/gr/semantics/structure_test.cc
    cpp/gr/splat/coverage_test.cc
    cpp/gr/splat/depth_prior_test.cc
    cpp/gr/splat/gaussians_test.cc
    cpp/gr/nav/arrows_test.cc
    cpp/gr/nav/features_test.cc
    cpp/gr/nav/lanes_test.cc
    cpp/gr/nav/localization_test.cc
    cpp/gr/nav/nav_map_test.cc
    cpp/gr/nav/position_test.cc
    cpp/gr/nav/slot_state_test.cc
    cpp/gr/nav/traffic_test.cc
    cpp/gr/io/recon_files_test.cc
    cpp/gr/io/trip_test.cc
    cpp/gr/io/value_test.cc
    cpp/gr/testing/e2e_test.cc
  )
  target_link_libraries(gr_tests PRIVATE gr_io)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(gr_tests PRIVATE -Wall -Wextra -Wno-unused-parameter -fno-exceptions)
    # The gr_io tests check its exceptions.
    set_source_files_properties(cpp/gr/io/recon_files_test.cc
                                cpp/gr/io/trip_test.cc cpp/gr/io/value_test.cc
                                PROPERTIES COMPILE_OPTIONS -fexceptions)
  endif()
  add_test(NAME gr_tests COMMAND gr_tests)
endif()
