// Camera extrinsic and time-offset refinement on the shared trajectory.
//
// Row v of an image stamped t is exposed at t + time_offset + v * line_time,
// and its pose is trajectory(time) * vehicle_from_camera. Image
// correspondences (from a dense matcher such as DINO / RoMa features) form
// tracks; their landmarks are triangulated and kept on the LiDAR surface by
// point-to-plane factors, which makes the extrinsic translation observable.
// The calibration (time offset, rotation and translation corrections of
// vehicle_from_camera) and the landmarks are estimated jointly by
// Levenberg-Marquardt with a Schur complement on the landmarks.
//
// Camera poses never move on their own: only these global calibration
// parameters change, so every image stays on the one pose set of the drive.

#ifndef GARAGE_RECON_GR_CALIBRATION_CAMERA_CALIBRATION_H_
#define GARAGE_RECON_GR_CALIBRATION_CAMERA_CALIBRATION_H_

#include <vector>

#include "gr/core/pose.h"
#include "gr/core/types.h"
#include "gr/odometry/trajectory.h"
#include "gr/semantics/projector.h"
#include "gr/sensors/camera.h"

namespace gr {

struct CameraCalibration {
  Pose vehicle_from_camera;
  double time_offset = 0.0;  // s
  double line_time = 0.0;    // s per row; 0 = global shutter.
};

// One image of the camera: its stamp and the trajectory of its drive (which
// must outlive the calibration call).
struct CalibrationImage {
  const ContinuousTrajectory* trajectory = nullptr;
  double stamp = 0.0;
};

struct TrackObservation {
  int image = 0;
  Vec2 uv = Vec2::Zero();
};
using Track = std::vector<TrackObservation>;

struct CameraCalibrationConfig {
  bool estimate_time_offset = true;
  bool estimate_rotation = true;
  bool estimate_translation = true;
  double pixel_sigma = 0.5;
  double huber = 1.5;             // Robust threshold, in sigmas.
  double max_reprojection = 6.0;  // px; larger errors are dropped as outliers.
  double min_parallax_deg = 1.0;  // Between the rays of a track.
  double plane_sigma = 0.02;      // m, landmark to LiDAR surface.
  double plane_radius = 0.2;      // m, search radius for the surface point.
  // Weak priors that keep unobservable directions at the initial values.
  double time_offset_prior = 0.05;  // s
  double rotation_prior_deg = 2.0;
  double translation_prior = 0.1;  // m
  int iterations = 15;
};

struct CameraCalibrationResult {
  CameraCalibration calibration;
  double rms_before = 0.0;  // px, landmarks refined, calibration fixed.
  double rms_after = 0.0;   // px
  int landmarks = 0;
  int observations = 0;
  int iterations = 0;
  // Standard deviations from the inverse Schur complement: time offset (s),
  // rotation (rad) and translation (m).
  double time_offset_sigma = 0.0;
  Vec3 rotation_sigma = Vec3::Zero();
  Vec3 translation_sigma = Vec3::Zero();
};

// `surface` (optional, with one normal per point in `normals`) is the LiDAR
// map for the point-to-plane factors.
CameraCalibrationResult CalibrateCamera(
    const CameraModel& camera, const CameraCalibration& initial,
    const std::vector<CalibrationImage>& images,
    const std::vector<Track>& tracks, const PointCloudIndex* surface,
    const Points* normals, const CameraCalibrationConfig& config);

// Pixel of a world point in an image with rolling shutter: the row is
// iterated to self-consistency. nullopt outside the field of view.
std::optional<Vec2> ProjectWithCalibration(const CameraModel& camera,
                                           const CameraCalibration& calibration,
                                           const CalibrationImage& image,
                                           const Vec3& point_world);

}  // namespace gr

#endif  // GARAGE_RECON_GR_CALIBRATION_CAMERA_CALIBRATION_H_
