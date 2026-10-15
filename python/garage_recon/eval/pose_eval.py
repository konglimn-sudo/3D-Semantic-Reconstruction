"""Evaluation of the pose set against synthetic ground truth.

* IMU bias: the back-end's estimate per trip against the simulated bias.
* Camera calibration: estimated extrinsic and time offset per camera
  against the truth (and the error of the nominal rig.yaml for reference).
* Camera exposure: for sampled images, ground-truth surface points are
  projected with the pose set's exposure poses (rolling shutter, time
  offset, calibration) and compared with their true pixels. The v1 method
  (frame poses interpolated at the image stamp, rig.yaml extrinsic, global
  shutter) is scored the same way as a baseline.
"""

from __future__ import annotations

import numpy as np

from garage_recon import _core
from garage_recon import geometry
from garage_recon.data.synthetic import ground_truth
from garage_recon.recon import pose_set as pose_set_lib


def _rotation_deg(a: np.ndarray, b: np.ndarray) -> float:
    _, angle = geometry.pose_error(a[None], b[None])
    return float(np.degrees(angle[0]))


def imu_bias_errors(pose_set: pose_set_lib.PoseSet,
                    world: ground_truth.SyntheticWorld) -> dict:
    """Mean estimated bias per trip against the truth."""
    out = {}
    for trip in pose_set.trips:
        if len(trip.biases) == 0:
            continue
        estimate = np.asarray(trip.biases).mean(axis=0)
        truth = world.imu_bias(trip.trip_id)
        out[trip.trip_id] = {
            "gyro_error_deg_s": float(np.degrees(
                np.linalg.norm(estimate[:3] - truth[:3]))),
            "gyro_truth_deg_s": float(np.degrees(np.linalg.norm(truth[:3]))),
            "accel_error_m_s2": float(np.linalg.norm(estimate[3:] - truth[3:])),
            "accel_truth_m_s2": float(np.linalg.norm(truth[3:])),
        }
    return out


def calibration_errors(pose_set: pose_set_lib.PoseSet,
                       world: ground_truth.SyntheticWorld) -> dict:
    """Extrinsic and time-offset errors of every camera (estimate, nominal)."""
    out = {}
    nominal = pose_set.datasets[0].rig.cameras
    for name, calib in pose_set.rig(0).cameras.items():
        truth = world.camera(name)
        entry = {}
        for label, estimate in (("estimate", calib),
                                ("nominal", nominal[name])):
            delta = geometry.inv(truth.T_vehicle_cam) @ estimate.T_vehicle_cam
            entry[label] = {
                "rotation_deg": _rotation_deg(truth.T_vehicle_cam,
                                              estimate.T_vehicle_cam),
                "translation_mm": float(np.linalg.norm(delta[:3, 3]) * 1e3),
                "time_offset_ms": float(
                    (estimate.time_offset - truth.time_offset) * 1e3),
            }
        out[name] = entry
    return out


def _sample_pixels(depth: np.ndarray, step: int) -> np.ndarray:
    h, w = depth.shape
    vv, uu = np.mgrid[step // 2:h:step, step // 2:w:step]
    keep = np.isfinite(depth[vv.ravel(), uu.ravel()])
    return np.column_stack([uu.ravel() + 0.5, vv.ravel() + 0.5])[keep]


def camera_reprojection(pose_set: pose_set_lib.PoseSet,
                        world: ground_truth.SyntheticWorld,
                        frame_poses: dict[str, tuple[np.ndarray, np.ndarray]],
                        images_per_camera: int = 12, step: int = 6) -> dict:
    """Pixel error of ground-truth points under the pose set's exposures.

    Args:
        pose_set: The evaluated pose set.
        world: Ground truth.
        frame_poses: Per trip (stamps, world <- vehicle) of the LiDAR frames,
            for the v1 baseline.
        images_per_camera: Images sampled per camera and trip.
        step: Pixel grid step of the sampled points.

    Returns:
        Per camera: mean / p95 pixel error of the pose set and of the v1
        baseline, and the number of points.
    """
    out = {}
    for t, data in enumerate(pose_set.datasets):
        rig = pose_set.rig(t)
        stamps_v1, vehicle_v1 = frame_poses[data.trip_id]
        for name in rig.fisheye_cameras:
            calib = rig.cameras[name]
            model = calib.make_model()
            nominal = data.rig.cameras[name]
            stamps = data.camera_stamps_ns(name)
            pick = np.linspace(0, len(stamps) - 1,
                               min(images_per_camera, len(stamps))).astype(int)
            errors, errors_v1 = [], []
            for ns in stamps[pick]:
                ns = int(ns)
                render = world.render(data.trip_id, name, ns)
                uv = _sample_pixels(render["depth"], step)
                rays, ok = model.unproject(uv)
                rows = world.row_poses(data.trip_id, name, ns)
                poses = rows[np.clip(uv[:, 1].astype(int), 0, len(rows) - 1)]
                depth = render["depth"][uv[:, 1].astype(int),
                                        uv[:, 0].astype(int)]
                points = (np.einsum("nij,nj->ni", poses[:, :3, :3],
                                    rays * depth[:, None]) + poses[:, :3, 3])
                points, uv = points[ok], uv[ok]
                exposure = pose_set.camera_exposure(t, name, ns)
                uv_est, _, _, _ = _core.project_visibility(
                    model, geometry.inv(exposure), points)
                v1 = (geometry.interpolate_poses(stamps_v1, vehicle_v1,
                                                 [ns * 1e-9])[0]
                      @ nominal.T_vehicle_cam)
                uv_v1, _, _, _ = _core.project_visibility(
                    model, geometry.inv(v1), points)
                for estimate, sink in ((uv_est, errors), (uv_v1, errors_v1)):
                    error = np.linalg.norm(estimate - uv, axis=1)
                    sink.append(error[np.isfinite(error)])
            errors = np.concatenate(errors) if errors else np.zeros(0)
            errors_v1 = np.concatenate(errors_v1) if errors_v1 else np.zeros(0)
            key = name if len(pose_set.datasets) == 1 else (
                f"{data.trip_id}/{name}")
            out[key] = {
                "points": int(len(errors)),
                "mean_px": float(errors.mean()) if len(errors) else None,
                "p95_px": (float(np.percentile(errors, 95)) if len(errors)
                           else None),
                "v1_mean_px": (float(errors_v1.mean()) if len(errors_v1)
                               else None),
                "v1_p95_px": (float(np.percentile(errors_v1, 95))
                              if len(errors_v1) else None),
            }
    return out


def evaluate_pose_set(pose_set: pose_set_lib.PoseSet,
                      world: ground_truth.SyntheticWorld,
                      frame_poses: dict[str, tuple[np.ndarray, np.ndarray]]
                      ) -> dict:
    """All pose-set metrics."""
    report = {"pose_set_id": pose_set.pose_set_id,
              "imu_bias": imu_bias_errors(pose_set, world)}
    if world.calibration is not None:
        report["calibration"] = calibration_errors(pose_set, world)
        report["camera_reprojection"] = camera_reprojection(pose_set, world,
                                                            frame_poses)
    return report
