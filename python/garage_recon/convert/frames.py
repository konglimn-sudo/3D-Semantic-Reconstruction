"""Vehicle frame and rig of a converted recording, shared by the converters.

Rigs that are not cars (handheld, backpack) have no natural vehicle frame
on the floor. With `vehicle.level` the converter makes one: z up from the
gravity the IMU measures while the rig stands still at the start, and with
`vehicle.floor: auto` the origin on the floor plane under the reference
sensor (RANSAC plane in sweeps spread over the recording). The pipeline
assumes both: the vehicle origin is on the floor (levels, ego volume) and
the world frame of a trip without an INS pose is the vehicle frame at the
start (gravity factors).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from garage_recon import _core
from garage_recon.convert import profile as profile_lib
from garage_recon.data import schema


def level_rotation(accel_reference: np.ndarray,
                   forward: np.ndarray) -> np.ndarray:
    """vehicle <- reference rotation with z along the measured "up".

    Args:
        accel_reference: Mean specific force at rest in the reference frame
            (points up).
        forward: Reference-frame direction that becomes the vehicle x axis
            once made horizontal.

    Raises:
        ValueError: `forward` is (nearly) vertical.
    """
    z = accel_reference / np.linalg.norm(accel_reference)
    x = forward - np.dot(forward, z) * z
    if np.linalg.norm(x) < 0.3:
        raise ValueError("vehicle.forward is nearly vertical")
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def floor_height(sweeps: list[np.ndarray], R_level_reference: np.ndarray,
                 min_radius: float = 1.5, max_radius: float = 8.0,
                 threshold: float = 0.03, max_tilt_deg: float = 25.0
                 ) -> tuple[float, int]:
    """Height of the reference origin above the floor plane.

    Args:
        sweeps: Points in the reference frame, one array per sweep, taken
            over the whole recording: a carried rig may start on the ground
            and tilts while it is carried.
        R_level_reference: Rotation to the gravity-aligned frame at the start
            (selects the points below the rig).
        min_radius: Horizontal distance band of the floor points (the area
            right under a carried rig is hidden by the person).
        max_radius: Outer edge of the band (drainage slopes further away).
        threshold: RANSAC inlier distance (m).
        max_tilt_deg: Rig tilt accepted relative to the start (the distance
            to the plane does not depend on it).

    Returns:
        (median height over the sweeps, sweeps with a floor plane).

    Raises:
        ValueError: No sweep showed a floor plane below the rig.
    """
    heights = []
    min_normal_z = np.cos(np.radians(max_tilt_deg))
    for points in sweeps:
        level = points @ R_level_reference.T
        radius = np.hypot(level[:, 0], level[:, 1])
        below = level[(radius > min_radius) & (radius < max_radius)
                      & (level[:, 2] < -0.3) & (level[:, 2] > -4.0)]
        if len(below) < 200:
            continue
        normal, d, inliers = _core.ransac_plane(np.ascontiguousarray(below),
                                                threshold, 300)
        normal = np.asarray(normal).ravel()
        if abs(normal[2]) < min_normal_z or len(inliers) < 0.3 * len(below):
            continue
        heights.append(abs(d) / np.linalg.norm(normal))
    if not heights:
        raise ValueError("no floor plane found below the rig in the sampled "
                         "sweeps")
    return float(np.median(heights)), len(heights)


def vehicle_frame(profile: profile_lib.Profile,
                  level_rows: np.ndarray | None,
                  floor_sweeps: Callable[[], list[np.ndarray]],
                  log=print) -> tuple[np.ndarray, dict]:
    """vehicle <- reference transform and how it was obtained.

    Args:
        profile: The conversion profile (`vehicle`, `imu`).
        level_rows: (N, 6) IMU rows over the levelling window: gyro (rad/s)
            and accelerometer before `imu.accel_scale`. Used with
            `vehicle.level`.
        floor_sweeps: Returns LiDAR sweeps in the reference frame spread
            over the recording; called with `vehicle.floor: auto`.
        log: Logger for warnings.

    Raises:
        ValueError: Levelling without enough IMU samples, or no floor plane.
    """
    vehicle = profile.vehicle
    out = vehicle.T_vehicle_reference.copy()
    info = {}
    if vehicle.level:
        imu = profile.imu
        if imu is None:
            raise ValueError("vehicle.level needs the IMU")
        rows = np.asarray([] if level_rows is None else level_rows,
                          float).reshape(-1, 6)
        if len(rows) < 10:
            raise ValueError("too few IMU samples to level the rig")
        R_ref_imu = imu.T_reference_sensor[:3, :3]
        accel = R_ref_imu @ (rows[:, 3:].mean(axis=0) * imu.accel_scale)
        gyro = float(np.linalg.norm(rows[:, :3].mean(axis=0)))
        spread = float(np.linalg.norm(rows[:, 3:].std(axis=0))
                       * imu.accel_scale)
        out[:3, :3] = level_rotation(accel, vehicle.forward)
        out[:3, 3] = 0.0
        tilt = np.degrees(np.arccos(np.clip(
            accel[2] / np.linalg.norm(accel), -1.0, 1.0)))
        info.update(level_tilt_deg=round(float(tilt), 3),
                    level_gravity=round(float(np.linalg.norm(accel)), 4),
                    level_gyro_rad_s=round(gyro, 4),
                    level_accel_spread=round(spread, 4))
        if gyro > 0.05 or spread > 0.3:
            log(f"[convert] warning: the rig moves during the levelling "
                f"window (gyro {gyro:.3f} rad/s, accel spread "
                f"{spread:.2f} m/s^2)")
    if vehicle.floor is not None:
        if vehicle.floor == "auto":
            height, used = floor_height(floor_sweeps(), out[:3, :3])
            info.update(floor_height=round(height, 4), floor_sweeps=used)
        else:
            height = float(vehicle.floor)
        out[:3, 3] = [vehicle.reference_xy[0], vehicle.reference_xy[1],
                      height]
    return out, info


def rig(profile: profile_lib.Profile,
        T_vehicle_reference: np.ndarray) -> schema.Rig:
    """The rig.yaml content (cameras at their output resolution)."""
    lidars = {
        lidar.name: schema.LidarCalib(
            lidar.name, T_vehicle_reference @ lidar.T_reference_sensor,
            lidar.sweep_duration) for lidar in profile.lidars}
    cameras = {}
    for cam in profile.cameras:
        width = int(round(cam.width * cam.scale))
        height = int(round(cam.height * cam.scale))
        params = list(cam.params)
        sx, sy = width / cam.width, height / cam.height
        params[0] *= sx
        params[1] *= sy
        params[2] = (params[2] + 0.5) * sx - 0.5
        params[3] = (params[3] + 0.5) * sy - 0.5
        cameras[cam.name] = schema.CameraCalib(
            cam.name, cam.model, width, height, params,
            T_vehicle_reference @ cam.T_reference_sensor,
            cam.rolling_shutter_line_time / sy, cam.time_offset)
    T_vehicle_imu = (T_vehicle_reference @ profile.imu.T_reference_sensor
                     if profile.imu is not None else np.eye(4))
    return schema.Rig(lidars, cameras, list(profile.stereo_pairs),
                      T_vehicle_imu,
                      profile.wheel.scale if profile.wheel is not None
                      else 1.0)
