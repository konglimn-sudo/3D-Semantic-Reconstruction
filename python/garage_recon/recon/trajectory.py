"""The final trajectory of every trip and the final keyframe sweeps.

After the back-end each trip gets one continuous-time trajectory (C++
ContinuousTrajectory): the optimised poses of all its LiDAR frames as knots,
and in between the gyro corrected by the back-end's bias estimates plus the
wheel speed. Together with the rig calibration it forms the pose set
(pose_set.py). The keyframe sweeps are then deskewed again with this
trajectory, since the odometry deskewed them with its provisional motion.
"""

from __future__ import annotations

import pathlib

import numpy as np
from scipy import spatial

from garage_recon import _core
from garage_recon import geometry
from garage_recon.data import dataset
from garage_recon.recon import keyframes as keyframes_lib
from garage_recon.recon import odometry
from garage_recon.recon import pose_set as pose_set_lib


def build_pose_set(datasets: list[dataset.TripDataset],
                   frame_poses: list[tuple[np.ndarray, np.ndarray]],
                   keyframe_table: dict) -> pose_set_lib.PoseSet:
    """Pose set from the back-end results.

    Args:
        datasets: The trips.
        frame_poses: Per trip (stamps (N,), world <- lidar (N, 4, 4)) of every
            LiDAR frame.
        keyframe_table: Arrays of graph/kf_poses.npz: trip, stamp and, with
            IMU factors, bias (K, 6) and inertial_valid (K,).

    Returns:
        The pose set with the trips' nominal rig calibration.
    """
    trips = []
    kf_trip = np.asarray(keyframe_table["trip"])
    kf_stamp = np.asarray(keyframe_table["stamp"])
    biases = np.asarray(keyframe_table.get("bias", np.zeros((0, 6))))
    valid = np.asarray(keyframe_table.get("inertial_valid", np.zeros(0)),
                       bool)
    for t, (ds, (stamps, lidar_poses)) in enumerate(zip(datasets,
                                                         frame_poses)):
        lidar_from_vehicle = geometry.inv(ds.rig.primary_lidar.T_vehicle_lidar)
        if len(valid) == len(kf_trip):
            mine = np.where((kf_trip == t) & valid)[0]
        else:
            mine = np.zeros(0, int)
        mine = mine[np.argsort(kf_stamp[mine])]
        trips.append(pose_set_lib.TripPoses(
            ds.trip_id, np.asarray(stamps, float),
            np.asarray(lidar_poses) @ lidar_from_vehicle,
            kf_stamp[mine].astype(float),
            biases[mine].reshape(-1, 6) if len(mine) else np.zeros((0, 6)),
            ds.rig))
    return pose_set_lib.PoseSet(trips, datasets)


def redeskew_keyframes(pose_set: pose_set_lib.PoseSet,
                       odoms: list[odometry.TripOdometry], cfg,
                       log=print) -> dict:
    """Writes kf_final/ sweeps deskewed with the final trajectories.

    Args:
        pose_set: The final pose set.
        odoms: Odometry of every trip (keyframe frames, kf/ location).
        cfg: Pipeline configuration (odometry ranges, keyframe voxel).
        log: Logger.

    Returns:
        Statistics: keyframes written and the mean point shift relative to
        the odometry deskew (cm).
    """
    written = 0
    shifts = []
    min_range = float(cfg.odometry.min_range)
    max_range = float(cfg.odometry.max_range)
    voxel = float(cfg.keyframes.voxel)
    for t, (data, odom) in enumerate(zip(pose_set.datasets, odoms)):
        trajectory = pose_set.trajectory(t)
        T_vl = pose_set.rig(t).primary_lidar.T_vehicle_lidar
        out_dir = odom.kf_dir.parent / keyframes_lib.FINAL_SWEEPS
        out_dir.mkdir(parents=True, exist_ok=True)
        for frame in odom.kf_frames:
            sweep = data.load_lidar(int(frame))
            points, intensity = _core.deskew_keyframe_sweep(
                sweep.points, sweep.intensity, sweep.time, sweep.stamp,
                trajectory, T_vl, min_range, max_range, voxel)
            np.savez(out_dir / f"{int(frame):06d}.npz",
                     points=points.astype(np.float32),
                     intensity=intensity.astype(np.float16))
            written += 1
            if written % 10 == 1:  # Shift against the odometry deskew.
                old, _ = odometry.load_keyframe(odom.kf_path(int(frame)))
                shifts.append(_median_shift(old, points))
        log(f"[trajectory] {data.trip_id}: {len(odom.kf_frames)} keyframe "
            f"sweeps deskewed (gyro {100 * trajectory.gyro_coverage:.0f}%, "
            f"wheel {100 * trajectory.wheel_coverage:.0f}% of knot "
            f"intervals)")
    shifts = [s for s in shifts if np.isfinite(s)]
    return {"keyframes": written,
            "median_shift_cm": float(np.median(shifts) * 100) if shifts
            else None}


def _median_shift(old: np.ndarray, new: np.ndarray) -> float:
    """Median distance between nearest points of two versions of a sweep."""
    if len(old) == 0 or len(new) == 0:
        return float("nan")
    distance, _ = spatial.cKDTree(old).query(new[::10],
                                             distance_upper_bound=0.5)
    distance = distance[np.isfinite(distance)]
    return float(np.median(distance)) if len(distance) else float("nan")


def save_frame_poses(work_dir: pathlib.Path, trip_id: str, stamps: np.ndarray,
                     lidar: np.ndarray, T_vehicle_lidar: np.ndarray) -> None:
    """Writes trips/<trip>/poses.{npz,tum} (LiDAR frame poses)."""
    vehicle = lidar @ geometry.inv(T_vehicle_lidar)
    directory = work_dir / "trips" / trip_id
    np.savez(directory / "poses.npz", stamps=stamps, lidar=lidar,
             vehicle=vehicle)
    geometry.write_tum(directory / "poses.tum", stamps, vehicle)
