"""Pose-graph back-end (sections 4.1 / 4.2).

The whole stage runs in C++ (`_core.run_graph_stage`): odometry chains of all
trips, entrance anchors (INS pose / GNSS fixes), accelerometer gravity
(roll / pitch) factors, loop closure (level and position gating, Scan Context,
multi-scale ICP verification, PCM) and scan-to-map refinement. This module
gathers the per-trip inputs and converts the results.
"""

from __future__ import annotations

import numpy as np

from .. import _core
from ..core_config import graph_config, graph_voxels
from ..data.dataset import TripDataset
from .keyframes import KeyframeSet
from .odometry import TripOdometry, entrance_pose


def trip_inputs(ds: TripDataset, odom: TripOdometry) -> _core.TripInputs:
    ti = _core.TripInputs()
    ti.T_vehicle_lidar = ds.rig.primary_lidar.T_vehicle_lidar
    ti.initial_pose = entrance_pose(ds)  # At the first LiDAR stamp.
    ti.gnss = ds.gnss()
    ti.frame_stamps = odom.stamps
    ti.frame_odometry = odom.lidar_poses
    ti.imu = ds.imu_samples()
    ti.wheel = ds.wheel_samples()
    return ti


def run_graph(kfs: KeyframeSet, datasets: list[TripDataset], odoms: list[TripOdometry], cfg, log=print,
              appearance: np.ndarray | None = None) -> dict:
    """Returns {poses (K, 4, 4), loops, anchors, gravity, summary, loops_tested, loops_verified}.

    `appearance`: (N, 2) keyframe pairs proposed by image retrieval (appearance.py)."""
    core = kfs.to_core(graph_voxels(cfg))
    return _core.run_graph_stage(core, [trip_inputs(ds, o) for ds, o in zip(datasets, odoms)], graph_config(cfg), log,
                                 appearance)


def loop_json(lc) -> dict:
    return {"i": int(lc.i), "j": int(lc.j), "T_ji": np.asarray(lc.T_ji).tolist(), "fitness": float(lc.fitness),
            "rmse": float(lc.rmse), "sc_distance": float(lc.sc_distance)}


def frame_poses(kfs: KeyframeSet, kf_poses: np.ndarray, trip: int, stamps: np.ndarray,
                odom_frames: np.ndarray) -> np.ndarray:
    """All-frame poses of a trip: odometry corrected by the keyframe
    corrections, interpolated in time."""
    idx = kfs.indices_of_trip(trip)
    return _core.correct_frame_poses(kfs.stamp[idx], kfs.odom[idx], kf_poses[idx], stamps, odom_frames)
