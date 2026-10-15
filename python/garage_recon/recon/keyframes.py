"""Keyframes of all trips in one table (the nodes of the pose graph).

The on-disk index and its C++ counterpart for the back-end. Keyframe sweeps
exist twice: kf/ as the odometry deskewed them (used by the back-end) and
kf_final/ deskewed again with the final trajectory (used by every stage
after it).
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from garage_recon import _core
from garage_recon.recon import odometry

ODOMETRY_SWEEPS = "kf"
FINAL_SWEEPS = "kf_final"


@dataclasses.dataclass
class KeyframeSet:
    """Keyframe table.

    Attributes:
        trip_ids: Trip id per trip index.
        trip: (K,) trip index of every keyframe.
        frame: (K,) LiDAR frame index within the trip.
        stamp: (K,) stamps (s).
        odom: (K, 4, 4) world <- lidar from odometry.
        paths: Sweep file of every keyframe.
    """

    trip_ids: list[str]
    trip: np.ndarray
    frame: np.ndarray
    stamp: np.ndarray
    odom: np.ndarray
    paths: list[pathlib.Path]

    @classmethod
    def from_odometry(cls, odoms: list[odometry.TripOdometry],
                      sweeps: str = ODOMETRY_SWEEPS) -> KeyframeSet:
        """Keyframes of the trips; `sweeps` selects kf/ or kf_final/."""
        trip, frame, stamp, poses, paths = [], [], [], [], []
        for t, odom in enumerate(odoms):
            directory = odom.kf_dir.parent / sweeps
            for f in odom.kf_frames:
                trip.append(t)
                frame.append(int(f))
                stamp.append(float(odom.stamps[f]))
                poses.append(odom.lidar_poses[f])
                paths.append(directory / f"{int(f):06d}.npz")
        return cls([o.trip_id for o in odoms], np.asarray(trip, int),
                   np.asarray(frame, int), np.asarray(stamp),
                   np.asarray(poses).reshape(-1, 4, 4), paths)

    def __len__(self) -> int:
        return len(self.trip)

    def load(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(points (M, 3) in the LiDAR frame, intensity (M,)) of keyframe k."""
        return odometry.load_keyframe(self.paths[k])

    def indices_of_trip(self, t: int) -> np.ndarray:
        return np.where(self.trip == t)[0]

    def to_core(self, voxels: list[float]) -> _core.KeyframeSet:
        """C++ keyframe table with the sweeps kept at `voxels` (m)."""
        core = _core.KeyframeSet(list(voxels))
        for k in range(len(self)):
            core.add(int(self.trip[k]), int(self.frame[k]),
                     float(self.stamp[k]), self.odom[k], self.load(k)[0])
        return core
