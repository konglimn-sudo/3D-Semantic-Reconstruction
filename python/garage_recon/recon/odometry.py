"""Per-trip LiDAR odometry (section 4.1).

The algorithm is the C++ `OdometryPipeline` (wheel-speed + gyro motion prior,
IMU in-sweep deskew, point-to-plane registration, keyframe selection); this
module streams a trip's sweeps into it and stores the trajectory and the
keyframe sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import _core
from ..core_config import keyframe_config, odometry_config
from ..data.dataset import TripDataset
from ..geometry import inv, read_tum, write_tum


@dataclass
class TripOdometry:
    trip_id: str
    stamps: np.ndarray  # (N,) LiDAR frame stamps (s)
    lidar_poses: np.ndarray  # (N, 4, 4) world <- lidar (odometry, anchored at the start)
    kf_frames: np.ndarray  # (K,) frame indices of keyframes
    kf_dir: Path

    def kf_path(self, frame: int) -> Path:
        return self.kf_dir / f"{int(frame):06d}.npz"

    def save(self, out_dir: Path, T_lidar_vehicle: np.ndarray) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_tum(out_dir / "odometry.tum", self.stamps, self.lidar_poses @ T_lidar_vehicle)
        np.savez(out_dir / "odometry.npz", stamps=self.stamps, lidar_poses=self.lidar_poses, kf_frames=self.kf_frames)

    @classmethod
    def load(cls, trip_id: str, out_dir: Path) -> "TripOdometry":
        with np.load(out_dir / "odometry.npz") as z:
            return cls(trip_id, z["stamps"], z["lidar_poses"], z["kf_frames"], out_dir / "kf")


def load_keyframe(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["points"].astype(np.float64), z["intensity"].astype(np.float32)


def motion_prior(ds: TripDataset) -> _core.MotionPrior:
    """Wheel speed + gyro dead reckoning of a trip.

    Without wheel data the prior only has the gyro (`valid` is false,
    `has_gyro` true): odometry then predicts the rotation from it.
    """
    imu, wheel = ds.imu_samples(), ds.wheel_samples()
    return _core.MotionPrior(imu, wheel) if len(imu) > 1 else _core.MotionPrior()


def entrance_pose(ds: TripDataset, prior: _core.MotionPrior | None = None) -> np.ndarray | None:
    """INS pose at the entrance moved to the first LiDAR stamp (world <- vehicle).

    The INS fix and the first sweep are not simultaneous (at 1.4 m/s, 0.1 s is
    14 cm along the track); the wheel / gyro prior bridges the gap.
    """
    T_ins = ds.initial_pose()
    t_ins = ds.initial_pose_time()
    if T_ins is None or t_ins is None or len(ds) == 0:
        return T_ins
    t_first = float(ds.lidar_stamps[0])
    if abs(t_first - t_ins) < 1e-6:
        return T_ins
    prior = prior if prior is not None else motion_prior(ds)
    if not prior.valid:
        return T_ins
    if t_ins < t_first:
        delta = prior.delta(t_ins, t_first)
        return T_ins if delta is None else T_ins @ delta
    delta = prior.delta(t_first, t_ins)
    return T_ins if delta is None else T_ins @ inv(delta)


def run_trip_odometry(ds: TripDataset, cfg, out_dir: Path, log=print) -> TripOdometry:
    T_vl = ds.rig.primary_lidar.T_vehicle_lidar
    T_wv0 = entrance_pose(ds)
    stamps = ds.lidar_stamps
    kf_dir = out_dir / "kf"
    kf_dir.mkdir(parents=True, exist_ok=True)

    external = None
    ext = cfg.odometry.get("external_trajectory")
    if ext:
        ext_path = Path(str(ext).format(trip=ds.root))
        t_ext, P_ext = read_tum(ext_path)
        external = _core.interpolate_poses(t_ext, P_ext, stamps) @ T_vl
        log(f"[odometry] {ds.trip_id}: using external trajectory {ext_path}")

    pipe = _core.OdometryPipeline(odometry_config(cfg), keyframe_config(cfg), T_vl,
                                  np.eye(4) if T_wv0 is None else T_wv0, motion_prior(ds))
    n = len(stamps)
    for k in range(n):
        f = ds.load_lidar(k)
        _, is_kf, pts, intensity = pipe.process(f.points, f.intensity, f.time, f.stamp, k == n - 1,
                                                None if external is None else external[k])
        if is_kf:
            np.savez(kf_dir / f"{k:06d}.npz", points=pts.astype(np.float32), intensity=intensity.astype(np.float16))
    poses = pipe.poses()
    result = TripOdometry(ds.trip_id, stamps, poses, pipe.keyframes().astype(int), kf_dir)
    result.save(out_dir, inv(T_vl))
    path_len = float(np.sum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)))
    log(f"[odometry] {ds.trip_id}: {n} frames, {len(result.kf_frames)} keyframes, path {path_len:.1f} m"
        f"{' (wheel/IMU prior)' if pipe.uses_prior else ''}"
        f"{' (gyro prior)' if pipe.uses_gyro_prior else ''}")
    return result
