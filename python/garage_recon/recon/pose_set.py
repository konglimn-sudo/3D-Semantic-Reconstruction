"""The one pose set of a reconstruction (the pose service).

After the back-end, every stage takes poses from here and nowhere else:

* the continuous-time trajectory of every trip (C++ ContinuousTrajectory:
  optimised frame poses as knots, bias-corrected gyro and wheel speed in
  between);
* the sensor calibration: LiDAR / IMU extrinsics, camera extrinsics, camera
  time offsets and rolling-shutter line times.

Two content hashes identify it. `trajectory_id` covers what LiDAR-only stages
depend on (trajectories, IMU / wheel data, LiDAR and IMU extrinsics);
`pose_set_id` additionally covers the cameras. Stage outputs record the ids
they were built with (see provenance.py), so outputs of an older pose set are
detected and re-run.

On disk (work/pose_set/):
    pose_set.json        ids, per trip: trip id, dataset directory, rig.
    <trip_id>.npz        knot_stamps, knot_poses (world <- vehicle),
                         bias_stamps, biases ([gyro, accel]).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import numpy as np

from garage_recon import _core
from garage_recon.data import dataset
from garage_recon.data import schema

_DIRECTORY = "pose_set"


@dataclasses.dataclass
class TripPoses:
    """Trajectory knots, IMU bias estimates and calibration of one trip.

    Attributes:
        trip_id: Trip identifier.
        knot_stamps: (N,) stamps of the optimised frame poses.
        knot_poses: (N, 4, 4) world <- vehicle at those stamps.
        bias_stamps: (K,) stamps of the IMU bias estimates (may be empty).
        biases: (K, 6) [gyro (rad/s), accel (m/s^2)] bias estimates.
        rig: Calibration of the trip's sensors.
    """

    trip_id: str
    knot_stamps: np.ndarray
    knot_poses: np.ndarray
    bias_stamps: np.ndarray
    biases: np.ndarray
    rig: schema.Rig


def _digest(*parts) -> str:
    sha = hashlib.sha256()
    for part in parts:
        if isinstance(part, np.ndarray):
            sha.update(np.ascontiguousarray(part, np.float64).tobytes())
        else:
            sha.update(json.dumps(part, sort_keys=True).encode())
    return sha.hexdigest()[:16]


class PoseSet:
    """Poses of every sensor sample of every trip, from one source."""

    def __init__(self, trips: list[TripPoses],
                 datasets: list[dataset.TripDataset]):
        """Creates the pose set.

        Args:
            trips: Per-trip knots, biases and calibration.
            datasets: The trips' datasets (IMU / wheel samples), same order.

        Raises:
            ValueError: If trips and datasets do not match.
        """
        if [t.trip_id for t in trips] != [d.trip_id for d in datasets]:
            raise ValueError("pose set trips and datasets do not match")
        self.trips = trips
        self.datasets = datasets
        self._trajectories: dict[int, _core.ContinuousTrajectory] = {}
        self._imu = [d.imu() for d in datasets]
        self._wheel = [d.wheel() for d in datasets]
        lidar_parts = []
        camera_parts = []
        for trip, imu, wheel in zip(trips, self._imu, self._wheel):
            rig = trip.rig.to_dict()
            cameras = rig.pop("cameras")
            lidar_parts += [trip.trip_id, trip.knot_stamps, trip.knot_poses,
                            trip.bias_stamps, trip.biases, rig,
                            np.zeros(0) if imu is None else imu,
                            np.zeros(0) if wheel is None else wheel]
            camera_parts.append(cameras)
        self.trajectory_id = _digest(*lidar_parts)
        self.pose_set_id = _digest(self.trajectory_id, *camera_parts)

    # -------------------------------------------------------------- access
    def index(self, trip: int | str) -> int:
        """Index of a trip given its index or id."""
        if isinstance(trip, str):
            return [t.trip_id for t in self.trips].index(trip)
        return int(trip)

    def rig(self, trip: int | str) -> schema.Rig:
        return self.trips[self.index(trip)].rig

    def trajectory(self, trip: int | str) -> _core.ContinuousTrajectory:
        """World <- vehicle of a trip as a continuous-time trajectory."""
        k = self.index(trip)
        if k not in self._trajectories:
            poses = self.trips[k]
            data = self.datasets[k]
            biases = np.asarray(poses.biases, float).reshape(-1, 6)
            self._trajectories[k] = _core.ContinuousTrajectory(
                poses.knot_stamps, poses.knot_poses,
                imu=data.imu_samples(), bias_stamps=poses.bias_stamps,
                gyro_biases=biases[:, :3], wheel=data.wheel_samples())
        return self._trajectories[k]

    def world_from_vehicle(self, trip: int | str, times) -> np.ndarray:
        """(N, 4, 4) world <- vehicle at `times`."""
        return self.trajectory(trip).poses_at(np.atleast_1d(times))

    def world_from_lidar(self, trip: int | str, times) -> np.ndarray:
        """(N, 4, 4) world <- primary LiDAR at `times`."""
        lidar = self.rig(trip).primary_lidar.T_vehicle_lidar
        return self.world_from_vehicle(trip, times) @ lidar

    def row_times(self, trip: int | str, camera: str,
                  stamp_ns: int) -> tuple[float, float]:
        """Exposure times of the first and last row of an image."""
        calib = self.rig(trip).cameras[camera]
        first = stamp_ns * 1e-9 + calib.time_offset
        return first, first + (calib.height - 1) * (
            calib.rolling_shutter_line_time)

    def camera_exposure(self, trip: int | str, camera: str,
                        stamp_ns: int) -> np.ndarray:
        """World <- camera of an image.

        Returns:
            (2, 4, 4) poses of the first and last rows for a rolling-shutter
            camera, (4, 4) for a global-shutter one.
        """
        calib = self.rig(trip).cameras[camera]
        first, last = self.row_times(trip, camera, stamp_ns)
        if calib.rolling_shutter_line_time <= 0.0:
            return self.world_from_vehicle(trip, first)[0] @ calib.T_vehicle_cam
        return self.world_from_vehicle(trip, [first, last]) @ (
            calib.T_vehicle_cam)

    def camera_center_pose(self, trip: int | str, camera: str,
                           stamp_ns: int) -> np.ndarray:
        """(4, 4) world <- camera at the middle row of an image."""
        first, last = self.row_times(trip, camera, stamp_ns)
        calib = self.rig(trip).cameras[camera]
        return (self.world_from_vehicle(trip, 0.5 * (first + last))[0]
                @ calib.T_vehicle_cam)

    # ---------------------------------------------------------- versions
    def with_cameras(self, cameras: dict[str, schema.CameraCalib]) -> PoseSet:
        """A new pose set with the given camera calibration in every trip."""
        trips = []
        for trip in self.trips:
            rig = schema.Rig.from_dict(trip.rig.to_dict())
            for name, calib in cameras.items():
                if name in rig.cameras:
                    rig.cameras[name] = dataclasses.replace(calib)
            trips.append(dataclasses.replace(trip, rig=rig))
        return PoseSet(trips, self.datasets)

    # ---------------------------------------------------------------- I/O
    def save(self, work_dir: str | pathlib.Path) -> pathlib.Path:
        """Writes work/pose_set/ and returns that directory."""
        out = pathlib.Path(work_dir) / _DIRECTORY
        out.mkdir(parents=True, exist_ok=True)
        records = []
        for trip, data in zip(self.trips, self.datasets):
            np.savez(out / f"{trip.trip_id}.npz", knot_stamps=trip.knot_stamps,
                     knot_poses=trip.knot_poses, bias_stamps=trip.bias_stamps,
                     biases=np.asarray(trip.biases, float).reshape(-1, 6))
            records.append({"trip_id": trip.trip_id,
                            "dataset": str(data.root),
                            "rig": trip.rig.to_dict()})
        (out / "pose_set.json").write_text(json.dumps({
            "pose_set_id": self.pose_set_id,
            "trajectory_id": self.trajectory_id,
            "trips": records,
        }, indent=1))
        return out

    @classmethod
    def load(cls, work_dir: str | pathlib.Path,
             datasets: list[dataset.TripDataset]) -> PoseSet:
        """Loads the pose set saved by `save`.

        Raises:
            FileNotFoundError: If the work directory has no pose set.
        """
        directory = pathlib.Path(work_dir) / _DIRECTORY
        meta = json.loads((directory / "pose_set.json").read_text())
        trips = []
        for record in meta["trips"]:
            with np.load(directory / f"{record['trip_id']}.npz") as z:
                trips.append(TripPoses(
                    record["trip_id"], z["knot_stamps"], z["knot_poses"],
                    z["bias_stamps"], z["biases"],
                    schema.Rig.from_dict(record["rig"])))
        by_id = {d.trip_id: d for d in datasets}
        return cls(trips, [by_id[t.trip_id] for t in trips])

    @staticmethod
    def exists(work_dir: str | pathlib.Path) -> bool:
        return (pathlib.Path(work_dir) / _DIRECTORY / "pose_set.json").exists()
