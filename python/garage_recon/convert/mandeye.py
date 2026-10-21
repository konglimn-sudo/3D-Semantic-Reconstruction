"""Mandeye recording -> trip directory (the data/dataset.py layout).

Mandeye (github.com/JanuszBedkowski/mandeye_controller), a handheld mapping
unit, records a Livox LiDAR and its built-in IMU without ROS, as numbered
chunks of 10-20 s:

    lidarNNNN.laz   the points of the chunk in the LiDAR frame (LAS 1.2);
                    gps_time is the device clock (s), point_source_id the
                    LiDAR index (lidarNNNN.sn: index -> serial number)
    imuNNNN.csv     t (ns, device clock), gx gy gz (rad/s), ax ay az (g),
                    imu id; space-separated, no header
    gnssNNNN.gnss   NMEA log (empty underground; not read)

The LiDAR delivers one continuous point stream; the converter cuts it into
sweeps of `sweep_duration` (stamp = the last point, per-point times <= 0),
also across chunk boundaries.

The profile is the one of the ROS bag converter (convert/profile.py) with
the chunk file prefix in place of a topic (`lidars.<name>.topic: lidar`,
`imu.topic: imu`). Of the LiDAR entry, the time, stamp and intensity fields
do not apply (LAS gps_time and intensity are read); `intensity_scale` maps
the LAS intensity to [0, 1].
"""

from __future__ import annotations

import pathlib
from typing import Iterator

import laspy
import numpy as np
import yaml

from garage_recon import geometry
from garage_recon.convert import frames
from garage_recon.convert import profile as profile_lib
from garage_recon.data import dataset


def chunk_files(root: str | pathlib.Path, prefix: str,
                suffixes: tuple[str, ...]) -> list[pathlib.Path]:
    """The <prefix>NNNN<suffix> files of a recording in chunk order."""
    numbered = []
    for path in pathlib.Path(root).iterdir():
        number = path.stem[len(prefix):]
        if (path.stem.startswith(prefix) and number.isdigit()
                and path.suffix.lower() in suffixes):
            numbered.append((int(number), path))
    return [path for _, path in sorted(numbered)]


def read_imu(root: str | pathlib.Path,
             imu: profile_lib.ImuProfile) -> np.ndarray:
    """IMU rows [t (s), gx, gy, gz, ax, ay, az], time-sorted.

    The accelerometer stays in the logged unit (g); `imu.accel_scale`
    converts it.

    Raises:
        ValueError: No IMU chunk, or samples of more than one IMU.
    """
    paths = chunk_files(root, imu.topic, (".csv",))
    if not paths:
        raise ValueError(f"no {imu.topic}NNNN.csv in {root}")
    data = np.concatenate([rows for rows in (
        np.loadtxt(path, ndmin=2) for path in paths) if rows.size])
    ids = np.unique(data[:, 7]) if data.shape[1] > 7 else np.zeros(1)
    if len(ids) > 1:
        raise ValueError(f"samples of several IMUs ({ids}); one IMU per "
                         f"recording is supported")
    rows = np.column_stack([data[:, 0] * 1e-9 + imu.time_offset,
                            data[:, 1:7]])
    rows = rows[np.argsort(rows[:, 0], kind="stable")]
    # A sample repeated at a chunk boundary.
    return rows[np.concatenate([[True], np.diff(rows[:, 0]) > 0])]


def read_points(path: pathlib.Path, lidar: profile_lib.LidarProfile
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time (s), xyz, intensity in [0, 1]) of one chunk, time-sorted.

    Raises:
        ValueError: Points of more than one LiDAR.
    """
    cloud = laspy.read(path)
    sources = np.unique(np.asarray(cloud.point_source_id))
    if len(sources) > 1:
        raise ValueError(f"{path.name}: points of several LiDARs "
                         f"({sources}); one LiDAR per recording is supported")
    t = np.asarray(cloud.gps_time, float) + lidar.time_offset
    xyz = np.column_stack([np.asarray(cloud.x), np.asarray(cloud.y),
                           np.asarray(cloud.z)]).astype(float)
    intensity = np.clip(np.asarray(cloud.intensity, float)
                        * lidar.intensity_scale, 0.0, 1.0)
    if len(t) > 1 and np.any(np.diff(t) < 0):
        order = np.argsort(t, kind="stable")
        t, xyz, intensity = t[order], xyz[order], intensity[order]
    return t, xyz, intensity


def sweeps(paths: list[pathlib.Path], lidar: profile_lib.LidarProfile,
           start: float, stop: float
           ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Consecutive sweeps of `lidar.sweep_duration` in [start, stop).

    Yields:
        (time (s), xyz, intensity) of each sweep; a last sweep shorter than
        half the duration is dropped.
    """
    duration = lidar.sweep_duration
    origin = None
    pending = None
    for path in paths:
        t, xyz, intensity = read_points(path, lidar)
        keep = (t >= start) & (t < stop)
        if not keep.any():
            if len(t) and t[0] >= stop:
                break
            continue
        chunk = [t[keep], xyz[keep], intensity[keep]]
        if pending is not None:
            chunk = [np.concatenate([a, b]) for a, b in zip(pending, chunk)]
            if np.any(np.diff(chunk[0]) < 0):  # Overlapping chunks.
                order = np.argsort(chunk[0], kind="stable")
                chunk = [a[order] for a in chunk]
        if origin is None:
            origin = float(chunk[0][0])
        bins = np.floor((chunk[0] - origin) / duration).astype(np.int64)
        bounds = [0, *(np.flatnonzero(np.diff(bins)) + 1), len(bins)]
        # The last sweep of the chunk may continue in the next chunk.
        for a, b in zip(bounds[:-2], bounds[1:-1]):
            yield chunk[0][a:b], chunk[1][a:b], chunk[2][a:b]
        a = bounds[-2]
        pending = [chunk[0][a:], chunk[1][a:], chunk[2][a:]]
    if pending is not None and (pending[0][-1] - pending[0][0]
                                >= 0.5 * duration):
        yield tuple(pending)


def convert(root: str | pathlib.Path, profile: profile_lib.Profile,
            out: str | pathlib.Path, trip_id: str | None = None,
            start: float = 0.0, duration: float | None = None,
            log=print) -> pathlib.Path:
    """Converts a Mandeye recording directory into a trip directory.

    Args:
        root: Directory with the lidarNNNN.laz and imuNNNN.csv chunks.
        profile: Conversion profile with one LiDAR and the IMU.
        out: Trip directory to create.
        trip_id: Defaults to the directory name.
        start: Seconds to skip at the beginning of the recording.
        duration: Seconds to convert (None: to the end).
        log: Logger.

    Returns:
        The trip directory.

    Raises:
        ValueError: Not one LiDAR, no IMU, or no data in the time range.
    """
    root, out = pathlib.Path(root), pathlib.Path(out)
    if len(profile.lidars) != 1 or profile.imu is None:
        raise ValueError("a Mandeye profile needs one LiDAR and the IMU")
    lidar, imu_profile = profile.lidars[0], profile.imu
    paths = chunk_files(root, lidar.topic, (".laz", ".las"))
    if not paths:
        raise ValueError(f"no {lidar.topic}NNNN.laz in {root}")
    imu = read_imu(root, imu_profile)
    t_start = float(imu[0, 0]) + start
    t_stop = np.inf if duration is None else t_start + duration
    imu = imu[(imu[:, 0] >= t_start) & (imu[:, 0] < t_stop)]
    if len(imu) == 0:
        raise ValueError("no IMU sample in the selected time range")
    level_rows = imu[imu[:, 0] < t_start + profile.vehicle.level_window, 1:]

    # The sweeps are stored in the LiDAR frame; rig.yaml is written once the
    # vehicle frame is known.
    writer = dataset.TripWriter(out, frames.rig(profile, np.eye(4)), {})
    samples = np.linspace(t_start, min(t_stop, float(imu[-1, 0])),
                          profile.vehicle.floor_sweeps, endpoint=False)
    floor, next_sample, count = [], 0, 0
    for t, xyz, intensity in sweeps(paths, lidar, t_start, t_stop):
        end = float(t[-1])
        with np.errstate(invalid="ignore"):
            rng = np.linalg.norm(xyz, axis=1)
        keep = (np.isfinite(rng) & (rng >= lidar.min_range)
                & (rng <= lidar.max_range))
        if not keep.any():
            continue
        writer.write_lidar(lidar.name, int(round(end * 1e9)), xyz[keep],
                           intensity[keep], (t - end)[keep],
                           lidar.storage_step)
        count += 1
        if next_sample < len(samples) and end >= samples[next_sample]:
            floor.append(geometry.transform(lidar.T_reference_sensor,
                                            xyz[keep]))
            while next_sample < len(samples) and end >= samples[next_sample]:
                next_sample += 1
        if count % 500 == 0:
            log(f"[convert] {lidar.name}: {count} sweeps")
    if count == 0:
        raise ValueError("no LiDAR sweep in the selected time range")

    T_vehicle_reference, info = frames.vehicle_frame(
        profile, level_rows, lambda: floor, log)
    if info:
        log(f"[convert] vehicle frame: {info}")
    frames.rig(profile, T_vehicle_reference).save(out / "rig.yaml")
    rows = imu.copy()
    rows[:, 4:7] *= imu_profile.accel_scale
    writer.write_csv("imu.csv", "t,gx,gy,gz,ax,ay,az", rows)
    meta = {"trip_id": trip_id or out.name, "garage_id": profile.garage_id,
            "conversion": {
                "mandeye": str(root), "chunks": len(paths),
                "start": float(start), "duration": duration,
                "T_vehicle_reference": T_vehicle_reference.tolist(),
                "reference": profile.reference, **info,
                "counts": {lidar.name: count, "imu": len(rows)}}}
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    log(f"[convert] {out}: {lidar.name} {count} sweeps, imu {len(rows)}")
    return out
