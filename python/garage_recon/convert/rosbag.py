"""ROS bag -> trip directory (the data/dataset.py layout).

Reads ROS 1 bags and ROS 2 bags (sqlite3 / mcap) with the pure-Python
`rosbags` package; no ROS installation is needed. What to read and how the
rig is calibrated comes from a conversion profile (convert/profile.py).

Per stream:
    LiDAR    sensor_msgs/PointCloud2 -> lidar/<name>/<end stamp>.npz with the
             per-point time relative to the end of the sweep (<= 0)
    IMU      sensor_msgs/Imu -> imu.csv (IMU frame; rig.yaml rotates it)
    wheel    TwistStamped / Odometry / JointState / Float -> wheel.csv
    GNSS     sensor_msgs/NavSatFix -> gnss.csv in a local ENU frame
    cameras  sensor_msgs/Image or CompressedImage -> cameras/<name>/*.jpg,
             optionally down-scaled (intrinsics scaled to match) and
             thinned to a maximum rate
    INS      nav_msgs/Odometry at the first sweep -> meta.yaml initial_pose

The vehicle frame of rigs that are not cars (levelled, on the floor) comes
from convert/frames.py.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import multiprocessing
import pathlib

import cv2
import numpy as np
from rosbags import highlevel
from scipy.spatial import transform as spatial_transform
import yaml

from garage_recon import _core
from garage_recon import geometry
from garage_recon.convert import frames
from garage_recon.convert import profile as profile_lib
from garage_recon.data import dataset
from garage_recon.data import schema

# sensor_msgs/PointField datatypes.
_POINT_FIELD_TYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
                      7: "f4", 8: "f8"}
# ROS Bayer encodings -> OpenCV conversions (OpenCV names the pattern by the
# second row).
_BAYER = {"bayer_rggb8": cv2.COLOR_BayerBG2RGB,
          "bayer_bggr8": cv2.COLOR_BayerRG2RGB,
          "bayer_gbrg8": cv2.COLOR_BayerGR2RGB,
          "bayer_grbg8": cv2.COLOR_BayerGB2RGB}
_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def stamp_ns(header) -> int:
    """Header stamp in integer nanoseconds."""
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def pointcloud2_array(msg) -> np.ndarray:
    """The points of a sensor_msgs/PointCloud2 as a structured array."""
    names, formats, offsets = [], [], []
    order = ">" if msg.is_bigendian else "<"
    for field in msg.fields:
        if field.datatype not in _POINT_FIELD_TYPES:
            continue
        names.append(field.name)
        base = order + _POINT_FIELD_TYPES[field.datatype]
        formats.append((base, int(field.count)) if field.count > 1 else base)
        offsets.append(int(field.offset))
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                      "itemsize": int(msg.point_step)})
    data = np.asarray(msg.data, np.uint8)
    rows = max(int(msg.height), 1)
    width = int(msg.width)
    used = width * int(msg.point_step)
    if int(msg.row_step) != used:  # Padding at the end of every row.
        data = data.reshape(rows, int(msg.row_step))[:, :used]
    return np.ascontiguousarray(data).reshape(-1)[:rows * used].view(dtype)


@dataclasses.dataclass
class Sweep:
    """One decoded LiDAR sweep (sensor frame)."""

    stamp_ns: int  # End of the sweep, on the common clock.
    xyz: np.ndarray
    intensity: np.ndarray
    time: np.ndarray  # s relative to the stamp (<= 0).


def decode_sweep(msg, lidar: profile_lib.LidarProfile) -> Sweep | None:
    """A LiDAR sweep with per-point times relative to its end.

    Returns:
        The sweep, or None if no point is within the range limits.
    """
    points = pointcloud2_array(msg)
    xyz = np.column_stack([points["x"], points["y"],
                           points["z"]]).astype(np.float64)
    header = stamp_ns(msg.header)
    if lidar.time_field and lidar.time_field in points.dtype.names:
        t = points[lidar.time_field].astype(np.float64) * lidar.time_scale
    else:
        t = np.zeros(len(points))
    with np.errstate(invalid="ignore"):
        rng = np.linalg.norm(xyz, axis=1)
    keep = (np.isfinite(rng) & (rng >= lidar.min_range)
            & (rng <= lidar.max_range))
    if not keep.any():
        return None
    # The sweep's time span comes from its valid returns only: Ouster pixels
    # without an echo (x = y = z = 0) can carry an undefined `t` (Hilti 2021:
    # +4.1 s), which moved the whole sweep 4 s late. Returns timed more than
    # a sweep away from the median are dropped as well.
    median = float(np.median(t[keep]))
    keep &= np.abs(t - median) <= lidar.sweep_duration
    t_max = float(t[keep].max())
    if lidar.stamp == "points":  # Absolute per-point times.
        end = t_max
    elif lidar.stamp == "start":
        end = header * 1e-9 + t_max - float(t[keep].min())
    else:
        end = header * 1e-9
    relative = t - t_max
    if lidar.intensity_field and lidar.intensity_field in points.dtype.names:
        intensity = np.clip(points[lidar.intensity_field][keep].astype(
            np.float64) * lidar.intensity_scale, 0.0, 1.0)
    else:
        intensity = np.zeros(int(keep.sum()))
    end_ns = int(round((end + lidar.time_offset) * 1e9))
    return Sweep(end_ns, xyz[keep], intensity, relative[keep])


def decode_image(msg, msgtype: str) -> np.ndarray:
    """RGB image of a sensor_msgs/Image or CompressedImage.

    Raises:
        ValueError: An encoding that is not supported.
    """
    if msgtype.endswith("CompressedImage"):
        bgr = cv2.imdecode(np.asarray(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot decode compressed image ({msg.format})")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    encoding = msg.encoding.lower()
    data = np.asarray(msg.data, np.uint8)
    rows, cols, step = int(msg.height), int(msg.width), int(msg.step)
    if encoding in ("rgb8", "bgr8"):
        image = data.reshape(rows, step)[:, :cols * 3].reshape(rows, cols, 3)
        return (image.copy() if encoding == "rgb8" else
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if encoding in ("mono8", "8uc1"):
        image = data.reshape(rows, step)[:, :cols]
        return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_GRAY2RGB)
    if encoding in _BAYER:
        image = np.ascontiguousarray(data.reshape(rows, step)[:, :cols])
        return cv2.cvtColor(image, _BAYER[encoding])
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def _joint_indices(msg, wheel: profile_lib.WheelProfile, values) -> list[int]:
    names = list(msg.name)
    return ([names.index(j) for j in wheel.joints] if wheel.joints else
            list(range(len(values))))


def wheel_distance(msg, wheel: profile_lib.WheelProfile) -> float:
    """Distance travelled (m) from accumulated encoder counts (the mean of
    the wheels), for `kind: joint_position`."""
    indices = _joint_indices(msg, wheel, msg.position)
    return float(np.mean([msg.position[i] for i in indices])
                 ) * wheel.meters_per_count


def speeds_from_distance(rows: np.ndarray, window: int) -> np.ndarray:
    """[t, speed] by central differences of [t, distance] over `window`
    samples (encoder quantisation averages out)."""
    half = max(int(window) // 2, 1)
    if len(rows) <= 2 * half:
        return np.zeros((0, 2))
    t, d = rows[:, 0], rows[:, 1]
    dt = t[2 * half:] - t[:-2 * half]
    ok = dt > 1e-6
    speed = (d[2 * half:] - d[:-2 * half])[ok] / dt[ok]
    return np.column_stack([t[half:-half][ok], speed])


def wheel_speed(msg, wheel: profile_lib.WheelProfile) -> float:
    """Vehicle speed (m/s, before the rig's wheel scale)."""
    if wheel.kind == "twist":
        return float(msg.twist.linear.x)
    if wheel.kind == "odometry":
        return float(msg.twist.twist.linear.x)
    if wheel.kind == "float":
        return float(msg.data)
    indices = _joint_indices(msg, wheel, msg.velocity)
    return float(np.mean([msg.velocity[i] for i in indices])) * wheel.radius


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat)**2)
    return np.array([(n + alt) * np.cos(lat) * np.cos(lon),
                     (n + alt) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - _WGS84_E2) + alt) * np.sin(lat)])


def enu_rotation(lat: float, lon: float) -> np.ndarray:
    """Rows: east, north, up at a geodetic position (ECEF -> ENU)."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])


def geodetic_to_enu(lat: float, lon: float, alt: float,
                    origin: list[float]) -> np.ndarray:
    """Local ENU coordinates (m) of a fix relative to `origin` (lat, lon,
    alt)."""
    delta = geodetic_to_ecef(lat, lon, alt) - geodetic_to_ecef(*origin)
    return enu_rotation(origin[0], origin[1]) @ delta


def _pose_matrix(pose) -> np.ndarray:
    """(4, 4) from a geometry_msgs/Pose."""
    q = pose.orientation
    return geometry.make_T(
        spatial_transform.Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
        [pose.position.x, pose.position.y, pose.position.z])


@dataclasses.dataclass
class ClockCorrection:
    """Correction (s) of a LiDAR stamp onto the IMU clock: linear through
    (times, lags), held at the ends unless `extrapolate`, plus the LiDAR
    IMU's own delay. A class (not a closure) so that it can be pickled."""

    times: np.ndarray
    lags: np.ndarray
    delay: float
    extrapolate: bool = False

    def __call__(self, t: float) -> float:
        if self.extrapolate:
            slope = (self.lags[1] - self.lags[0]) / (self.times[1]
                                                     - self.times[0])
            return float(self.lags[0] + slope * (t - self.times[0])
                         + self.delay)
        return float(np.interp(t, self.times, self.lags)) + self.delay


class Converter:
    """Converts one recording (one or more bag files) into a trip."""

    def __init__(self, bags: list[str | pathlib.Path],
                 profile: profile_lib.Profile, log=print):
        self.bags = [pathlib.Path(b) for b in bags]
        self.profile = profile
        self.log = log
        self.reader = highlevel.AnyReader(self.bags)
        self.reader.open()
        topics = {c.topic: c for c in self.reader.connections}
        missing = [t for t in profile.topics() if t not in topics]
        if missing:
            raise ValueError(f"topics not in the bag: {missing}; available: "
                             f"{sorted(topics)}")
        self.connections = topics

    def close(self) -> None:
        self.reader.close()

    def _messages(self, topics: list[str], start_ns: int | None = None,
                  stop_ns: int | None = None):
        """Yields (connection, bag time in ns, message) in time order."""
        connections = [self.connections[t] for t in sorted(set(topics))]
        for connection, bag_ns, raw in self.reader.messages(
                connections=connections, start=start_ns, stop=stop_ns):
            yield (connection, bag_ns,
                   self.reader.deserialize(raw, connection.msgtype))

    def vehicle_frame(self, start_ns: int, stop_ns: int | None = None
                      ) -> tuple[np.ndarray, dict]:
        """vehicle <- reference transform and how it was obtained."""
        vehicle = self.profile.vehicle
        imu = self.profile.imu
        rows = None
        if vehicle.level and imu is not None:
            stop = start_ns + int(vehicle.level_window * 1e9)
            rows = np.array([[m.angular_velocity.x, m.angular_velocity.y,
                              m.angular_velocity.z, m.linear_acceleration.x,
                              m.linear_acceleration.y,
                              m.linear_acceleration.z]
                             for _, _, m in self._messages(
                                 [imu.topic], start_ns, stop)])

        def floor_sweeps():
            # Sweeps spread over the recording: the rig's carrying height,
            # not where it was put down at the start.
            lidar = self.profile.lidars[0]
            stop = stop_ns if stop_ns is not None else self.reader.end_time
            sweeps = []
            for t in np.linspace(start_ns, stop, vehicle.floor_sweeps,
                                 endpoint=False):
                for _, _, msg in self._messages([lidar.topic], int(t)):
                    sweep = decode_sweep(msg, lidar)
                    if sweep is not None:
                        sweeps.append(geometry.transform(
                            lidar.T_reference_sensor, sweep.xyz))
                    break
            return sweeps

        return frames.vehicle_frame(self.profile, rows, floor_sweeps,
                                    self.log)

    def clock_sync(self, lidar: profile_lib.LidarProfile, start_ns: int,
                   stop_ns: int | None):
        """Correction (s) of the LiDAR stamps onto the main IMU clock.

        Returns:
            (function of the LiDAR stamp in s -> correction in s, info), or
            (None, info) when the rig never turned enough to measure it.
        """
        sync = lidar.clock_sync
        if self.profile.imu is None:
            raise ValueError("clock_sync needs the main IMU")
        rates = {sync.imu_topic: [], self.profile.imu.topic: []}
        for connection, _, msg in self._messages(list(rates), start_ns,
                                                 stop_ns):
            w = msg.angular_velocity
            rates[connection.topic].append([stamp_ns(msg.header) * 1e-9,
                                            w.x, w.y, w.z])
        reference = np.array(rates[self.profile.imu.topic])
        reference[:, 0] += self.profile.imu.time_offset
        other = np.array(rates[sync.imu_topic])
        config = _core.ClockSyncConfig()
        config.window = float(sync.window)
        config.max_lag = float(sync.max_lag)
        result = _core.estimate_clock_offset(
            reference[:, 0].tolist(), np.ascontiguousarray(reference[:, 1:]),
            other[:, 0].tolist(), np.ascontiguousarray(other[:, 1:]), config)
        windows = np.asarray(result["windows"])
        info = {"imu_topic": sync.imu_topic, "imu_delay": sync.imu_delay,
                "model": sync.model, "windows": len(windows)}
        if not result["valid"]:
            self.log(f"[convert] warning: {lidar.name} clock offset not "
                     f"measurable (no rotation); using time_offset only")
            return None, info
        t0 = float(other[0, 0])
        info.update(
            offset_ms=round(float(1000 * (result["offset"] + result["drift"] * (
                t0 - result["reference_time"]))), 2),
            drift_ms_per_min=round(float(60000 * result["drift"]), 3),
            window_lags_ms=[[round(float(c - t0), 1), round(float(1000 * l), 2),
                             round(float(np.degrees(r)), 3)]
                            for c, l, r, _ in windows])
        if sync.model == "linear":
            reference_time = float(result["reference_time"])
            offset = float(result["offset"])
            drift = float(result["drift"])
            correction = ClockCorrection(
                np.array([reference_time - 1.0, reference_time + 1.0]),
                np.array([offset - drift, offset + drift]), sync.imu_delay,
                extrapolate=True)
        else:
            # Piecewise linear through the window lags, median of three
            # neighbours against single bad windows; held at the ends.
            lags = windows[:, 1].copy()
            if len(lags) >= 3:
                padded = np.concatenate([lags[:1], lags, lags[-1:]])
                lags = np.median(np.stack([padded[:-2], padded[1:-1],
                                           padded[2:]]), axis=0)
            correction = ClockCorrection(windows[:, 0].copy(), lags,
                                         sync.imu_delay)
        lags_ms = 1000 * windows[:, 1]
        self.log(f"[convert] {lidar.name} clock: {sync.imu_topic} vs the IMU "
                 f"in {len(windows)} windows, lag {lags_ms.min():+.1f} .. "
                 f"{lags_ms.max():+.1f} ms, + delay "
                 f"{1000 * sync.imu_delay:+.1f} ms ({sync.model})")
        return correction, info

    def read_window(self, writer: dataset.TripWriter, rig: schema.Rig,
                    corrections: dict, start_ns: int, stop_ns: int) -> dict:
        """Writes the sweeps and images of [start_ns, stop_ns) (bag time).

        Returns:
            The rows of the streams written at the end ("imu", "wheel",
            "gnss": (t, lat, lon, alt, covariance, covariance type), "ins":
            (t, pose)), the "counts" per sensor and the stamp of the first
            sweep of the primary LiDAR ("first_sweep", None if none).
        """
        p = self.profile
        lidar_by_topic = {lidar.topic: lidar for lidar in p.lidars}
        camera_by_topic = {cam.topic: cam for cam in p.cameras}
        imu_rows, wheel_rows, gnss_fixes, ins = [], [], [], []
        counts = {name: 0 for name in
                  [lidar.name for lidar in p.lidars]
                  + [cam.name for cam in p.cameras]}
        last_image = {cam.name: -np.inf for cam in p.cameras}
        first_sweep = None
        for connection, bag_ns, msg in self._messages(p.topics(), start_ns,
                                                      stop_ns):
            topic = connection.topic
            # Headerless messages (e.g. std_msgs/Float64) use the bag time.
            t = stamp_ns(msg.header) if hasattr(msg, "header") else bag_ns
            lidar = lidar_by_topic.get(topic)
            if lidar is not None:
                sweep = decode_sweep(msg, lidar)
                correction = corrections.get(lidar.name)
                if sweep is not None and correction is not None:
                    sweep.stamp_ns += int(round(
                        correction(sweep.stamp_ns * 1e-9) * 1e9))
                if sweep is not None:
                    writer.write_lidar(lidar.name, sweep.stamp_ns, sweep.xyz,
                                       sweep.intensity, sweep.time,
                                       lidar.storage_step)
                    counts[lidar.name] += 1
                    if lidar is p.lidars[0] and first_sweep is None:
                        first_sweep = sweep.stamp_ns
                    if counts[lidar.name] % 500 == 0:
                        self.log(f"[convert] {lidar.name}: "
                                 f"{counts[lidar.name]} sweeps")
            cam = camera_by_topic.get(topic)
            if cam is not None and (
                    cam.max_rate <= 0 or
                    (t - last_image[cam.name]) * 1e-9 >= 0.999 / cam.max_rate):
                rgb = decode_image(msg, connection.msgtype)
                size = (rig.cameras[cam.name].width,
                        rig.cameras[cam.name].height)
                if (rgb.shape[1], rgb.shape[0]) != size:
                    rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
                writer.write_image(cam.name, t, rgb, cam.quality)
                last_image[cam.name] = t
                counts[cam.name] += 1
            if p.imu is not None and topic == p.imu.topic:
                imu_rows.append([
                    t * 1e-9 + p.imu.time_offset,
                    msg.angular_velocity.x, msg.angular_velocity.y,
                    msg.angular_velocity.z,
                    msg.linear_acceleration.x * p.imu.accel_scale,
                    msg.linear_acceleration.y * p.imu.accel_scale,
                    msg.linear_acceleration.z * p.imu.accel_scale])
            if p.wheel is not None and topic == p.wheel.topic:
                value = (wheel_distance(msg, p.wheel)
                         if p.wheel.kind == "joint_position" else
                         wheel_speed(msg, p.wheel))
                wheel_rows.append([t * 1e-9 + p.wheel.time_offset, value])
            if (p.gnss is not None and topic == p.gnss.topic
                    and int(msg.status.status) >= p.gnss.min_status):
                gnss_fixes.append((t, msg.latitude, msg.longitude,
                                   msg.altitude,
                                   np.asarray(msg.position_covariance, float),
                                   int(msg.position_covariance_type)))
            if p.initial_pose is not None and topic == p.initial_pose.topic:
                ins.append((t, _pose_matrix(msg.pose.pose)))
        return {"imu": imu_rows, "wheel": wheel_rows, "gnss": gnss_fixes,
                "ins": ins, "counts": counts, "first_sweep": first_sweep}

    def run(self, out: str | pathlib.Path, trip_id: str | None = None,
            start: float = 0.0, duration: float | None = None,
            jobs: int = 1) -> pathlib.Path:
        """Writes the trip.

        Args:
            out: Trip directory to create.
            trip_id: Defaults to the directory name.
            start: Seconds to skip at the beginning of the recording.
            duration: Seconds to convert (None: to the end).
            jobs: Processes reading consecutive time windows in parallel
                (bz2-compressed bags decode at ~20 MB/s per process).

        Returns:
            The trip directory.
        """
        p = self.profile
        out = pathlib.Path(out)
        start_ns = self.reader.start_time + int(start * 1e9)
        stop_ns = (None if duration is None else
                   start_ns + int(duration * 1e9))
        T_vehicle_reference, info = self.vehicle_frame(start_ns, stop_ns)
        if info:
            self.log(f"[convert] vehicle frame: {info}")
        rig = frames.rig(p, T_vehicle_reference)
        corrections, sync_info = {}, {}
        for lidar in p.lidars:
            if lidar.clock_sync is not None:
                corrections[lidar.name], sync_info[lidar.name] = (
                    self.clock_sync(lidar, start_ns, stop_ns))
        writer = dataset.TripWriter(out, rig, {})
        stop = self.reader.end_time + 1 if stop_ns is None else stop_ns
        if jobs <= 1:
            parts = [self.read_window(writer, rig, corrections, start_ns,
                                      stop)]
        else:
            # Integer ns: float64 would round the stamps by 256 ns.
            edges = [start_ns + (stop - start_ns) * k // jobs
                     for k in range(jobs + 1)]
            job = {"bags": self.bags, "profile": p, "out": out, "rig": rig,
                   "corrections": corrections}
            # Spawned, not forked: the parent may run OpenMP / OpenCV
            # threads.
            with concurrent.futures.ProcessPoolExecutor(
                    jobs, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                futures = [pool.submit(_convert_window, job, (a, b))
                           for a, b in zip(edges[:-1], edges[1:])]
                for k, _ in enumerate(
                        concurrent.futures.as_completed(futures)):
                    self.log(f"[convert] window {k + 1}/{jobs} done")
                parts = [future.result() for future in futures]
            rig.save(out / "rig.yaml")  # Also written by the workers.
        imu_rows = [row for part in parts for row in part["imu"]]
        wheel_rows = [row for part in parts for row in part["wheel"]]
        gnss_fixes = [fix for part in parts for fix in part["gnss"]]
        ins = [pose for part in parts for pose in part["ins"]]
        counts = {name: sum(part["counts"][name] for part in parts)
                  for name in parts[0]["counts"]}
        first_sweep = next((part["first_sweep"] for part in parts
                            if part["first_sweep"] is not None), None)
        if first_sweep is None:
            raise ValueError("no LiDAR sweep in the selected time range")

        if imu_rows:
            writer.write_csv("imu.csv", "t,gx,gy,gz,ax,ay,az",
                             np.array(imu_rows))
        if wheel_rows and p.wheel.kind == "joint_position":
            wheel_rows = speeds_from_distance(np.array(wheel_rows),
                                              p.wheel.window).tolist()
        if wheel_rows:
            writer.write_csv("wheel.csv", "t,speed", np.array(wheel_rows))
        meta = {"trip_id": trip_id or out.name, "garage_id": p.garage_id}
        if gnss_fixes:
            origin = p.gnss.origin or list(gnss_fixes[0][1:4])
            rows = []
            for t, lat, lon, alt, cov, cov_type in gnss_fixes:
                enu = geodetic_to_enu(lat, lon, alt, origin)
                std = (float(np.sqrt(max(cov[0], cov[4], 1e-6)))
                       if cov_type > 0 else 1.0)
                rows.append([t * 1e-9 + p.gnss.time_offset, *enu, std])
            writer.write_csv("gnss.csv", "t,x,y,z,std", np.array(rows))
            meta["gnss_origin"] = [float(v) for v in origin]
        if ins:
            stamps = np.array([s for s, _ in ins])
            k = int(np.argmin(np.abs(stamps - first_sweep)))
            # world <- vehicle = world <- sensor . sensor <- reference .
            # reference <- vehicle.
            pose = (ins[k][1]
                    @ np.linalg.inv(p.initial_pose.T_reference_sensor)
                    @ np.linalg.inv(T_vehicle_reference))
            meta["initial_pose"] = pose.tolist()
            meta["initial_pose_time"] = float(stamps[k] * 1e-9)
        meta["conversion"] = {
            "bags": [str(b) for b in self.bags],
            "start": float(start), "duration": duration,
            "T_vehicle_reference": T_vehicle_reference.tolist(),
            "reference": p.reference, **info,
            **({"clock_sync": sync_info} if sync_info else {}),
            "counts": {**counts, "imu": len(imu_rows),
                       "wheel": len(wheel_rows), "gnss": len(gnss_fixes)}}
        (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        self.log(f"[convert] {out}: " + ", ".join(
            f"{k} {v}" for k, v in meta["conversion"]["counts"].items()))
        return out


def _convert_window(job: dict, bounds: tuple[int, int]) -> dict:
    """Worker: one time window of the recording (Converter.read_window)."""
    converter = Converter(job["bags"], job["profile"], log=_quiet)
    try:
        writer = dataset.TripWriter(job["out"], job["rig"], {})
        return converter.read_window(writer, job["rig"], job["corrections"],
                                     *bounds)
    finally:
        converter.close()


def _quiet(*args) -> None:
    del args


def convert(bags: list[str | pathlib.Path], profile: profile_lib.Profile,
            out: str | pathlib.Path, trip_id: str | None = None,
            start: float = 0.0, duration: float | None = None,
            log=print, jobs: int = 1) -> pathlib.Path:
    """Converts a recording into a trip directory (see `Converter.run`)."""
    converter = Converter(bags, profile, log)
    try:
        return converter.run(out, trip_id, start, duration, jobs)
    finally:
        converter.close()
