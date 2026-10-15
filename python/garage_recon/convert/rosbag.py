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

Rigs that are not cars (handheld, backpack) have no natural vehicle frame
on the floor. With `vehicle.level` the converter makes one: z up from the
gravity the IMU measures while the rig stands still at the start, and with
`vehicle.floor: auto` the origin on the floor plane under the reference
sensor (RANSAC plane in the first sweeps). The pipeline assumes both: the
vehicle origin is on the floor (levels, ego volume) and the world frame of a
trip without an INS pose is the vehicle frame at the start (gravity factors).
"""

from __future__ import annotations

import dataclasses
import pathlib

import cv2
import numpy as np
from rosbags import highlevel
from scipy.spatial import transform as spatial_transform
import yaml

from garage_recon import _core
from garage_recon import geometry
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
    t_max = float(t.max()) if len(t) else 0.0
    if lidar.stamp == "points":  # Absolute per-point times.
        end = t_max
    elif lidar.stamp == "start":
        end = header * 1e-9 + t_max - float(t.min() if len(t) else 0.0)
    else:
        end = header * 1e-9
    relative = t - t_max
    with np.errstate(invalid="ignore"):
        rng = np.linalg.norm(xyz, axis=1)
    keep = (np.isfinite(rng) & (rng >= lidar.min_range)
            & (rng <= lidar.max_range))
    if not keep.any():
        return None
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
        out = vehicle.T_vehicle_reference.copy()
        info = {}
        if vehicle.level:
            imu = self.profile.imu
            if imu is None:
                raise ValueError("vehicle.level needs the IMU")
            stop = start_ns + int(vehicle.level_window * 1e9)
            rows = np.array([[m.angular_velocity.x, m.angular_velocity.y,
                              m.angular_velocity.z, m.linear_acceleration.x,
                              m.linear_acceleration.y,
                              m.linear_acceleration.z]
                             for _, _, m in self._messages(
                                 [imu.topic], start_ns, stop)])
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
                self.log(f"[convert] warning: the rig moves during the "
                         f"levelling window (gyro {gyro:.3f} rad/s, accel "
                         f"spread {spread:.2f} m/s^2)")
        if vehicle.floor is not None:
            if vehicle.floor == "auto":
                # Sweeps spread over the recording: the rig's carrying
                # height, not where it was put down at the start.
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
                height, used = floor_height(sweeps, out[:3, :3])
                info.update(floor_height=round(height, 4), floor_sweeps=used)
            else:
                height = float(vehicle.floor)
            out[:3, 3] = [vehicle.reference_xy[0], vehicle.reference_xy[1],
                          height]
        return out, info

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
            offset, drift = result["offset"], result["drift"]
            reference_time = result["reference_time"]

            def correction(t):
                return (offset + drift * (t - reference_time)
                        + sync.imu_delay)
        else:
            # Piecewise linear through the window lags, median of three
            # neighbours against single bad windows; held at the ends.
            lags = windows[:, 1].copy()
            if len(lags) >= 3:
                padded = np.concatenate([lags[:1], lags, lags[-1:]])
                lags = np.median(np.stack([padded[:-2], padded[1:-1],
                                           padded[2:]]), axis=0)
            centers = windows[:, 0]

            def correction(t):
                return float(np.interp(t, centers, lags)) + sync.imu_delay
        lags_ms = 1000 * windows[:, 1]
        self.log(f"[convert] {lidar.name} clock: {sync.imu_topic} vs the IMU "
                 f"in {len(windows)} windows, lag {lags_ms.min():+.1f} .. "
                 f"{lags_ms.max():+.1f} ms, + delay "
                 f"{1000 * sync.imu_delay:+.1f} ms ({sync.model})")
        return correction, info

    def rig(self, T_vehicle_reference: np.ndarray) -> schema.Rig:
        """The rig.yaml content (cameras at their output resolution)."""
        p = self.profile
        lidars = {
            lidar.name: schema.LidarCalib(
                lidar.name, T_vehicle_reference @ lidar.T_reference_sensor,
                lidar.sweep_duration) for lidar in p.lidars}
        cameras = {}
        for cam in p.cameras:
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
        T_vehicle_imu = (T_vehicle_reference @ p.imu.T_reference_sensor
                         if p.imu is not None else np.eye(4))
        return schema.Rig(lidars, cameras, list(p.stereo_pairs),
                          T_vehicle_imu,
                          p.wheel.scale if p.wheel is not None else 1.0)

    def run(self, out: str | pathlib.Path, trip_id: str | None = None,
            start: float = 0.0, duration: float | None = None
            ) -> pathlib.Path:
        """Writes the trip.

        Args:
            out: Trip directory to create.
            trip_id: Defaults to the directory name.
            start: Seconds to skip at the beginning of the recording.
            duration: Seconds to convert (None: to the end).

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
        rig = self.rig(T_vehicle_reference)
        corrections, sync_info = {}, {}
        for lidar in p.lidars:
            if lidar.clock_sync is not None:
                corrections[lidar.name], sync_info[lidar.name] = (
                    self.clock_sync(lidar, start_ns, stop_ns))
        writer = dataset.TripWriter(out, rig, {})
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
                gnss_fixes.append((t, msg))
            if p.initial_pose is not None and topic == p.initial_pose.topic:
                ins.append((t, _pose_matrix(msg.pose.pose)))
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
            first = gnss_fixes[0][1]
            origin = p.gnss.origin or [first.latitude, first.longitude,
                                       first.altitude]
            rows = []
            for t, fix in gnss_fixes:
                enu = geodetic_to_enu(fix.latitude, fix.longitude,
                                      fix.altitude, origin)
                cov = np.asarray(fix.position_covariance, float)
                std = (float(np.sqrt(max(cov[0], cov[4], 1e-6)))
                       if fix.position_covariance_type > 0 else 1.0)
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


def convert(bags: list[str | pathlib.Path], profile: profile_lib.Profile,
            out: str | pathlib.Path, trip_id: str | None = None,
            start: float = 0.0, duration: float | None = None,
            log=print) -> pathlib.Path:
    """Converts a recording into a trip directory (see `Converter.run`)."""
    converter = Converter(bags, profile, log)
    try:
        return converter.run(out, trip_id, start, duration)
    finally:
        converter.close()
