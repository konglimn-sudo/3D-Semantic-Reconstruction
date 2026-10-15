"""Conversion profiles: how a recording maps onto the trip layout.

A profile (YAML) names the topics of a recording and describes the rig: the
extrinsics of every sensor relative to one reference sensor, intrinsics, time
offsets and where the vehicle frame sits on the reference sensor.

    garage_id: hkust_gz_underground
    reference: body_imu        # frame of every T_reference_sensor below
    vehicle:
      T_vehicle_reference: <transform>   # default: identity
      level: false   # true: vehicle z = up from the static IMU at the start
      floor: null    # m from the reference origin down to the floor, "auto"
                     # (from the first sweeps) or null (keep the translation)
      forward: [1, 0, 0]   # with level: true, the vehicle x axis is this
                           # reference-frame direction made horizontal
      reference_xy: [0, 0] # with floor: where the reference sits above the
                           # ground in vehicle x / y (cars: the origin at
                           # the rear axle, where the lateral velocity is 0)
    lidars:
      <name>:
        topic: /os_cloud_node/points
        T_reference_sensor: <transform>
        stamp: end           # header stamp = end | start of the sweep, or
                             # "points" (absolute per-point times)
        time_field: t        # per-point time field (null: none)
        time_scale: 1.0e-9   # field value -> seconds
        intensity_field: intensity
        intensity_scale: 0.001   # field value -> [0, 1]
        min_range: 0.5
        max_range: 60.0
        storage_step: 0.002  # m; int16 storage of the coordinates (0: float)
        sweep_duration: 0.1
        time_offset: 0.0     # s added to the stamps (onto the IMU clock)
        clock_sync:          # optional: offset estimated per recording
          imu_topic: /os_cloud_node/imu/data_raw   # IMU on the LiDAR clock
          imu_delay: 0.0     # s: that IMU's own delay (per LiDAR model)
          window: 30.0       # s per estimate
          model: piecewise   # or linear (offset + drift)
    imu: {topic, T_reference_sensor, time_offset: 0, accel_scale: 1}
    wheel: {topic, kind: twist | odometry | joint_state | joint_position |
            float, joints: [names] (joint_*), radius: 1 (joint_state:
            velocity in rad/s), meters_per_count: 1 and window: 5
            (joint_position: encoder counts, differentiated over `window`
            samples), scale: 1 (rig wheel_speed_scale), time_offset: 0}
    gnss: {topic, origin: [lat, lon, alt] (default: the first fix),
           min_status: 0, time_offset: 0}
    cameras:
      <name>: {topic, model, width, height, params, T_reference_sensor,
               time_offset: 0, rolling_shutter_line_time: 0, scale: 1,
               max_rate: 0 (Hz; 0 keeps every image), quality: 92}
    stereo_pairs: [[left, right]]
    initial_pose: {topic, T_reference_sensor}   # nav_msgs/Odometry (INS)

<transform> is a 4 x 4 matrix (nested lists) or
    {quaternion_wxyz: [w, x, y, z], translation: [x, y, z], invert: false}
giving the sensor in the reference frame (p_reference = R p_sensor + t);
`invert: true` reads the opposite direction (reference in the sensor frame),
as many calibration tools write it.

Time offsets bring every stream onto one clock (the IMU's): a stamp t of a
stream with offset dt becomes t + dt. Camera offsets go to rig.yaml instead,
where the calibration stage refines them.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
from scipy.spatial import transform as spatial_transform
import yaml


def parse_transform(value) -> np.ndarray:
    """(4, 4) transform from a matrix or a quaternion / translation dict.

    Raises:
        ValueError: Neither a 4 x 4 matrix nor a quaternion dict.
    """
    if value is None:
        return np.eye(4)
    if isinstance(value, dict):
        w, x, y, z = (float(v) for v in value["quaternion_wxyz"])
        out = np.eye(4)
        out[:3, :3] = spatial_transform.Rotation.from_quat(
            [x, y, z, w]).as_matrix()
        out[:3, 3] = np.asarray(value.get("translation", [0.0, 0.0, 0.0]),
                                float)
        return np.linalg.inv(out) if value.get("invert", False) else out
    matrix = np.asarray(value, float)
    if matrix.shape != (4, 4):
        raise ValueError(f"a transform must be 4 x 4, got {matrix.shape}")
    return matrix


@dataclasses.dataclass
class ClockSyncProfile:
    """Per-recording LiDAR clock offset from an IMU on the LiDAR's clock.

    The offset of the LiDAR clock to the main IMU is not a constant (it
    changes between recordings and drifts within one). It is measured by
    correlating the LiDAR's own IMU with the main IMU (C++
    `estimate_clock_offset`); `imu_delay` is the LiDAR IMU's own latency
    relative to its sweeps, a constant of the LiDAR model.
    """

    imu_topic: str
    imu_delay: float = 0.0
    window: float = 30.0
    model: str = "piecewise"
    max_lag: float = 0.15


@dataclasses.dataclass
class LidarProfile:
    """One LiDAR topic and its rig entry."""

    name: str
    topic: str
    T_reference_sensor: np.ndarray
    stamp: str = "end"
    time_field: str | None = "t"
    time_scale: float = 1e-9
    intensity_field: str | None = "intensity"
    intensity_scale: float = 1.0
    min_range: float = 0.5
    max_range: float = 60.0
    storage_step: float = 0.002
    sweep_duration: float = 0.1
    time_offset: float = 0.0
    clock_sync: ClockSyncProfile | None = None


@dataclasses.dataclass
class ImuProfile:
    topic: str
    T_reference_sensor: np.ndarray
    time_offset: float = 0.0
    accel_scale: float = 1.0


@dataclasses.dataclass
class WheelProfile:
    """Vehicle speed along its x axis from one of several message kinds."""

    topic: str
    kind: str = "twist"
    joints: list[str] = dataclasses.field(default_factory=list)
    radius: float = 1.0
    meters_per_count: float = 1.0
    window: int = 5
    scale: float = 1.0
    time_offset: float = 0.0


@dataclasses.dataclass
class GnssProfile:
    topic: str
    origin: list[float] | None = None  # lat, lon (deg), alt (m).
    min_status: int = 0
    time_offset: float = 0.0


@dataclasses.dataclass
class CameraProfile:
    """One camera topic, its calibration at full resolution and sampling."""

    name: str
    topic: str
    model: str
    width: int
    height: int
    params: list[float]
    T_reference_sensor: np.ndarray
    time_offset: float = 0.0
    rolling_shutter_line_time: float = 0.0
    scale: float = 1.0
    max_rate: float = 0.0
    quality: int = 92


@dataclasses.dataclass
class InitialPoseProfile:
    topic: str
    T_reference_sensor: np.ndarray


@dataclasses.dataclass
class VehicleProfile:
    """Where the vehicle frame sits on the reference sensor."""

    T_vehicle_reference: np.ndarray
    level: bool = False
    floor: float | str | None = None
    forward: np.ndarray = dataclasses.field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    level_window: float = 1.0  # s of IMU at the start (must be static).
    floor_sweeps: int = 20  # Spread over the recording.
    reference_xy: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(2))


@dataclasses.dataclass
class Profile:
    """A complete conversion profile."""

    garage_id: str
    reference: str
    vehicle: VehicleProfile
    lidars: list[LidarProfile]
    imu: ImuProfile | None = None
    wheel: WheelProfile | None = None
    gnss: GnssProfile | None = None
    cameras: list[CameraProfile] = dataclasses.field(default_factory=list)
    stereo_pairs: list[tuple[str, str]] = dataclasses.field(
        default_factory=list)
    initial_pose: InitialPoseProfile | None = None
    source: dict = dataclasses.field(default_factory=dict)

    def topics(self) -> list[str]:
        """Every topic the conversion reads."""
        out = [lidar.topic for lidar in self.lidars]
        out += [lidar.clock_sync.imu_topic for lidar in self.lidars
                if lidar.clock_sync is not None]
        out += [c.topic for c in self.cameras]
        for item in (self.imu, self.wheel, self.gnss, self.initial_pose):
            if item is not None:
                out.append(item.topic)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> Profile:
        """Profile from the parsed YAML (see the module docstring).

        Raises:
            ValueError: No LiDAR, or an unknown wheel kind / stamp mode.
        """
        vehicle = data.get("vehicle") or {}
        vehicle_profile = VehicleProfile(
            parse_transform(vehicle.get("T_vehicle_reference")),
            bool(vehicle.get("level", False)), vehicle.get("floor"),
            np.asarray(vehicle.get("forward", [1.0, 0.0, 0.0]), float),
            float(vehicle.get("level_window", 1.0)),
            int(vehicle.get("floor_sweeps", 20)),
            np.asarray(vehicle.get("reference_xy", [0.0, 0.0]), float))
        lidars = []
        for name, entry in (data.get("lidars") or {}).items():
            entry = dict(entry)
            transform = parse_transform(entry.pop("T_reference_sensor", None))
            sync = entry.pop("clock_sync", None)
            lidar = LidarProfile(name, entry.pop("topic"), transform,
                                 clock_sync=(ClockSyncProfile(**sync)
                                             if sync else None), **entry)
            if lidar.clock_sync and lidar.clock_sync.model not in (
                    "piecewise", "linear"):
                raise ValueError(f"lidar {name}: unknown clock_sync model "
                                 f"{lidar.clock_sync.model!r}")
            if lidar.stamp not in ("end", "start", "points"):
                raise ValueError(f"lidar {name}: unknown stamp {lidar.stamp!r}")
            lidars.append(lidar)
        if not lidars:
            raise ValueError("a profile needs at least one LiDAR")

        def section(key, kind, has_transform=False):
            entry = data.get(key)
            if not entry:
                return None
            entry = dict(entry)
            if has_transform:
                entry["T_reference_sensor"] = parse_transform(
                    entry.get("T_reference_sensor"))
            return kind(**entry)

        wheel = section("wheel", WheelProfile)
        if wheel is not None and wheel.kind not in (
                "twist", "odometry", "joint_state", "joint_position", "float"):
            raise ValueError(f"unknown wheel kind {wheel.kind!r}")
        cameras = []
        for name, entry in (data.get("cameras") or {}).items():
            entry = dict(entry)
            transform = parse_transform(entry.pop("T_reference_sensor", None))
            cameras.append(CameraProfile(
                name, entry.pop("topic"), entry.pop("model"),
                int(entry.pop("width")), int(entry.pop("height")),
                [float(p) for p in entry.pop("params")], transform, **entry))
        return cls(
            garage_id=str(data.get("garage_id", "garage")),
            reference=str(data.get("reference", "reference")),
            vehicle=vehicle_profile, lidars=lidars,
            imu=section("imu", ImuProfile, True), wheel=wheel,
            gnss=section("gnss", GnssProfile), cameras=cameras,
            stereo_pairs=[tuple(p) for p in data.get("stereo_pairs", [])],
            initial_pose=section("initial_pose", InitialPoseProfile, True),
            source=data)


def load(path: str | pathlib.Path) -> Profile:
    """Reads a profile YAML file."""
    return Profile.from_dict(yaml.safe_load(pathlib.Path(path).read_text()))
