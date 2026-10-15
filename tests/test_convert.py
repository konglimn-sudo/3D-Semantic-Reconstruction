"""ROS bag conversion: a synthetic ROS 1 bag is converted into a trip and
compared with what was written (LiDAR, IMU, wheel, GNSS, cameras, INS pose,
rig), and a tilted handheld rig gets a levelled vehicle frame on the
floor."""

import cv2
import numpy as np
import pytest
from rosbags import rosbag1
from rosbags import typesys
from scipy.spatial import transform as spatial_transform
import yaml

from garage_recon import geometry
from garage_recon.convert import profile as profile_lib
from garage_recon.convert import rosbag
from garage_recon.data import dataset

TYPES = typesys.get_typestore(typesys.Stores.ROS1_NOETIC)
START = 1_700_000_000.0
ORIGIN = [22.3, 113.5, 10.0]  # lat, lon, alt
# Reference frame: an IMU 1.3 m above the floor, tilted by roll 8 deg and
# pitch -5 deg.
R_LEVEL_REFERENCE = spatial_transform.Rotation.from_euler(
    "xyz", [8.0, -5.0, 0.0], degrees=True).as_matrix()
HEIGHT = 1.3
T_REFERENCE_LIDAR = geometry.make_T(
    spatial_transform.Rotation.from_euler(
        "z", 1.0, degrees=True).as_matrix(), [0.01, 0.02, 0.05])
T_REFERENCE_CAM = geometry.make_T(
    np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
    [0.05, 0.08, -0.04])


def _msg(msgtype, **fields):
    return TYPES.types[msgtype](**fields)


def _header(t: float):
    sec = int(t)
    return _msg("std_msgs/msg/Header", seq=0, frame_id="",
                stamp=_msg("builtin_interfaces/msg/Time", sec=sec,
                           nanosec=int(round((t - sec) * 1e9))))


def _vec(x, y, z):
    return _msg("geometry_msgs/msg/Vector3", x=float(x), y=float(y),
                z=float(z))


def _sweep_points(rng) -> np.ndarray:
    """Floor points (1.5 - 8 m away) and a wall, in the LiDAR frame."""
    radius = rng.uniform(1.6, 7.5, 3000)
    angle = rng.uniform(-np.pi, np.pi, 3000)
    floor = np.column_stack([radius * np.cos(angle), radius * np.sin(angle),
                             np.full(3000, -HEIGHT)])
    wall = np.column_stack([np.full(500, 6.0), rng.uniform(-3, 3, 500),
                            rng.uniform(-1.0, 2.0, 500)])
    level = np.concatenate([floor, wall])
    reference = level @ R_LEVEL_REFERENCE  # Level -> reference frame.
    return geometry.transform(np.linalg.inv(T_REFERENCE_LIDAR), reference)


def _pointcloud(t_end: float, xyz: np.ndarray, rows: int = 4):
    """An organised Ouster-like cloud: x y z (f4), intensity (u2), 2 pad
    bytes, t (u4, ns from the sweep start); header stamp = end; one padded
    row byte block; invalid zero points in between."""
    n = len(xyz)
    width = int(np.ceil(n / rows))
    total = rows * width
    points = np.zeros(total, np.dtype({
        "names": ["x", "y", "z", "intensity", "t"],
        "formats": ["<f4", "<f4", "<f4", "<u2", "<u4"],
        "offsets": [0, 4, 8, 12, 16], "itemsize": 20}))
    points["x"][:n], points["y"][:n], points["z"][:n] = xyz.T
    points["intensity"][:n] = np.arange(n) % 1000
    points["t"] = np.linspace(0, 99_000_000, total).astype(np.uint32)
    raw = points.view(np.uint8).reshape(rows, width * 20)
    padded = np.zeros((rows, width * 20 + 8), np.uint8)
    padded[:, :width * 20] = raw
    fields = [_msg("sensor_msgs/msg/PointField", name=name, offset=off,
                   datatype=kind, count=1)
              for name, off, kind in (("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
                                      ("intensity", 12, 4), ("t", 16, 6))]
    return _msg("sensor_msgs/msg/PointCloud2", header=_header(t_end),
                height=rows, width=width, fields=fields, is_bigendian=False,
                point_step=20, row_step=width * 20 + 8,
                data=padded.reshape(-1), is_dense=False), points


def _enu_to_geodetic(enu: np.ndarray) -> tuple[float, float, float]:
    """Inverse of the local ENU conversion (exact to well below 1 mm over
    100 m)."""
    lat0 = np.radians(ORIGIN[0])
    e2, a = 6.69437999014e-3, 6378137.0
    n = a / np.sqrt(1.0 - e2 * np.sin(lat0)**2)
    m = a * (1.0 - e2) / (1.0 - e2 * np.sin(lat0)**2)**1.5
    lat = ORIGIN[0] + np.degrees(enu[1] / (m + ORIGIN[2]))
    lon = ORIGIN[1] + np.degrees(enu[0] / ((n + ORIGIN[2]) * np.cos(lat0)))
    return lat, lon, ORIGIN[2] + enu[2]


def _write_bag(path) -> dict:
    """A 2 s recording; returns what was written for the comparisons."""
    rng = np.random.default_rng(0)
    truth = {"sweeps": [], "imu": [], "wheel": [], "gnss": [], "images": []}
    gravity = R_LEVEL_REFERENCE.T @ np.array([0.0, 0.0, 9.81])
    messages = []
    for k in range(20):  # LiDAR, 10 Hz, stamp = end of the sweep.
        t_end = START + 0.1 * (k + 1)
        msg, points = _pointcloud(t_end, _sweep_points(rng))
        messages.append(("/points", "sensor_msgs/msg/PointCloud2", t_end, msg))
        truth["sweeps"].append((t_end, points))
    for k in range(400):  # IMU, 200 Hz, static.
        t = START + 0.005 * k
        gyro = rng.normal(0.0, 1e-3, 3)
        accel = gravity + rng.normal(0.0, 0.01, 3)
        messages.append(("/imu", "sensor_msgs/msg/Imu", t, _msg(
            "sensor_msgs/msg/Imu", header=_header(t),
            orientation=_msg("geometry_msgs/msg/Quaternion", x=0.0, y=0.0,
                             z=0.0, w=1.0),
            orientation_covariance=np.zeros(9),
            angular_velocity=_vec(*gyro),
            angular_velocity_covariance=np.zeros(9),
            linear_acceleration=_vec(*accel),
            linear_acceleration_covariance=np.zeros(9))))
        truth["imu"].append([t, *gyro, *accel])
    for k in range(40):  # Wheel, 20 Hz.
        t = START + 0.05 * k
        messages.append(("/speed", "geometry_msgs/msg/TwistStamped", t, _msg(
            "geometry_msgs/msg/TwistStamped", header=_header(t),
            twist=_msg("geometry_msgs/msg/Twist", linear=_vec(1.25, 0, 0),
                       angular=_vec(0, 0, 0)))))
        truth["wheel"].append([t, 1.25])
    for k in range(4):  # GNSS, 2 Hz, 40 m east / 30 m north of the first.
        t = START + 0.5 * k
        enu = np.array([40.0 * k / 3, 30.0 * k / 3, 0.5 * k])
        lat, lon, alt = _enu_to_geodetic(enu)
        cov = np.zeros(9)
        cov[0] = cov[4] = 0.03**2
        messages.append(("/fix", "sensor_msgs/msg/NavSatFix", t, _msg(
            "sensor_msgs/msg/NavSatFix", header=_header(t),
            status=_msg("sensor_msgs/msg/NavSatStatus", status=0, service=1),
            latitude=lat, longitude=lon, altitude=alt,
            position_covariance=cov, position_covariance_type=2)))
        truth["gnss"].append([t, *enu, 0.03])
    for k in range(20):  # Camera, 10 Hz, 64 x 48 JPEG.
        t = START + 0.1 * k + 0.03
        image = np.zeros((48, 64, 3), np.uint8)
        image[:, :, 0] = np.arange(64) * 4
        image[:, :, 1] = (np.arange(48) * 5)[:, None]
        image[:, :, 2] = 10 * k
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, 98])
        assert ok
        messages.append(("/cam/compressed", "sensor_msgs/msg/CompressedImage",
                         t, _msg("sensor_msgs/msg/CompressedImage",
                                 header=_header(t), format="jpeg",
                                 data=jpeg.reshape(-1))))
        truth["images"].append((t, image))
    world_sensor = geometry.make_T(geometry.rot_z(0.7), [5.0, -2.0, 0.3])
    messages.append(("/ins", "nav_msgs/msg/Odometry", START + 0.12, _msg(
        "nav_msgs/msg/Odometry", header=_header(START + 0.12),
        child_frame_id="ins",
        pose=_msg("geometry_msgs/msg/PoseWithCovariance", covariance=np.zeros(
            36), pose=_msg(
                "geometry_msgs/msg/Pose",
                position=_msg("geometry_msgs/msg/Point", x=5.0, y=-2.0, z=0.3),
                orientation=_msg("geometry_msgs/msg/Quaternion", x=0.0,
                                 y=0.0, z=float(np.sin(0.35)),
                                 w=float(np.cos(0.35))))),
        twist=_msg("geometry_msgs/msg/TwistWithCovariance",
                   covariance=np.zeros(36), twist=_msg(
                       "geometry_msgs/msg/Twist", linear=_vec(0, 0, 0),
                       angular=_vec(0, 0, 0))))))
    truth["world_sensor"] = world_sensor
    with rosbag1.Writer(path) as writer:
        connections = {}
        for topic, msgtype, t, msg in sorted(messages, key=lambda m: m[2]):
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, msgtype,
                                                           typestore=TYPES)
            writer.write(connections[topic], int(t * 1e9),
                         TYPES.serialize_ros1(msg, msgtype))
    return truth


def _profile(level: bool) -> dict:
    return {
        "garage_id": "test_garage",
        "reference": "imu",
        "vehicle": {"level": level, "floor": "auto" if level else 1.3},
        "lidars": {"os": {
            "topic": "/points",
            "T_reference_sensor": T_REFERENCE_LIDAR.tolist(),
            "stamp": "end", "time_field": "t", "time_scale": 1e-9,
            "intensity_field": "intensity", "intensity_scale": 1e-3,
            "min_range": 0.5, "max_range": 60.0, "storage_step": 0.002}},
        "imu": {"topic": "/imu"},
        "wheel": {"topic": "/speed", "kind": "twist", "scale": 1.02},
        "gnss": {"topic": "/fix", "origin": ORIGIN},
        "cameras": {"cam": {
            "topic": "/cam/compressed", "model": "pinhole", "width": 64,
            "height": 48, "params": [40.0, 41.0, 31.5, 23.5, -0.1, 0.05, 0.0,
                                     0.0],
            "T_reference_sensor": {
                "quaternion_wxyz": list(np.roll(
                    spatial_transform.Rotation.from_matrix(
                        T_REFERENCE_CAM[:3, :3]).as_quat(), 1)),
                "translation": T_REFERENCE_CAM[:3, 3].tolist()},
            "time_offset": 0.017, "scale": 0.5, "max_rate": 5.0}},
        "initial_pose": {"topic": "/ins"},
    }


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    root = tmp_path_factory.mktemp("bag")
    truth = _write_bag(root / "rec.bag")
    return root, truth


def test_transform_parsing():
    rotation = spatial_transform.Rotation.from_euler("xyz", [10, -20, 30],
                                                     degrees=True)
    matrix = geometry.make_T(rotation.as_matrix(), [1.0, 2.0, 3.0])
    x, y, z, w = rotation.as_quat()
    parsed = profile_lib.parse_transform(
        {"quaternion_wxyz": [w, x, y, z], "translation": [1.0, 2.0, 3.0]})
    assert np.allclose(parsed, matrix)
    inverted = profile_lib.parse_transform(
        {"quaternion_wxyz": [w, x, y, z], "translation": [1.0, 2.0, 3.0],
         "invert": True})
    assert np.allclose(inverted, np.linalg.inv(matrix))
    assert np.allclose(profile_lib.parse_transform(matrix.tolist()), matrix)


def test_round_trip(recording, tmp_path):
    root, truth = recording
    profile = profile_lib.Profile.from_dict(_profile(level=False))
    trip = rosbag.convert([root / "rec.bag"], profile, tmp_path / "trip",
                          log=lambda *_: None)
    ds = dataset.TripDataset(trip)
    # LiDAR: stamps, points, per-point times relative to the end.
    assert len(ds) == 20
    for k, (t_end, points) in enumerate(truth["sweeps"]):
        frame = ds.load_lidar(k)
        assert abs(frame.stamp - t_end) < 1e-6
        valid = np.linalg.norm(np.column_stack(
            [points["x"], points["y"], points["z"]]), axis=1) >= 0.5
        expected = np.column_stack([points["x"], points["y"],
                                    points["z"]])[valid]
        assert len(frame.points) == valid.sum()
        assert np.abs(frame.points - expected).max() <= 0.0011
        t = points["t"].astype(np.int64)
        relative = (t[valid] - t.max()) * 1e-9
        assert np.allclose(frame.time, relative, atol=1e-6)
        assert frame.time.max() <= 0.0
        assert np.allclose(frame.intensity,
                           points["intensity"][valid] * 1e-3, atol=1e-3)
    # IMU, wheel, GNSS.
    assert np.allclose(ds.imu(), np.array(truth["imu"]), atol=1e-7)
    assert np.allclose(ds.wheel(), np.array(truth["wheel"]), atol=1e-7)
    assert np.allclose(ds.gnss(), np.array(truth["gnss"]), atol=2e-3)
    # Cameras: thinned to 5 Hz, half resolution, intrinsics scaled.
    stamps = ds.camera_stamps_ns("cam")
    assert len(stamps) == 10
    camera = ds.rig.cameras["cam"]
    assert (camera.width, camera.height) == (32, 24)
    assert np.allclose(camera.params[:4], [20.0, 20.5, 15.5, 11.5])
    assert camera.params[4:] == [-0.1, 0.05, 0.0, 0.0]
    assert camera.time_offset == pytest.approx(0.017)
    image = ds.load_image("cam", stamps[1])
    t, original = truth["images"][2]
    assert abs(stamps[1] * 1e-9 - t) < 1e-6
    expected = cv2.resize(original, (32, 24), interpolation=cv2.INTER_AREA)
    assert np.abs(image.astype(int) - expected).mean() < 3.0
    # Rig: the vehicle frame is the reference 1.3 m higher.
    T_vehicle_reference = geometry.make_T(t=[0.0, 0.0, 1.3])
    assert np.allclose(ds.rig.primary_lidar.T_vehicle_lidar,
                       T_vehicle_reference @ T_REFERENCE_LIDAR)
    assert np.allclose(camera.T_vehicle_cam,
                       T_vehicle_reference @ T_REFERENCE_CAM, atol=1e-9)
    assert ds.rig.wheel_speed_scale == pytest.approx(1.02)
    # INS pose at the first sweep: world <- vehicle.
    meta = yaml.safe_load((trip / "meta.yaml").read_text())
    assert meta["garage_id"] == "test_garage"
    assert np.allclose(ds.initial_pose(), truth["world_sensor"]
                       @ np.linalg.inv(T_vehicle_reference), atol=1e-9)
    assert ds.initial_pose_time() == pytest.approx(START + 0.12)


def test_handheld_rig_is_levelled_onto_the_floor(recording, tmp_path):
    root, _ = recording
    profile = profile_lib.Profile.from_dict(_profile(level=True))
    trip = rosbag.convert([root / "rec.bag"], profile, tmp_path / "trip",
                          duration=0.95, log=lambda *_: None)
    meta = yaml.safe_load((trip / "meta.yaml").read_text())
    T_vehicle_reference = np.asarray(
        meta["conversion"]["T_vehicle_reference"])
    # z up: gravity (level frame) maps to the vehicle -z.
    up = T_vehicle_reference[:3, :3] @ R_LEVEL_REFERENCE.T @ [0.0, 0.0, 1.0]
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=2e-3)
    assert meta["conversion"]["level_tilt_deg"] == pytest.approx(
        np.degrees(np.arccos(R_LEVEL_REFERENCE[2, 2])), abs=0.1)
    assert T_vehicle_reference[2, 3] == pytest.approx(HEIGHT, abs=0.01)
    # A floor point right below the reference origin is at vehicle z = 0.
    below = R_LEVEL_REFERENCE.T @ [0.0, 0.0, -HEIGHT]
    assert abs(geometry.transform(T_vehicle_reference, below[None])[0, 2]) \
        < 0.01
    assert len(dataset.TripDataset(trip)) == 9  # 0.95 s window.


def _rate(t: float) -> np.ndarray:
    """Angular rate of a turning, swaying rig (rad/s)."""
    return np.array([0.3 * np.sin(1.7 * t) + 0.1 * np.sin(5.3 * t + 1.0),
                     0.2 * np.sin(2.3 * t + 0.4) + 0.1 * np.cos(7.1 * t),
                     0.5 * np.sin(0.9 * t + 2.0) + 0.2 * np.sin(3.7 * t)])


def _imu_msg(t: float, rate: np.ndarray):
    return _msg("sensor_msgs/msg/Imu", header=_header(t),
                orientation=_msg("geometry_msgs/msg/Quaternion", x=0.0,
                                 y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.zeros(9),
                angular_velocity=_vec(*rate),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration=_vec(0.0, 0.0, 9.81),
                linear_acceleration_covariance=np.zeros(9))


def test_lidar_clock_is_synchronised_per_recording(tmp_path):
    # The LiDAR (and its IMU) clock runs 45 ms + 0.1 ms/s behind the IMU.
    def lag(t_lidar):
        return 0.045 + 1e-4 * (t_lidar - START)
    rng = np.random.default_rng(1)
    messages = []
    for k in range(int(60 / 0.005)):  # Main IMU, 200 Hz.
        t = START + 0.005 * k
        messages.append(("/imu", "sensor_msgs/msg/Imu", t,
                         _imu_msg(t, _rate(t - START))))
    for k in range(int(59 / 0.01)):  # LiDAR IMU, 100 Hz, own clock.
        t = START + 0.5 + 0.01 * k
        messages.append(("/lidar_imu", "sensor_msgs/msg/Imu", t,
                         _imu_msg(t, _rate(t + lag(t) - START))))
    sweep_stamps = START + 1.0 + 0.1 * np.arange(500)
    for t in sweep_stamps:  # Stamped by the LiDAR clock.
        msg, _ = _pointcloud(t, _sweep_points(rng)[:200])
        messages.append(("/points", "sensor_msgs/msg/PointCloud2", t, msg))
    with rosbag1.Writer(tmp_path / "sync.bag") as writer:
        connections = {}
        for topic, msgtype, t, msg in sorted(messages, key=lambda m: m[2]):
            if topic not in connections:
                connections[topic] = writer.add_connection(topic, msgtype,
                                                           typestore=TYPES)
            writer.write(connections[topic], int(t * 1e9),
                         TYPES.serialize_ros1(msg, msgtype))
    data = _profile(level=False)
    del data["wheel"], data["gnss"], data["cameras"], data["initial_pose"]
    data["lidars"]["os"]["clock_sync"] = {
        "imu_topic": "/lidar_imu", "imu_delay": 0.005, "window": 10.0,
        "model": "linear"}
    trip = rosbag.convert([tmp_path / "sync.bag"],
                          profile_lib.Profile.from_dict(data),
                          tmp_path / "trip", log=lambda *_: None)
    stamps = dataset.TripDataset(trip).lidar_stamps
    expected = sweep_stamps + np.array([lag(t) for t in sweep_stamps]) + 0.005
    assert np.abs(stamps - expected).max() < 1e-3
    meta = yaml.safe_load((trip / "meta.yaml").read_text())
    sync = meta["conversion"]["clock_sync"]["os"]
    assert sync["windows"] >= 4
    assert sync["drift_ms_per_min"] == pytest.approx(6.0, abs=0.5)
