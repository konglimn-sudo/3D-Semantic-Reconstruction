"""Mandeye conversion: a synthetic two-chunk recording (LAS point stream of a
tilted handheld rig above a floor, IMU in g) is converted into a trip and
compared with what was written; a sweep spans the chunk boundary."""

import numpy as np
import pytest
from scipy.spatial import transform as spatial_transform
import yaml

laspy = pytest.importorskip("laspy")  # The dev extra.

from garage_recon import geometry  # noqa: E402
from garage_recon.convert import mandeye  # noqa: E402
from garage_recon.convert import profile as profile_lib  # noqa: E402
from garage_recon.data import dataset  # noqa: E402

T0 = 60.0  # Device clock (s).
DURATION = 1.23  # 12 full sweeps of 0.1 s; the last 0.03 s are dropped.
SPLIT = T0 + 0.55  # Chunk boundary inside sweep 5.
# Points per second; no point falls (within rounding) on a sweep boundary.
RATE = 19_997.3
G = 9.80665
HEIGHT = 1.1
# LiDAR (reference) frame tilted by roll 6 deg and pitch -25 deg.
R_LEVEL_REFERENCE = spatial_transform.Rotation.from_euler(
    "xyz", [6.0, -25.0, 0.0], degrees=True).as_matrix()
T_REFERENCE_IMU = geometry.make_T(t=[0.011, 0.02329, -0.04412])


def _stream(rng):
    """Time, LiDAR-frame points and intensity of the whole recording; every
    tenth point has no return (0, 0, 0)."""
    n = int(DURATION * RATE)
    t = T0 + np.arange(n) / RATE
    radius = rng.uniform(1.6, 7.5, n)
    angle = rng.uniform(-np.pi, np.pi, n)
    level = np.column_stack([radius * np.cos(angle), radius * np.sin(angle),
                             np.full(n, -HEIGHT)])
    wall = np.arange(n) % 7 == 0
    level[wall] = np.column_stack([np.full(wall.sum(), 6.0),
                                   rng.uniform(-3, 3, wall.sum()),
                                   rng.uniform(-1.0, 2.0, wall.sum())])
    xyz = level @ R_LEVEL_REFERENCE  # Level -> reference frame.
    xyz[np.arange(n) % 10 == 3] = 0.0
    intensity = (np.arange(n) % 256).astype(np.uint16)
    return t, xyz, intensity


def _write_las(path, t, xyz, intensity):
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.scales = [1e-4, 1e-4, 1e-4]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.intensity = intensity
    las.gps_time = t
    las.point_source_id = np.zeros(len(t), np.uint16)
    las.write(str(path))


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    root = tmp_path_factory.mktemp("mandeye")
    rng = np.random.default_rng(0)
    t, xyz, intensity = _stream(rng)
    first = t < SPLIT
    _write_las(root / "lidar0000.las", t[first], xyz[first], intensity[first])
    _write_las(root / "lidar0001.las", t[~first], xyz[~first],
               intensity[~first])
    # Static IMU at 200 Hz, gravity in g; the chunk boundary sample is
    # logged twice.
    stamps = T0 - 0.004 + 0.005 * np.arange(int(DURATION / 0.005) + 1)
    gravity = R_LEVEL_REFERENCE.T @ np.array([0.0, 0.0, 1.0])
    imu = np.column_stack([
        np.round(stamps * 1e9), rng.normal(0.0, 1e-3, (len(stamps), 3)),
        gravity + rng.normal(0.0, 1e-3, (len(stamps), 3)),
        np.zeros(len(stamps))])
    split = int(np.searchsorted(stamps, SPLIT))
    np.savetxt(root / "imu0000.csv", imu[:split + 1], fmt="%.12g")
    np.savetxt(root / "imu0001.csv", imu[split:], fmt="%.12g")
    (root / "gnss0000.gnss").write_text("")
    return root, (t, xyz, intensity), imu


def _profile() -> profile_lib.Profile:
    return profile_lib.Profile.from_dict({
        "garage_id": "test_garage", "reference": "livox",
        "vehicle": {"level": True, "floor": "auto", "level_window": 0.5,
                    "floor_sweeps": 5},
        "lidars": {"livox": {"topic": "lidar",
                             "intensity_scale": 1.0 / 255.0,
                             "min_range": 0.5, "storage_step": 0.002}},
        "imu": {"topic": "imu", "T_reference_sensor": T_REFERENCE_IMU.tolist(),
                "accel_scale": G}})


def test_sweeps_across_chunks(recording, tmp_path):
    root, (t, xyz, intensity), imu = recording
    trip = mandeye.convert(root, _profile(), tmp_path / "trip",
                           log=lambda *_: None)
    ds = dataset.TripDataset(trip)
    assert len(ds) == 12
    origin = t[0]
    valid = np.linalg.norm(xyz, axis=1) >= 0.5
    for k in range(12):
        sweep = (t >= origin + 0.1 * k) & (t < origin + 0.1 * (k + 1))
        frame = ds.load_lidar(k)
        assert frame.stamp == pytest.approx(t[sweep][-1], abs=1e-6)
        assert len(frame.points) == (sweep & valid).sum()
        assert np.abs(frame.points - xyz[sweep & valid]).max() <= 0.0011
        assert np.allclose(frame.time, (t - t[sweep][-1])[sweep & valid],
                           atol=1e-6)
        assert frame.time.max() <= 0.0 and frame.time.min() > -0.1
        assert np.allclose(frame.intensity,
                           intensity[sweep & valid] / 255.0, atol=1e-3)
    # IMU in s and m/s^2; the repeated boundary sample appears once.
    rows = ds.imu()
    assert len(rows) == len(imu)
    assert np.allclose(rows[:, 0], imu[:, 0] * 1e-9, atol=1e-8)
    assert np.allclose(rows[:, 4:7], imu[:, 4:7] * G, atol=1e-6)


def test_tilted_rig_is_levelled_onto_the_floor(recording, tmp_path):
    root, _, _ = recording
    trip = mandeye.convert(root, _profile(), tmp_path / "trip",
                           log=lambda *_: None)
    meta = yaml.safe_load((trip / "meta.yaml").read_text())
    T_vehicle_reference = np.asarray(
        meta["conversion"]["T_vehicle_reference"])
    up = T_vehicle_reference[:3, :3] @ R_LEVEL_REFERENCE.T @ [0.0, 0.0, 1.0]
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=2e-3)
    assert T_vehicle_reference[2, 3] == pytest.approx(HEIGHT, abs=0.01)
    rig = dataset.TripDataset(trip).rig
    assert np.allclose(rig.primary_lidar.T_vehicle_lidar, T_vehicle_reference)
    assert np.allclose(rig.T_vehicle_imu,
                       T_vehicle_reference @ T_REFERENCE_IMU)
    assert meta["garage_id"] == "test_garage"


def test_time_window(recording, tmp_path):
    root, _, _ = recording
    trip = mandeye.convert(root, _profile(), tmp_path / "trip", start=0.3,
                           duration=0.6, log=lambda *_: None)
    ds = dataset.TripDataset(trip)
    assert len(ds) == 6
    imu_start = T0 - 0.004
    assert ds.lidar_stamps[0] > imu_start + 0.3
    assert ds.lidar_stamps[-1] < imu_start + 0.9
    assert ds.imu()[:, 0].min() >= imu_start + 0.3 - 1e-9
