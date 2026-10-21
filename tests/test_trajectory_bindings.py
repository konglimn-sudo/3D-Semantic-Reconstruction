"""Python checks of the pose-set bindings: trajectory, deskew, exposures,
camera calibration and identity association."""

import numpy as np
import pytest

from garage_recon import _core
from garage_recon import geometry


def _circle(t: np.ndarray, yaw_rate: float = 0.4,
            speed: float = 3.0) -> np.ndarray:
    yaw = yaw_rate * t
    radius = speed / yaw_rate
    poses = np.zeros((len(t), 4, 4))
    for k, angle in enumerate(yaw):
        poses[k] = geometry.make_T(geometry.rot_z(angle),
                                   [radius * np.sin(angle),
                                    radius * (1 - np.cos(angle)), 0.0])
    return poses


def test_trajectory_passes_knots_and_validates_input():
    stamps = np.arange(0.0, 10.0, 0.5)
    trajectory = _core.ContinuousTrajectory(stamps, _circle(stamps))
    assert trajectory.start_time == 0.0 and trajectory.end_time == 9.5
    assert np.allclose(trajectory.at(2.0), _circle(np.array([2.0]))[0])
    assert trajectory.poses_at([1.0, 3.25]).shape == (2, 4, 4)
    with pytest.raises(ValueError):
        _core.ContinuousTrajectory(stamps[::-1], _circle(stamps))
    with pytest.raises(ValueError):
        _core.ContinuousTrajectory(stamps, _circle(stamps)[:3])


def test_gyro_and_wheel_shape_the_curve_between_knots():
    t = np.arange(0.0, 10.0, 0.005)
    imu = np.column_stack([t, np.zeros(len(t)), np.zeros(len(t)),
                           np.full(len(t), 0.4), np.zeros(len(t)),
                           np.full(len(t), 1.2), np.full(len(t), 9.81)])
    wheel = np.column_stack([t, np.full(len(t), 3.0)])
    stamps = np.arange(1.0, 9.0, 0.6)
    knots_only = _core.ContinuousTrajectory(stamps, _circle(stamps))
    inertial = _core.ContinuousTrajectory(
        stamps, _circle(stamps), imu=_core.ImuSamples(imu),
        wheel=_core.WheelSamples(wheel))
    assert inertial.gyro_coverage == 1.0 and inertial.wheel_coverage == 1.0
    query = np.arange(1.3, 8.5, 0.6)  # Mid-intervals.
    truth = _circle(query)[:, :3, 3]
    error_knots = np.linalg.norm(knots_only.poses_at(query)[:, :3, 3] - truth,
                                 axis=1).max()
    error_inertial = np.linalg.norm(
        inertial.poses_at(query)[:, :3, 3] - truth, axis=1).max()
    assert error_inertial < 1e-3 < error_knots


def test_deskew_and_exposure_bindings():
    stamps = np.arange(0.0, 10.0, 0.05)
    trajectory = _core.ContinuousTrajectory(stamps, _circle(stamps))
    points = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    out = _core.deskew_sweep(points, np.array([-0.1, 0.0]), 5.0, trajectory,
                             np.eye(4))
    assert np.allclose(out[1], points[1]) and not np.allclose(out[0], points[0])
    camera = _core.make_camera("pinhole", 64, 48, [40, 40, 32, 24])
    pose = geometry.make_T(None, [0.0, 0.0, 5.0])  # camera <- world
    single, _, _ = _core.render_points(camera, pose, np.zeros((1, 3)))
    pair, _, _ = _core.render_points(camera, np.stack([pose, pose]),
                                     np.zeros((1, 3)))
    assert np.isfinite(single).sum() == np.isfinite(pair).sum() > 0
    with pytest.raises(ValueError):
        _core.render_points(camera, np.zeros((3, 4, 4)), np.zeros((1, 3)))


def test_identity_associator_binding():
    floor = np.column_stack([np.repeat(np.arange(0, 4, 0.1), 40),
                             np.tile(np.arange(-2, 2, 0.1), 40),
                             np.zeros(1600)])
    associator = _core.IdentityAssociator(floor)
    camera = _core.make_camera("pinhole", 64, 48, [40, 40, 32, 24])
    look_down = geometry.make_T(np.diag([1.0, -1.0, -1.0]), [2.0, 0.0, 3.0])
    masks = np.zeros((48, 64), np.int32)
    records = associator.add_view(camera, look_down, masks)
    assert list(records["mask"]) == [0] and records["identity"][0] == 0
    associator.add_view(camera, look_down, masks + 7)  # Other mask ids.
    result = associator.finish()
    assert result["num_identities"] == 1
    assert (result["point_identity"] >= 0).sum() > 100
    with pytest.raises(ValueError):
        associator.add_view(camera, look_down, np.zeros((10, 10), np.int32))


def _plane_points(rng: np.random.Generator) -> np.ndarray:
    """A floor, a ceiling and two walls of a 10 x 6 x 3 m room (world)."""
    floor = np.column_stack([rng.uniform(0, 10, 4000), rng.uniform(0, 6, 4000),
                             np.zeros(4000)])
    ceiling = floor + [0.0, 0.0, 3.0]
    wall_x = np.column_stack([np.zeros(2000), rng.uniform(0, 6, 2000),
                              rng.uniform(0, 3, 2000)])
    wall_y = np.column_stack([rng.uniform(0, 10, 2000), np.zeros(2000),
                              rng.uniform(0, 3, 2000)])
    return np.concatenate([floor, ceiling, wall_x, wall_y])


def test_bundle_adjustment_planes_binding():
    rng = np.random.default_rng(3)
    world = _plane_points(rng)
    keyframes = _core.KeyframeSet([0.1])
    poses = []
    for k in range(3):
        pose = geometry.make_T(geometry.rot_z(0.2 * k), [3.0 + k, 3.0, 1.8])
        poses.append(pose)
        local = (world - pose[:3, 3]) @ pose[:3, :3]
        keyframes.add(0, k, float(k), pose, local)
    config = _core.LidarBaConfig()
    config.min_range = 1.0  # A small room: the default (5 m) keeps few returns.
    planes = _core.extract_planes(keyframes, np.asarray(poses), config)
    offsets = planes["offsets"].astype(int)
    assert len(planes["rms"]) > 50
    assert np.all(np.diff(offsets) >= 2)  # Every plane: >= 2 keyframes.
    # Noise-free points: flat planes, except a few voxels at the room's
    # corners that keep a little of the other surface.
    assert np.median(planes["rms"]) < 1e-3
    assert (planes["rms"] < 0.005).mean() > 0.95
    assert planes["rms"].max() <= config.max_rms
    assert set(np.unique(planes["keyframes"]).astype(int)) == {0, 1, 2}
    with pytest.raises(ValueError):
        _core.extract_planes(keyframes, np.asarray(poses[:2]), config)


def test_map_fusion_binding():
    fusion = _core.MapFusionConfig()
    accumulator = _core.MapAccumulator(0.1, 20.0, fusion)
    sweeps = [(geometry.make_T(None, [-2.0, 0.0, 0.0]), [[2.02, 0.02, 0.0]]),
              (geometry.make_T(None, [-3.0, 0.0, 0.0]), [[3.08, 0.02, 0.0]])]
    for pose, sweep in sweeps:
        accumulator.add_ranges(pose, np.asarray(sweep))
    for k, (pose, sweep) in enumerate(sweeps):
        accumulator.add(k, pose, np.asarray(sweep), np.array([0.5]))
    result = accumulator.result()
    assert result["observations"].tolist() == [2]
    assert abs(result["xyz"][0, 0] - 0.05) < 1e-3
    sigma = _core.point_sigmas(np.array([[1.0, 0.0, 0.0]]),
                               result["observations"], result["spread"],
                               result["model_sigma"], 0.003)
    assert abs(sigma[0] - 0.0214) < 5e-4
