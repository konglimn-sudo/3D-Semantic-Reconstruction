"""Localization in the navigation map (C++ Localizer through nav.localize)."""

import json
import types

import numpy as np
import pytest

from garage_recon import _core
from garage_recon import geometry
from garage_recon.nav import localize
from garage_recon.semantics import taxonomy


def test_a_trip_localizes_in_its_map(tiny_run, tmp_path):
    work = tiny_run["root"] / "work"
    trip = tiny_run["root"] / "data" / "trip_00"
    report = localize.run(work, trip, tmp_path, tiny_run["cfg"],
                          prior="entrance",
                          reference=tiny_run["root"] / "data" / "gt" /
                          "trip_00", log=lambda *_: None)
    assert report["tracked"] == report["sweeps"]
    assert report["registered_share_of_tracked"] >= 0.95
    # Against the map (rigidly aligned); the absolute error adds the tiny
    # map's own offset (16 beams, one trip: a few cm).
    assert report["errors"]["position_aligned_m"]["p95"] < 0.03
    assert report["errors"]["position_m"]["p95"] < 0.15
    assert report["errors"]["yaw_deg"]["p95"] < 0.5
    assert report["last_position"]["level"] == 0
    assert (tmp_path / "poses.tum").exists()
    saved = json.loads((tmp_path / "localization.json").read_text())
    assert saved["tracked"] == report["tracked"]


def test_the_localization_map_keeps_the_permanent_structure(tiny_run):
    nav = tiny_run["root"] / "work" / "semantics" / "nav"
    assert len(localize.load_map(nav)) > 1000
    # Vehicles (parked or moving) and dynamic points are left out; parked
    # cars stay with keep_parked.
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0]])
    labels = np.array([taxonomy.cid("wall"), taxonomy.cid("vehicle"),
                       taxonomy.cid("vehicle_moving"), taxonomy.cid("column")])
    point_map = types.SimpleNamespace(
        xyz=xyz, dynamic=np.array([False, False, False, True]))
    kept = localize.structure_points(point_map, labels, 0.1)
    assert kept[:, 0].tolist() == [0.0]
    kept = localize.structure_points(point_map, labels, 0.1, keep_parked=True)
    assert sorted(kept[:, 0].tolist()) == [0.0, 1.0]


def test_levelled_start_points_z_along_gravity():
    class Rig:  # pylint: disable=too-few-public-methods
        T_vehicle_imu = np.eye(4)

    class Trip:  # pylint: disable=too-few-public-methods
        rig = Rig()

        def imu(self):
            # The vehicle stands on a ramp pitched 8 deg nose up: in its
            # frame the specific force (up) leans forward.
            pitch = np.radians(8.0)
            up = np.array([np.sin(pitch), 0.0, np.cos(pitch)]) * 9.81
            t = np.linspace(0.0, 2.0, 200)
            return np.column_stack([t, np.zeros((200, 3)),
                                    np.tile(up, (200, 1))])

    T = localize.levelled_start(Trip(), 1.0, 1.0)
    up_world = T[:3, :3] @ np.array([np.sin(np.radians(8.0)), 0.0,
                                     np.cos(np.radians(8.0))])
    assert up_world == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    # The vehicle's forward axis keeps its heading (x) and climbs.
    forward = T[:3, :3] @ np.array([1.0, 0.0, 0.0])
    assert forward[1] == pytest.approx(0.0, abs=1e-9)
    assert np.degrees(np.arcsin(forward[2])) == pytest.approx(8.0)
    assert geometry.yaw_of(T) == pytest.approx(0.0, abs=1e-9)


def test_placement_and_localizer_bindings():
    rng = np.random.default_rng(0)
    # An L-shaped room: walls along x and y, a floor.
    walls = np.vstack([
        np.column_stack([rng.uniform(0, 20, 4000), np.zeros(4000),
                         rng.uniform(0, 3, 4000)]),
        np.column_stack([np.zeros(4000), rng.uniform(0, 12, 4000),
                         rng.uniform(0, 3, 4000)]),
        np.column_stack([rng.uniform(0, 20, 4000), np.full(4000, 12.0),
                         rng.uniform(0, 3, 4000)]),
        np.column_stack([np.full(2000, 20.0), rng.uniform(0, 6, 2000),
                         rng.uniform(0, 3, 2000)]),
        np.column_stack([rng.uniform(0, 20, 8000), rng.uniform(0, 12, 8000),
                         np.zeros(8000)]),
    ])
    T = np.eye(4)
    T[:3, :3] = geometry.rot_z(0.3)
    T[:3, 3] = [4.0, -2.0, 0.0]
    moved = geometry.transform(geometry.inv(T), walls)
    placement = _core.place_gravity_aligned(moved, walls)
    assert set(placement) >= {"placed", "verdict", "T", "fitness"}
    assert placement["hypotheses"] >= 1
    config = _core.LocalizerConfig()
    assert config.placement.min_fitness == pytest.approx(0.7)
    localizer = _core.Localizer(walls, config)
    assert localizer.map_size > 0
    with pytest.raises(ValueError):
        localizer.process(walls[:10], np.zeros(3), 0.0, np.eye(4))
    result = localizer.process(walls[:2000], None, 0.0, np.eye(4))
    assert result["status"] == "initializing"


def test_trip_localizer_and_nav_position_bindings():
    rng = np.random.default_rng(1)
    floor = np.column_stack([rng.uniform(-10, 10, 5000),
                             rng.uniform(-10, 10, 5000), np.zeros(5000)])
    config = _core.TripLocalizerConfig()
    assert config.level_window == pytest.approx(1.0)
    logs = []
    trip = _core.TripLocalizer(floor, config, np.eye(4), np.eye(4),
                               _core.MotionPrior(), np.eye(4), logs.append)
    assert trip.map_size > 0 and logs[0].startswith("[localize] map")
    # Odometry only: nothing recorded.
    assert trip.process(floor[:500], None, None, 0.1, 0, False, False) is None
    r = trip.process(floor[:500], None, None, 0.2, 1, True, True)
    assert set(r) >= {"status", "T_map_vehicle", "T_odom_vehicle", "ms",
                      "sigma_t", "index"}
    summary = trip.summary()
    assert summary["sweeps"] == 1
    assert set(summary) >= {"first_tracked", "ms_median", "degenerate_share"}

    edges = [(0, "aisle", np.array([[0.0, 0.0], [10.0, 0.0]])),
             (0, "connector", np.array([[3.0, 0.0], [3.0, 1.0]]))]
    slots = [(0, "L0-001", np.array([[2.0, 0.5], [4.0, 0.5], [4.0, 3.0],
                                     [2.0, 3.0]]))]
    floors = [(0, -5.0, -5.0, 1.0, np.zeros((20, 20)))]
    T = np.eye(4)
    T[:3, 3] = [3.0, 0.8, 0.1]
    p = _core.nav_position(T, edges, slots, floors)
    assert p["level"] == 0 and p["slot"] == "L0-001"
    assert p["edge"]["edge"] == 0
    assert p["edge"]["along"] == pytest.approx(3.0)
    assert p["edge"]["lateral"] == pytest.approx(0.8)
