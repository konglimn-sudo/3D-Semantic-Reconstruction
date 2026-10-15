"""Semantic navigation map: the tiny garage end to end, and its parts."""

import json
import re

import numpy as np
import pytest
import yaml

from garage_recon import config as config_lib
from garage_recon.nav import address
from garage_recon.nav import graph as graph_lib
from garage_recon.nav import layers as layers_lib
from garage_recon.recon import bev
from garage_recon.semantics import numbers


def test_every_slot_is_reachable_without_touching_structure(tiny_run):
    cfg = tiny_run["cfg"]
    nav = tiny_run["root"] / "work" / "semantics" / "nav"
    qa = json.loads((nav / "qa.json").read_text())
    assert qa["slots"] > 0
    assert qa["reachable"] == qa["slots"]
    assert qa["routes_checked"] == qa["slots"]
    assert qa["routes_touching_structure"] == []
    assert qa["route_min_clearance_m"] >= 0.5 * cfg.nav.vehicle.width
    assert qa["driven_path_on_graph"]["0"] > 0.9
    goals = json.loads((nav / "goals.json").read_text())["goals"]
    for goal in goals:
        assert goal["reachable"] and goal["route_length"] > 0.0
        assert len(goal["park"]["xyz"]) == 3 and goal["approach"] is not None
        # Head-in and back-in headings are opposite.
        diff = goal["park"]["yaw_head_in"] - goal["park"]["yaw_back_in"]
        assert abs(abs(np.angle(np.exp(1j * diff))) - np.pi) < 1e-3


def test_ros_map_matches_its_layers(tiny_run):
    nav = tiny_run["root"] / "work" / "semantics" / "nav"
    meta = yaml.safe_load((nav / "level_0.yaml").read_text())
    data = (nav / meta["image"]).read_bytes()
    header = data.split(b"\n", 3)
    width, height = map(int, header[1].split())
    image = np.frombuffer(header[3], np.uint8).reshape(height, width)
    with np.load(nav / "level_0_layers.npz") as z:
        structure, grid = z["structure"], z["grid"]
    assert (width, height) == (int(grid[3]), int(grid[4]))
    assert meta["resolution"] == pytest.approx(float(grid[2]))
    occupied = np.flipud(image) == 0
    assert np.array_equal(occupied, structure)
    assert set(np.unique(image)) <= {0, 205, 254}


def test_slot_numbers_match_the_ground_truth(tiny_run):
    slots = [s.to_json() for s in tiny_run["sem"].slots()]
    truth = json.loads((tiny_run["root"] / "data" / "gt" / "scene.json")
                       .read_text())["slots"]
    correct = 0
    for slot in slots:
        same = [g for g in truth if g["level"] == slot["level"]]
        centers = np.array([np.mean(g["corners"], axis=0) for g in same])
        k = int(np.argmin(np.linalg.norm(
            centers - np.mean(slot["corners"], axis=0), axis=1)))
        number = int(re.findall(r"\d+", same[k]["slot_id"])[-1])
        correct += (slot["number"] is not None
                    and int(slot["number"]) == number
                    and slot["zone"] == same[k]["row"])
    assert correct >= 0.95 * len(slots)


def test_address_query_finds_the_slot(tiny_run):
    nav = tiny_run["root"] / "work" / "semantics" / "nav"
    data = json.loads((nav / "goals.json").read_text())
    goals, names = data["goals"], data["level_names"]
    goal = next(g for g in goals if g["number"])
    query = f"{names[goal['level']]} {goal['zone']}区 {int(goal['number'])}号"
    found = address.find(goals, query, names)
    assert [g["slot_id"] for g in found] == [goal["slot_id"]]
    assert address.find(goals, goal["slot_id"], names)[0] is not None


def test_parse_query_forms():
    names = ["B1", "B2"]
    assert address.parse_query("B2 C区 105号", names) == {
        "level": 1, "zone": "C", "number": "105"}
    assert address.parse_query("负二层C区105", names)["level"] == 1
    assert address.parse_query("b1-a-012", names) == {
        "level": 0, "zone": "A", "number": "012"}
    assert address.parse_query("C105", names) == {
        "level": None, "zone": "C", "number": "105"}
    assert address.parse_query("B3 A 1", names)["level"] == -1


def test_split_text():
    assert numbers.split_text("C-105") == ("C", "105")
    assert numbers.split_text("b12") == ("B", "12")
    assert numbers.split_text("L0-A-007") == ("A", "007")
    assert numbers.split_text("105") == (None, "105")
    assert numbers.split_text("exit") == (None, None)


def _corridor_layers(cfg) -> layers_lib.LevelLayers:
    """A 30 x 12 m level: a wall across x = 15 with a 4 m gap at the top."""
    grid = bev.BevGrid(0, 0.0, 0.0, 0.1, 300, 120)
    shape = (grid.height, grid.width)
    structure = np.zeros(shape, bool)
    structure[:, 150:152] = True
    structure[80:, 150:152] = False  # The gap (y 8..12 m).
    structure[[0, -1], :] = True
    structure[:, [0, -1]] = True
    floor = ~structure
    from garage_recon import _core  # pylint: disable=import-outside-toplevel
    clearance = _core.distance_transform(structure.astype(np.uint8)) * grid.res
    zeros = np.zeros(shape, bool)
    return layers_lib.LevelLayers(
        grid, floor, zeros, structure, zeros, zeros, zeros,
        np.zeros(shape, np.int32), zeros, np.zeros(shape), clearance)


def test_aisle_graph_keeps_clearance_and_connectors_avoid_walls():
    cfg = config_lib.load_config()
    layers = _corridor_layers(cfg)
    graph = graph_lib.NavGraph()
    assert graph_lib.add_level_graph(graph, layers, cfg) > 0
    half = 0.5 * cfg.nav.vehicle.width + cfg.nav.vehicle.margin
    for edge in graph.edges:
        assert layers.sample("clearance", edge.points[:, :2]).min() >= (
            half - layers.grid.res)
    # Both sides of the wall are on one connected graph (through the gap).
    left = graph_lib.connect(graph, layers, [5.0, 2.0, 0.0], 0, 20.0)
    right = graph_lib.connect(graph, layers, [25.0, 2.0, 0.0], 0, 20.0)
    assert left is not None and right is not None
    path = graph.route_graph().shortest_path(left.node, right.node)
    assert path["found"]
    points = graph.path_points(path["nodes"], path["edges"])
    assert not layers.sample("structure", points[:, :2], True).any()
    assert points[:, 1].max() > 8.0  # Through the gap.
    # A point boxed in by walls gets no connector.
    layers.structure[30:50, 30:32] = layers.structure[30:50, 50:52] = True
    layers.structure[30:32, 30:52] = layers.structure[48:50, 30:52] = True
    assert graph_lib.connect(graph, layers, [4.0, 4.0, 0.0], 0, 0.5) is None


def test_ramp_ends_follow_the_slope():
    class Ramp:  # pylint: disable=too-few-public-methods
        plane = np.array([0.15, 0.0, -3.0])  # z rises with x.
        polygon = np.array([[10, 0], [30, 0], [30, 6], [10, 6]], float)

        def z_at(self, xy):
            return self.plane[0] * xy[:, 0] + self.plane[1] * xy[:, 1] + (
                self.plane[2])

    top, bottom, width = graph_lib.ramp_ends(Ramp())
    assert top[0] == pytest.approx(30.0) and bottom[0] == pytest.approx(10.0)
    assert top[2] == pytest.approx(1.5) and bottom[2] == pytest.approx(-1.5)
    assert width == pytest.approx(6.0)


def test_a_slot_across_the_aisle_is_bridged():
    cfg = config_lib.load_config()
    layers = _corridor_layers(cfg)
    layers.structure[:, 150:152] = False
    layers.structure[45:, :] = True  # A 4.4 m wide corridor along x.
    layers.floor = ~layers.structure
    from garage_recon import _core  # pylint: disable=import-outside-toplevel
    layers.clearance = _core.distance_transform(
        layers.structure.astype(np.uint8)) * layers.grid.res
    layers.slot[:, 140:165] = 1  # A misdetected slot across the corridor.
    graph = graph_lib.NavGraph()
    graph_lib.add_level_graph(graph, layers, cfg)
    components = graph_lib._components  # pylint: disable=protected-access
    assert len(set(components(graph))) >= 2
    assert graph_lib.bridge_components(graph, layers, cfg) >= 1
    assert len(set(components(graph))) == 1
    bridge = next(e for e in graph.edges if e.kind == "bridge")
    assert not layers.sample("structure", bridge.points[:, :2], True).any()
