"""Traffic direction, lanes from slot rows, height limits and points of
interest of the navigation map."""

import numpy as np
import pytest

from garage_recon import _core
from garage_recon import config as config_lib
from garage_recon.nav import attributes
from garage_recon.nav import graph as graph_lib
from garage_recon.nav import lanes
from garage_recon.nav import layers as layers_lib
from garage_recon.nav import traffic
from garage_recon.recon import bev
from garage_recon.semantics import slots as slots_lib


def _loop_graph() -> graph_lib.NavGraph:
    """A 40 x 20 m loop of aisles: corners 0 (0, 0), 1 (40, 0), 2 (40, 20),
    3 (0, 20); the bottom aisle is split at x = 20 (node 4)."""
    graph = graph_lib.NavGraph()
    corners = [(0, 0), (40, 0), (40, 20), (0, 20), (20, 0)]
    for x, y in corners:
        graph.add_node([x, y, 0.0], 0)

    def aisle(a, b):
        points = np.array([graph.nodes[a], graph.nodes[b]])
        graph.add_edge(graph_lib.Edge(a, b, points, "aisle", 0, 6.0))

    aisle(0, 4)
    aisle(4, 1)
    aisle(1, 2)
    aisle(2, 3)
    aisle(3, 0)
    return graph


def _arrow(x, y, dx, dy=0.0) -> dict:
    return {"level": 0, "center": [x, y], "direction": [dx, dy]}


def test_centred_arrows_make_a_one_way_corridor():
    cfg = config_lib.load_config()
    graph = _loop_graph()
    # Two arrows on the centre line of the bottom aisle, pointing -x.
    report = traffic.apply_arrows(graph, [_arrow(10, 0.2, -1), _arrow(30, -0.3,
                                                                      -1)],
                                  cfg, log=lambda *_: None)
    assert report["one_way_edges"] == 2
    bottom = [e for e in graph.edges if e.one_way]
    assert all(e.points[0, 0] > e.points[-1, 0] for e in bottom)
    route = graph.route_graph()
    # From the left end to the right end the route goes round the loop.
    assert route.distances(0)[1] == pytest.approx(20 + 40 + 20)
    assert route.distances(1)[0] == pytest.approx(40)


@pytest.mark.parametrize("arrows, verdict", [
    # One arrow per lane, each on its own side: two-way.
    ([_arrow(10, -1.5, 1), _arrow(30, 1.5, -1)], "two-way"),
    # Only the arrows of one lane were seen.
    ([_arrow(10, -1.5, 1), _arrow(30, -1.5, 1)], "two-way (one lane seen)"),
    # A single centred arrow is not enough ...
    ([_arrow(10, 0.0, 1)], "unknown (one arrow)"),
    # ... nor centred arrows with an opposite one close by.
    ([_arrow(10, 0.0, 1), _arrow(30, 0.0, 1), _arrow(31, 1.6, -1)],
     "two-way"),
])
def test_other_arrow_patterns_keep_two_way(arrows, verdict):
    cfg = config_lib.load_config()
    graph = _loop_graph()
    report = traffic.apply_arrows(graph, arrows, cfg, log=lambda *_: None)
    assert report["one_way_edges"] == 0
    assert [c["verdict"] for c in report["corridors"]] == [verdict]


def _slot(row: str, x0: float, y_entrance: float,
          into: float) -> slots_lib.Slot:
    """A 2.5 x 5 m slot with its entrance on y = y_entrance, opening
    towards -into (into = +1: the slot lies above the entrance)."""
    y_back = y_entrance + 5.0 * into
    corners = [[x0, y_entrance], [x0 + 2.5, y_entrance],
               [x0 + 2.5, y_back], [x0, y_back]]
    if into < 0:  # Keep entrance-left, entrance-right order.
        corners = [corners[1], corners[0], corners[3], corners[2]]
    return slots_lib.Slot(f"{row}-{x0}", 0, corners, 0.0, 2.5, 5.0, False,
                          False, 1.0, row)


def test_lanes_run_midway_between_facing_rows_and_in_front_of_single_rows():
    cfg = config_lib.load_config()
    # Row A below y = 0 opening up, row B above y = 7 opening down (a 7 m
    # aisle), row C alone further up, opening down at y = 30.
    slots = ([_slot("A", 2.5 * k, 0.0, -1) for k in range(8)]
             + [_slot("B", 2.5 * k, 7.0, 1) for k in range(8)]
             + [_slot("C", 2.5 * k, 30.0, 1) for k in range(4)])
    navigable = np.ones((500, 400), np.uint8)  # From (-10, -10) at 0.1 m.
    plan = _core.plan_lanes(
        [(s.row, np.asarray(s.corners, float), 0.0, 2.5) for s in slots],
        navigable, -10.0, -10.0, 0.1, [], lanes.lane_config(cfg))
    assert plan["rows"] == 3 and plan["followed"] == 0
    ys = sorted(round(float(np.asarray(l)[:, 1].mean()), 2)
                for l in plan["lanes"])
    assert ys == pytest.approx([3.5, 30.0 - cfg.nav.lanes.lane_offset])
    pair = next(np.asarray(l) for l in plan["lanes"]
                if abs(np.asarray(l)[0, 1] - 3.5) < 0.1)
    assert sorted(pair[:, 0]) == pytest.approx([0.0, 20.0], abs=0.1)


def _open_lot(cfg) -> layers_lib.LevelLayers:
    """A 60 x 40 m open lot with a wall round it."""
    grid = bev.BevGrid(0, 0.0, 0.0, 0.1, 600, 400)
    shape = (grid.height, grid.width)
    structure = np.zeros(shape, bool)
    structure[[0, -1], :] = True
    structure[:, [0, -1]] = True
    clearance = (_core.distance_transform(structure.astype(np.uint8))
                 * grid.res).astype(np.float32)
    zeros = np.zeros(shape, bool)
    return layers_lib.LevelLayers(
        grid, ~structure, zeros, structure, zeros, zeros, zeros,
        np.zeros(shape, np.int32), zeros, np.zeros(shape, np.float32),
        clearance, np.full(shape, np.inf, np.float32))


def test_a_lane_replaces_the_web_of_an_open_lot():
    cfg = config_lib.load_config()
    layers = _open_lot(cfg)
    # One row of slots along the south wall, opening north at y = 8.
    slots = [_slot("A", 10.0 + 2.5 * k, 8.0, -1) for k in range(16)]
    for index, slot in enumerate(slots):
        layers_lib.fill_polygon(layers.slot.shape, layers.grid, slot.corners,
                                layers.slot, index + 1)
    graph = graph_lib.NavGraph()
    graph_lib.add_level_graph(graph, layers, cfg)
    report = lanes.add_level_lanes(graph, layers, slots, cfg)
    assert report["lanes"] == 1
    lane = next(e for e in graph.edges if e.kind == "lane")
    assert lane.points[:, 1] == pytest.approx(8.0 + cfg.nav.lanes.lane_offset)
    assert lane.length > 30.0
    # The lane is part of one connected graph with what is left.
    graph_lib.bridge_components(graph, layers, cfg)
    used = {n for e in graph.edges for n in (e.a, e.b)}
    labels = graph_lib._components(graph)  # pylint: disable=protected-access
    assert len({labels[n] for n in used}) == 1


def test_height_signs_and_headroom():
    assert attributes.height_limit("限高2.2m") == pytest.approx(2.2)
    assert attributes.height_limit("HEIGHT 2.10M") == pytest.approx(2.1)
    assert attributes.height_limit("限高 3米") == pytest.approx(3.0)
    assert attributes.height_limit("5A区") is None
    assert attributes.height_limit("限高20") is None  # Not a height.
    cfg = config_lib.load_config()
    layers = _open_lot(cfg)
    layers.headroom[100:120, 200:400] = 2.4  # A duct across x = 20..40.
    graph = graph_lib.NavGraph()
    graph.add_node([5, 11, 0], 0)
    graph.add_node([55, 11, 0], 0)
    graph.add_edge(graph_lib.Edge(0, 1, np.array([[5, 11, 0], [55, 11, 0.0]]),
                                  "aisle", 0, 6.0))
    attributes.annotate_headroom(graph, {0: layers}, cfg)
    assert graph.edges[0].headroom == pytest.approx(2.4)
    signs = attributes.apply_height_signs(
        graph, [{"text": "限高2.2m", "level": 0, "xyz": [30, 13, 2.5]}], 15.0)
    assert len(signs) == 1 and graph.edges[0].height_limit == pytest.approx(2.2)


def test_points_of_interest_and_queries():
    graph = _loop_graph()
    landmarks = [
        {"text": "EXIT", "texts": ["EXIT", "出口"], "kind": "exit", "level": 0,
         "xyz": [38.0, 10.0, 2.4]},
        {"text": "2号门出口", "kind": "exit", "level": 0,
         "xyz": [2.0, 18.0, 2.4]},
        {"text": "5A", "kind": "zone", "level": 0, "xyz": [20, 20, 2.4]},
    ]
    pois = attributes.points_of_interest(graph, landmarks, 0, 15.0)
    assert [p["kind"] for p in pois] == ["entrance", "exit", "exit"]
    assert all(p["node"] is not None for p in pois)
    assert len(attributes.find_pois(pois, "去出口")) == 2
    named = attributes.find_pois(pois, "2号门出口")
    assert [p["name"] for p in named] == ["2号门出口"]
    assert [p["kind"] for p in attributes.find_pois(pois, "入口")] == [
        "entrance"]
    assert attributes.find_pois(pois, "B1 C区 105") == []
