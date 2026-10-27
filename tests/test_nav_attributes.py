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


def test_ramp_ends_and_road_features_become_points_of_interest():
    import types
    cfg = config_lib.load_config()
    graph = _loop_graph()
    # A ramp from node 2 (level 0) down to a node on level 1.
    lower = graph.add_node([40.0, 44.0, -3.6], 1)
    graph.add_edge(graph_lib.Edge(
        2, lower, np.array([graph.nodes[2], graph.nodes[lower]]), "ramp", 0,
        6.0))
    ramps = attributes.ramp_pois(graph, ["B1", "B2"])
    assert [p["direction"] for p in ramps] == ["down", "up"]
    assert ramps[0]["name"].startswith("坡道口 B1→B2")
    # A barrier gate cabinet beside the bottom aisle at x = 10 and a speed
    # bump across the right aisle at y = 10.
    cabinet = np.column_stack([np.full(40, 10.0) + np.tile([0, .1], 20),
                               np.repeat(np.linspace(1.2, 1.6, 20), 2),
                               np.full(40, 0.5)])
    bump = np.column_stack([np.repeat(np.linspace(37, 43, 61), 3),
                            np.tile([9.9, 10.0, 10.1], 61), np.full(183, 0.05)])
    xyz = np.vstack([cabinet, bump])
    labels = np.array([19] * len(cabinet) + [10] * len(bump))
    point_map = types.SimpleNamespace(xyz=xyz, level=np.zeros(len(xyz), int),
                                      dynamic=None)
    pois, report = attributes.road_features(graph, point_map, labels, cfg)
    assert report == {"barrier": 1, "speed_bump": 1}
    assert len(pois) == 1 and pois[0]["kind"] == "barrier"
    assert pois[0]["node"] is not None
    assert graph.nodes[pois[0]["node"]][0] == pytest.approx(10.05, abs=0.1)
    assert "barrier" in graph.edges[0].features
    assert any("speed_bump" in e.features for e in graph.edges
               if e.a in (1, 2) and e.b in (1, 2))
    found = attributes.find_pois(ramps + pois, "去最近的坡道")
    assert {p["kind"] for p in found} == {"ramp"}
    assert attributes.find_pois(ramps + pois, "道闸")[0]["kind"] == "barrier"
    restored = graph_lib.NavGraph.from_json(graph.to_json())
    assert restored.edges[0].features == graph.edges[0].features


def test_slot_attributes_from_signs_and_queries():
    from garage_recon.nav import address
    from garage_recon.semantics import numbers
    assert numbers.text_attributes("新能源充电车位") == ["ev"]
    assert numbers.text_attributes("无障碍专用") == ["accessible", "reserved"]
    assert numbers.text_kind("充电车位") == "attribute"

    def goal(slot_id, x, level=0, attributes=()):
        return {"slot_id": slot_id, "level": level, "number": slot_id[-3:],
                "zone": "A", "corners": [[x, 0], [x + 2.5, 0], [x + 2.5, 5],
                                         [x, 5]],
                "attributes": list(attributes)}

    goals = [goal("L0-A-001", 0.0, attributes=["ev"]), goal("L0-A-002", 2.5),
             goal("L0-A-003", 5.0), goal("L1-A-001", 0.0, level=1)]
    signs = [{"kind": "attribute", "text": "无障碍", "texts": ["无障碍"],
              "level": 0, "xyz": [6.0, 6.5, 2.0]},
             {"kind": "attribute", "text": "充电车位", "texts": ["充电车位"],
              "level": 0, "xyz": [50.0, 6.5, 2.0]}]  # Too far from any slot.
    assert attributes.attach_sign_attributes(goals, signs, 3.0) == 1
    assert goals[2]["attributes"] == ["accessible"]
    names = ["B1", "B2"]
    assert [g["slot_id"] for g in address.find(goals, "充电车位", names)] == [
        "L0-A-001"]
    assert [g["slot_id"] for g in address.find(goals, "B1 无障碍车位",
                                               names)] == ["L0-A-003"]
    assert address.find(goals, "B2 充电车位", names) == []
    # Numbers still work.
    assert [g["slot_id"] for g in address.find(goals, "B1 A-002", names)] == [
        "L0-A-002"]


def test_one_arrow_and_drives_or_signs_make_a_corridor_one_way():
    cfg = config_lib.load_config()
    quiet = lambda *_: None  # noqa: E731

    def drive(x0, x1):
        x = np.linspace(x0, x1, 60)
        return (0, np.column_stack([x, np.full(60, 1.0 if x1 < x0 else -1.0)]))

    graph = _loop_graph()
    report = traffic.apply_arrows(graph, [_arrow(10, 0.1, -1)], cfg, quiet,
                                  drives=[drive(39, 1), drive(38, 2)])
    assert report["corridors"][0]["verdict"] == "one-way (driven)"
    assert report["corridors"][0]["passes"] == [0, 4]
    assert report["one_way_edges"] == 2
    # A no-entry sign read at the bottom-right corner, facing +x (its
    # readers come from x > 40 ... and turn in): traffic leaves there.
    graph = _loop_graph()
    sign = {"kind": "no_entry", "text": "禁止驶入", "level": 0,
            "xyz": [37.0, -2.5, 2.0], "facing": [1.0, 0.0]}
    report = traffic.apply_arrows(graph, [], cfg, quiet, landmarks=[sign])
    assert report["signs"] == 1
    verdicts = [c["verdict"] for c in report["corridors"]]
    assert verdicts == ["one-way (sign)"]
    bottom = [e for e in graph.edges if e.one_way]
    assert all(e.points[-1, 0] > e.points[0, 0] for e in bottom)  # +x.


def test_text_landmarks_face_their_readers():
    from garage_recon.semantics import numbers
    sign = np.array([10.0, 0.0, 2.5])
    readings = [("禁止驶入", sign + [0.02 * k, 0, 0], 1.0,
                 np.array([10.0 + k, -8.0, 1.5])) for k in range(-3, 4)]
    marks = numbers.text_landmarks(readings, 3.0, 2, 1.0)
    assert marks[0]["kind"] == "no_entry"
    assert marks[0]["facing"] == pytest.approx([0.0, -1.0], abs=0.05)
    # Without cameras: no facing.
    marks = numbers.text_landmarks([r[:3] for r in readings], 3.0, 2, 1.0)
    assert "facing" not in marks[0]


def test_update_checks_whether_the_vehicle_still_passes():
    from garage_recon.nav import update
    # A 30 m long, 6 m wide aisle (walls at y = 0 and 6), its edge along
    # y = 3; a fence from the south wall to y = 2.4 or across it.
    grid = bev.BevGrid(0, -1.0, -1.0, 0.1, 320, 80)

    def layers_with(fence_top):
        rows, cols = np.mgrid[0:80, 0:320]
        y = grid.y0 + (rows + 0.5) * grid.res
        x = grid.x0 + (cols + 0.5) * grid.res
        structure = (y < 0.0) | (y > 6.0)
        structure |= (x > 14.0) & (x < 14.3) & (y < fence_top)
        clearance = (_core.distance_transform(structure.astype(np.uint8))
                     * grid.res).astype(np.float32)
        z = np.zeros(structure.shape, bool)
        return layers_lib.LevelLayers(grid, ~structure, z, structure, z, z,
                                      z, np.zeros(structure.shape, np.int32),
                                      z, np.zeros(structure.shape),
                                      clearance, None)

    edge = graph_lib.Edge(0, 1, np.array([[1.0, 3.0, 0.0], [29.0, 3.0, 0.0]]),
                          "aisle", 0, 6.0)
    ok, width = update.passable(layers_with(2.4), edge, 1.25, 4.0)
    assert ok and width == pytest.approx(1.75, abs=0.15)
    ok, _ = update.passable(layers_with(6.0), edge, 1.25, 4.0)
    assert not ok
