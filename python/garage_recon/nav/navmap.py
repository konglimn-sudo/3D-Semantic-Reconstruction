"""Builds the semantic navigation map of a reconstruction (stage "nav").

Inputs are the semantic results: the point map with labels, levels and
ramps, slots (with OCR numbers when the numbers stage ran) and the driven
paths. Outputs go to work/semantics/nav/ (see export.py) with qa.json.
"""

from __future__ import annotations

import collections
import json

import numpy as np

from garage_recon.nav import address
from garage_recon.nav import export
from garage_recon.nav import goals as goals_lib
from garage_recon.nav import graph as graph_lib
from garage_recon.nav import layers as layers_lib
from garage_recon.nav import qa
from garage_recon.recon import bev


def _origin(recon, levels, cfg) -> tuple[np.ndarray, int]:
    """Route origin: configured [x, y, level], or where the first trip
    starts (the vehicle entered the garage there)."""
    configured = cfg.nav.get("origin")
    if configured:
        x, y, level = configured
        z = float(levels.floor_z(np.array([[x, y]]), int(level))[0])
        return np.array([x, y, z]), int(level)
    pose = recon.vehicle_poses(recon.datasets[0])[1][0]
    xyz = pose[:3, 3].copy()
    # The level whose floor is nearest to the vehicle origin (which sits on
    # the floor; robust to a carried rig that starts on the ground).
    floors = [float(levels.floor_z(xyz[None, :2], k)[0])
              for k in range(levels.n)]
    level = int(np.argmin(np.abs(np.asarray(floors) - xyz[2])))
    xyz[2] = floors[level]
    return xyz, level


def build(semantics, cfg, log=print) -> dict:
    """Writes the navigation map and returns the QA report."""
    recon = semantics.recon
    out = semantics.out / "nav"
    out.mkdir(parents=True, exist_ok=True)
    point_map = semantics.point_map()
    labels = semantics.labels()
    levels = recon.levels()
    slots = (semantics.slots() if (semantics.out / "slots.json").exists()
             else [])
    paths = semantics.paths_by_level()
    layers_by_level = {}
    for level in range(levels.n):
        if not np.any(point_map.level == level):
            continue
        grid = bev.level_grid(point_map, level, cfg.nav.res, cfg.nav.margin)
        layers_by_level[level] = layers_lib.level_layers(
            point_map, labels, levels, slots, paths.get(level), grid, cfg)
    graph = graph_lib.build_graph(layers_by_level, levels, cfg, log)
    origin_xyz, origin_level = _origin(recon, levels, cfg)
    origin = goals_lib.origin_node(graph, layers_by_level, origin_xyz,
                                   origin_level, cfg)
    goals = goals_lib.add_slot_goals(graph, layers_by_level, slots, cfg)
    names = address.level_names(levels.n, cfg)
    path = semantics.out / "landmarks.json"
    landmarks = json.loads(path.read_text()) if path.exists() else []
    address.assign_zones(goals, cfg, landmarks)
    for goal in goals:
        goal["level_name"] = names[goal["level"]]
        goal["address"] = address.address(goal, names)
    goals_lib.add_routes(graph, goals, origin)
    report = qa.check(graph, layers_by_level, goals, origin, paths, cfg)
    report["origin"] = {"xyz": np.round(origin_xyz, 3).tolist(),
                        "level": origin_level}
    report["landmarks"] = {k: sum(l["kind"] == k for l in landmarks)
                           for k in sorted({l["kind"] for l in landmarks})}
    report["zones_from_signs"] = sum(g.get("zone_source") == "sign"
                                     for g in goals)
    # Slots a zone query ("5A区") finds, per level and zone.
    report["zone_slots"] = dict(collections.Counter(
        f"{names[g['level']]}-{g['zone']}" for g in goals if g.get("zone")))
    for level, layers in layers_by_level.items():
        export.write_ros_map(out, layers, f"level_{level}")
        np.savez_compressed(out / f"level_{level}_layers.npz",
                            **layers.to_npz())
        export.write_preview(out / f"level_{level}_preview.png", layers,
                             graph, goals)
    export.write_graph(out, graph)
    export.write_goals(out, goals, names)
    # Exits, gates, zone and level signs: landmarks of the navigation map.
    (out / "landmarks.json").write_text(json.dumps(landmarks, indent=1,
                                                   ensure_ascii=False))
    (out / "qa.json").write_text(json.dumps(report, indent=1,
                                            ensure_ascii=False))
    log(f"[nav] {report['nodes']} nodes, "
        f"{sum(v['count'] for v in report['edges'].values())} edges; "
        f"{report['reachable']}/{report['slots']} slots reachable, "
        f"{len(report['slots_driven_through'])} driven through "
        f"(graph gaps {len(report['unreachable_graph_gap'])}, cut off "
        f"{len(report['unreachable_cut_off'])}); routes touching structure "
        f"{len(report['routes_touching_structure'])}, min clearance "
        f"{report['route_min_clearance_m']} m; driven path on graph "
        f"{report['driven_path_on_graph']}")
    return report


def load_goals(semantics) -> tuple[list[dict], list[str]]:
    data = json.loads((semantics.out / "nav" / "goals.json").read_text())
    return data["goals"], data["level_names"]


def load_graph(semantics) -> graph_lib.NavGraph:
    return graph_lib.NavGraph.from_json(json.loads(
        (semantics.out / "nav" / "graph.json").read_text()))
