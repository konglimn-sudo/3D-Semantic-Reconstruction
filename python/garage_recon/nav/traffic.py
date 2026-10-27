"""Traffic direction of the aisles from the floor arrows (markings stage).

An arrow belongs to the nearest aisle (or lane) edge of its level that runs
along it. Edges that continue each other without a junction form a
corridor (the stretch between two intersections; slot connectors do not
count as junctions). Per corridor:

    arrows in both directions                 two-way;
    at least `min_arrows` in one direction,   one-way in that direction
    all near the centre line (within          (an arrow per lane sits on its
    `center_tolerance`), and no opposite      lane, off the centre line);
    arrow within `opposite_radius` of them
    otherwise, or no arrows                   two-way (unknown).

A single centred arrow (or none) is completed by other evidence: a no-entry
sign (禁止驶入) at a corridor end, or a one-way sign (单行) with the arrow
or with the survey drives, or the arrow with `min_passes` drives along it
and none against (gr/nav/traffic.h). Drives alone never make a corridor
one-way.

The skeleton wiggles at junctions, where a lane's arrow can fall on it; the
count and the opposite arrows nearby keep such stretches two-way.

One-way edges are turned to run in the direction of travel (edge a -> b).
The rules are C++ (gr/nav/traffic.h); this module edits the graph.
"""

from __future__ import annotations

import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon.nav import graph as graph_lib

DRIVABLE_KINDS = ("aisle", "lane")


def _reverse(edge: graph_lib.Edge) -> None:
    edge.a, edge.b = edge.b, edge.a
    edge.points = edge.points[::-1].copy()


def traffic_config(cfg) -> _core.TrafficConfig:
    return core_config.fill(_core.TrafficConfig(), cfg.nav.get("traffic"))


SIGN_KINDS = ("no_entry", "one_way")


def apply_arrows(graph: graph_lib.NavGraph, arrows: list[dict], cfg,
                 log=print, drives=(), landmarks=()) -> dict:
    """Marks one-way corridors (C++ DecideTraffic) from the floor arrows,
    the survey drives ([(level, (N, 2) xy in time order)]) and the no-entry
    / one-way signs among the text landmarks; returns a report for
    qa.json."""
    edges = [(e.a, e.b, e.level, e.kind in DRIVABLE_KINDS,
              np.ascontiguousarray(e.points[:, :2], float))
             for e in graph.edges]
    signs = [(int(m["level"]), np.asarray(m["xyz"][:2], float), m["kind"],
              None if m.get("facing") is None
              else np.asarray(m["facing"], float))
             for m in landmarks
             if m.get("kind") in SIGN_KINDS and m.get("level") is not None]
    assigned, corridors = _core.decide_traffic(
        edges, [(a["level"], np.asarray(a["center"], float),
                 np.asarray(a["direction"], float)) for a in arrows],
        traffic_config(cfg),
        [(int(level), np.ascontiguousarray(xy[:, :2], float))
         for level, xy in drives if len(xy) >= 2], signs)
    report = {"arrows": len(arrows), "arrows_assigned": int(assigned),
              "signs": len(signs), "corridors": [], "one_way_edges": 0}
    for corridor in corridors:
        report["corridors"].append({
            "level": corridor["level"], "edges": len(corridor["steps"]),
            "length": round(float(corridor["length"]), 1),
            "arrows": corridor["arrows"],
            "passes": [corridor["passes_along"], corridor["passes_against"]],
            "signs": {"no_entry": corridor["no_entry_signs"],
                      "one_way": corridor["one_way_signs"]},
            "verdict": corridor["verdict"]})
        if not corridor["verdict"].startswith("one-way"):
            continue
        for edge_index, forward in corridor["steps"]:
            edge = graph.edges[edge_index]
            if (corridor["travel"] > 0) != forward:
                _reverse(edge)
            edge.one_way = True
            report["one_way_edges"] += 1
    one_way = [c for c in report["corridors"]
               if c["verdict"].startswith("one-way")]
    log(f"[nav] arrows: {report['arrows_assigned']}/{report['arrows']} on "
        f"aisles, {report['signs']} one-way / no-entry signs, "
        f"{len(report['corridors'])} corridors with arrows or signs, "
        f"{len(one_way)} one-way ({sum(c['length'] for c in one_way):.0f} m)")
    return report
