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


def apply_arrows(graph: graph_lib.NavGraph, arrows: list[dict], cfg,
                 log=print) -> dict:
    """Marks one-way corridors (C++ DecideTraffic); returns a report for
    qa.json."""
    edges = [(e.a, e.b, e.level, e.kind in DRIVABLE_KINDS,
              np.ascontiguousarray(e.points[:, :2], float))
             for e in graph.edges]
    assigned, corridors = _core.decide_traffic(
        edges, [(a["level"], np.asarray(a["center"], float),
                 np.asarray(a["direction"], float)) for a in arrows],
        traffic_config(cfg))
    report = {"arrows": len(arrows), "arrows_assigned": int(assigned),
              "corridors": [], "one_way_edges": 0}
    for corridor in corridors:
        report["corridors"].append({
            "level": corridor["level"], "edges": len(corridor["steps"]),
            "length": round(float(corridor["length"]), 1),
            "arrows": corridor["arrows"], "verdict": corridor["verdict"]})
        if corridor["verdict"] != "one-way":
            continue
        for edge_index, forward in corridor["steps"]:
            edge = graph.edges[edge_index]
            if (corridor["travel"] > 0) != forward:
                _reverse(edge)
            edge.one_way = True
            report["one_way_edges"] += 1
    one_way = [c for c in report["corridors"] if c["verdict"] == "one-way"]
    log(f"[nav] arrows: {report['arrows_assigned']}/{report['arrows']} on "
        f"aisles, {len(report['corridors'])} corridors with arrows, "
        f"{len(one_way)} one-way ({sum(c['length'] for c in one_way):.0f} m)")
    return report
