"""Lanes along the rows of slots.

The aisle skeleton follows the middle of the free space. Between two rows
that is the lane, but in an open lot (one row facing open ground, a wide
apron) the skeleton wanders through the open space and forms a web, away
from where vehicles drive. Every row of slots therefore proposes a lane
parallel to its entrance line:

    facing rows   rows of the same level whose entrances face each other
                  `min_aisle`..`max_aisle` apart: the lane runs midway;
    single row    `lane_offset` in front of the entrance line;
    walls         in open areas (free ground `wall_open_width` beyond the
                  lane), `wall_offset` from long walls' faces;
    drives        long straight stretches of the survey drives (an open lot
                  without walls).

Lane pieces must keep the clearance of the aisle graph (drivable, outside
slots and ramps, half the vehicle width plus the margin from obstacles).
Where the skeleton already runs along a lane (`skip_followed` of it within
`follow_distance`, e.g. between two rows of a garage) the skeleton stays.
Skeleton edges that run along a lane (within `replace_distance`, at most
`replace_angle` off, over `replace_share` of their length) are dropped; the
lanes are tied into the rest of the graph at their crossings and ends, and
pieces left apart are bridged as before (graph.bridge_components). The lane
geometry is C++ (gr/nav/lanes.h); this module edits the graph.
"""

from __future__ import annotations

import collections

import numpy as np
import shapely

from garage_recon import _core
from garage_recon import core_config
from garage_recon.nav import graph as graph_lib


def lane_config(cfg) -> _core.LaneConfig:
    return core_config.fill(_core.LaneConfig(), cfg.nav.lanes)


def _navigable(layers, cfg) -> np.ndarray:
    """Cells a lane may use: drivable with the vehicle's clearance, off the
    ramps, and outside slots unless the survey vehicle drove there (a slot
    detected on an aisle, or an empty slot crossed)."""
    nav = cfg.nav
    half = 0.5 * nav.vehicle.width + nav.vehicle.margin
    off_slots = layers.slot == 0
    if layers.driven is not None:
        off_slots = off_slots | layers.driven
    return (layers.drivable & off_slots & ~layers.ramp
            & (layers.clearance >= half))


def _merge(graph: graph_lib.NavGraph, keep: int, drop: int) -> None:
    """Rewires the edges of node `drop` to node `keep`."""
    if keep == drop:
        return
    for edge in graph.edges:
        if edge.a == drop:
            edge.a = keep
            edge.points[0] = graph.nodes[keep]
        if edge.b == drop:
            edge.b = keep
            edge.points[-1] = graph.nodes[keep]


def add_level_lanes(graph: graph_lib.NavGraph, layers, slots, cfg,
                    walls=(), drives=()) -> dict:
    """Lanes of one level's slot rows and of its walls in open areas into
    its graph (lane geometry: C++ PlanLanes / CoveredByLanes); returns a
    report."""
    lanes_cfg = cfg.nav.lanes
    level = layers.grid.level
    report = {"rows": 0, "walls": 0, "drives": 0, "lanes": 0,
              "lane_length_m": 0.0, "replaced_edges": 0, "links": 0}
    slots = [s for s in slots if s.level == level]
    walls = [w for w in walls if w.level == level]
    drives = [np.ascontiguousarray(xy[:, :2], float)
              for lv, xy in drives if lv == level and len(xy) >= 2]
    if not lanes_cfg.get("enabled", True) or not (slots or walls or drives):
        return report
    config = lane_config(cfg)
    skeleton = [np.ascontiguousarray(e.points[:, :2], float)
                for e in graph.edges
                if e.level == level and e.kind == "aisle"]
    plan = _core.plan_lanes(
        [(s.row, np.asarray(s.corners, float), float(s.z), float(s.width))
         for s in slots],
        np.ascontiguousarray(_navigable(layers, cfg), np.uint8),
        layers.grid.x0, layers.grid.y0, layers.grid.res, skeleton, config,
        [(np.asarray(w.p0, float)[:2], np.asarray(w.p1, float)[:2],
          float(w.thickness)) for w in walls], drives)
    report["rows"] = plan["rows"]
    report["walls"] = plan["walls"]
    report["drives"] = plan["drives"]
    if plan["followed"]:
        report["followed"] = plan["followed"]
    lanes = [np.asarray(l, float) for l in plan["lanes"]]
    if not lanes:
        return report
    # Skeleton edges along a lane give way to it; the branches they carried
    # are re-attached to the lanes below.
    candidates = [k for k, e in enumerate(graph.edges)
                  if e.level == level and e.kind == "aisle"]
    covered = _core.covered_by_lanes(
        [np.ascontiguousarray(graph.edges[k].points[:, :2], float)
         for k in candidates], lanes, config)
    drop = {k for k, c in zip(candidates, covered) if c}
    orphans = set()
    for k in drop:
        orphans.update((graph.edges[k].a, graph.edges[k].b))
    report["replaced_edges"] = len(drop)
    graph.edges = [e for k, e in enumerate(graph.edges) if k not in drop]
    for piece in lanes:
        col, row, _ = layers.cells(piece)
        z = layers.floor_z[row, col]
        z = np.where(np.isfinite(z), z, np.nanmedian(layers.floor_z))
        ends = [graph.add_node([*piece[k], z[k]], level) for k in (0, 1)]
        n = max(2, int(np.ceil(np.linalg.norm(piece[1] - piece[0])
                               / layers.grid.res)) + 1)
        xy = piece[0] + np.linspace(0, 1, n)[:, None] * (piece[1] - piece[0])
        width = 2.0 * float(layers.sample("clearance", xy, 0.0).min())
        graph.add_edge(graph_lib.Edge(ends[0], ends[1],
                                      np.column_stack([piece, z]), "lane",
                                      level, width))
        report["lanes"] += 1
        report["lane_length_m"] += float(np.linalg.norm(piece[1] - piece[0]))
    report["links"] = _tie_in(graph, layers, level, cfg, orphans)
    report["pruned_edges"] = _prune_fragments(graph, level,
                                              lanes_cfg.min_fragment)
    report["lane_length_m"] = round(report["lane_length_m"], 1)
    return report


def _prune_fragments(graph: graph_lib.NavGraph, level: int,
                     min_length: float) -> int:
    """Drops the pieces of skeleton the replaced edges left apart: graph
    components of the level without a lane and shorter than `min_length`
    (m). Returns the number of edges dropped."""
    # pylint: disable-next=protected-access
    labels = graph_lib._components(graph)
    length = collections.Counter()
    has_lane = set()
    for edge in graph.edges:
        if edge.level != level:
            continue
        length[labels[edge.a]] += edge.length
        if edge.kind == "lane":
            has_lane.add(labels[edge.a])
    small = {c for c, total in length.items()
             if total < min_length and c not in has_lane}
    keep = [e for e in graph.edges
            if not (e.level == level and labels[e.a] in small)]
    dropped = len(graph.edges) - len(keep)
    graph.edges = keep
    return dropped


def _tie_in(graph: graph_lib.NavGraph, layers, level: int, cfg,
            orphans: set[int]) -> int:
    """Joins the lanes of a level to the rest of its graph: at crossings
    (both edges split, nodes merged), from loose lane ends and from the
    `orphans` (nodes of replaced skeleton edges that other edges still use),
    by straight free links. Returns the number of joins."""
    lanes_cfg = cfg.nav.lanes
    joins = 0
    # Crossings, one at a time (splitting changes the edges).
    changed = True
    while changed:
        changed = False
        lanes = [k for k, e in enumerate(graph.edges)
                 if e.level == level and e.kind == "lane"]
        for k in lanes:
            lane = shapely.LineString(graph.edges[k].points[:, :2])
            for j, other in enumerate(graph.edges):
                if j == k or other.level != level or other.kind not in (
                        "aisle", "lane", "bridge"):
                    continue
                if {other.a, other.b} & {graph.edges[k].a, graph.edges[k].b}:
                    continue
                cross = lane.intersection(
                    shapely.LineString(other.points[:, :2]))
                if cross.is_empty or cross.geom_type != "Point":
                    continue
                point = np.array([cross.x, cross.y])
                s_lane = lane.project(cross)
                s_other = shapely.LineString(other.points[:, :2]).project(
                    cross)
                a = graph.split(k, s_lane)
                b = graph.split(j, s_other)
                graph.nodes[a][:2] = point
                _merge(graph, a, b)
                joins += 1
                changed = True
                break
            if changed:
                break
    degree = collections.Counter()
    for edge in graph.edges:
        degree[edge.a] += 1
        degree[edge.b] += 1
    # Branches of the replaced edges: to the nearest lane.
    reach = lanes_cfg.replace_distance + lanes_cfg.max_link
    for node in sorted(orphans):
        if degree[node] == 0:
            continue
        xyz = graph.nodes[node]
        for near in graph.nearest(xyz[:2], level, kinds=("lane",),
                                  max_distance=reach):
            edge = graph.edges[near.edge]
            if node in (edge.a, edge.b):
                break
            free, clearance = graph_lib.segment_free(layers, xyz, near.point)
            if not free or clearance < 0.5 * cfg.nav.vehicle.width:
                continue
            access = graph.split(near.edge, near.s)
            graph.add_edge(graph_lib.Edge(
                node, access, np.vstack([xyz, graph.nodes[access]]), "lane",
                level, 2.0 * clearance))
            degree[node] += 1
            degree[access] += 1
            joins += 1
            break
    # Loose lane ends: a straight free link to the nearest other edge.
    for k in [k for k, e in enumerate(graph.edges)
              if e.level == level and e.kind == "lane"]:
        for end in (graph.edges[k].a, graph.edges[k].b):
            if degree[end] != 1:
                continue
            xyz = graph.nodes[end]
            for near in graph.nearest(xyz[:2], level,
                                      max_distance=lanes_cfg.max_link):
                edge = graph.edges[near.edge]
                if near.edge == k or end in (edge.a, edge.b):
                    continue
                free, clearance = graph_lib.segment_free(layers, xyz,
                                                         near.point)
                if not free or clearance < 0.5 * cfg.nav.vehicle.width:
                    continue
                access = graph.split(near.edge, near.s)
                target = graph.nodes[access]
                graph.add_edge(graph_lib.Edge(
                    end, access, np.vstack([xyz, target]), "lane", level,
                    2.0 * clearance))
                degree[end] += 1
                degree[access] += 1
                joins += 1
                break
    return joins
