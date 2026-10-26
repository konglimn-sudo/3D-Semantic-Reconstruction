"""Navigation goals of the parking slots.

Every slot gets:
    park       the final pose: slot centre, heading into the slot (head-in)
               or out of it (back-in);
    approach   where the route leaves the aisle centre line, found along the
               ray from the entrance into the aisle, with the aisle heading;
    node       a graph node just outside the entrance, joined to the approach
               by a connector that avoids walls and columns (parked cars are
               ignored: occupancy changes).
The route length from the origin (where the garage is entered) tells whether
the slot is reachable.
"""

from __future__ import annotations

import numpy as np
import shapely

from garage_recon.nav import graph as graph_lib
from garage_recon.nav import layers as layers_lib


def _wrap(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def slot_frame(slot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(entrance midpoint, centre, unit direction into the slot), xy."""
    corners = np.asarray(slot.corners, float)
    entrance = 0.5 * (corners[0] + corners[1])
    back = 0.5 * (corners[2] + corners[3])
    into = back - entrance
    return entrance, corners.mean(axis=0), into / max(np.linalg.norm(into),
                                                      1e-9)


def add_slot_goals(graph: graph_lib.NavGraph, layers_by_level: dict, slots,
                   cfg) -> list[dict]:
    """Joins every slot to the graph and returns its goal record."""
    nav = cfg.nav
    goals = []
    for index, slot in enumerate(slots):
        entrance, center, into = slot_frame(slot)
        yaw_in = float(np.arctan2(into[1], into[0]))
        layers = layers_by_level.get(slot.level)
        start = np.array([*(entrance - nav.entrance_offset * into), slot.z])
        joined = None if layers is None else graph_lib.connect(
            graph, layers, start, slot.level, nav.max_approach,
            direction=-into, ignore_parked=True)
        blocked = 0
        driven_share = 0.0
        interior = shapely.Polygon(slot.corners).buffer(-nav.slot_inset)
        if layers is not None and not interior.is_empty:
            inside = layers_lib.fill_polygon(
                layers.structure.shape, layers.grid,
                np.asarray(interior.exterior.coords)) > 0
            blocked = int((layers.structure & inside).sum())
            driven_share = float((layers.driven & inside).sum()
                                 / max(inside.sum(), 1))
        goal = {
            "slot_id": slot.slot_id,
            "level": int(slot.level),
            "row": slot.row,
            "number": getattr(slot, "number", None),
            "zone": getattr(slot, "zone", None),
            "park": {"xyz": [round(float(center[0]), 3),
                             round(float(center[1]), 3), round(slot.z, 3)],
                     "yaw_head_in": round(yaw_in, 4),
                     "yaw_back_in": round(_wrap(yaw_in + np.pi), 4)},
            "entrance": [round(float(entrance[0]), 3),
                         round(float(entrance[1]), 3), round(slot.z, 3)],
            "corners": np.round(np.asarray(slot.corners, float), 3).tolist(),
            "width": round(float(slot.width), 3),
            "depth": round(float(slot.depth), 3),
            "occupied_at_capture": slot.occupied,
            "inferred": bool(slot.inferred),
            "blocked_cells": blocked,
            # Driven through during the capture (an empty slot crossed, or
            # a slot detected on an aisle): shown for review.
            "driven_through": driven_share > nav.max_driven_share,
            "node": None, "access_node": None, "approach": None,
            "connector_length": None,
        }
        if joined is not None:
            access = graph.nodes[joined.access_node]
            heading = float(np.arctan2(joined.tangent[1], joined.tangent[0]))
            goal.update({
                "node": joined.node, "access_node": joined.access_node,
                "connector_length": round(joined.length, 3),
                "approach": {"xyz": np.round(access, 3).tolist(),
                             "yaw": [round(heading, 4),
                                     round(_wrap(heading + np.pi), 4)]}})
        goals.append(goal)
    return goals


def origin_node(graph: graph_lib.NavGraph, layers_by_level: dict, xyz,
                level: int, cfg) -> int | None:
    """Graph node of the route origin (e.g. the garage entrance)."""
    layers = layers_by_level.get(level)
    if layers is None:
        return None
    joined = graph_lib.connect(graph, layers, xyz, level,
                               cfg.nav.max_origin_attach)
    return None if joined is None else joined.node


def add_routes(graph: graph_lib.NavGraph, goals: list[dict],
               origin: int | None) -> None:
    """Route length from the origin to every goal (None if unreachable) and
    the lowest height limit along the route (`route_height_limit`, m; None
    if nothing hangs over it): a taller vehicle needs another route."""
    route_graph = graph.route_graph()
    distances = (route_graph.distances(origin) if origin is not None
                 else np.full(len(graph.nodes), np.inf))
    for goal in goals:
        d = distances[goal["node"]] if goal["node"] is not None else np.inf
        goal["reachable"] = bool(np.isfinite(d))
        goal["route_length"] = round(float(d), 2) if np.isfinite(d) else None
        goal["route_height_limit"] = None
        if goal["reachable"]:
            path = route_graph.shortest_path(origin, goal["node"])
            limits = [graph.edges[e].height_limit for e in path["edges"]]
            limits = [v for v in limits if v is not None]
            goal["route_height_limit"] = min(limits) if limits else None
