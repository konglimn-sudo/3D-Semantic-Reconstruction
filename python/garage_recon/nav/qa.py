"""Checks of the navigation map.

For every slot the route from the origin is planned on the graph and its
aisle and connector parts are sampled against the layers: a route must not
touch a structure cell and should keep half the vehicle width from
obstacles. Slots the graph cannot reach are re-planned with grid A* (C++) on
the drivable cells that keep that clearance: a grid path means the graph
has a gap, no path means the slot is cut off in the map itself. The graph's
completeness is measured against where the vehicle actually drove.
"""

from __future__ import annotations

import numpy as np

from garage_recon import _core


def _route_samples(graph, nodes, edges, step: float):
    """(xy, level) samples of the aisle and connector edges of a route."""
    out = []
    for k, e in enumerate(edges):
        edge = graph.edges[e]
        if edge.kind in ("ramp", "bridge"):
            continue
        points = edge.points if edge.a == nodes[k] else edge.points[::-1]
        for p, q in zip(points[:-1], points[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(q[:2] - p[:2]) / step)) + 1)
            xy = p[:2] + np.linspace(0, 1, n)[:, None] * (q[:2] - p[:2])
            out.append((xy, edge.level, edge.kind))
    return out


def grid_path_exists(layers, start_xy, goal_xy, half_width: float) -> bool:
    """Grid A* between two points over drivable cells with the clearance."""
    cost = np.where(layers.drivable & (layers.clearance >= half_width), 0.0,
                    -1.0).astype(np.float32)
    col, row, inside = layers.cells(np.asarray([start_xy[:2], goal_xy[:2]]))
    if not inside.all():
        return False
    # Let the end cells through (slot entrances lie at the aisle border).
    for c, r in zip(col, row):
        cost[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = np.maximum(
            cost[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3],
            np.where(layers.drivable[max(r - 2, 0):r + 3,
                                     max(c - 2, 0):c + 3], 0.0, -1.0))
    path = _core.grid_astar(cost, (int(col[0]), int(row[0])),
                            (int(col[1]), int(row[1])))
    return len(path) > 0


def check(graph, layers_by_level: dict, goals: list[dict], origin: int | None,
          paths_by_level: dict, cfg) -> dict:
    """QA report of the navigation map."""
    nav = cfg.nav
    half = 0.5 * nav.vehicle.width
    route_graph = graph.route_graph()
    structure_hits, tight, checked = [], [], 0
    min_clearance = []
    for goal in goals:
        if not goal.get("reachable"):
            continue
        path = route_graph.shortest_path(origin, goal["node"])
        checked += 1
        worst = np.inf
        hit = False
        for xy, level, kind in _route_samples(graph, path["nodes"],
                                              path["edges"],
                                              0.5 * nav.res):
            layers = layers_by_level[level]
            hit |= bool(layers.sample("structure", xy, False).any())
            if kind == "aisle":
                worst = min(worst, float(layers.sample("clearance", xy,
                                                       0.0).min()))
        if hit:
            structure_hits.append(goal["slot_id"])
        if np.isfinite(worst):
            min_clearance.append(worst)
            if worst < half:
                tight.append(goal["slot_id"])
    # One-way aisles can trap a vehicle: the way back out must exist too.
    back = (graph.route_graph(reverse=True).distances(origin)
            if origin is not None else np.full(len(graph.nodes), np.inf))
    no_way_back = [g["slot_id"] for g in goals
                   if g.get("reachable") and g["node"] is not None
                   and not np.isfinite(back[g["node"]])]
    unreachable = [g for g in goals if not g.get("reachable")]
    # Reachable if the one-way aisles could be driven both ways.
    either_way = (graph.route_graph(two_way=True).distances(origin)
                  if origin is not None else np.full(len(graph.nodes), np.inf))
    one_way = [g for g in unreachable if g["node"] is not None
               and np.isfinite(either_way[g["node"]])]
    unreachable = [g for g in unreachable if g not in one_way]
    gaps, cut_off = [], []
    origin_xyz = graph.nodes[origin] if origin is not None else None
    origin_level = graph.node_level[origin] if origin is not None else None
    for goal in unreachable:
        layers = layers_by_level.get(goal["level"])
        if (layers is None or origin_xyz is None
                or goal["level"] != origin_level):
            continue
        start = np.asarray(goal["entrance"][:2])
        if grid_path_exists(layers, origin_xyz, start,
                            half + nav.vehicle.margin):
            gaps.append(goal["slot_id"])
        else:
            cut_off.append(goal["slot_id"])
    coverage = {}
    for level, path in paths_by_level.items():
        if level not in layers_by_level or not len(path):
            continue
        samples = path[::max(1, len(path) // 2000), :2]
        best = np.full(len(samples), np.inf)
        for edge in graph.edges:
            if edge.level != level or edge.kind == "connector":
                continue
            p = edge.points[:, :2]
            for a, b in zip(p[:-1], p[1:]):
                ab = b - a
                t = np.clip((samples - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
                best = np.minimum(best, np.linalg.norm(
                    samples - (a + t[:, None] * ab), axis=1))
        coverage[str(level)] = round(float(np.mean(best < nav.coverage_radius)),
                                     3)
    kinds = {}
    for edge in graph.edges:
        kinds.setdefault(edge.kind, [0, 0.0])
        kinds[edge.kind][0] += 1
        kinds[edge.kind][1] += edge.length
    report = {
        "nodes": len(graph.nodes),
        "edges": {k: {"count": v[0], "length_m": round(v[1], 1)}
                  for k, v in kinds.items()},
        "origin_node": origin,
        "slots": len(goals),
        "reachable": len(goals) - len(unreachable) - len(one_way),
        "unreachable": [g["slot_id"] for g in unreachable + one_way],
        "unreachable_one_way": [g["slot_id"] for g in one_way],
        "unreachable_graph_gap": gaps,
        "unreachable_cut_off": cut_off,
        "no_connector": [g["slot_id"] for g in goals if g["node"] is None],
        "slots_driven_through": [g["slot_id"] for g in goals
                                 if g.get("driven_through")],
        "blocked_slots": [g["slot_id"] for g in goals
                          if g["blocked_cells"] > nav.blocked_cells],
        "bridges": [{"edge": k, "width_m": round(e.width, 2),
                     "length_m": round(e.length, 1),
                     "xy": np.round(e.points[len(e.points) // 2, :2],
                                    1).tolist()}
                    for k, e in enumerate(graph.edges) if e.kind == "bridge"],
        "routes_checked": checked,
        "routes_touching_structure": structure_hits,
        "routes_below_half_width": tight,
        "route_min_clearance_m": (round(float(np.min(min_clearance)), 3)
                                  if min_clearance else None),
        "driven_path_on_graph": coverage,
        "numbered_slots": int(sum(1 for g in goals if g.get("number"))),
        "one_way_edges": sum(e.one_way for e in graph.edges),
        "one_way_length_m": round(sum(e.length for e in graph.edges
                                      if e.one_way), 1),
        "no_way_back_to_origin": no_way_back,
    }
    return report
