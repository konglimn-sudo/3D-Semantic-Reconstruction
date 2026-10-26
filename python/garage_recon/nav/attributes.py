"""Attributes of the route graph and the points of interest.

    headroom        per edge: the lowest static object above the floor
                    within half the vehicle width of the edge (the
                    `headroom` layer), so that a taller vehicle can be routed
                    around low ducts; ramps are skipped (their floor is not
                    the level's).
    posted limits   height-limit signs read by OCR ("限高2.2m", "HEIGHT
                    2.1M") set `posted_height` of the nearest edge.
    points of       the route origin (where the capture entered the garage)
    interest        and every exit / gate sign, joined to the nearest
                    drivable edge, so that "去出口" routes to the nearest one.
"""

from __future__ import annotations

import re

import numpy as np

from garage_recon.nav import graph as graph_lib
from garage_recon.nav import traffic

_HEIGHT_SIGN = re.compile(
    r"(?:限高|HEIGHT|LIMIT|H)[:：=]?([0-9](?:\.[0-9]{1,2})?)(?![0-9])(?:M|米)?")


def swept_samples(points: np.ndarray, half_width: float,
                  step: float) -> np.ndarray:
    """(K, 2) samples along a polyline and half_width to either side."""
    out = []
    for p, q in zip(points[:-1, :2], points[1:, :2]):
        length = float(np.linalg.norm(q - p))
        if length < 1e-9:
            continue
        n = max(2, int(np.ceil(length / step)) + 1)
        line = p + np.linspace(0.0, 1.0, n)[:, None] * (q - p)
        normal = np.array([p[1] - q[1], q[0] - p[0]]) / length
        for offset in (-half_width, 0.0, half_width):
            out.append(line + offset * normal)
    return np.vstack(out) if out else points[:, :2].copy()


def annotate_headroom(graph: graph_lib.NavGraph, layers_by_level: dict,
                      cfg) -> None:
    """Sets `headroom` (m, None for open sky) of every edge but ramps."""
    half = 0.5 * cfg.nav.vehicle.width
    for edge in graph.edges:
        layers = layers_by_level.get(edge.level)
        if (edge.kind == "ramp" or layers is None
                or layers.headroom is None):
            continue
        samples = swept_samples(edge.points, half, layers.grid.res)
        value = float(np.min(layers.sample("headroom", samples, np.inf)))
        edge.headroom = round(value, 2) if np.isfinite(value) else None


def height_limit(text: str) -> float | None:
    """The height of a height-limit sign text (m), None if not one."""
    match = _HEIGHT_SIGN.search(re.sub(r"\s+", "", str(text).upper()))
    if match is None:
        return None
    value = float(match.group(1))
    return value if 1.5 <= value <= 5.0 else None


def apply_height_signs(graph: graph_lib.NavGraph, landmarks: list[dict],
                       max_distance: float) -> list[dict]:
    """Posted height limits on the nearest drivable edge of each sign."""
    applied = []
    for landmark in landmarks:
        texts = landmark.get("texts") or [landmark.get("text", "")]
        values = [v for v in map(height_limit, texts) if v is not None]
        if not values or landmark.get("level") is None:
            continue
        near = graph.nearest(np.asarray(landmark["xyz"])[:2],
                             landmark["level"], kinds=traffic.DRIVABLE_KINDS,
                             max_distance=max_distance)
        if not near:
            continue
        edge = graph.edges[near[0].edge]
        value = min(values)
        edge.posted_height = (value if edge.posted_height is None
                              else min(edge.posted_height, value))
        applied.append({"text": texts[0], "height_m": value,
                        "edge": near[0].edge})
    return applied


def points_of_interest(graph: graph_lib.NavGraph, landmarks: list[dict],
                       origin: int | None, max_distance: float
                       ) -> list[dict]:
    """The origin and the exit / gate signs, each with a graph node."""
    pois = []
    if origin is not None:
        pois.append({"kind": "entrance", "name": "入口",
                     "level": graph.node_level[origin],
                     "xyz": np.round(graph.nodes[origin], 3).tolist(),
                     "node": origin})
    for landmark in landmarks:
        if landmark.get("kind") not in ("exit", "gate"):
            continue
        if landmark.get("level") is None:
            continue
        xyz = np.asarray(landmark["xyz"], float)
        near = graph.nearest(xyz[:2], landmark["level"],
                             kinds=traffic.DRIVABLE_KINDS,
                             max_distance=max_distance)
        node = None if not near else graph.split(near[0].edge, near[0].s)
        pois.append({"kind": landmark["kind"], "name": landmark["text"],
                     "texts": landmark.get("texts", [landmark["text"]]),
                     "level": int(landmark["level"]),
                     "xyz": np.round(xyz, 3).tolist(),
                     "access": (None if node is None else
                                np.round(graph.nodes[node], 3).tolist()),
                     "node": node})
    return pois


def find_pois(pois: list[dict], text: str) -> list[dict]:
    """Points of interest a query asks for ("出口", "去2号门出口", "入口",
    "exit"): the named ones first, else every one of the asked kind."""
    key = re.sub(r"\s+|^(去|到|导航到|前往)|^(最近的?)", "",
                 str(text).upper())
    kinds = []
    if re.search(r"出口|EXIT", key):
        kinds.append("exit")
    if re.search(r"门|GATE", key):
        kinds.append("gate")
    if re.search(r"入口|ENTRANCE", key):
        kinds.append("entrance")

    def score(poi: dict) -> int:
        """Length of the longest of its texts the query contains (or that
        contains the query)."""
        best = 0
        for t in poi.get("texts", [poi["name"]]):
            t = re.sub(r"\s+", "", t.upper())
            if len(t) > 1 and (t in key or key in t):
                best = max(best, min(len(t), len(key)))
        return best

    candidates = [p for p in pois if p["node"] is not None
                  and (not kinds or p["kind"] in kinds)]
    scores = [score(p) for p in candidates]
    if scores and max(scores) > 0:
        return [p for p, s in zip(candidates, scores) if s == max(scores)]
    return [p for p in candidates if p["kind"] in kinds]
