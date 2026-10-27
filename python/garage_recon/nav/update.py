"""Navigation map update from a trip's structural changes.

`localize --changes` compares a new trip with the map (C++ ChangeDetector)
and lists the added and removed structure with the drivable edges whose
vehicle corridor it reaches (changes.json). This module applies them to the
navigation map of the reconstruction (command update-map):

    added       burnt into the level's structure layer; the clearance is
                recomputed, and every reached edge is checked with an A*
                search on the cells that keep the vehicle's half width,
                within `band` metres of the edge: if the vehicle still gets
                through, the edge keeps its route and records the narrowest
                width; otherwise it is blocked (routes avoid it, goals
                behind it become unreachable);
    removed     reached edges are flagged for review: their headroom (a
                removed duct) is only measured again by a new
                reconstruction.

Slot goals get their routes again. The previous graph, goals and layers stay
as *.before_update.* files.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np

from garage_recon import _core
from garage_recon.nav import export
from garage_recon.nav import goals as goals_lib
from garage_recon.nav import graph as graph_lib
from garage_recon.nav import navmap


def _burn(layers, lo: list[float], hi: list[float]) -> np.ndarray:
    """Mask of the cells of an axis-aligned box (xy)."""
    g = layers.grid
    mask = np.zeros(layers.structure.shape, bool)
    c0, r0 = (int(np.floor((lo[k] - (g.x0, g.y0)[k]) / g.res)) for k in (0, 1))
    c1, r1 = (int(np.floor((hi[k] - (g.x0, g.y0)[k]) / g.res)) for k in (0, 1))
    c0, c1 = max(c0, 0), min(c1, g.width - 1)
    r0, r1 = max(r0, 0), min(r1, g.height - 1)
    if c0 <= c1 and r0 <= r1:
        mask[r0:r1 + 1, c0:c1 + 1] = True
    return mask


def passable(layers, edge: graph_lib.Edge, half: float,
             band: float) -> tuple[bool, float]:
    """Whether the vehicle (`half` width + margin) gets from one end of the
    edge to the other within `band` m of it, and the clearance of the
    widest way through (m; on the edge line when there is none).

    Each end moves to the nearest cell where the vehicle fits (an end may
    lie next to the obstacle). The widest way is the largest clearance
    threshold that still connects the ends (bisection, A* per step).
    """
    g = layers.grid
    rows, cols = np.mgrid[0:g.height, 0:g.width]
    xy = np.column_stack([g.x0 + (cols.ravel() + 0.5) * g.res,
                          g.y0 + (rows.ravel() + 0.5) * g.res])
    near = np.full(len(xy), np.inf)
    for a, b in zip(edge.points[:-1, :2], edge.points[1:, :2]):
        ab = b - a
        t = np.clip((xy - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
        near = np.minimum(near, np.linalg.norm(xy - (a + t[:, None] * ab),
                                                axis=1))
    band_mask = near.reshape(rows.shape) <= band
    clearance = layers.clearance
    on_line = float(np.min(layers.sample("clearance", edge.points[:, :2],
                                         0.0)))

    def connects(threshold: float) -> bool:
        ok = band_mask & (clearance >= threshold)
        free_rows, free_cols = np.nonzero(ok)
        if not len(free_rows):
            return False
        ends = []
        for p in (edge.points[0, :2], edge.points[-1, :2]):
            col, row, _ = layers.cells(p[None])
            d2 = (free_cols - col[0]) ** 2 + (free_rows - row[0]) ** 2
            k = int(np.argmin(d2))
            if d2[k] * g.res ** 2 > band ** 2:
                return False
            ends.append(np.array([free_cols[k], free_rows[k]]))
        cost = np.where(ok, 1.0, -1.0).astype(np.float32)
        return len(_core.grid_astar(cost, ends[0], ends[1])) > 0

    if not connects(half):
        return False, on_line
    lo, hi = half, float(np.max(np.where(band_mask, clearance, 0.0)))
    while hi - lo > 0.05:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if connects(mid) else (lo, mid)
    return True, lo


def apply_changes(work: pathlib.Path, changes: dict, cfg,
                  log=print) -> dict:
    """Applies changes.json to work/semantics/nav; returns the report
    (also written as nav/updates.json)."""
    nav_dir = pathlib.Path(work) / "semantics" / "nav"
    vehicle = cfg.nav.vehicle
    half = 0.5 * float(vehicle.width) + float(vehicle.margin)
    band = float((cfg.nav.get("update") or {}).get("band", 4.0))
    graph = graph_lib.NavGraph.from_json(json.loads(
        (nav_dir / "graph.json").read_text()))
    goals_data = json.loads((nav_dir / "goals.json").read_text())
    layers_by_level = navmap.load_layers(nav_dir)
    report = {"changes": len(changes["changes"]), "narrowed": [],
              "blocked": [], "review": []}
    touched = set()
    for change in changes["changes"]:
        level = change["level"]
        layers = layers_by_level.get(level)
        if layers is None or not change["edges"]:
            continue
        if change["kind"] == "removed":
            report["review"] += [e for e in change["edges"]
                                 if e not in report["review"]]
            continue
        burnt = _burn(layers, change["min"], change["max"])
        layers.structure = layers.structure | burnt
        distance = _core.distance_transform(burnt.astype(np.uint8))
        layers.clearance = np.minimum(
            layers.clearance, (distance * layers.grid.res).astype(np.float32))
        touched.update((level, e) for e in change["edges"])
    for level, k in sorted(touched):
        edge = graph.edges[k]
        ok, width = passable(layers_by_level[level], edge, half, band)
        edge.width = round(min(edge.width, 2.0 * width), 3)
        if ok:
            report["narrowed"].append({"edge": k, "width_m": edge.width})
        else:
            edge.blocked = True
            report["blocked"].append(k)
    origin = json.loads((nav_dir / "qa.json").read_text()).get("origin_node")
    goals = goals_data["goals"]
    before = sum(g.get("reachable", False) for g in goals)
    goals_lib.add_routes(graph, goals, origin)
    report["reachable_before"] = before
    report["reachable_after"] = sum(g["reachable"] for g in goals)
    for name in ("graph.json", "goals.json"):
        backup = nav_dir / name.replace(".json", ".before_update.json")
        if not backup.exists():
            shutil.copy(nav_dir / name, backup)
    for level, layers in layers_by_level.items():
        path = nav_dir / f"level_{level}_layers.npz"
        backup = nav_dir / f"level_{level}_layers.before_update.npz"
        if not backup.exists():
            shutil.copy(path, backup)
        np.savez_compressed(path, **layers.to_npz())
    export.write_graph(nav_dir, graph)
    (nav_dir / "goals.json").write_text(json.dumps(goals_data,
                                                    ensure_ascii=False))
    (nav_dir / "updates.json").write_text(json.dumps(report, indent=1))
    log(f"[update-map] {report['changes']} changes: "
        f"{len(report['narrowed'])} edges narrowed, "
        f"{len(report['blocked'])} blocked, {len(report['review'])} for "
        f"review (headroom); slots reachable {report['reachable_before']} "
        f"-> {report['reachable_after']}")
    return report
