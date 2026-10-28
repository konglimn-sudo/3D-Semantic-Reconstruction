"""Export of reconstructions to an interactive web viewer.

`export_viewer` writes a static directory that a browser opens over HTTP
(`garage-recon viewer --serve`, or any static file server):

    <out>/index.html        the viewer (three.js; drag to rotate, right-drag
                            or shift-drag to pan, wheel to zoom at the
                            cursor, double-click to focus a point)
    <out>/manifest.json     the datasets
    <out>/<name>/scene.json frame, attribute layout, label table, trips,
                            vectors (slots, markings, columns, walls) and
                            the navigation map (route graph, slot goals)
    <out>/<name>/points.bin positions (int16 x 3, `scale` m around `center`)
                            and per-point uint8 attributes
    <out>/<name>/tracks.bin trajectories (float32 x 3 per pose)
    <out>/<name>/files/     map.ply (all attributes) and trajectories (TUM)

Point attributes: intensity (0-255), trip, semantic label (taxonomy id, 0
without semantics), dynamic (0 static, 1 moving, 2 changed between trips),
sigma (mm, 255 = unknown or larger) and height above the level floor (2 cm
steps, 255 = unknown). The map is voxel-thinned to at most `max_points`.
"""

from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np

from garage_recon import _core
from garage_recon import geometry
from garage_recon.io import export
from garage_recon.recon import static_map
from garage_recon.semantics import taxonomy

_TEMPLATE = pathlib.Path(__file__).with_name("viewer_template.html")
_TRIP_COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
                (145, 30, 180), (70, 240, 240), (240, 50, 230),
                (210, 245, 60)]


def _thin(xyz: np.ndarray, max_points: int) -> np.ndarray:
    """Indices of a voxel thinning to at most `max_points` points."""
    keep = np.arange(len(xyz))
    voxel = 0.02
    while len(keep) > max_points:
        keep = keep[_core.voxel_downsample_indices(
            np.ascontiguousarray(xyz[keep]), voxel)]
        voxel *= 1.15
    return np.sort(keep)


def _vectors(work: pathlib.Path, center: np.ndarray) -> dict:
    """Slots, markings, columns and walls as polylines (local frame)."""
    semantics = work / "semantics"
    levels_path = work / "map" / "levels.json"
    level_z = {}
    if levels_path.exists():
        for level in json.loads(levels_path.read_text())["levels"]:
            level_z[int(level["id"])] = float(level["z"])

    def floor(level: int) -> float:
        return level_z.get(int(level), 0.0)

    def local(points) -> list:
        return np.round(np.asarray(points, float) - center, 3).tolist()

    out = {"slots": [], "markings": [], "columns": [], "walls": []}
    path = semantics / "slots.json"
    if path.exists():
        for slot in json.loads(path.read_text()):
            z = float(slot.get("z", floor(slot["level"]))) + 0.03
            ring = [[x, y, z] for x, y in slot["corners"]]
            out["slots"].append({"id": slot["slot_id"],
                                 "occupied": bool(slot.get("occupied")),
                                 "inferred": bool(slot.get("inferred")),
                                 "points": local(ring + ring[:1])})
    path = semantics / "markings.json"
    if path.exists():
        for line in json.loads(path.read_text()):
            z = floor(line["level"]) + 0.03
            out["markings"].append({
                "cls": line.get("cls", "line"),
                "points": local([[*line["p0"], z], [*line["p1"], z]])})
    path = semantics / "structure.json"
    if path.exists():
        structure = json.loads(path.read_text())
        for column in structure.get("columns", []):
            cx, cy = column["center"][:2]
            sx, sy = column["size"]
            yaw = float(column.get("yaw", 0.0))
            c, s = np.cos(yaw), np.sin(yaw)
            corners = [(cx + a * c - b * s, cy + a * s + b * c)
                       for a, b in ((-sx / 2, -sy / 2), (sx / 2, -sy / 2),
                                    (sx / 2, sy / 2), (-sx / 2, sy / 2))]
            z0 = float(column.get("z0", floor(column["level"])))
            z1 = float(column.get("z1", z0 + 2.5))
            out["columns"].append({
                "id": column.get("id", ""),
                "bottom": local([[x, y, z0] for x, y in corners]),
                "top": local([[x, y, z1] for x, y in corners])})
        for wall in structure.get("walls", []):
            z = floor(wall["level"]) + 0.05
            out["walls"].append(
                {"points": local([[*wall["p0"], z], [*wall["p1"], z]])})
    return out


def _nav(work: pathlib.Path, center: np.ndarray) -> dict | None:
    """Route graph, slot goals and points of interest of the navigation map
    (local frame)."""
    nav = work / "semantics" / "nav"
    if not (nav / "graph.json").exists() or not (nav / "goals.json").exists():
        return None
    graph = json.loads((nav / "graph.json").read_text())
    goals = json.loads((nav / "goals.json").read_text())
    pois = (json.loads((nav / "pois.json").read_text())
            if (nav / "pois.json").exists() else [])
    qa = (json.loads((nav / "qa.json").read_text())
          if (nav / "qa.json").exists() else {})

    def local(points) -> list:
        return np.round(np.asarray(points, float) - center, 3).tolist()

    return {
        "level_names": goals["level_names"],
        "origin": qa.get("origin_node"),
        "nodes": local([n["xyz"] for n in graph["nodes"]]),
        "edges": [{"a": e["a"], "b": e["b"], "kind": e["kind"],
                   "length": e["length"], "one_way": e["one_way"],
                   "limit": min([v for v in (e.get("headroom"),
                                             e.get("posted_height"))
                                 if v is not None], default=None),
                   "points": local(e["points"])} for e in graph["edges"]],
        "pois": [{"kind": p["kind"], "name": p["name"],
                  "texts": p.get("texts", [p["name"]]), "level": p["level"],
                  "node": p["node"], "xyz": local(p["xyz"]),
                  "route_length": p.get("route_length")} for p in pois],
        "goals": [{"id": g["slot_id"], "address": g["address"],
                   "level": g["level"], "zone": g.get("zone"),
                   "number": g.get("number"), "node": g["node"],
                   "park": local(g["park"]["xyz"]),
                   "yaw": g["park"]["yaw_head_in"],
                   "reachable": g.get("reachable", False),
                   "route_length": g.get("route_length"),
                   "occupied": g.get("occupied_at_capture")}
                  for g in goals["goals"]],
    }


def export_dataset(work: str | pathlib.Path, out: str | pathlib.Path,
                   name: str, folder: str, max_points: int = 3_000_000,
                   log=print) -> dict:
    """Writes one dataset of the viewer into out/folder; returns its
    manifest entry."""
    work, out = pathlib.Path(work), pathlib.Path(out)
    directory = out / folder
    (directory / "files").mkdir(parents=True, exist_ok=True)
    point_map = static_map.load_point_map(work / "map")
    n = len(point_map)
    labels = np.zeros(n, np.uint8)
    labels_path = work / "semantics" / "labels.npz"
    if labels_path.exists():
        with np.load(labels_path) as z:
            if len(z["labels"]) == n:
                labels = z["labels"].astype(np.uint8)
    dynamic = np.zeros(n, np.uint8)
    if point_map.dynamic is not None:
        kind = getattr(point_map, "dynamic_kind", None)
        dynamic = (np.asarray(kind, np.uint8) if kind is not None else
                   np.asarray(point_map.dynamic, np.uint8))
        dynamic[~np.asarray(point_map.dynamic, bool)] = 0
    trip = (np.asarray(point_map.trip, np.uint8)
            if getattr(point_map, "trip", None) is not None else
            np.zeros(n, np.uint8))
    sigma = getattr(point_map, "sigma", None)
    sigma_mm = (np.full(n, 255, np.uint8) if sigma is None else np.clip(
        np.nan_to_num(np.asarray(sigma) * 1000.0, nan=255.0), 0, 255
    ).astype(np.uint8))
    height = getattr(point_map, "height", None)
    height_q = (np.full(n, 255, np.uint8) if height is None else np.clip(
        np.nan_to_num(np.asarray(height) / 0.02, nan=255.0), 0, 255
    ).astype(np.uint8))
    intensity = point_map.intensity
    intensity_u8 = (np.zeros(n, np.uint8) if intensity is None else
                    np.clip(np.asarray(intensity, float) * 255.0, 0, 255
                            ).astype(np.uint8))

    # Full-resolution exports for other tools.
    export.write_ply(directory / "files" / "map.ply", point_map.xyz,
                     {"intensity": intensity_u8, "trip": trip,
                      "label": labels, "dynamic": dynamic,
                      "sigma_mm": sigma_mm})

    keep = _thin(point_map.xyz, max_points)
    xyz = point_map.xyz[keep]
    center = np.round(np.median(xyz, axis=0), 2)
    extent = np.abs(xyz - center).max()
    scale = 0.01 if extent < 327.0 else float(np.ceil(extent / 327.67 * 100)
                                              / 100)
    positions = np.round((xyz - center) / scale).astype(np.int16)
    attributes = {"intensity": intensity_u8[keep], "trip": trip[keep],
                  "label": labels[keep], "dynamic": dynamic[keep],
                  "sigma": sigma_mm[keep], "height": height_q[keep]}
    layout, offset = [], 0
    with open(directory / "points.bin", "wb") as f:
        f.write(positions.tobytes())
        layout.append({"name": "position", "type": "int16", "size": 3,
                       "offset": 0})
        offset = positions.nbytes
        for key, values in attributes.items():
            pad = (-offset) % 4  # Typed arrays need aligned offsets.
            f.write(b"\0" * pad)
            offset += pad
            f.write(values.tobytes())
            layout.append({"name": key, "type": "uint8", "size": 1,
                           "offset": offset})
            offset += values.nbytes

    # Trajectories (vehicle origins) of every trip.
    tracks, track_layout, offset = [], [], 0
    trips_dir = work / "trips"
    trip_names = sorted(p.name for p in trips_dir.iterdir()
                        if (p / "poses.tum").exists()) if trips_dir.exists() \
        else []
    for index, trip_name in enumerate(trip_names):
        stamps, poses = geometry.read_tum(trips_dir / trip_name / "poses.tum")
        shutil.copy(trips_dir / trip_name / "poses.tum",
                    directory / "files" / f"{trip_name}.tum")
        path = (poses[:, :3, 3] - center).astype(np.float32)
        tracks.append(path)
        track_layout.append({"trip": index, "name": trip_name,
                             "offset": offset, "count": len(path),
                             "duration_s": round(float(stamps[-1]
                                                       - stamps[0]), 1)})
        offset += path.nbytes
    if tracks:
        (directory / "tracks.bin").write_bytes(
            np.concatenate(tracks).tobytes())

    present = sorted(set(np.unique(attributes["label"]).tolist()))
    scene = {
        "name": name, "source": str(work), "points": int(len(keep)),
        "points_full": int(n), "center": center.tolist(), "scale": scale,
        "layout": layout,
        "bounds": {"min": (xyz.min(axis=0) - center).round(2).tolist(),
                   "max": (xyz.max(axis=0) - center).round(2).tolist()},
        "z_range": np.percentile(xyz[:, 2] - center[2], [1, 99]).round(
            2).tolist(),
        "labels": [{"id": c.id, "name": c.name, "color": list(c.color)}
                   for c in taxonomy.CLASSES if c.id in present],
        "trips": [{"id": t["trip"], "name": t["name"],
                   "color": list(_TRIP_COLORS[t["trip"] % len(_TRIP_COLORS)]),
                   "duration_s": t["duration_s"]} for t in track_layout],
        "tracks": track_layout,
        "vectors": _vectors(work, center),
        "nav": _nav(work, center),
    }
    (directory / "scene.json").write_text(json.dumps(scene))
    log(f"[viewer] {name}: {len(keep):,} of {n:,} points, "
        f"{len(trip_names)} trajectories, "
        f"{sum(len(v) for v in scene['vectors'].values())} vectors")
    return {"name": name, "path": folder, "points": int(len(keep))}


def export_viewer(works: list[str | pathlib.Path], names: list[str],
                  out: str | pathlib.Path, max_points: int = 3_000_000,
                  log=print) -> pathlib.Path:
    """Writes the viewer with one dataset per work directory.

    Raises:
        ValueError: `names` and `works` differ in length.
    """
    if len(names) != len(works):
        raise ValueError("one name per work directory")
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = [export_dataset(w, out, n, f"dataset_{i}", max_points, log)
                for i, (w, n) in enumerate(zip(works, names))]
    (out / "manifest.json").write_text(json.dumps({"datasets": datasets}))
    shutil.copy(_TEMPLATE, out / "index.html")
    log(f"[viewer] wrote {out / 'index.html'}")
    return out
