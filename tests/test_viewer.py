"""Viewer export of the tiny reconstruction: files, layout and contents."""

import json

import numpy as np

from garage_recon.io import viewer
from garage_recon.recon import static_map


def test_viewer_export(tiny_run, tmp_path):
    work = tiny_run["root"] / "work"
    out = viewer.export_viewer([work], ["tiny"], tmp_path / "viewer",
                               max_points=50_000, log=lambda *_: None)
    assert (out / "index.html").read_text().startswith("<!doctype html>")
    manifest = json.loads((out / "manifest.json").read_text())
    entry = manifest["datasets"][0]
    assert entry["name"] == "tiny" and entry["points"] <= 50_000
    directory = out / entry["path"]
    scene = json.loads((directory / "scene.json").read_text())
    data = (directory / "points.bin").read_bytes()
    n = scene["points"]
    layout = {item["name"]: item for item in scene["layout"]}
    assert set(layout) == {"position", "intensity", "trip", "label",
                           "dynamic", "sigma", "height"}
    for item in scene["layout"]:
        assert item["offset"] % 4 == 0  # Typed array alignment.
        size = 6 * n if item["name"] == "position" else n
        assert item["offset"] + size <= len(data)
    # Positions round-trip to within the quantisation.
    positions = np.frombuffer(data, np.int16, n * 3).reshape(n, 3)
    xyz = positions * scene["scale"] + np.asarray(scene["center"])
    full = static_map.PointMap.load(work / "map" / "point_map.npz").xyz
    lo, hi = full.min(axis=0) - 0.01, full.max(axis=0) + 0.01
    assert np.all((xyz >= lo) & (xyz <= hi))
    labels = np.frombuffer(data, np.uint8, n, layout["label"]["offset"])
    assert {l["id"] for l in scene["labels"]} == set(np.unique(labels))
    assert len(scene["trips"]) == len(scene["tracks"]) >= 1
    tracks = np.frombuffer((directory / "tracks.bin").read_bytes(), np.float32)
    assert len(tracks) == 3 * sum(t["count"] for t in scene["tracks"])
    assert scene["vectors"]["slots"] and scene["vectors"]["columns"]
    assert (directory / "files" / "map.ply").exists()
