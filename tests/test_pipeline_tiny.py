"""End-to-end: synthetic tiny garage -> section 4 -> section 5 -> evaluation."""

import copy
import dataclasses
import json
import shutil

import numpy as np

from garage_recon import provenance
from garage_recon.recon import pipeline as recon_pipeline
from garage_recon.recon import static_map


def test_trajectory_and_map(tiny_run):
    report = tiny_run["report"]
    for trajectory in report["trajectory"].values():
        assert trajectory["ate_rmse"] < 0.10
        assert trajectory["ate_aligned_rmse"] < 0.05
    assert report["map_accuracy"]["p90"] < 0.10


def test_structure_slots_and_objects(tiny_run):
    report = tiny_run["report"]
    assert report["columns"]["recall"] >= 0.9
    assert report["columns"]["precision"] >= 0.9
    assert report["columns"]["center_err_mean"] < 0.10
    assert report["slots"]["f1"] >= 0.9
    assert report["slots"]["corner_err_mean"] < 0.15
    assert report["slots"]["entrance_correct"] == 1.0
    for vehicles in report["static_vehicles"].values():
        assert vehicles["f1"] >= 0.8
    for moving in report["moving_objects"].values():
        assert moving["tracked"] == moving["gt"]


def test_pose_set_is_shared_and_current(tiny_run):
    recon = tiny_run["recon"]
    pose_set = recon.pose_set()
    assert provenance.stale_stages(recon.work, pose_set) == []
    entries = provenance.entries(recon.work)
    assert {"recon.map", "semantics.frames", "semantics.language"} <= set(
        entries)
    frames = sorted((recon.work / "semantics" / "frames").rglob("*.json"))
    record = json.loads(next(f for f in frames
                             if f.name != "summary.json").read_text())
    assert record["pose_set_id"] == pose_set.pose_set_id
    assert np.asarray(record["T_world_cam"]).shape == (2, 4, 4)  # Rolling.


def test_imu_bias_and_camera_calibration(tiny_run):
    poses = tiny_run["report"]["poses"]
    for errors in poses["imu_bias"].values():
        assert errors["gyro_error_deg_s"] < 0.02
        assert errors["accel_error_m_s2"] < 0.02
    for camera in poses["calibration"].values():
        # The estimate is at least as good as rig.yaml.
        assert (camera["estimate"]["rotation_deg"]
                <= camera["nominal"]["rotation_deg"] + 0.05)
    for camera in poses["camera_reprojection"].values():
        assert camera["mean_px"] < camera["v1_mean_px"]


def test_language_route(tiny_run):
    language = tiny_run["report"]["language"]
    assert language["points"]["labelled"] > 0.85
    assert language["points"]["accuracy"] > 0.8
    assert language["identities"]["purity"] > 0.9
    assert language["cross_view_consistency"] > 0.8


def test_deliverables_exist(tiny_run):
    work = tiny_run["root"] / "work"
    semantics = work / "semantics"
    assert (semantics / "export" / "labeled_map.ply").exists()
    osm = (semantics / "export" / "lanelet2_map.osm").read_text()
    assert osm.count('v="parking_space"') >= 20
    frames = list((semantics / "frames").rglob("*.json"))
    assert len(frames) > 50
    record = json.loads(next(f for f in frames
                             if f.name != "summary.json").read_text())
    assert {"slots", "markings", "boxes", "consistency_score"} <= record.keys()
    occupancy = list((semantics / "occupancy").rglob("*.npz"))
    assert occupancy
    with np.load(occupancy[0]) as z:
        assert (z["semantics"].shape == z["mask_lidar"].shape
                == z["mask_camera"].shape)


def test_stages_are_rerunnable(tiny_run):
    semantics = tiny_run["sem"]
    before = (semantics.out / "slots.json").read_text()
    static = json.loads(
        (semantics.out / "objects" / "static.json").read_text())["trip_00"]
    semantics.run(["slots", "objects"])
    assert (semantics.out / "slots.json").read_text() == before
    after = json.loads(
        (semantics.out / "objects" / "static.json").read_text())["trip_00"]
    assert len(after) == len(static)


def test_map_built_in_tiles_is_the_map_built_in_one_piece(tiny_run, tmp_path):
    recon = tiny_run["recon"]
    work = tmp_path / "work"
    shutil.copytree(recon.work, work,
                    ignore=shutil.ignore_patterns("semantics", "map", "bev"))
    cfg = copy.deepcopy(tiny_run["cfg"])
    cfg.map.tile_size = 8.0
    tiled = recon_pipeline.ReconPipeline(
        [d.root for d in recon.datasets], work, cfg, lambda *a, **k: None,
        recon.models)
    tiled.stage_map()
    meta = json.loads((work / "map" / "tiles.json").read_text())
    assert len(meta["tiles"]) >= 6
    assert not (work / "map" / "point_map.npz").exists()
    whole = recon.point_map()
    pieces = tiled.point_map()
    assert sum(t["points"] for t in meta["tiles"]) == len(whole)
    for field in dataclasses.fields(static_map.PointMap):
        a, b = getattr(whole, field.name), getattr(pieces, field.name)
        if a is None:
            assert b is None, field.name
            continue
        assert a.dtype == b.dtype, field.name
        np.testing.assert_array_equal(a, b, err_msg=field.name)
    assert ((work / "map" / "levels.json").read_text()
            == (recon.work / "map" / "levels.json").read_text())
    # A region loads only the tiles that touch it.
    x0, y0 = whole.xyz[:, :2].min(axis=0)
    part = tiled.point_map(bounds=(x0, y0, x0 + 1.0, y0 + 1.0))
    assert 0 < len(part) < len(whole)
