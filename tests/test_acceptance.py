"""Acceptance against a handheld ground truth: readers, registration, column
check points, the C++ comparison bindings and the end-to-end report."""

import json

import numpy as np
import pytest

from garage_recon import _core
from garage_recon import geometry
from garage_recon.acceptance import handheld as handheld_lib
from garage_recon.acceptance import registration
from garage_recon.acceptance import report as report_lib
from garage_recon.data.synthetic import handheld as synthetic_handheld


def _room(spacing: float = 0.05) -> np.ndarray:
    """Points on a 12 x 8 x 3 m room with two columns (z up)."""
    rng = np.random.default_rng(0)
    out = []
    xs, ys, zs = (np.arange(0, 12, spacing), np.arange(0, 8, spacing),
                  np.arange(0, 3, spacing))
    gx, gy = np.meshgrid(xs, ys)
    for z in (0.0, 3.0):  # Floor, ceiling.
        out.append(np.column_stack([gx.ravel(), gy.ravel(),
                                    np.full(gx.size, z)]))
    for x in (0.0, 12.0):
        a, b = np.meshgrid(ys, zs)
        out.append(np.column_stack([np.full(a.size, x), a.ravel(), b.ravel()]))
    for y in (0.0, 8.0):
        a, b = np.meshgrid(xs, zs)
        out.append(np.column_stack([a.ravel(), np.full(a.size, y), b.ravel()]))
    for cx, cy in ((4.0, 3.0), (8.5, 5.5)):  # 0.5 m columns.
        s = np.arange(-0.25, 0.25, spacing)
        a, b = np.meshgrid(s, zs)
        for dx, dy, horizontal in ((-0.25, 0, False), (0.25, 0, False),
                                   (0, -0.25, True), (0, 0.25, True)):
            if horizontal:
                points = np.column_stack(
                    [cx + a.ravel(), np.full(a.size, cy + dy), b.ravel()])
            else:
                points = np.column_stack(
                    [np.full(a.size, cx + dx), cy + a.ravel(), b.ravel()])
            out.append(points)
    points = np.concatenate(out)
    return points + rng.normal(0.0, 0.002, points.shape)


def test_cloud_formats_round_trip(tmp_path):
    xyz = np.array([[1.0, 2.0, 3.0], [4.5, -5.25, 6.0], [7.0, 8.0, -9.5]])
    intensity = np.array([0.0, 0.5, 1.0])
    synthetic_handheld.write_ply(tmp_path / "a.ply", xyz, intensity)
    (tmp_path / "b.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\n"
        "property double y\nproperty double z\nend_header\n"
        + "\n".join(" ".join(map(str, p)) for p in xyz) + "\n")
    (tmp_path / "c.pcd").write_text(
        "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\n"
        "DATA ascii\n" + "\n".join(" ".join(map(str, [*p, i]))
                                    for p, i in zip(xyz, intensity)) + "\n")
    np.savetxt(tmp_path / "d.csv", np.column_stack([xyz, intensity]),
               delimiter=",", header="x,y,z,intensity", comments="")
    np.savez(tmp_path / "e.npz", xyz=xyz, intensity=intensity)
    # Binary PCD as written by Leica Cyclone / CloudCompare exports: a
    # capitalised intensity and a 4-byte padding field (COUNT 4).
    record = np.zeros(3, np.dtype([("Intensity", "<f4"), ("rgb", "<f4"),
                                   ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("_", "u1", (4,))]))
    record["Intensity"] = intensity
    record["x"], record["y"], record["z"] = xyz.T
    (tmp_path / "f.pcd").write_bytes(
        b"# .PCD v0.7\nVERSION 0.7\nFIELDS Intensity rgb x y z _\n"
        b"SIZE 4 4 4 4 4 1\nTYPE F F F F F U\nCOUNT 1 1 1 1 1 4\nWIDTH 3\n"
        b"HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 3\nDATA binary\n"
        + record.tobytes())
    for name in ("a.ply", "b.ply", "c.pcd", "d.csv", "e.npz", "f.pcd"):
        points, values = handheld_lib.read_cloud(tmp_path / name)
        assert np.allclose(points, xyz, atol=1e-5), name
        if values is not None:
            scale = 255.0 if name == "a.ply" else 1.0
            assert np.allclose(values / scale, intensity, atol=0.01), name
    with pytest.raises(ValueError):
        handheld_lib.read_cloud(tmp_path / "x.e57")


def test_rigid_fit_recovers_a_transform():
    rng = np.random.default_rng(1)
    source = rng.uniform(-10, 10, (20, 3))
    truth = geometry.make_T(geometry.rot_z(0.4), [3.0, -2.0, 0.5])
    fitted = registration.rigid_fit(source, registration.transform(source,
                                                                   truth))
    assert np.allclose(fitted, truth, atol=1e-9)


def test_columns_are_measured_from_their_faces():
    room = _room()
    shifted = room + np.array([0.012, -0.008, 0.005])  # The "map".
    columns = [{"id": "A", "level": 0, "center": [4.0, 3.0, 1.5],
                "size": [0.5, 0.5], "yaw_deg": 0.0, "height": 3.0},
               {"id": "B", "level": 0, "center": [8.5, 5.5, 1.5],
                "size": [0.5, 0.5], "yaw_deg": 0.0, "height": 3.0}]
    # Normals are not needed: faces come from point positions.
    result = registration.measure_columns(shifted, columns, np.eye(4))
    for m in result:
        assert m.faces == 4
        assert np.allclose(m.measured - m.truth, [0.012, -0.008, 0.005],
                           atol=0.004)
    ids = registration.control_column_ids(
        [dict(c, center=c["center"]) for c in columns], per_level=4)
    assert set(ids) == {"A", "B"}


def test_comparison_bindings():
    room = _room()
    yaw, offset = 0.6, np.array([25.0, -7.0, 1.2])
    truth = registration.transform(room, geometry.make_T(geometry.rot_z(yaw),
                                                         offset))
    coarse = _core.align_gravity_aligned(room, truth)
    assert coarse is not None
    error = np.linalg.inv(geometry.make_T(geometry.rot_z(yaw), offset)) @ \
        np.asarray(coarse["T"])
    assert np.linalg.norm(error[:3, 3]) < 0.6
    signed, nearest, normals = _core.surface_distances(
        room + [0.0, 0.0, 0.01], room, 0.15, 0.3)
    floor = (np.abs(room[:, 2]) < 1e-1) & np.isfinite(signed) & (
        normals[:, 2] > 0.99)
    assert floor.sum() > 1000
    assert abs(np.median(signed[floor]) - 0.01) < 0.002
    assert np.nanmax(nearest) <= 0.3


def test_acceptance_report_on_the_tiny_garage(tiny_run):
    root = tiny_run["root"]
    settings = {"azimuth_step": 1.0, "elevation": (-70.0, 70.0, 2.0),
                "step": 3.0}
    handheld_dir = synthetic_handheld.simulate_handheld(
        root / "data", settings=settings, log=lambda *_: None)
    report = report_lib.run_acceptance(root / "work", handheld_dir,
                                       tiny_run["cfg"], lambda *_: None)
    assert report["registration"]["method"] == "control"
    assert len(report["registration"]["control_ids"]) >= 3
    columns = report["checkpoints"]["columns"]["stats"]
    assert columns["count"] >= 3
    assert columns["rmse_3d"] < 0.05
    assert report["surface"]["all"]["p95"] < 0.05
    assert report["surface"]["sigma"]["within_2_sigma"] > 0.7
    assert report["vectors"]["slots"]["matched"] >= 10
    verdict = {row["criterion"]: row for row in report["verdict"]}
    assert verdict["checkpoint_rmse_3d"]["pass"]
    work = root / "work" / "acceptance"
    assert (work / "report.md").read_text().startswith("# 测量级验收报告")
    assert json.loads((work / "report.json").read_text())["verdict"]
    assert list(work.glob("deviation_level_*.png"))
