"""Scanner export import: cropping, thinning and drafted levels / columns
on a synthetic dense scan."""

import json

import numpy as np

from garage_recon import geometry
from garage_recon.acceptance import handheld as handheld_lib
from garage_recon.acceptance import scan_import

COLUMNS = [((4.0, 3.0), (0.6, 0.6), 12.0), ((9.0, 5.5), (0.5, 0.8), -20.0)]


def _box_faces(center, size, yaw, z0, z1, spacing, rng):
    """Points on the vertical faces of a box."""
    out = []
    sx, sy = size
    zs = np.arange(z0, z1, spacing)
    for along, half_along, half_across, axis in ((sy, sy / 2, sx / 2, 0),
                                                 (sx, sx / 2, sy / 2, 1)):
        s = np.arange(-half_along, half_along, spacing)
        a, z = np.meshgrid(s, zs)
        for side in (-1.0, 1.0):
            local = np.zeros((a.size, 2))
            local[:, axis] = side * half_across
            local[:, 1 - axis] = a.ravel()
            out.append(np.column_stack([local, z.ravel()]))
    points = np.concatenate(out)
    points[:, :2] = points[:, :2] @ geometry.rot_z(np.radians(yaw))[:2, :2].T
    points[:, :2] += center
    return points + rng.normal(0.0, 0.002, points.shape)


def _scan(rng, spacing=0.02) -> np.ndarray:
    """A 14 x 9 m level (floor z = 0, ceiling 3 m), two columns, a wall, a
    parked car (1.5 m tall) and far-away points outside the crop."""
    xs, ys = np.arange(0, 14, spacing), np.arange(0, 9, spacing)
    gx, gy = np.meshgrid(xs, ys)
    parts = [np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]),
             np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 3.0)])]
    for center, size, yaw in COLUMNS:
        parts.append(_box_faces(center, size, yaw, 0.0, 3.0, spacing, rng))
    parts.append(_box_faces((7.0, 8.8), (12.0, 0.3), 0.0, 0.0, 3.0, spacing,
                            rng))  # Wall.
    parts.append(_box_faces((11.5, 2.0), (1.8, 4.5), 0.0, 0.2, 1.5, spacing,
                            rng))  # Car.
    parts.append(rng.uniform([50, 50, 0], [60, 60, 3], (5000, 3)))
    return np.concatenate(parts)


def test_import_drafts_levels_and_columns(tmp_path):
    rng = np.random.default_rng(0)
    xyz = _scan(rng)
    np.savez(tmp_path / "export.npz", xyz=xyz)
    path = np.column_stack([np.linspace(1, 13, 200), np.full(200, 4.5),
                            np.full(200, 1.4)])
    stamps = np.arange(200) * 0.1
    geometry.write_tum(tmp_path / "path.tum", stamps,
                       np.stack([geometry.make_T(t=p) for p in path]))
    out = scan_import.import_scan(tmp_path / "export.npz", tmp_path / "gt",
                                  voxel=0.03, trajectory=tmp_path / "path.tum",
                                  margin=6.0, log=lambda *_: None)
    truth = handheld_lib.load(out)
    assert truth.xyz.max(axis=0)[0] < 30.0  # Far points cropped.
    info = json.loads((out / "import.json").read_text())
    assert info["points"] < info["points_read"]
    assert len(truth.features["levels"]) == 1
    assert abs(truth.features["levels"][0]["z"]) < 0.01
    columns = truth.features["columns"]
    assert len(columns) == 2  # Not the wall, not the car.
    for (center, size, yaw), column in zip(
            COLUMNS, sorted(columns, key=lambda c: c["center"][0])):
        assert np.allclose(column["center"][:2], center, atol=0.005)
        # Same rectangle up to a quarter turn of the axes.
        if abs(column["yaw_deg"] - yaw) > 45:
            column["size"] = column["size"][::-1]
            column["yaw_deg"] += 90 * np.sign(yaw - column["yaw_deg"])
        assert abs(column["yaw_deg"] - yaw) < 0.5
        assert np.allclose(column["size"], size, atol=0.01)
        assert column["draft"]
        assert 2.5 < column["height"] < 3.2
