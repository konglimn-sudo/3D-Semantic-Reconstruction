"""Registration of the vehicle map to the handheld frame (plan section 3.5).

Two datums:
    auto     the best rigid fit of the whole map: a coarse gravity-aligned
             search (C++ `align_gravity_aligned`) and multi-scale point-to-
             plane ICP of the map structure against the handheld cloud.
    control  survey practice: only the control points define the frame (a
             least-squares rigid fit); check points stay out of the fit and
             measure the accuracy. Control points are surveyed points with
             map coordinates (points.csv) or, without them, the control
             columns measured in the map (four per level, at the corners of
             the column grid).

Columns are measured in the map from the points on their faces (centre in
x / y, a face pair or a single face with the known size) and the floor
around them (height): a 3D check point per column that depends only on the
LiDAR geometry, not on the semantic pipeline.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import spatial

from garage_recon import _core
from garage_recon.acceptance import handheld as handheld_lib


@dataclasses.dataclass
class ColumnMeasurement:
    """One column as a 3D check point, in the handheld frame."""

    column_id: str
    level: int
    truth: np.ndarray  # (3,) centre x, y and floor z.
    measured: np.ndarray | None  # (3,) or None if not measurable.
    faces: int  # Faces seen in the map.
    map_xyz: np.ndarray | None = None  # `measured` in the map frame.


@dataclasses.dataclass
class Registration:
    """Map -> handheld transform and how it was obtained."""

    method: str
    gt_from_map: np.ndarray
    coarse: dict | None
    icp: dict | None
    control_ids: list[str]
    control_residuals: list[dict]


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares rigid transform (no scale) mapping source onto target."""
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((target - mean_t).T @ (source - mean_s))
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = mean_t - rotation @ mean_s
    return out


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _face_offset(axis: np.ndarray, across: np.ndarray, valid: np.ndarray,
                 half: float, half_across: float, band: float,
                 min_points: int) -> tuple[float | None, int]:
    """Offset of a column along one axis from its two faces on that axis.

    Returns:
        (mean offset of the seen faces or None, number of faces seen).
    """
    found = []
    for side in (1.0, -1.0):
        on_face = (valid & (np.abs(axis - side * half) < band)
                   & (np.abs(across) < half_across - 0.05))
        if on_face.sum() >= min_points:
            found.append(np.median(axis[on_face]) - side * half)
    return (float(np.mean(found)) if found else None), len(found)


def measure_columns(map_xyz: np.ndarray, columns: list[dict],
                    gt_from_map: np.ndarray, band: float = 0.06,
                    min_points: int = 15) -> list[ColumnMeasurement]:
    """Measures every ground-truth column in the map.

    Args:
        map_xyz: (N, 3) static map points, map frame.
        columns: Ground-truth columns (features.json).
        gt_from_map: Current map -> handheld transform; the faces are
            searched within `band` of their expected position.
        band: Half width of the face band (m).
        min_points: Points needed on a face or on the floor ring.

    Returns:
        One measurement per column; `measured` is None if no face of one of
        the two axes was seen.
    """
    points = transform(map_xyz, gt_from_map)
    tree = spatial.cKDTree(points[:, :2])
    map_from_gt = np.linalg.inv(gt_from_map)
    out = []
    for column in columns:
        center = np.asarray(column["center"], float)
        sx, sy = column["size"]
        height = float(column["height"])
        yaw = np.radians(column.get("yaw_deg", 0.0))
        floor = center[2] - 0.5 * height
        truth = np.array([center[0], center[1], floor])
        near = tree.query_ball_point(center[:2], 0.5 * np.hypot(sx, sy) + 1.3)
        local = points[near] - center
        cos, sin = np.cos(yaw), np.sin(yaw)
        a = local[:, 0] * cos + local[:, 1] * sin
        b = -local[:, 0] * sin + local[:, 1] * cos
        z = points[near, 2]
        inside_height = (z > floor + 0.3) & (z < floor + height - 0.3)
        da, faces_a = _face_offset(a, b, inside_height, sx / 2, sy / 2, band,
                                   min_points)
        db, faces_b = _face_offset(b, a, inside_height, sy / 2, sx / 2, band,
                                   min_points)
        faces = faces_a + faces_b
        if da is None or db is None:
            out.append(ColumnMeasurement(column["id"], int(column["level"]),
                                         truth, None, faces))
            continue
        outline = np.maximum(np.abs(a) - sx / 2, np.abs(b) - sy / 2)
        ring = (outline > 0.3) & (outline < 1.2) & (np.abs(z - floor) < 0.1)
        dz = np.median(z[ring] - floor) if ring.sum() >= min_points else 0.0
        measured = np.array([center[0] + da * cos - db * sin,
                             center[1] + da * sin + db * cos, floor + dz])
        out.append(ColumnMeasurement(
            column["id"], int(column["level"]), truth, measured, faces,
            transform(measured[None, :], map_from_gt)[0]))
    return out


def control_column_ids(columns: list[dict], per_level: int = 4) -> list[str]:
    """Columns at the corners of each level's column grid."""
    ids = []
    for level in sorted({int(c["level"]) for c in columns}):
        members = [c for c in columns if int(c["level"]) == level]
        xy = np.array([c["center"][:2] for c in members])
        chosen = []
        for direction in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            order = np.argsort(-(xy @ np.asarray(direction, float)))
            for k in order:
                if members[k]["id"] not in chosen:
                    chosen.append(members[k]["id"])
                    break
            if len(chosen) == per_level:
                break
        ids += chosen
    return ids


def register(map_xyz: np.ndarray, truth: handheld_lib.HandheldGroundTruth,
             cfg, log=print) -> Registration:
    """Map -> handheld transform.

    Args:
        map_xyz: (N, 3) static structure points of the map.
        truth: The handheld ground truth.
        cfg: The `acceptance` config section.
        log: Progress output.

    Returns:
        The registration.

    Raises:
        RuntimeError: The coarse search failed, or the control datum has
            fewer than three control points.
    """
    coarse_config = _core.CoarseAlignConfig()
    coarse = _core.align_gravity_aligned(map_xyz, truth.xyz, coarse_config)
    if coarse is None:
        raise RuntimeError("coarse registration failed (too little vertical "
                           "structure)")
    log(f"[accept] coarse registration: score {coarse['score']:.2f}, "
        f"ambiguity {coarse['ambiguity']:.2f}")
    icp = _core.icp_multiscale(map_xyz, truth.xyz, coarse["T"],
                               list(cfg.icp_schedule), float(cfg.icp_voxel))
    gt_from_map = np.asarray(icp["T"])
    log(f"[accept] ICP: rmse {100 * icp['rmse']:.2f} cm, fitness "
        f"{icp['fitness']:.2f}")
    coarse_out = {"T": np.asarray(coarse["T"]).tolist(),
                  "score": coarse["score"], "ambiguity": coarse["ambiguity"]}
    icp_out = {"rmse": icp["rmse"], "fitness": icp["fitness"]}
    if cfg.method == "auto":
        return Registration("auto", gt_from_map, coarse_out, icp_out, [], [])
    if cfg.method != "control":
        raise ValueError(f"unknown registration method {cfg.method!r}")

    surveyed = [p for p in truth.points
                if p.role == "control" and p.map_xyz is not None]
    ids = ([p.point_id for p in surveyed] if surveyed else
           control_column_ids(truth.features["columns"],
                              int(cfg.control_columns_per_level)))
    for _ in range(2):  # Re-measure the columns in the refined frame.
        if surveyed:
            source = np.array([p.map_xyz for p in surveyed])
            target = np.array([p.xyz for p in surveyed])
        else:
            measured = [m for m in measure_columns(
                map_xyz, truth.features["columns"], gt_from_map)
                        if m.column_id in ids and m.measured is not None]
            source = np.array([m.map_xyz for m in measured])
            target = np.array([m.truth for m in measured])
            ids = [m.column_id for m in measured]
        if len(source) < 3:
            raise RuntimeError("the control datum needs at least three "
                               "control points")
        gt_from_map = rigid_fit(source, target)
    residuals = transform(source, gt_from_map) - target
    control = [{"id": i, "dx": float(r[0]), "dy": float(r[1]),
                "dz": float(r[2])} for i, r in zip(ids, residuals)]
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    log(f"[accept] control datum: {len(ids)} points, residual RMS "
        f"{100 * rms:.2f} cm")
    return Registration("control", gt_from_map, coarse_out, icp_out, ids,
                        control)
