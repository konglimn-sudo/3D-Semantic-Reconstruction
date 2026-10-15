"""Turns a scanner export into a handheld ground-truth directory.

Scanner exports are large (a terrestrial laser scanner merge of a garage
level has 1e8 points or more) and cover more than the vehicle drove. The
import
    * reads the export (any format of `handheld.read_cloud`),
    * crops it to the surroundings of a trajectory given in the cloud's
      frame (the scanner's or the vehicle's path) or to a bounding box,
    * thins it on a voxel grid (C++ `voxel_downsample_indices`, in chunks),
    * drafts features.json: levels (floor heights under the trajectory) and
      the rectangular columns found in the cloud. Every drafted feature is
      marked `"draft": true`; an annotator reviews them before the
      acceptance is signed off.

Column drafting works on a horizontal band above the floor: cells (2 cm)
whose vertical extent covers most of the band are structure (parked cars
are lower or have glass, walls are long); compact rectangular blobs are
column candidates; each face is then measured from the points near it
(median offset), which gives the centre, size and yaw to millimetres for a
dense scan. Round columns are skipped (the acceptance measures faces).

Layout written:
    <out>/cloud.npz      xyz (float32, the scan frame), intensity if any
    <out>/features.json  drafted levels / columns (when requested)
    <out>/import.json    source, crop, voxel and counts
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
from scipy import ndimage
from scipy import spatial

from garage_recon import _core
from garage_recon import geometry
from garage_recon.acceptance import handheld as handheld_lib


def voxel_thin(xyz: np.ndarray, voxel: float,
               chunk: int = 20_000_000) -> np.ndarray:
    """Indices of one point per voxel (chunked to bound the memory)."""
    keep = []
    for start in range(0, len(xyz), chunk):
        part = np.ascontiguousarray(xyz[start:start + chunk], np.float64)
        keep.append(start + _core.voxel_downsample_indices(part, voxel))
    keep = np.concatenate(keep) if keep else np.zeros(0, np.int64)
    if len(xyz) > chunk:  # Voxels split between chunks.
        merged = _core.voxel_downsample_indices(
            np.ascontiguousarray(xyz[keep], np.float64), voxel)
        keep = keep[merged]
    return np.sort(keep)


def crop_to_path(xyz: np.ndarray, path: np.ndarray,
                 margin: float) -> np.ndarray:
    """Mask of points within `margin` (horizontal) of a path and between 5 m
    below and 8 m above it."""
    step = max(margin / 4.0, 0.5)
    keep_path = [0]
    for k in range(1, len(path)):
        if np.linalg.norm(path[k] - path[keep_path[-1]]) >= step:
            keep_path.append(k)
    nodes = path[keep_path]
    tree = spatial.cKDTree(nodes[:, :2])
    mask = np.zeros(len(xyz), bool)
    for start in range(0, len(xyz), 5_000_000):
        part = xyz[start:start + 5_000_000]
        dist, nearest = tree.query(part[:, :2], distance_upper_bound=margin)
        hit = np.isfinite(dist)
        dz = np.full(len(part), np.inf)
        dz[hit] = part[hit, 2] - nodes[nearest[hit], 2]
        mask[start:start + len(part)] = hit & (dz > -5.0) & (dz < 8.0)
    return mask


def floor_under_path(xyz: np.ndarray, path: np.ndarray, radius: float = 1.0,
                     min_drop: float = 0.4, max_drop: float = 2.5,
                     stride: int = 10) -> np.ndarray:
    """Floor height under every `stride`-th path sample (NaN if not seen).

    The floor is the densest 2 cm height bin of the points within `radius`
    (horizontal) and `min_drop` .. `max_drop` below the path.
    """
    tree = spatial.cKDTree(xyz[:, :2])
    samples = path[::stride]
    out = np.full(len(samples), np.nan)
    for k, p in enumerate(samples):
        near = tree.query_ball_point(p[:2], radius)
        z = xyz[near, 2]
        z = z[(z < p[2] - min_drop) & (z > p[2] - max_drop)]
        if len(z) < 50:
            continue
        bins = np.floor(z / 0.02)
        values, counts = np.unique(bins, return_counts=True)
        best = values[np.argmax(counts)]
        out[k] = float(np.median(z[np.abs(bins - best) <= 1]))
    return out


def draft_levels(floor_z: np.ndarray, min_separation: float = 2.0,
                 min_samples: int = 20) -> list[float]:
    """Level floor heights from the floor under the path (flat clusters)."""
    z = np.sort(floor_z[np.isfinite(floor_z)])
    if not len(z):
        return []
    clusters, current = [], [z[0]]
    for value in z[1:]:
        if value - current[-1] > 0.15:
            clusters.append(current)
            current = []
        current.append(value)
    clusters.append(current)
    levels = sorted((float(np.median(c)) for c in clusters
                     if len(c) >= min_samples), reverse=True)
    out = []
    for level in levels:  # Ramps make small clusters between the floors.
        if all(abs(level - other) >= min_separation for other in out):
            out.append(level)
    return out


def _structure_mask(points: np.ndarray, floor: float, cell: float,
                    band: tuple[float, float], coverage: float):
    """Raster of cells whose points cover most of the height band.

    A cell counts the 10 cm height bins occupied in it or its 8 neighbours,
    so sparse scans (spacing about the cell size) still form solid faces.

    Returns:
        (mask, origin of cell (0, 0), points in the band) or Nones.
    """
    z0, z1 = floor + band[0], floor + band[1]
    inside = points[(points[:, 2] >= z0) & (points[:, 2] < z1)]
    if not len(inside):
        return None, None, None
    origin = inside[:, :2].min(axis=0) - 2.0 * cell
    ij = np.floor((inside[:, :2] - origin) / cell).astype(np.int64)
    shape = tuple(ij.max(axis=0) + 3)
    num_bins = int(np.ceil((z1 - z0) / 0.1))
    zbin = np.minimum(((inside[:, 2] - z0) / 0.1).astype(np.int64),
                      num_bins - 1)
    occupied = np.unique((ij[:, 0] * shape[1] + ij[:, 1]) * num_bins + zbin)
    cells, bins = occupied // num_bins, occupied % num_bins
    ci, cj = cells // shape[1], cells % shape[1]
    keys = [((ci + di) * shape[1] + (cj + dj)) * num_bins + bins
            for di in (-1, 0, 1) for dj in (-1, 0, 1)]
    dilated = np.unique(np.concatenate(keys)) // num_bins
    counted, count = np.unique(dilated, return_counts=True)
    mask = np.zeros(shape[0] * shape[1], bool)
    mask[counted[count >= coverage * num_bins]] = True
    # The neighbourhood widens every face by one cell on each side.
    mask = ndimage.binary_erosion(mask.reshape(shape))
    return mask, origin, inside


def _measure_rectangle(points: np.ndarray, center: np.ndarray, size,
                       yaw: float, band: float = 0.06,
                       min_points: int = 30):
    """Refines a rectangle (centre, size, yaw) from the points on its faces.

    Returns:
        (centre (2,), size (2,), yaw) or None if a face is missing.
    """
    for _ in range(2):
        cos, sin = np.cos(yaw), np.sin(yaw)
        local = points[:, :2] - center
        a = local[:, 0] * cos + local[:, 1] * sin
        b = -local[:, 0] * sin + local[:, 1] * cos
        offsets, slopes = [], []
        # A yaw error d tilts the faces normal to a by -d (a = c - d b) and
        # those normal to b by +d (b = c + d a).
        for axis, across, half, half_across, sign in (
                (a, b, size[0] / 2, size[1] / 2, -1.0),
                (b, a, size[1] / 2, size[0] / 2, 1.0)):
            faces = []
            for side in (-1.0, 1.0):
                on = ((np.abs(axis - side * half) < band)
                      & (np.abs(across) < half_across - 0.05))
                if on.sum() < min_points:
                    return None
                faces.append(float(np.median(axis[on])))
                slopes.append(sign * np.polyfit(across[on], axis[on], 1)[0])
            offsets.append(faces)
        (a0, a1), (b0, b1) = offsets
        center = center + np.array([cos, sin]) * (a0 + a1) / 2 + \
            np.array([-sin, cos]) * (b0 + b1) / 2
        size = (a1 - a0, b1 - b0)
        yaw += float(np.arctan(np.median(slopes)))
    return center, np.array(size), yaw


def draft_columns(points: np.ndarray, levels: list[float],
                  cell: float = 0.04, band: tuple[float, float] = (0.3, 2.2),
                  coverage: float = 0.85,
                  size_range: tuple[float, float] = (0.25, 1.6),
                  log=print) -> list[dict]:
    """Rectangular columns of every level, measured from their faces."""
    out = []
    for level_id, floor in enumerate(levels):
        mask, origin, inside = _structure_mask(points, floor, cell, band,
                                               coverage)
        if mask is None:
            continue
        mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask))
        labels, count = ndimage.label(mask)
        tree = spatial.cKDTree(inside[:, :2])
        # Everything up to 6 m above the floor: the column top (slab or
        # beam underside) for the height.
        column_zone = points[(points[:, 2] > floor + band[0])
                             & (points[:, 2] < floor + 6.0)]
        zone_tree = spatial.cKDTree(column_zone[:, :2])
        found = 0
        for index, cells in enumerate(ndimage.find_objects(labels), 1):
            component = np.argwhere(labels[cells] == index) + [
                cells[0].start, cells[1].start]
            if len(component) * cell * cell < 0.05:
                continue
            xy = (component + 0.5) * cell + origin
            (cx, cy), (w, h), angle = cv2.minAreaRect(xy.astype(np.float32))
            if not (size_range[0] <= min(w, h) and
                    max(w, h) <= size_range[1]):
                continue
            if len(component) * cell * cell < 0.8 * w * h:  # Not a rectangle.
                continue
            near = inside[tree.query_ball_point([cx, cy],
                                                0.5 * np.hypot(w, h) + 0.2)]
            measured = _measure_rectangle(near, np.array([cx, cy]), (w, h),
                                          np.radians(angle))
            if measured is None:
                continue
            center, size, yaw = measured
            # Normalise: yaw in [-45, 45) deg, size along the rotated axes.
            turns = int(np.round(yaw / (np.pi / 2)))
            yaw -= turns * np.pi / 2
            if turns % 2:
                size = size[::-1]
            # The faces run up to the slab or beam: top of the face shell.
            shell = column_zone[zone_tree.query_ball_point(
                center, 0.5 * float(np.hypot(*size)) + 0.1)]
            cos, sin = np.cos(yaw), np.sin(yaw)
            local = shell[:, :2] - center
            outline = np.maximum(
                np.abs(local[:, 0] * cos + local[:, 1] * sin) - size[0] / 2,
                np.abs(-local[:, 0] * sin + local[:, 1] * cos) - size[1] / 2)
            top = shell[np.abs(outline) < 0.05, 2]
            height = (float(np.clip(np.percentile(top, 99) - floor, 2.0, 6.0))
                      if len(top) else 2.5)
            found += 1
            out.append({"id": f"L{level_id}_C{found:03d}", "level": level_id,
                        "center": [float(center[0]), float(center[1]),
                                   floor + height / 2],
                        "size": [float(size[0]), float(size[1])],
                        "yaw_deg": float(np.degrees(yaw)),
                        "height": height, "draft": True})
        log(f"[import] level {level_id} (floor {floor:.2f} m): {found} "
            f"columns drafted from {count} structure blobs")
    return out


def import_scan(cloud: str | pathlib.Path, out: str | pathlib.Path,
                voxel: float = 0.02, trajectory: str | None = None,
                margin: float = 15.0, bbox: list[float] | None = None,
                draft: bool = True, log=print) -> pathlib.Path:
    """Writes a handheld ground-truth directory from a scanner export.

    Args:
        cloud: The export (PLY / LAS / LAZ / PCD / XYZ / NPZ).
        out: Directory to write.
        voxel: Output spacing (m); 0 keeps every point.
        trajectory: TUM trajectory in the cloud's frame (the scanner's or
            the vehicle's path): crop region and floor heights.
        margin: Crop distance from the trajectory (m).
        bbox: [xmin, ymin, zmin, xmax, ymax, zmax] crop instead.
        draft: Draft features.json (levels need the trajectory).
        log: Progress output.

    Returns:
        The output directory.
    """
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xyz, intensity = handheld_lib.read_cloud(cloud)
    info = {"source": str(cloud), "points_read": int(len(xyz))}
    log(f"[import] {cloud}: {len(xyz):,} points")
    path = None
    if trajectory is not None:
        _, poses = geometry.read_tum(trajectory)
        path = poses[:, :3, 3]
        mask = crop_to_path(xyz, path, margin)
        info.update(trajectory=str(trajectory), margin=margin)
    elif bbox is not None:
        lo, hi = np.asarray(bbox[:3]), np.asarray(bbox[3:])
        mask = np.all((xyz >= lo) & (xyz <= hi), axis=1)
        info["bbox"] = list(map(float, bbox))
    else:
        mask = np.ones(len(xyz), bool)
    xyz = xyz[mask]
    intensity = intensity[mask] if intensity is not None else None
    log(f"[import] cropped to {len(xyz):,} points")
    if voxel > 0:
        keep = voxel_thin(xyz, voxel)
        xyz = xyz[keep]
        intensity = intensity[keep] if intensity is not None else None
        log(f"[import] {voxel * 100:.0f} cm voxels: {len(xyz):,} points")
    arrays = {"xyz": xyz.astype(np.float32)}
    if intensity is not None:
        arrays["intensity"] = intensity.astype(np.float32)
    np.savez(out / "cloud.npz", **arrays)
    info.update(voxel=voxel, points=int(len(xyz)))
    if draft:
        levels = []
        if path is not None:
            floors = floor_under_path(xyz, path)
            levels = draft_levels(floors)
            info["floor_samples"] = int(np.isfinite(floors).sum())
        columns = draft_columns(xyz, levels, log=log) if levels else []
        features = {"levels": [{"id": i, "z": z, "draft": True}
                               for i, z in enumerate(levels)],
                    "columns": columns, "slots": [], "lines": [],
                    "draft": True}
        (out / "features.json").write_text(json.dumps(features, indent=1))
        info.update(levels=levels, columns=len(columns))
        log(f"[import] drafted {len(levels)} levels, {len(columns)} columns "
            f"(review before sign-off)")
    (out / "import.json").write_text(json.dumps(info, indent=1))
    return out
