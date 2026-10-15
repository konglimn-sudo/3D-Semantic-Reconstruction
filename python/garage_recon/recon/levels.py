"""Level (B1/B2/...) and ramp segmentation (section 4.3 / 5 module 1).

The vehicle drives on the floor, so its trajectory is the most robust source
of floor heights: flat stretches give the levels, sloped stretches give the
ramps. Map points then refine a per-level floor height grid (drainage slopes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage

from .. import _core


@dataclass
class FloorGrid:
    x0: float
    y0: float
    res: float
    z: np.ndarray  # (H, W) floor height

    def sample(self, xy: np.ndarray) -> np.ndarray:
        i = np.clip(((xy[:, 0] - self.x0) / self.res).astype(int), 0, self.z.shape[1] - 1)
        j = np.clip(((xy[:, 1] - self.y0) / self.res).astype(int), 0, self.z.shape[0] - 1)
        return self.z[j, i]


@dataclass
class Ramp:
    level_from: int  # upper level
    level_to: int  # lower level
    plane: np.ndarray  # z = a x + b y + c
    polygon: np.ndarray  # (M, 2) xy outline
    slope: float

    def contains(self, xy: np.ndarray) -> np.ndarray:
        import shapely

        return shapely.contains_xy(shapely.Polygon(self.polygon), xy[:, 0], xy[:, 1])

    def z_at(self, xy: np.ndarray) -> np.ndarray:
        return self.plane[0] * xy[:, 0] + self.plane[1] * xy[:, 1] + self.plane[2]


@dataclass
class LevelModel:
    z: np.ndarray  # (L,) floor height per level, level 0 = top
    ceiling: np.ndarray  # (L,) ceiling height per level
    grids: list[FloorGrid] = field(default_factory=list)
    ramps: list[Ramp] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.z)

    def floor_z(self, xy: np.ndarray, level: int) -> np.ndarray:
        if level < len(self.grids) and self.grids[level] is not None:
            return self.grids[level].sample(xy)
        return np.full(len(xy), self.z[level])

    def assign(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Level index, height above that floor and on-ramp flag per point.

        A point belongs to the highest floor that is not above it; points over
        a ramp outline that sit closer to the ramp surface use the ramp instead.
        """
        n = len(xyz)
        level = np.full(n, -1, np.int16)
        height = np.full(n, np.inf, np.float32)
        on_ramp = np.zeros(n, bool)
        for l in range(self.n):
            h = xyz[:, 2] - self.floor_z(xyz[:, :2], l)
            better = (h >= -0.3) & (h < height)
            level[better] = l
            height[better] = h[better]
        for r in self.ramps:
            inside = r.contains(xyz[:, :2])
            h = xyz[:, 2] - r.z_at(xyz[:, :2])
            better = inside & (h >= -0.3) & (h < height)
            level[better] = r.level_from
            height[better] = h[better]
            on_ramp |= better
        height[~np.isfinite(height)] = np.nan
        return level, height, on_ramp

    def to_json(self) -> dict:
        return {
            "levels": [{"id": i, "z": float(z), "ceiling": float(c)} for i, (z, c) in enumerate(zip(self.z, self.ceiling))],
            "ramps": [{"level_from": r.level_from, "level_to": r.level_to, "plane": r.plane.tolist(),
                       "polygon": r.polygon.tolist(), "slope": r.slope} for r in self.ramps],
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "levels.json").write_text(json.dumps(self.to_json(), indent=2))
        np.savez(directory / "floor_grids.npz",
                 **{f"g{i}": g.z for i, g in enumerate(self.grids)},
                 meta=np.array([[g.x0, g.y0, g.res] for g in self.grids]))

    @classmethod
    def load(cls, directory: Path) -> "LevelModel":
        d = json.loads((directory / "levels.json").read_text())
        grids = []
        with np.load(directory / "floor_grids.npz") as z:
            for i, m in enumerate(z["meta"]):
                grids.append(FloorGrid(float(m[0]), float(m[1]), float(m[2]), z[f"g{i}"]))
        ramps = [Ramp(r["level_from"], r["level_to"], np.asarray(r["plane"]), np.asarray(r["polygon"]), r["slope"])
                 for r in d["ramps"]]
        return cls(np.array([l["z"] for l in d["levels"]]), np.array([l["ceiling"] for l in d["levels"]]), grids,
                   ramps)


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for k, v in enumerate(mask):
        if v and start is None:
            start = k
        if not v and start is not None:
            out.append((start, k))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def detect_levels(vehicle_paths: list[np.ndarray], xyz: np.ndarray, normals: np.ndarray, cfg, log=print) -> LevelModel:
    lc = cfg.levels
    flat_z, ramp_segments = [], []
    for P in vehicle_paths:
        p = P[:, :3, 3]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])
        slope = np.zeros(len(p))
        for k in range(len(p)):
            a = np.searchsorted(s, s[k] - 1.0)
            b = min(len(p) - 1, np.searchsorted(s, s[k] + 1.0))
            ds = s[b] - s[a]
            slope[k] = (p[b, 2] - p[a, 2]) / ds if ds > 0.2 else 0.0
        flat = np.abs(slope) < lc.flat_slope
        flat_z.append(p[flat, 2])
        for a, b in _segments(np.abs(slope) >= 2 * lc.flat_slope):
            if s[b - 1] - s[a] > 3.0:
                ramp_segments.append(p[a:b])
    zs = np.sort(np.concatenate(flat_z)) if flat_z else np.zeros(0)
    clusters, cur = [], [zs[0]] if len(zs) else []
    for z in zs[1:]:
        if z - cur[-1] > 0.5 * lc.min_separation:
            clusters.append(cur)
            cur = []
        cur.append(z)
    if cur:
        clusters.append(cur)
    level_z = np.array(sorted([np.median(c) for c in clusters if len(c) >= lc.min_samples], reverse=True))
    if len(level_z) == 0:
        level_z = np.array([float(np.median(zs))]) if len(zs) else np.array([0.0])

    # Ceiling per level from downward-facing surfaces above the floor.
    nz = normals[:, 2]
    ceiling = []
    for i, z in enumerate(level_z):
        top = level_z[i - 1] if i > 0 else z + 6.0
        sel = (nz < -0.85) & (xyz[:, 2] > z + 2.0) & (xyz[:, 2] < top)
        ceiling.append(float(np.percentile(xyz[sel, 2], 90)) if sel.sum() > 50 else z + 3.0)

    ramps = _fit_ramps(ramp_segments, level_z, xyz, nz, log)

    # Floor height grid per level: per-cell median of upward-facing points near
    # the level, excluding ramp surfaces (robust to stray far-range returns).
    up = nz > 0.85
    on_ramp = np.zeros(len(xyz), bool)
    for r in ramps:
        inside = r.contains(xyz[:, :2])
        on_ramp |= inside & (np.abs(xyz[:, 2] - r.z_at(xyz[:, :2])) < 0.1)
    lo, hi = xyz[:, :2].min(axis=0) - 1.0, xyz[:, :2].max(axis=0) + 1.0
    res = float(lc.grid_res)
    W, H = int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res))
    grids = []
    for z in level_z:
        sel = up & ~on_ramp & (np.abs(xyz[:, 2] - z) < lc.floor_band)
        g = _cell_median(xyz[sel, :2], xyz[sel, 2], lo[0], lo[1], res, W, H)
        if np.isfinite(g).any():
            missing = ~np.isfinite(g)
            _, (jj, ii) = ndimage.distance_transform_edt(missing, return_indices=True)
            g = ndimage.median_filter(g[jj, ii], size=3)
        else:
            g = np.full((H, W), z)
        grids.append(FloorGrid(float(lo[0]), float(lo[1]), res, g))
    log(f"[levels] {len(level_z)} levels at z={np.round(level_z, 2).tolist()}, ceilings "
        f"{np.round(ceiling, 2).tolist()}, {len(ramps)} ramps "
        f"{[round(r.slope, 3) for r in ramps]}")
    return LevelModel(level_z, np.asarray(ceiling), grids, ramps)


def _cell_median(xy, z, x0, y0, res, W, H) -> np.ndarray:
    i = ((xy[:, 0] - x0) / res).astype(int)
    j = ((xy[:, 1] - y0) / res).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < W) & (j < H)
    cell, zz = j[ok] * W + i[ok], z[ok]
    out = np.full(H * W, np.nan)
    if len(cell) == 0:
        return out.reshape(H, W)
    order = np.lexsort((zz, cell))
    cell, zz = cell[order], zz[order]
    starts = np.r_[0, np.nonzero(np.diff(cell))[0] + 1]
    ends = np.r_[starts[1:], len(cell)]
    out[cell[starts]] = zz[(starts + ends - 1) // 2]
    return out.reshape(H, W)


def _fit_ramps(segments, level_z, xyz, nz, log) -> list[Ramp]:
    """Plane per driven ramp: fitted on floor points beside the middle part of
    the sloped path (the ends blend into the level floors), outlined by the
    points that fit the plane but are on neither level."""
    from scipy.spatial import ConvexHull, cKDTree

    ramps: list[Ramp] = []
    up = nz > 0.8
    for seg in segments:
        z_top, z_bot = seg[:, 2].max(), seg[:, 2].min()
        lf = int(np.argmin(np.abs(level_z - z_top)))
        lt = int(np.argmin(np.abs(level_z - z_bot)))
        if lf == lt:
            continue
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(seg[:, :2], axis=0), axis=1))])
        core = seg[(s > 0.15 * s[-1]) & (s < 0.85 * s[-1])]
        if len(core) < 3:
            continue
        tree = cKDTree(core[:, :2])
        d, nn = tree.query(xyz[:, :2], distance_upper_bound=3.5)
        near = np.isfinite(d)
        near[near] &= up[near] & (np.abs(xyz[near, 2] - core[nn[near], 2]) < 0.2)
        pts = xyz[near]
        if len(pts) < 50:
            continue
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        plane, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        for _ in range(2):  # trim outliers and refit
            keep = np.abs(A @ plane - pts[:, 2]) < 0.05
            plane, *_ = np.linalg.lstsq(A[keep], pts[keep, 2], rcond=None)
        if any(abs(r.plane - plane).max() < 0.05 and r.level_from == lf for r in ramps):
            continue  # same ramp driven twice
        full = cKDTree(seg[:, :2])
        d2, _ = full.query(xyz[:, :2], distance_upper_bound=4.5)
        zp = plane[0] * xyz[:, 0] + plane[1] * xyz[:, 1] + plane[2]
        on = np.isfinite(d2) & up & (np.abs(xyz[:, 2] - zp) < 0.05)
        on &= (np.abs(xyz[:, 2] - level_z[lf]) > 0.1) & (np.abs(xyz[:, 2] - level_z[lt]) > 0.1)
        if on.sum() < 50:
            continue
        P = xyz[on, :2]
        hull = P[ConvexHull(P).vertices]
        ramps.append(Ramp(lf, lt, plane, hull, float(np.hypot(plane[0], plane[1]))))
    return ramps
