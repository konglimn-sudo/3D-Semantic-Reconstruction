"""4D object labels (section 5, module 6): per-keyframe detections ->
multi-object tracking -> track-level box refinement.

Static tracks (parked cars) aggregate the points of every observation into a
single box, which is far more complete than any single sweep; moving tracks
get smoothed trajectories, velocity-aligned yaw and an object-frame size.
`ClusterDetector` is the geometric baseline; a learned 3D detector can be
plugged in through `semantics.adapters.Detector3D`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from scipy import ndimage

from .. import _core
from ..geometry import transform
from ..recon.levels import LevelModel
from .structure import rot2
from .taxonomy import BY_ID, STRUCTURE_IDS, cid


@dataclass
class Detection:
    kf: int
    stamp: float
    center: np.ndarray  # (3,) world
    size: np.ndarray  # (3,) length, width, height
    yaw: float
    cls: str
    score: float
    points: np.ndarray = field(repr=False)


@dataclass
class Box3D:
    track_id: int
    cls: str
    center: list[float]
    size: list[float]
    yaw: float
    static: bool
    velocity: list[float]
    num_points: int
    score: float = 1.0

    def corners(self) -> np.ndarray:
        l, w, h = self.size
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        offs = np.array([[sx * l / 2, sy * w / 2, sz * h / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return np.asarray(self.center) + offs @ R.T

    def contains(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        d = pts - np.asarray(self.center)
        lx = c * d[:, 0] - s * d[:, 1]
        ly = s * d[:, 0] + c * d[:, 1]
        half = np.asarray(self.size) / 2 + margin
        return (np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1]) & (np.abs(d[:, 2]) <= half[2])

    def to_json(self) -> dict:
        return asdict(self)

    def to_core(self) -> _core.ObjectBox:
        return _core.ObjectBox(int(self.track_id), cid(self.cls), np.asarray(self.center, float),
                               np.asarray(self.size, float), float(self.yaw), bool(self.static),
                               np.asarray(self.velocity, float), int(self.num_points), float(self.score))

    @classmethod
    def from_core(cls, b: _core.ObjectBox) -> "Box3D":
        return cls(int(b.track_id), BY_ID[int(b.label)].name, [float(v) for v in b.center], [float(v) for v in b.size],
                   float(b.yaw), bool(b.is_static), [float(v) for v in b.velocity], int(b.num_points), float(b.score))


class VoxelLabelIndex:
    """Fast "is this location static structure?" lookup on the labelled map."""

    def __init__(self, xyz: np.ndarray, labels: np.ndarray, voxel: float):
        self.voxel = voxel
        struct = np.isin(labels, list(STRUCTURE_IDS))
        self.keys = np.unique(self._encode(np.floor(xyz[struct] / voxel).astype(np.int64)))

    @staticmethod
    def _encode(v: np.ndarray) -> np.ndarray:
        v = v + (1 << 20)
        return (v[:, 0] << 42) | (v[:, 1] << 21) | v[:, 2]

    def is_structure(self, pts: np.ndarray, dilate: int = 1) -> np.ndarray:
        hit = np.zeros(len(pts), bool)
        if len(self.keys) == 0 or len(pts) == 0:
            return hit
        base = np.floor(pts / self.voxel).astype(np.int64)
        r = range(-dilate, dilate + 1)
        for dx in r:
            for dy in r:
                for dz in r:
                    k = self._encode(base + np.array([dx, dy, dz]))
                    pos = np.clip(np.searchsorted(self.keys, k), 0, len(self.keys) - 1)
                    hit |= self.keys[pos] == k
        return hit


def _box_from_points(xy: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Minimum-area rectangle: centre, yaw of the long side, length, width."""
    rect = cv2.minAreaRect(xy.astype(np.float32))
    pts = cv2.boxPoints(rect)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    long_edge = e1 if l1 >= l2 else e2
    yaw = float(np.arctan2(long_edge[1], long_edge[0]))
    return np.asarray(rect[0], float), yaw, float(max(l1, l2)), float(min(l1, l2))


def classify(size: np.ndarray, partial: bool = True) -> str | None:
    l, w, h = size
    if 2.5 <= l <= 6.5 and h >= 0.9 and (w >= 1.2 or (partial and w >= 0.3)) and w <= 2.8:
        return "vehicle"
    if l <= 1.2 and w <= 1.0 and 1.0 <= h <= 2.1:
        return "pedestrian"
    if 1.2 < l <= 2.5 and w <= 1.0 and 0.8 <= h <= 1.8:
        return "two_wheeler"
    if h >= 0.3 and l >= 0.3:
        return "other"
    return None


class ClusterDetector:
    """Geometric detector: points that are not static structure and lie in the
    object height band, clustered on a BEV grid."""

    def __init__(self, levels: LevelModel, index: VoxelLabelIndex, cfg):
        self.levels = levels
        self.index = index
        self.cfg = cfg.objects

    def detect(self, pts_world: np.ndarray, kf: int, stamp: float, origin: np.ndarray | None = None) -> list[Detection]:
        oc = self.cfg
        if origin is not None:  # far returns carry centimetre-to-decimetre height errors
            pts_world = pts_world[np.linalg.norm(pts_world - origin, axis=1) <= oc.detect_max_range]
        lv, h, _ = self.levels.assign(pts_world)
        keep = (h >= oc.height_range[0]) & (h <= oc.height_range[1])
        keep[keep] &= ~self.index.is_structure(pts_world[keep], int(oc.structure_dilation))
        P, H = pts_world[keep], h[keep]
        if len(P) < oc.min_points:
            return []
        res = float(oc.cluster_res)
        lo = P[:, :2].min(axis=0)
        ij = ((P[:, :2] - lo) / res).astype(int)
        grid = np.zeros(ij.max(axis=0)[::-1] + 1, bool)
        grid[ij[:, 1], ij[:, 0]] = True
        lab, n = ndimage.label(ndimage.binary_closing(grid, iterations=1) | grid, structure=np.ones((3, 3)))
        comp = lab[ij[:, 1], ij[:, 0]]
        dets = []
        for c in range(1, n + 1):
            m = comp == c
            if m.sum() < oc.min_points:
                continue
            cxy, yaw, length, width = _box_from_points(P[m, :2])
            floor = float(np.median(P[m, 2] - H[m]))
            top = float(np.percentile(P[m, 2], 98))
            size = np.array([max(length, 0.1), max(width, 0.1), max(top - floor, 0.1)])
            cls = classify(size)
            if cls is None:
                continue
            center = np.array([cxy[0], cxy[1], floor + size[2] / 2])
            dets.append(Detection(kf, stamp, center, size, yaw, cls, min(1.0, m.sum() / 200.0), P[m]))
        return dets


@dataclass
class Track:
    track_id: int
    dets: list[Detection] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    missed: int = 0

    @property
    def last(self) -> Detection:
        return self.dets[-1]


def track_detections(frames: list[list[Detection]], cfg, start_id: int = 0) -> list[Track]:
    """Greedy-free (Hungarian) association with constant-velocity prediction."""
    oc = cfg.objects
    active: list[Track] = []
    done: list[Track] = []
    next_id = start_id
    for dets in frames:
        if active and dets:
            C = np.full((len(active), len(dets)), 1e6)
            for a, t in enumerate(active):
                dt = dets[0].stamp - t.last.stamp
                pred = t.last.center[:2] + t.velocity * dt
                for b, d in enumerate(dets):
                    if (d.cls in ("pedestrian",)) != (t.last.cls in ("pedestrian",)):
                        continue
                    dist = np.linalg.norm(pred - d.center[:2])
                    if dist <= oc.match_gate:
                        C[a, b] = dist
            rows, cols = linear_sum_assignment(C)
            matched_t, matched_d = set(), set()
            for a, b in zip(rows, cols):
                if C[a, b] >= 1e6:
                    continue
                t, d = active[a], dets[b]
                dt = max(d.stamp - t.last.stamp, 1e-3)
                v = (d.center[:2] - t.last.center[:2]) / dt
                t.velocity = 0.5 * t.velocity + 0.5 * v if len(t.dets) > 1 else v
                t.dets.append(d)
                t.missed = 0
                matched_t.add(a)
                matched_d.add(b)
        else:
            matched_t, matched_d = set(), set()
        still = []
        for a, t in enumerate(active):
            if a not in matched_t:
                t.missed += 1
            (still if t.missed <= oc.max_missed else done).append(t)
        active = still
        for b, d in enumerate(dets):
            if b not in matched_d:
                active.append(Track(next_id, [d]))
                next_id += 1
    done += active
    return stitch_tracks([t for t in done if len(t.dets) >= oc.min_hits], cfg)


def stitch_tracks(tracks: list[Track], cfg, max_gap: float = 3.0) -> list[Track]:
    """Joins fragments of one object (e.g. split by a partial view that was
    classified differently): B starts after A ends, within `max_gap` seconds,
    where A's constant-velocity prediction puts it."""
    tracks = sorted(tracks, key=lambda t: t.dets[0].stamp)
    merged = True
    while merged:
        merged = False
        for a in tracks:
            for b in tracks:
                if a is b:
                    continue
                if b.dets[0].stamp < a.dets[0].stamp:
                    continue
                dt = b.dets[0].stamp - a.last.stamp
                if not (-1.0 <= dt <= max_gap):  # small overlap: duplicate detections of one object
                    continue
                pred = a.last.center[:2] + a.velocity * max(dt, 0.0)
                if np.linalg.norm(pred - b.dets[0].center[:2]) <= cfg.objects.match_gate * 1.5:
                    by_kf = {d.kf: d for d in a.dets}
                    for d in b.dets:  # keep the better-supported detection per keyframe
                        if d.kf not in by_kf or len(d.points) > len(by_kf[d.kf].points):
                            by_kf[d.kf] = d
                    a.dets = sorted(by_kf.values(), key=lambda d: d.stamp)
                    a.velocity = b.velocity
                    tracks.remove(b)
                    merged = True
                    break
            if merged:
                break
    return tracks


def refine_track(t: Track, levels: LevelModel, cfg) -> tuple[bool, dict[int, Box3D]]:
    """Returns (is_static, {keyframe: box})."""
    oc = cfg.objects
    centers = np.array([d.center for d in t.dets])
    disp = np.max(np.linalg.norm(centers[:, :2] - centers[0, :2], axis=1))
    classes = [d.cls for d in t.dets]
    cls = "vehicle" if "vehicle" in classes else max(set(classes), key=classes.count)
    out: dict[int, Box3D] = {}
    if disp < oc.static_displacement:
        P = np.vstack([d.points for d in t.dets])
        cxy, yaw, length, width = _box_from_points(P[:, :2])
        _, h, _ = levels.assign(P)
        floor = float(np.median(P[:, 2] - h))
        top = float(np.percentile(P[:, 2], 99))
        size = [length, width, top - floor]
        box = Box3D(t.track_id, cls, [cxy[0], cxy[1], floor + size[2] / 2], size, yaw, True, [0.0, 0.0, 0.0], len(P),
                    float(np.mean([d.score for d in t.dets])))
        kfs = [d.kf for d in t.dets]
        for k in range(min(kfs), max(kfs) + 1):
            out[k] = box
        return True, out
    stamps = np.array([d.stamp for d in t.dets])
    win = min(5, len(centers) - (1 - len(centers) % 2))
    smooth = centers.copy()
    if win >= 3:
        pad = win // 2
        for a in range(3):
            padded = np.pad(centers[:, a], pad, mode="edge")
            smooth[:, a] = np.convolve(padded, np.ones(win) / win, mode="valid")
    vel = np.gradient(smooth[:, :2], stamps, axis=0) if len(stamps) > 1 else np.zeros((1, 2))
    yaws = np.array([np.arctan2(v[1], v[0]) if np.linalg.norm(v) > 0.3 else d.yaw for v, d in zip(vel, t.dets)])
    local = []
    for d, c, y in zip(t.dets, smooth, yaws):
        cs, sn = np.cos(-y), np.sin(-y)
        q = d.points[:, :2] - c[:2]
        local.append(np.stack([cs * q[:, 0] - sn * q[:, 1], sn * q[:, 0] + cs * q[:, 1]], axis=1))
    L = np.vstack(local)
    length = float(np.percentile(L[:, 0], 98) - np.percentile(L[:, 0], 2))
    width = float(np.percentile(L[:, 1], 98) - np.percentile(L[:, 1], 2))
    height = float(np.median([d.size[2] for d in t.dets]))
    for d, c, y, v in zip(t.dets, smooth, yaws, vel):
        cls_m = "vehicle_moving" if cls == "vehicle" else cls
        out[d.kf] = Box3D(t.track_id, cls_m, [c[0], c[1], c[2]], [length, width, height], float(y), False,
                          [float(v[0]), float(v[1]), 0.0], len(d.points), d.score)
    return False, out


def _grid_box(P: np.ndarray, floor_z: float, yaw: float, path_g: np.ndarray, typical: tuple[float, float],
              track_id: int) -> Box3D:
    """Box of a parked object from its aggregated points, aligned with the
    garage grid. Cars park perpendicular to the aisle, so the length axis is
    taken perpendicular to the local driving direction unless the points say
    otherwise; sides that were never seen (the far end in the bay) are
    completed to the typical size, away from the driven path."""
    to_g = rot2(-yaw)
    g = P[:, :2] @ to_g.T
    lo = np.percentile(g, 1, axis=0)
    hi = np.percentile(g, 99, axis=0)
    ext = hi - lo
    c = (lo + hi) / 2
    near = path_g[np.argmin(np.linalg.norm(path_g - c, axis=1))] if len(path_g) else c - np.array([0.0, 5.0])
    if (max(ext) > 3.0 and min(ext) < 2.6) or len(path_g) < 2:
        la = int(np.argmax(ext))
    else:
        k = int(np.argmin(np.linalg.norm(path_g - c, axis=1)))
        d = path_g[min(len(path_g) - 1, k + 5)] - path_g[max(0, k - 5)]
        la = 1 if abs(d[0]) >= abs(d[1]) else 0  # aisle along x -> cars along y
    for axis, target in ((la, typical[0]), (1 - la, typical[1])):
        if hi[axis] - lo[axis] < target - 0.3:
            if near[axis] < c[axis]:
                hi[axis] = lo[axis] + target
            else:
                lo[axis] = hi[axis] - target
    c = (lo + hi) / 2
    ext = hi - lo
    height = float(np.percentile(P[:, 2], 98) - floor_z)
    cw = c @ rot2(yaw).T
    return Box3D(track_id, "vehicle", [float(cw[0]), float(cw[1]), floor_z + height / 2],
                 [float(ext[la]), float(ext[1 - la]), height], float(yaw + (0.0 if la == 0 else np.pi / 2)), True,
                 [0.0, 0.0, 0.0], len(P))


def _support_grid(floor_xy: np.ndarray, res: float):
    """Predicate: is an xy location over (dilated) observed floor?"""
    if len(floor_xy) == 0:
        return lambda xy: True
    lo = floor_xy[:, :2].min(axis=0) - 2 * res
    ij = ((floor_xy[:, :2] - lo) / res).astype(int)
    g = np.zeros(ij.max(axis=0)[::-1] + 3, bool)
    g[ij[:, 1], ij[:, 0]] = True
    g = ndimage.binary_closing(g, iterations=4) | g
    g = ndimage.binary_fill_holes(g)  # parked cars hide the floor under them

    def inside(xy) -> bool:
        i, j = ((np.asarray(xy) - lo) / res).astype(int)
        return bool(0 <= j < g.shape[0] and 0 <= i < g.shape[1] and g[j, i])

    return inside


def _slot_box(P: np.ndarray, floor_z: float, slot, typical: tuple[float, float], track_id: int) -> Box3D:
    """Box of a car parked in a known slot: length along the slot depth,
    completed towards the slot's back (never beyond it), width centred in the
    slot when only partly seen."""
    e, b = slot.entrance_mid, slot.back_mid
    depth = float(np.linalg.norm(b - e))
    u_ax = (b - e) / max(depth, 1e-9)
    v_ax = np.array([-u_ax[1], u_ax[0]])
    q = P[:, :2] - e
    u, v = q @ u_ax, q @ v_ax
    u0, u1 = np.percentile(u, 1), np.percentile(u, 99)
    v0, v1 = np.percentile(v, 1), np.percentile(v, 99)
    L, W = typical
    if u1 - u0 < L - 0.3:
        u1 = min(u0 + L, depth + 0.2)
        u0 = max(min(u0, u1 - L), -0.5)
    if v1 - v0 < W - 0.3:
        lim = max(slot.width / 2 - W / 2, 0.0)
        vc = float(np.clip((v0 + v1) / 2, -lim, lim))
        v0, v1 = vc - W / 2, vc + W / 2
    c = e + u_ax * (u0 + u1) / 2 + v_ax * (v0 + v1) / 2
    height = float(np.percentile(P[:, 2], 98) - floor_z)
    return Box3D(track_id, "vehicle", [float(c[0]), float(c[1]), floor_z + height / 2],
                 [float(u1 - u0), float(v1 - v0), height], float(np.arctan2(u_ax[1], u_ax[0])), True,
                 [0.0, 0.0, 0.0], len(P))


def static_objects_from_map(pm, labels: np.ndarray, levels: LevelModel, yaw: float, paths_by_level: dict,
                            n_trips: int, cfg, slots=(), log=print) -> tuple[dict[int, list[Box3D]], np.ndarray]:
    """Parked objects from the aggregated map.

    Object points (not structure, in the object height band) are clustered and
    each cluster is judged as a whole, because point-level visibility evidence
    is incomplete (parts of a moving car are never observed again):
      * >= `moving_fraction` of points moved within their session -> the smear
        of a moving object: not parked, all its points become dynamic;
      * mostly changed between sessions -> parked only in the sessions whose
        points it holds;
      * otherwise parked in every session.
    Returns (boxes per trip index, mask of map points re-labelled as moving)."""
    oc = cfg.objects
    h = pm.height
    kind = pm.dynamic_kind if pm.dynamic_kind is not None else pm.dynamic.astype(np.uint8) * 2
    cand = (labels == 0) & (h >= oc.height_range[0]) & (h <= oc.height_range[1]) & (pm.level >= 0)
    out: dict[int, list[Box3D]] = {t: [] for t in range(n_trips)}
    moving_pts = np.zeros(len(pm), bool)
    typical = tuple(oc.typical_vehicle_size[:2])
    next_id = 50000
    floor_pts = pm.xyz[(labels == cid("floor")) | (labels == cid("ramp"))]
    for lv in range(levels.n):
        support = _support_grid(floor_pts[np.abs(floor_pts[:, 2] - levels.z[lv]) < 1.0], 0.5)
        pending: dict = {}  # (slot, trips) -> point indices, so fragments of one car merge
        gidx = np.where(cand & (pm.level == lv))[0]
        if len(gidx) < oc.min_points:
            continue
        P, H = pm.xyz[gidx], h[gidx]
        res = float(oc.cluster_res)
        lo = P[:, :2].min(axis=0)
        ij = ((P[:, :2] - lo) / res).astype(int)
        grid = np.zeros(ij.max(axis=0)[::-1] + 1, bool)
        grid[ij[:, 1], ij[:, 0]] = True
        lab, n = ndimage.label(ndimage.binary_closing(grid, iterations=1) | grid, structure=np.ones((3, 3)))
        comp = lab[ij[:, 1], ij[:, 0]]
        path_g = paths_by_level.get(lv, np.zeros((0, 2))) @ rot2(-yaw).T
        lv_slots = [sl for sl in slots if sl.level == lv]
        in_slot = np.full(len(P), -1)
        for k_s, sl in enumerate(lv_slots):
            hit = shapely.contains_xy(sl.polygon(), P[:, 0], P[:, 1]) & (in_slot < 0)
            in_slot[hit] = k_s
        centers = np.array([np.mean(np.asarray(sl.corners), axis=0) for sl in lv_slots]) if lv_slots else None
        for c in range(1, n + 1):
            sel_c = comp == c
            if sel_c.sum() < oc.min_points:
                continue
            # Cars stand in bays: split clusters spanning several slots, but a
            # car that only spills over a line stays one object. Each part is
            # then judged on its own (moving smear / session change / parked).
            ids, counts = np.unique(in_slot[sel_c & (in_slot >= 0)], return_counts=True)
            keep = ids[counts >= max(oc.min_points, 0.2 * sel_c.sum())]
            if len(keep) == 0:
                parts = [(sel_c, -1)]
            else:
                idx = np.where(sel_c)[0]
                d = np.linalg.norm(P[idx, None, :2] - centers[keep][None], axis=2)
                owner = keep[np.argmin(d, axis=1)]
                parts = []
                for k_s in keep:
                    m = np.zeros(len(P), bool)
                    m[idx[owner == k_s]] = True
                    parts.append((m, int(k_s)))
            for sel, k_s in parts:
                kc = kind[gidx[sel]]
                if len(kc) == 0:
                    continue
                if np.mean(kc == 1) >= oc.moving_fraction:
                    moving_pts[gidx[sel]] = True
                    continue
                trips = None
                if np.mean(kc == 2) >= 0.5:
                    tr = pm.trip[gidx[sel]][kc == 2]
                    vals, cnt = np.unique(tr, return_counts=True)
                    trips = [int(v) for v, k in zip(vals, cnt) if k >= 0.2 * len(tr)]
                key = (k_s if k_s >= 0 else f"c{c}", None if trips is None else tuple(trips))
                pending.setdefault(key, []).append(np.where(sel)[0])
        for (k_s, trips), chunks in pending.items():
            sel = np.concatenate(chunks)
            if len(sel) < oc.parked_min_points:  # a car aggregated over many sweeps has far more
                continue
            Q = P[sel]
            span = np.ptp(Q[:, :2] @ rot2(-yaw).T, axis=0)
            if span.max() > 7.0:
                # The streak of a moving object (handled by tracking) holds
                # moved points; without any, it is a row of touching parked
                # cars or clutter along a wall, and stays static.
                if np.mean(kind[gidx[sel]] == 1) >= oc.streak_min_moved:
                    moving_pts[gidx[sel]] = True
                continue
            if span.max() < oc.parked_min_extent:
                continue
            # Cars have flat surfaces at two heights (bonnet, roof); a sloped
            # surface spread over many heights is structure (e.g. a ramp underside).
            nzs = np.abs(pm.normals[gidx[sel], 2])
            horiz = nzs > 0.8
            if horiz.mean() > 0.3:
                hz = H[sel][horiz]
                if np.percentile(hz, 95) - np.percentile(hz, 5) > oc.max_flat_height_span:
                    continue
            if not support(np.mean(Q[:, :2], axis=0)):
                continue  # not standing on observed floor (behind walls, outside)
            floor = float(np.median(Q[:, 2] - H[sel]))
            if isinstance(k_s, (int, np.integer)):
                box = _slot_box(Q, floor, lv_slots[int(k_s)], typical, next_id)
            else:
                box = _grid_box(Q, floor, yaw, path_g, typical, next_id)
            cls = classify(np.asarray(box.size), partial=False)
            if cls == "vehicle" and box.size[2] > oc.vehicle_max_height:
                cls = "other"  # taller than any car: usually unlabelled structure
            if cls is None or box.size[2] < 0.3 or (cls == "other" and not isinstance(k_s, (int, np.integer))):
                continue  # clutter outside bays is not reported as a parked object
            box.cls = cls
            next_id += 1
            for t in (range(n_trips) if trips is None else trips):
                out[t].append(box)
    log(f"[objects] parked objects per trip: {[len(v) for v in out.values()]}, "
        f"{moving_pts.sum()} map points re-labelled as moving")
    return out, moving_pts


def _nms(dets: list[Detection], radius: float) -> list[Detection]:
    """Keeps the best-supported detection among those closer than `radius`
    (fragments of one partially seen vehicle)."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -len(d.points)):
        if all(np.linalg.norm(d.center[:2] - k.center[:2]) >= radius for k in kept):
            kept.append(d)
    return kept


def run_objects(kfs, kf_poses: np.ndarray, levels: LevelModel, index: VoxelLabelIndex, pm, labels: np.ndarray,
                yaw: float, paths_by_level: dict, cfg, slots=(), log=print):
    """Returns (tracks_json, parked boxes per trip index, moving boxes per keyframe, map points that
    belong to moving objects)."""
    static, moving_pts = static_objects_from_map(pm, labels, levels, yaw, paths_by_level, len(kfs.trip_ids), cfg,
                                                 slots, log)
    det = ClusterDetector(levels, index, cfg)
    stride = int(cfg.objects.frame_stride)
    moving: dict[int, list[Box3D]] = {}
    tracks_json = []
    next_id = 0
    for t in range(len(kfs.trip_ids)):
        for b in static[t]:
            tracks_json.append({"trip": kfs.trip_ids[t], "track_id": b.track_id, "static": True, "cls": b.cls,
                                "frames": [], "box": b.to_json()})
        idx = kfs.indices_of_trip(t)[::stride]
        frames = []
        for k in idx:
            pts, _ = kfs.load(int(k))
            dets = det.detect(transform(kf_poses[k], pts), int(k), float(kfs.stamp[k]), kf_poses[k][:3, 3])
            # Parked objects are already known from the map; clutter is not tracked.
            dets = [d for d in dets if d.cls != "other"
                    and not any(b.contains(d.center[None], 0.5)[0] for b in static[t])]
            frames.append(_nms(dets, float(cfg.objects.nms_radius)))
        tracks = track_detections(frames, cfg, next_id)
        next_id += 10000
        n_moving = 0
        for tr in tracks:
            is_static, per_kf = refine_track(tr, levels, cfg)
            if is_static:
                continue
            n_moving += 1
            for k, b in per_kf.items():
                moving.setdefault(int(k), []).append(b)
            tracks_json.append({"trip": kfs.trip_ids[t], "track_id": tr.track_id, "static": False,
                                "cls": next(iter(per_kf.values())).cls, "frames": sorted(int(k) for k in per_kf),
                                "box": None})
        log(f"[objects] {kfs.trip_ids[t]}: {len(static[t])} parked, {n_moving} moving tracks "
            f"({sum(len(f) for f in frames)} unexplained detections)")
    return tracks_json, static, moving, moving_pts
