"""Floor markings from the per-level BEV (section 5, module 3).

Markings are paint, not geometry, so they are extracted in 2D on the floor
orthophoto (fisheye texture) and the LiDAR intensity raster, then vectorised:
fill sparse samples -> white top-hat -> threshold -> skeleton -> junction
split -> line fit -> collinear merge.

`MarkingSegmenter` is the slot for a learned model (SegFormer / Mask2Former
fine-tuned on garage BEV tiles, or SAM 3 prompted with "parking line"); the
classical path below is the default and needs no training data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from ..recon.bev import BevGrid
from . import vector_fit


@dataclass
class MarkingSegment:
    level: int
    p0: list[float]
    p1: list[float]
    width: float
    cls: str  # parking_line | lane_marking
    confidence: float
    inferred: bool = False
    # p0 / p1 measured on the orthophoto with observed floor beyond them
    # (vector_fit); unreliable ends may be hidden by cars or the wall base.
    end_ok: list[bool] = field(default_factory=lambda: [False, False])

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.p1, self.p0)))

    def to_json(self) -> dict:
        d = asdict(self)
        d["length"] = self.length
        return d


class MarkingSegmenter(Protocol):
    def segment(self, image: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Binary marking mask for a BEV image in [0, 1]."""


def _normalize(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    v = img[valid]
    if len(v) == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(v, 2), np.percentile(v, 99.5)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _fill_sparse(img: np.ndarray, valid: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised convolution: interpolates sparse LiDAR samples."""
    w = ndimage.gaussian_filter(valid.astype(float), sigma)
    v = ndimage.gaussian_filter(np.where(valid, img, 0.0), sigma)
    filled = np.where(w > 1e-3, v / np.maximum(w, 1e-6), 0.0)
    return filled, w > 0.15


def marking_layers(rasters: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, brightness in [0, 1], valid mask) for every available source."""
    layers = []
    inten = rasters.get("intensity")
    if inten is not None and np.isfinite(inten).any():
        filled, ok = _fill_sparse(np.nan_to_num(inten), np.isfinite(inten), 1.0)
        layers.append(("lidar_intensity", _normalize(filled, ok), ok))
    rgb = rasters.get("rgb")
    if rgb is not None and "texture_nobs" in rasters and (rasters["texture_nobs"] > 0).any():
        ok = rasters["texture_nobs"] > 0
        lum = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(float) / 255.0
        layers.append(("camera_texture", _normalize(lum, ok), ok))
    if not layers:
        raise ValueError("no intensity or texture raster available")
    hmax = rasters.get("height_max")
    if hmax is not None:
        covered = np.nan_to_num(hmax) > 0.3  # under an object: no paint visible
        layers = [(n, img, ok & ~covered) for n, img, ok in layers]
    return layers


def _line_kernel(length: int, angle: float) -> np.ndarray:
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2.0
    d = np.array([np.cos(angle), np.sin(angle)]) * c
    cv2.line(k, (int(round(c - d[0])), int(round(c - d[1]))), (int(round(c + d[0])), int(round(c + d[1]))), 1, 1)
    return k


def layer_mask(img: np.ndarray, valid: np.ndarray, res: float, cfg) -> np.ndarray:
    """White top-hat with a robust per-layer threshold (median + k * MAD)."""
    mc = cfg.markings
    bg = np.median(img[valid]) if valid.any() else 0.0
    work = np.where(valid, img, bg)
    k = max(3, int(round(mc.tophat_size / res)) | 1)
    u8 = (work * 255).astype(np.uint8)
    tophat = cv2.morphologyEx(u8, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    t = tophat[valid].astype(float)
    if len(t) == 0:
        return np.zeros_like(valid)
    med = np.median(t)
    mad = np.median(np.abs(t - med)) * 1.4826
    otsu, _ = cv2.threshold(tophat[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu, med + mc.mad_k * mad, mc.min_contrast * 255)
    return (tophat > thr) & valid


def elongated(mask: np.ndarray, res: float, yaw: float, cfg) -> np.ndarray:
    """Keeps paint that forms line pieces along the garage grid axes (removes
    texture speckle, text and blobs)."""
    L = max(3, int(round(cfg.markings.line_open_length / res)) | 1)
    Lc = max(3, int(round(cfg.markings.line_close_length / res)) | 1)
    m = mask.astype(np.uint8)
    out = np.zeros_like(m)
    for a in (yaw, yaw + np.pi / 2):
        # LiDAR samples paint ring by ring: bridge the dots along the line
        # direction first (no smearing across), then require line pieces.
        closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _line_kernel(Lc, a))
        out |= cv2.morphologyEx(closed, cv2.MORPH_OPEN, _line_kernel(L, a))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    lab, n = ndimage.label(out, structure=np.ones((3, 3)))
    if n:
        sizes = ndimage.sum(out, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes * res * res >= 0.05  # >= 0.05 m^2 of paint
        out = keep[lab].astype(np.uint8)
    return out.astype(bool)


def classical_mask(rasters: dict, res: float, yaw: float, cfg) -> np.ndarray:
    mask = None
    for _, img, valid in marking_layers(rasters):
        m = layer_mask(img, valid, res, cfg)
        mask = m if mask is None else (mask | m)
    return elongated(mask, res, yaw, cfg)


def grid_aligned(seg: MarkingSegment, yaw: float, max_angle_deg: float) -> bool:
    """Parking and lane lines follow the column grid; slanted fragments are
    skeleton artefacts (angled bays would need this filter relaxed)."""
    d = np.subtract(seg.p1, seg.p0)
    a = (np.arctan2(d[1], d[0]) - yaw) % (np.pi / 2)
    return min(a, np.pi / 2 - a) <= np.radians(max_angle_deg)


def both_sides_observed(seg: MarkingSegment, valid: np.ndarray, grid: BevGrid, offset: float = 0.2,
                        min_fraction: float = 0.5) -> bool:
    """Paint lies on the floor: both sides of a line must be observed floor.
    Rejects bright rims at the floor boundary (wall bases, raster edges)."""
    p0, p1 = np.asarray(seg.p0), np.asarray(seg.p1)
    d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
    n = np.array([-d[1], d[0]]) * (seg.width / 2 + offset)
    t = np.linspace(0.1, 0.9, 20)[:, None]
    line = p0 + t * (p1 - p0)
    for side in (line + n, line - n):
        cr = np.floor(grid.world_to_px(side)).astype(int)
        ok = (cr[:, 0] >= 0) & (cr[:, 1] >= 0) & (cr[:, 0] < valid.shape[1]) & (cr[:, 1] < valid.shape[0])
        frac = valid[cr[ok, 1], cr[ok, 0]].mean() if ok.any() else 0.0
        if frac < min_fraction:
            return False
    return True


def vectorize(mask: np.ndarray, grid: BevGrid, cfg, level: int) -> list[MarkingSegment]:
    mc = cfg.markings
    res = grid.res
    skel = skeletonize(mask)
    nb = ndimage.convolve(skel.astype(int), np.ones((3, 3), int), mode="constant") - 1
    junction = skel & (nb >= 3)
    branches = skel & ~ndimage.binary_dilation(junction, iterations=1)
    dist = ndimage.distance_transform_edt(mask)
    lab, n = ndimage.label(branches, structure=np.ones((3, 3)))
    segs: list[MarkingSegment] = []
    # Each branch within its own bounding box: np.where over the whole raster
    # per branch costs branches x cells (hours on a 400 m level with camera
    # texture).
    for c, box in enumerate(ndimage.find_objects(lab), start=1):
        if box is None:
            continue
        jj, ii = np.where(lab[box] == c)
        jj, ii = jj + box[0].start, ii + box[1].start
        if len(jj) * res < mc.min_length * 0.5:
            continue
        pts = np.stack([ii + 0.5, jj + 0.5], axis=1)
        mean = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
        d = vt[0]
        t = (pts - mean) @ d
        resid = np.abs((pts - mean) @ np.array([-d[1], d[0]]))
        if resid.mean() > 2.0:  # curved branch (arrow heads, text)
            continue
        p0, p1 = grid.px_to_world(np.array([mean + t.min() * d, mean + t.max() * d]))
        width = float(2.0 * np.mean(dist[jj, ii]) * res)
        segs.append(MarkingSegment(level, p0.tolist(), p1.tolist(), width, "parking_line", 1.0))
    segs = merge_collinear(segs, cfg)
    return [s for s in segs if s.length >= mc.min_length and mc.min_width <= s.width <= mc.max_width]


def classify_markings(segs: list[MarkingSegment], paths_xy: dict[int, np.ndarray], cfg) -> None:
    """Long lines running along the driven aisle are lane markings; the
    rest are parking-slot lines."""
    mc = cfg.markings
    for s in segs:
        s.cls = "parking_line"
        path = paths_xy.get(s.level)
        if path is None or len(path) < 3 or s.length < mc.lane_min_length:
            continue
        mid = (np.asarray(s.p0) + np.asarray(s.p1)) / 2
        k = int(np.argmin(np.linalg.norm(path - mid, axis=1)))
        a, b = path[max(0, k - 5)], path[min(len(path) - 1, k + 5)]
        t = b - a
        d = np.subtract(s.p1, s.p0)
        if np.linalg.norm(t) > 0.5 and abs(t @ d) / (np.linalg.norm(t) * np.linalg.norm(d)) > np.cos(np.radians(10)):
            s.cls = "lane_marking"


def merge_collinear(segs: list[MarkingSegment], cfg) -> list[MarkingSegment]:
    """Merges segments that are parallel (within `merge_angle`), on one line
    (both ends of the other within `merge_offset` of it) and at most
    `merge_gap` apart along it. Each segment absorbs every partner it finds,
    tested against all remaining segments at once (a level with camera
    texture has thousands of fragments; pairwise loops restarted after each
    merge took hours)."""
    mc = cfg.markings
    segs = list(segs)
    n = len(segs)
    if n < 2:
        return segs
    p0 = np.array([s.p0 for s in segs], float)
    p1 = np.array([s.p1 for s in segs], float)
    alive = np.ones(n, bool)
    cos_max = np.cos(np.radians(mc.merge_angle))

    def unit(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

    for a in range(n):
        while alive[a]:
            others = np.nonzero(alive)[0]
            others = others[others != a]
            if len(others) == 0:
                break
            da = unit(p1[a] - p0[a])
            db = unit(p1[others] - p0[others])
            nrm = np.array([-da[1], da[0]])
            ok = np.abs(db @ da) >= cos_max
            ok &= np.maximum(np.abs((p0[others] - p0[a]) @ nrm),
                             np.abs((p1[others] - p0[a]) @ nrm)) <= mc.merge_offset
            la = float(np.linalg.norm(p1[a] - p0[a]))
            t0 = (p0[others] - p0[a]) @ da
            t1 = (p1[others] - p0[a]) @ da
            gap = np.maximum(np.minimum(t0, t1) - la, -np.maximum(t0, t1))
            ok &= gap <= mc.merge_gap
            if not ok.any():
                break
            k = int(others[np.argmax(ok)])
            A, B = segs[a], segs[k]
            lo = min(0.0, t0[np.argmax(ok)], t1[np.argmax(ok)])
            hi = max(la, t0[np.argmax(ok)], t1[np.argmax(ok)])
            lb = B.length
            w = (A.width * la + B.width * lb) / max(la + lb, 1e-9)
            conf = min(1.0, (A.confidence * la + B.confidence * lb) / max(la + lb, 1e-9))
            start, end = p0[a] + lo * da, p0[a] + hi * da
            segs[a] = MarkingSegment(A.level, start.tolist(), end.tolist(), w, A.cls, conf)
            p0[a], p1[a] = start, end
            alive[k] = False
    return [segs[i] for i in range(n) if alive[i]]


def extract_markings(rasters: dict, grid: BevGrid, cfg, yaw: float = 0.0, segmenter: MarkingSegmenter | None = None,
                     log=print) -> list[MarkingSegment]:
    if segmenter is not None:
        layers = marking_layers(rasters)
        img = np.max([l[1] * l[2] for l in layers], axis=0)
        valid = np.any([l[2] for l in layers], axis=0)
        mask = segmenter.segment(img, valid)
    else:
        mask = classical_mask(rasters, grid.res, yaw, cfg)
    observed = np.any([l[2] for l in marking_layers(rasters)], axis=0)
    covered = np.nan_to_num(rasters["height_max"]) > 0.3 if "height_max" in rasters else np.zeros_like(observed)
    floor = ndimage.binary_closing(observed | covered, iterations=2)  # cars do not end the floor
    segs = [s for s in vectorize(mask, grid, cfg, grid.level)
            if both_sides_observed(s, floor, grid) and grid_aligned(s, yaw, cfg.markings.max_axis_angle)]
    refined = 0
    if cfg.markings.get("subpixel", True):
        refined = vector_fit.refine_segments(segs, rasters, grid)
    log(f"[markings] level {grid.level}: {len(segs)} segments, {refined} measured on the orthophoto "
        f"({sum(sum(s.end_ok) for s in segs)} reliable ends)")
    return segs
