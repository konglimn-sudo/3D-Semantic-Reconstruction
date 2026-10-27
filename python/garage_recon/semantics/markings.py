"""Floor markings from the per-level BEV (section 5, module 3).

Markings are paint, not geometry, so they are extracted in 2D on the floor
orthophoto (fisheye texture) and the LiDAR intensity raster, then vectorised:
fill sparse samples -> white top-hat -> threshold -> floor arrows (C++,
gr/nav/arrows.h) taken out -> skeleton -> junction split -> line fit ->
collinear merge. The raster steps use OpenCV; from the skeleton on the
algorithm is C++ (gr/semantics/markings.h, vector_fit.h).

`MarkingSegmenter` is the slot for a learned model (SegFormer / Mask2Former
fine-tuned on garage BEV tiles, or SAM 3 prompted with "parking line"); the
classical path below is the default and needs no training data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from .. import _core, core_config
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


def _rasters(rasters: dict) -> tuple:
    return (rasters.get("intensity"), rasters.get("rgb"),
            rasters.get("texture_nobs"), rasters.get("height_max"))


def paint_config(cfg) -> _core.PaintConfig:
    return core_config.fill(_core.PaintConfig(), cfg.markings)


def marking_layers(rasters: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, brightness in [0, 1], valid mask) for every available source
    (C++ MarkingLayers)."""
    layers = _core.marking_layers(*_rasters(rasters))
    if not layers:
        raise ValueError("no intensity or texture raster available")
    return [(name, image, valid > 0) for name, image, valid in layers]


def elongated(mask: np.ndarray, res: float, yaw: float, cfg) -> np.ndarray:
    """Keeps paint that forms line pieces along the garage grid axes (removes
    texture speckle, text and blobs; C++ Elongated)."""
    return _core.elongated(np.ascontiguousarray(mask, np.uint8), res, yaw,
                           paint_config(cfg)) > 0


def paint_mask(rasters: dict, res: float, cfg) -> np.ndarray:
    """Painted cells of all available layers, before the line filter (C++
    PaintMask: per layer a white top-hat above a robust threshold)."""
    marking_layers(rasters)  # Raises without a usable layer.
    return _core.paint_mask(*_rasters(rasters), res, paint_config(cfg)) > 0


def classical_mask(rasters: dict, res: float, yaw: float, cfg) -> np.ndarray:
    return elongated(paint_mask(rasters, res, cfg), res, yaw, cfg)


def detect_arrows(mask: np.ndarray, grid: BevGrid,
                  cfg) -> tuple[list[dict], np.ndarray]:
    """Floor arrows in a paint mask (C++ DetectArrows).

    Arrow shafts are straight paint pieces like parking lines; parallel
    shafts of the two lanes of an aisle would pass for slot separators, so
    the arrows are found first and their paint is kept out of the lines.

    Args:
        mask: Painted cells (height, width).
        grid: The level's grid.
        cfg: Configuration (markings.arrows: ArrowConfig fields).

    Returns:
        The arrows (level, center, direction, tail, tip, length, widths,
        area; metres) and the mask of their paint, slightly dilated.
    """
    # Bridge the gaps between LiDAR rings inside an arrow.
    closed = _core.close_mask(np.ascontiguousarray(mask, np.uint8),
                              "ellipse", 3)
    config = core_config.fill(_core.ArrowConfig(),
                              cfg.markings.get("arrows"))
    found = _core.detect_arrows(closed, grid.res, grid.x0, grid.y0, config)
    paint = np.zeros(mask.shape, bool)
    arrows = []
    for arrow in found:
        paint.flat[arrow["cells"]] = True
        arrows.append({
            "level": grid.level,
            **{k: np.round(np.asarray(arrow[k], float), 3).tolist()
               for k in ("center", "direction", "tail", "tip")},
            **{k: round(float(arrow[k]), 3)
               for k in ("length", "shaft_width", "head_width", "area")}})
    return arrows, _core.dilate_cross(paint.astype(np.uint8), 2) > 0


def _to_dict(seg: MarkingSegment) -> dict:
    return {"level": seg.level, "p0": seg.p0, "p1": seg.p1,
            "width": seg.width, "cls": seg.cls, "confidence": seg.confidence,
            "end_ok": list(seg.end_ok)}


def _from_dict(d: dict) -> MarkingSegment:
    return MarkingSegment(int(d["level"]), np.asarray(d["p0"]).tolist(),
                          np.asarray(d["p1"]).tolist(), float(d["width"]),
                          d["cls"], float(d["confidence"]),
                          end_ok=[bool(v) for v in d["end_ok"]])


def marking_config(cfg) -> _core.MarkingConfig:
    return core_config.fill(_core.MarkingConfig(), cfg.markings)


def vectorize(mask: np.ndarray, grid: BevGrid, cfg,
              level: int) -> list[MarkingSegment]:
    """Line segments of a paint mask (C++ VectorizeMarkings: skeleton
    (scikit-image's thinning, ported), branches fitted, collinear pieces
    merged, length / width filters)."""
    out = _core.vectorize_markings(
        np.ascontiguousarray(mask, np.uint8), grid.x0, grid.y0, grid.res,
        marking_config(cfg), level)
    return [_from_dict(d) for d in out]


def classify_markings(segs: list[MarkingSegment],
                      paths_xy: dict[int, np.ndarray], cfg) -> None:
    """Long lines running along the driven aisle are lane markings; the
    rest are parking-slot lines (C++ ClassifyMarkings)."""
    paths = {int(k): np.column_stack([np.asarray(v)[:, :2], np.zeros(len(v))])
             for k, v in paths_xy.items() if len(v)}
    out = _core.classify_markings([_to_dict(s) for s in segs], paths,
                                  marking_config(cfg))
    for seg, d in zip(segs, out):
        seg.cls = d["cls"]


def extract_markings(rasters: dict, grid: BevGrid, cfg, yaw: float = 0.0, segmenter: MarkingSegmenter | None = None,
                     log=print) -> tuple[list[MarkingSegment], list[dict]]:
    """Line segments and floor arrows of one level."""
    if segmenter is not None:
        layers = marking_layers(rasters)
        img = np.max([l[1] * l[2] for l in layers], axis=0)
        valid = np.any([l[2] for l in layers], axis=0)
        paint = segmenter.segment(img, valid)
    else:
        paint = paint_mask(rasters, grid.res, cfg)
    arrows, arrow_paint = detect_arrows(paint, grid, cfg)
    paint = paint & ~arrow_paint
    mask = (paint if segmenter is not None
            else elongated(paint, grid.res, yaw, cfg))
    observed = np.any([l[2] for l in marking_layers(rasters)], axis=0)
    covered = np.nan_to_num(rasters["height_max"]) > 0.3 if "height_max" in rasters else np.zeros_like(observed)
    floor = _core.close_mask((observed | covered).astype(np.uint8), "cross",
                             3, 2) > 0  # Cars do not end the floor.
    # Paint lies on observed floor (both sides) and follows the grid.
    kept = _core.filter_markings(
        [_to_dict(s) for s in vectorize(mask, grid, cfg, grid.level)],
        np.ascontiguousarray(floor, np.uint8), grid.x0, grid.y0, grid.res, yaw,
        float(cfg.markings.max_axis_angle))
    segs = [_from_dict(d) for d in kept]
    refined = 0
    if cfg.markings.get("subpixel", True):
        refined = vector_fit.refine_segments(segs, rasters, grid)
    ends = sum(sum(s.end_ok) for s in segs)
    log(f"[markings] level {grid.level}: {len(segs)} segments, {refined} "
        f"measured on the orthophoto ({ends} reliable ends), "
        f"{len(arrows)} arrows")
    return segs, arrows
