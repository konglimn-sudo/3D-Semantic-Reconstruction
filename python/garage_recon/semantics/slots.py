"""Parking slots from vectorised markings (section 5, module 3).

Separators are grouped into rows, consecutive separators with a plausible
spacing form slots, gaps of k typical widths are split into k *inferred*
slots (separators hidden by cars), and the entrance side is the side facing
the driven aisle.

Rows are often a little rotated against the structure grid (or the grid yaw
is off outdoors, without a column grid): each row's direction is refined
from its separator ends and the row is assembled in its own frame.
Back-to-back rows are told from single rows by the 10 / 90 % quantiles of
the separator ends, so a few long fragments do not split a single row.

An inferred slot needs a reason for the missing line: a hypothetical
separator on observed, empty floor without a paint ridge in the LiDAR
intensity would have been seen, so such gaps are not split (a ridge means
paint that vectorisation missed; hidden floor means a parked car). Inferred
slots the vehicle drove through, that overlap a slot with paint, or that lie
mostly on unobserved ground (behind walls) are dropped as well.

Corner order follows the common parking-slot convention: entrance-left,
entrance-right, back-right, back-left (looking into the slot).

The algorithm is C++ (gr/semantics/slots.h); this module holds the Slot type.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import shapely

from garage_recon import _core
from garage_recon.recon import bev
from garage_recon.semantics import markings as markings_lib
from garage_recon.semantics import structure as structure_lib


@dataclasses.dataclass
class Slot:
    """A parking slot.

    Attributes:
        slot_id: "L<level>-R<row>-<index>" (generated).
        level: Level index.
        corners: (4, 2) entrance-left, entrance-right, back-right, back-left.
        z: Floor height (m).
        width: Across the slot (m).
        depth: Into the slot (m).
        occupied: Whether a vehicle stood in it at capture time (None if
            not measured).
        inferred: Split from a gap of several slot widths.
        confidence: 1 with paint on both sides, lower for inferred slots.
        row: Row id ("R<n>").
        number: Painted number (OCR, row-repaired).
        zone: Zone letter(s) painted with the number.
        text_votes: OCR readings that voted for this slot.
    """

    slot_id: str
    level: int
    corners: list[list[float]]
    z: float
    width: float
    depth: float
    occupied: bool | None
    inferred: bool
    confidence: float
    row: str
    number: str | None = None
    zone: str | None = None
    text_votes: int = 0

    def polygon(self) -> shapely.Polygon:
        return shapely.Polygon(self.corners)

    @property
    def entrance_mid(self) -> np.ndarray:
        c = np.asarray(self.corners)
        return (c[0] + c[1]) / 2

    @property
    def back_mid(self) -> np.ndarray:
        c = np.asarray(self.corners)
        return (c[2] + c[3]) / 2

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


def slot_config(cfg) -> _core.SlotConfig:
    """C++ SlotConfig from the slots / rules sections."""
    sc = cfg.slots
    config = _core.SlotConfig()
    config.width_min, config.width_max = (float(v) for v in sc.width_range)
    config.depth_min, config.depth_max = (float(v) for v in sc.depth_range)
    for key in ("min_separator_length", "typical_depth", "angle_tol",
                "row_overlap", "infer_tolerance", "occupied_height",
                "occupied_fraction", "visible_separator_share",
                "paint_ridge_share", "inferred_max_overlap",
                "inferred_min_evidence"):
        if sc.get(key) is not None:
            setattr(config, key, sc[key])
    config.column_overlap_area = float(cfg.rules.column_overlap_area)
    return config


def assemble_slots(segments: list[markings_lib.MarkingSegment], level: int,
                   yaw: float, path_xy: np.ndarray,
                   columns: list[structure_lib.Column], grid: bev.BevGrid,
                   height_max: np.ndarray, floor_z: np.ndarray, cfg,
                   log=print, floor_count: np.ndarray | None = None,
                   intensity: np.ndarray | None = None) -> list[Slot]:
    """Slots of one level from its marking segments (C++ AssembleSlots).

    Args:
        segments: Vectorised markings of the level.
        level: Level index.
        yaw: Structure grid yaw (rad).
        path_xy: (N, 2) driven positions on the level.
        columns: Columns of the level (their faces bound bays).
        grid: The level's BEV grid.
        height_max: Highest object above the floor per cell (m).
        floor_z: Floor height per cell (m).
        cfg: Configuration (`slots`, `rules`).
        log: Logger.
        floor_count: Floor points per cell (evidence for inferred slots).
        intensity: LiDAR floor intensity per cell (paint ridges).

    Returns:
        The slots, with ids, occupancy at capture time and floor height.
    """
    path = np.asarray(path_xy, float).reshape(-1, 2)
    data = [{"level": s.level, "p0": s.p0, "p1": s.p1, "width": s.width,
             "cls": s.cls, "confidence": s.confidence,
             "end_ok": list(getattr(s, "end_ok", [False, False]))}
            for s in segments]

    def raster(values):
        return (None if values is None
                else np.ascontiguousarray(values, np.float32))

    out = _core.assemble_slots(
        data, level, float(yaw), np.column_stack([path, np.zeros(len(path))]),
        [(c.center[0], c.center[1], c.size[0], c.size[1]) for c in columns],
        grid.x0, grid.y0, grid.res, raster(height_max), raster(floor_z),
        raster(floor_count), raster(intensity), slot_config(cfg), log)
    return [Slot(d["slot_id"], int(d["level"]),
                 [list(map(float, c)) for c in d["corners"]], float(d["z"]),
                 float(d["width"]), float(d["depth"]), d["occupied"],
                 bool(d["inferred"]), float(d["confidence"]), d["row"])
            for d in out]
