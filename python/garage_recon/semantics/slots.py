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
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import shapely

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


@dataclasses.dataclass(frozen=True)
class _Separator:
    """A slot boundary in a row frame: across position, along extent."""

    c: float
    a0: float
    a1: float
    virtual: bool = False  # Column face, no paint.
    ok0: bool = False      # a0 / a1 measured with floor observed beyond.
    ok1: bool = False
    index: int = -1        # Marking segment it comes from.


def _row_tilt(seps: list[_Separator], min_change: float = 0.15,
              min_baseline: float = 2.0) -> float:
    """Angle (rad) of a row's entrance / back lines against the frame.

    Theil-Sen slope of the measured separator ends over their across
    position. It is applied only where it matters and is significant: the
    ends must move by at least `min_change` and by four robust residual
    sigmas across the row; otherwise 0 (end noise over short rows would
    otherwise rotate aligned rows).
    """
    sets = [[(m.c, m.a0) for m in seps if m.ok0],
            [(m.c, m.a1) for m in seps if m.ok1]]
    slopes = []
    for ends in sets:
        for i in range(len(ends)):
            for j in range(i + 1, len(ends)):
                dc = ends[j][0] - ends[i][0]
                if abs(dc) >= min_baseline:
                    slopes.append((ends[j][1] - ends[i][1]) / dc)
    if len(slopes) < 3:
        return 0.0
    beta = float(np.median(slopes))
    residuals, cs = [], []
    for ends in sets:
        if not ends:
            continue
        e = np.asarray(ends, float)
        r = e[:, 1] - beta * e[:, 0]
        residuals += list(r - np.median(r))
        cs += list(e[:, 0])
    sigma = 1.4826 * float(np.median(np.abs(residuals)))
    change = abs(beta) * (max(cs) - min(cs))
    if change < max(min_change, 4.0 * sigma):
        return 0.0
    return float(np.arctan(beta))


def _rows(seps: list[_Separator], overlap: float) -> list[list[_Separator]]:
    """Groups separators whose along-intervals overlap."""
    rows: list[list[_Separator]] = []
    for s in sorted(seps, key=lambda s: (s.a0 + s.a1) / 2):
        for r in rows:
            ra0, ra1 = min(m.a0 for m in r), max(m.a1 for m in r)
            inter = min(ra1, s.a1) - max(ra0, s.a0)
            if inter / max(min(ra1 - ra0, s.a1 - s.a0), 1e-6) >= overlap:
                r.append(s)
                break
        else:
            rows.append([s])
    return rows


def _row_ends(painted: list[_Separator], near: float = 0.15
              ) -> tuple[float, float, bool, bool]:
    """Along extent of a row and whether each end was measured.

    Rows share their entrance and back lines, so each end is the median of
    the measured separator ends near the row's extreme (ends farther in are
    lines that stop early, e.g. at the middle of back-to-back rows); without
    any, the envelope of the paint.
    """
    e0 = min(m.a0 for m in painted)
    e1 = max(m.a1 for m in painted)
    lo = [m.a0 for m in painted if m.ok0 and m.a0 < e0 + near]
    hi = [m.a1 for m in painted if m.ok1 and m.a1 > e1 - near]
    a0 = float(np.median(lo)) if lo else float(e0)
    a1 = float(np.median(hi)) if hi else float(e1)
    return a0, a1, bool(lo), bool(hi)


def _line_cells(c: float, a0: float, a1: float, orient: int,
                to_w: np.ndarray, grid: bev.BevGrid
                ) -> tuple[np.ndarray, np.ndarray, int]:
    """(col, row, samples) of a line across-position c, along a0..a1 (row
    frame), 0.3 m in from both ends; cells outside the grid are dropped."""
    along = np.linspace(a0 + 0.3, a1 - 0.3, max(int((a1 - a0) / 0.1), 2))
    pts_g = (np.column_stack([np.full_like(along, c), along]) if orient == 1
             else np.column_stack([along, np.full_like(along, c)]))
    px = np.floor(grid.world_to_px(pts_g @ to_w.T)).astype(int)
    inside = ((px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] < grid.width)
              & (px[:, 1] < grid.height))
    return px[inside, 0], px[inside, 1], len(px)


def _separator_evidence(c: float, a0: float, a1: float, orient: int,
                        to_w: np.ndarray, grid: bev.BevGrid,
                        floor_count: np.ndarray, height_max: np.ndarray,
                        intensity: np.ndarray | None,
                        occupied_height: float) -> tuple[float, float]:
    """(visible share, paint ridge) of a hypothetical separator: the share
    of it on observed floor with nothing standing on it, and the mean LiDAR
    intensity on the line minus that 0.35 m to both sides (NaN without
    intensity)."""
    col, row, n = _line_cells(c, a0, a1, orient, to_w, grid)
    if not len(col):
        return 0.0, float("nan")
    seen = floor_count[row, col] > 0
    empty = ~(np.nan_to_num(height_max[row, col], nan=0.0) >= occupied_height)
    visible = float(np.sum(seen & empty)) / n
    if intensity is None:
        return visible, float("nan")
    values = []
    for offset in (0.0, -0.35, 0.35):
        cc, rr, _ = _line_cells(c + offset, a0, a1, orient, to_w, grid)
        v = intensity[rr, cc] if len(cc) else np.array([np.nan])
        values.append(np.nanmean(v) if np.isfinite(v).any() else np.nan)
    return visible, float(values[0] - 0.5 * (values[1] + values[2]))


def _absent(visible: float, ridge: float, paint: float, sc) -> bool:
    """Whether a hypothetical separator is visibly not there: on observed,
    empty floor and without a paint ridge of at least `paint_ridge_share` of
    the row's painted separators."""
    if visible < sc.visible_separator_share:
        return False  # Hidden (parked car, unobserved): may be there.
    if not (np.isfinite(ridge) and np.isfinite(paint) and paint > 0):
        return True
    return ridge < sc.paint_ridge_share * paint


def _point(c: float, a: float, orient: int) -> np.ndarray:
    """Row-frame point at across position c, along position a."""
    return np.array([c, a]) if orient == 1 else np.array([a, c])


def _project(segments: list[markings_lib.MarkingSegment], indices,
             orient: int, row_yaw: float, sc, dmax: float
             ) -> list[_Separator]:
    """Separators of the given segments in the frame rotated by row_yaw."""
    to_g = structure_lib.rot2(-row_yaw)
    axis = np.array([0.0, 1.0]) if orient == 1 else np.array([1.0, 0.0])
    out = []
    for i in indices:
        s = segments[i]
        p0, p1 = np.asarray(s.p0) @ to_g.T, np.asarray(s.p1) @ to_g.T
        d = p1 - p0
        length = np.linalg.norm(d)
        if length < sc.min_separator_length or length > 2 * dmax + 0.5:
            continue
        angle = np.degrees(np.arccos(min(1.0, abs(d @ axis) / length)))
        if angle > sc.angle_tol:
            continue
        c = (p0[1 - orient] + p1[1 - orient]) / 2
        ok = list(getattr(s, "end_ok", [False, False]))
        if p0[orient] <= p1[orient]:
            out.append(_Separator(c, p0[orient], p1[orient], False, ok[0],
                                  ok[1], i))
        else:
            out.append(_Separator(c, p1[orient], p0[orient], False, ok[1],
                                  ok[0], i))
    return out


def assemble_slots(segments: list[markings_lib.MarkingSegment], level: int,
                   yaw: float, path_xy: np.ndarray,
                   columns: list[structure_lib.Column], grid: bev.BevGrid,
                   height_max: np.ndarray, floor_z: np.ndarray, cfg,
                   log=print, floor_count: np.ndarray | None = None,
                   intensity: np.ndarray | None = None) -> list[Slot]:
    """Slots of one level from its marking segments.

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
    sc = cfg.slots
    wmin, wmax = sc.width_range
    dmin, dmax = sc.depth_range
    col_polys = [shapely.box(c.center[0] - c.size[0] / 2,
                             c.center[1] - c.size[1] / 2,
                             c.center[0] + c.size[0] / 2,
                             c.center[1] + c.size[1] / 2) for c in columns]
    candidates = []
    for orient in (1, 0):  # 1: separators along grid y, slots along x.
        rows = []  # (separators, row yaw)
        groups = _rows(_project(segments, range(len(segments)), orient, yaw,
                                sc, dmax), sc.row_overlap)
        # Rows too short or noisy for their own direction take the median
        # of the significant ones.
        tilts = [_row_tilt(group) for group in groups]
        significant = [t for t in tilts if t != 0.0]
        default = (float(np.median(significant)) if len(significant) >= 2
                   else 0.0)
        for group, tilt in zip(groups, tilts):
            tilt = tilt if tilt != 0.0 else default
            row_yaw = yaw + (tilt if orient == 1 else -tilt)
            r = (_project(segments, [m.index for m in group], orient, row_yaw,
                          sc, dmax) if tilt else group)
            if not r:
                continue
            # Back-to-back rows share painted lines: split the whole group
            # at the middle of its extent (robust to fragments crossing the
            # boundary). A few long fragments do not make a single row
            # back-to-back.
            q0 = float(np.percentile([m.a0 for m in r], 10))
            q1 = float(np.percentile([m.a1 for m in r], 90))
            if q1 - q0 <= dmax + 0.5:
                if max(m.a1 for m in r) - min(m.a0 for m in r) > dmax + 0.5:
                    r = [dataclasses.replace(m, a0=max(m.a0, q0 - 0.3),
                                             a1=min(m.a1, q1 + 0.3))
                         for m in r]
                rows.append((r, row_yaw))
                continue
            e0, e1, ok0, ok1 = _row_ends(r)
            mid = (e0 + e1) / 2
            measured = ok0 and ok1  # The middle follows from both ends.
            low = [dataclasses.replace(
                m, a1=min(m.a1, mid), ok1=measured if m.a1 > mid else m.ok1)
                   for m in r if m.a0 < mid - 0.5]
            high = [dataclasses.replace(
                m, a0=max(m.a0, mid), ok0=measured if m.a0 < mid else m.ok0)
                    for m in r if m.a1 > mid + 0.5]
            rows += [(x, row_yaw) for x in (low, high) if x]
        # Columns at the row's corners bound bays as reliably as paint: add
        # their faces as virtual separators (they make hidden lines
        # inferable).
        for r, row_yaw in rows:
            to_g = structure_lib.rot2(-row_yaw)
            ra0, ra1 = min(m.a0 for m in r), max(m.a1 for m in r)
            for column in columns:
                cg = np.asarray(column.center) @ to_g.T
                half_across = np.asarray(column.size)[1 - orient] / 2
                if ra0 - 1.0 <= cg[orient] <= ra1 + 1.0:
                    for face in (cg[1 - orient] - half_across,
                                 cg[1 - orient] + half_across):
                        if all(abs(face - m.c) > 0.1 for m in r):
                            r.append(_Separator(float(face), ra0, ra1, True))
        gaps_all = []
        for r, _ in rows:
            g = np.diff(np.sort([m.c for m in r if not m.virtual]))
            gaps_all += list(g[(g >= wmin) & (g <= wmax)])
        w_global = float(np.median(gaps_all)) if gaps_all else 2.5
        # Slot depth from the rows whose two ends were both measured.
        measured_depths = []
        for r, _ in rows:
            painted = [m for m in r if not m.virtual]
            if painted:
                a0, a1, ok0, ok1 = _row_ends(painted)
                if ok0 and ok1 and dmin <= a1 - a0 <= dmax:
                    measured_depths.append(a1 - a0)
        typical_depth = (float(np.median(measured_depths)) if measured_depths
                         else float(sc.typical_depth))
        for r, row_yaw in rows:
            to_g = structure_lib.rot2(-row_yaw)
            to_w = structure_lib.rot2(row_yaw)
            path_g = path_xy @ to_g.T if len(path_xy) else np.zeros((0, 2))
            r = sorted(r, key=lambda m: m.c)
            painted = [m for m in r if not m.virtual]
            if len(r) < 2 or not painted:
                continue
            # Ends of the row: the measured ends of its separators (rows
            # share their entrance and back lines), else the envelope of the
            # paint.
            a0, a1, ok0, ok1 = _row_ends(painted)
            if a1 - a0 > dmax:
                continue
            if ok0 != ok1 and abs(a1 - a0 - typical_depth) < 0.4:
                # One end measured, the other hidden (wall base, cars) and
                # the paint about as long as a slot: place the hidden end at
                # the measured slot depth.
                if ok0:
                    a1 = a0 + typical_depth
                else:
                    a0 = a1 - typical_depth
            elif a1 - a0 < typical_depth - 0.25:
                # One end hidden (cars, wall base): keep the measured end and
                # extend the other; without measured ends extend away from
                # the driven aisle.
                if ok0 != ok1:
                    near_low = ok0
                else:
                    c_mid = float(np.median([m.c for m in r]))
                    lo_pt = _point(c_mid, a0, orient)
                    hi_pt = _point(c_mid, a1, orient)
                    near_low = (
                        not len(path_g)
                        or np.min(np.linalg.norm(path_g - lo_pt, axis=1))
                        <= np.min(np.linalg.norm(path_g - hi_pt, axis=1)))
                if near_low:
                    a1 = a0 + typical_depth
                else:
                    a0 = a1 - typical_depth
            if not dmin <= a1 - a0 <= dmax:
                continue
            gaps = np.diff([m.c for m in r])
            ok = np.diff([m.c for m in painted])
            ok = ok[(ok >= wmin) & (ok <= wmax)]
            w_typ = float(np.median(ok)) if len(ok) else w_global
            # Paint contrast of this row's separators, for the hidden ones.
            ridges = []
            if floor_count is not None and intensity is not None:
                for m in painted:
                    _, ridge = _separator_evidence(
                        m.c, max(m.a0, a0), min(m.a1, a1), orient, to_w, grid,
                        floor_count, height_max, intensity,
                        sc.occupied_height)
                    if np.isfinite(ridge):
                        ridges.append(ridge)
            paint = float(np.median(ridges)) if ridges else float("nan")
            row_key = f"{orient}:{(r[0].a0 + r[0].a1) / 2:.1f}:{row_yaw:.4f}"
            for k, g in enumerate(gaps):
                c0 = r[k].c
                if r[k].virtual and r[k + 1].virtual:
                    continue  # Both bounds virtual: no paint evidence at all.
                if wmin <= g <= wmax:
                    pieces, inferred = [(c0, c0 + g)], False
                elif g > wmax:
                    m = int(round(g / w_typ))
                    if m < 2 or abs(g - m * w_typ) > sc.infer_tolerance * m:
                        continue
                    if floor_count is not None and any(
                            _absent(*_separator_evidence(
                                c0 + i * g / m, a0, a1, orient, to_w, grid,
                                floor_count, height_max, intensity,
                                sc.occupied_height), paint, sc)
                            for i in range(1, m)):
                        continue  # The missing line would have been seen.
                    pieces = [(c0 + i * g / m, c0 + (i + 1) * g / m)
                              for i in range(m)]
                    inferred = True
                else:
                    continue
                for x0, x1 in pieces:
                    candidates.append((orient, x0, x1, a0, a1, inferred,
                                       row_key, row_yaw))
    slots: list[Slot] = []
    seen = None
    if floor_count is not None:
        seen = (floor_count > 0) | (np.nan_to_num(height_max, nan=0.0)
                                    >= sc.occupied_height)
    for orient, c0, c1, a0, a1, inferred, row_key, row_yaw in candidates:
        to_g = structure_lib.rot2(-row_yaw)
        to_w = structure_lib.rot2(row_yaw)
        path_g = path_xy @ to_g.T if len(path_xy) else np.zeros((0, 2))
        if len(path_g):
            d0 = np.min(np.linalg.norm(
                path_g - _point((c0 + c1) / 2, a0, orient), axis=1))
            d1 = np.min(np.linalg.norm(
                path_g - _point((c0 + c1) / 2, a1, orient), axis=1))
            entrance_low = d0 <= d1
        else:
            entrance_low = True
        if orient == 1:
            corners = ([[c0, a0], [c1, a0], [c1, a1], [c0, a1]] if entrance_low
                       else [[c1, a1], [c0, a1], [c0, a0], [c1, a0]])
        else:
            corners = ([[a0, c1], [a0, c0], [a1, c0], [a1, c1]] if entrance_low
                       else [[a1, c0], [a1, c1], [a0, c1], [a0, c0]])
        cw = (np.asarray(corners) @ to_w.T).tolist()
        poly = shapely.Polygon(cw)
        if any(poly.intersection(cp).area > cfg.rules.column_overlap_area
               for cp in col_polys):
            continue
        if inferred and len(path_xy):
            core = poly.buffer(-0.3)
            if not core.is_empty and shapely.contains_xy(
                    core, path_xy[:, 0], path_xy[:, 1]).any():
                continue  # The vehicle drove through it: an aisle.
        if inferred and seen is not None:
            # Evidence for the slot itself: observed floor or a parked car.
            # Inferred slots in unobserved space (behind walls) are guesses.
            mask = np.zeros(seen.shape, np.uint8)
            px = np.floor(grid.world_to_px(np.asarray(cw))).astype(np.int32)
            cv2.fillPoly(mask, [px], 1)
            inside = mask.astype(bool)
            if (not inside.any()
                    or seen[inside].mean() < sc.inferred_min_evidence):
                continue
        slots.append(Slot("", level, cw, 0.0, float(c1 - c0), float(a1 - a0),
                          None, bool(inferred), 0.6 if inferred else 1.0,
                          row_key))
    # Remove duplicates (both orientations / overlapping rows).
    slots.sort(key=lambda s: -s.confidence)
    kept: list[Slot] = []
    for s in slots:
        p = s.polygon()
        share = sc.inferred_max_overlap if s.inferred else 0.5
        if all(p.intersection(k.polygon()).area < share * p.area
               for k in kept):
            kept.append(s)
    # Occupancy at capture time, floor height and ids.
    rows = sorted({s.row for s in kept},
                  key=lambda r: (r.split(":")[0], float(r.split(":")[1])))
    row_index = {r: i + 1 for i, r in enumerate(rows)}
    hm = np.nan_to_num(height_max, nan=0.0)
    for s in kept:
        poly = s.polygon().buffer(-0.3)
        if poly.is_empty:
            continue
        px = grid.world_to_px(np.asarray(poly.exterior.coords))
        mask = np.zeros(hm.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 1)
        inside = mask.astype(bool)
        if inside.any():
            s.occupied = bool((hm[inside] >= sc.occupied_height).mean()
                              >= sc.occupied_fraction)
            s.z = float(np.nanmedian(floor_z[inside]))
    by_row: dict[str, list[Slot]] = {}
    for s in kept:
        by_row.setdefault(s.row, []).append(s)
    to_g = structure_lib.rot2(-yaw)
    for r, members in by_row.items():
        c = np.array([np.mean(np.asarray(m.corners) @ to_g.T, axis=0)
                      for m in members])
        orient = int(r.split(":")[0])
        order = np.argsort(c[:, 0] if orient == 1 else c[:, 1])
        for k, idx in enumerate(order):
            m = members[idx]
            m.row = f"R{row_index[r]:02d}"
            m.slot_id = f"L{level}-{m.row}-{k + 1:03d}"
    log(f"[slots] level {level}: {len(kept)} slots "
        f"({sum(s.inferred for s in kept)} inferred, "
        f"{sum(bool(s.occupied) for s in kept)} occupied at capture)")
    return kept
