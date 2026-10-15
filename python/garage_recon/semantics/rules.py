"""Consistency rules (section 5, "6.4"): violations go to the review queue."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .slots import Slot
from .structure import Column


def _issue(severity: str, kind: str, obj: str, message: str, location=None) -> dict:
    return {"severity": severity, "kind": kind, "object": obj, "message": message,
            "location": None if location is None else [float(v) for v in location]}


def check_slots(slots: list[Slot], columns: list[Column], paths_xy: dict[int, np.ndarray], cfg) -> list[dict]:
    rc = cfg.rules
    out = []
    ids = Counter(s.slot_id for s in slots)
    for s in slots:
        c = np.mean(np.asarray(s.corners), axis=0)
        if ids[s.slot_id] > 1:
            out.append(_issue("error", "duplicate_id", s.slot_id, "slot id used more than once", c))
        if not (rc.slot_width_range[0] <= s.width <= rc.slot_width_range[1]):
            out.append(_issue("warning", "slot_width", s.slot_id, f"width {s.width:.2f} m out of range", c))
        if not (rc.slot_depth_range[0] <= s.depth <= rc.slot_depth_range[1]):
            out.append(_issue("warning", "slot_depth", s.slot_id, f"depth {s.depth:.2f} m out of range", c))
        path = paths_xy.get(s.level)
        if path is not None and len(path):
            d = float(np.min(np.linalg.norm(path - s.entrance_mid, axis=1)))
            if d > rc.aisle_max_distance:
                out.append(_issue("warning", "no_aisle", s.slot_id, f"entrance {d:.1f} m from any driven aisle", c))
        if s.inferred:
            out.append(_issue("info", "inferred", s.slot_id, "separator inferred from row regularity; verify", c))
    return out


def check_columns(columns: list[Column]) -> list[dict]:
    out = []
    labels = Counter((c.level, c.axis_label) for c in columns)
    for c in columns:
        if labels[(c.level, c.axis_label)] > 1:
            out.append(_issue("warning", "column_axis", c.id, "two columns snapped to the same grid node", c.center))
        if c.completed:
            out.append(_issue("info", "column_completed", c.id, "extent completed from typical size (unseen side)",
                              c.center))
    return out


def fix_slot_numbers(slots_in_row: list[Slot], numbers: list[str | None]) -> list[str | None]:
    """Repairs OCR'd numbers of one row using the consecutive-numbering prior:
    fits number = a + k (or a - k) to the confidently parsed readings and
    fills / corrects the rest."""
    k = np.arange(len(numbers))
    parsed = [(i, m.group()) for i, n in enumerate(numbers) if n and (m := re.fullmatch(r"\D*(\d+)\D*", n))]
    parsed = [(i, re.search(r"\d+", g).group()) for i, g in parsed]
    if len(parsed) < 2:
        return numbers
    idx, val = np.array([p[0] for p in parsed]), np.array([int(p[1]) for p in parsed])
    best = None
    for sign in (1, -1):
        offs = val - sign * idx
        a = Counter(offs).most_common(1)[0]
        if best is None or a[1] > best[1]:
            best = (a[0], a[1], sign)
    a, support, sign = best
    if support < 2:
        return numbers
    width = max(len(p[1]) for p in parsed)  # keep leading zeros of the painted numbers
    return [str(a + sign * i).zfill(width) for i in k]
