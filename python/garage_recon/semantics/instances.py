"""3D object instances of the language route, and their text (README 5.2).

The identity association (stage identities) merges the views of one object
into an identity; the language stage gives it a class. Views that the
association could not link split one object into several identities (on
FP2 a ceiling light became three on average), so identities of one class
whose robust boxes overlap are merged into one instance: union of the pairs
whose 5-95 % boxes, grown by `pad`, overlap `overlap` of the smaller box,
best overlap first, while the merged box stays within `max_extent`.

An instance has its map points, centroid, 5-95 % box (min, max), level and
the identities it merges. The numbers stage then attaches each OCR text
landmark to the sign (or other object) it is written on: the nearest such
instance whose box, grown by `max_distance`, contains the text.
"""

from __future__ import annotations

import numpy as np

from garage_recon.semantics import taxonomy


def _boxes(points: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """5-95 % boxes (lo, hi) of point sets; stray points do not grow them."""
    lo = np.array([np.percentile(p, 5, axis=0) for p in points]).reshape(-1, 3)
    hi = np.array([np.percentile(p, 95, axis=0) for p in points]).reshape(-1, 3)
    return lo, hi


def merge_groups(lo: np.ndarray, hi: np.ndarray, overlap: float, pad: float,
                 max_extent: float) -> np.ndarray:
    """Group index of each box: boxes overlapping `overlap` of the smaller
    one (grown by `pad`) are merged, best overlap first, while the merged
    box's longest horizontal side stays within `max_extent`."""
    n = len(lo)
    group = np.arange(n)
    if n < 2 or overlap <= 0.0:
        return group
    a, b = lo - pad, hi + pad
    volume = np.prod(b - a, axis=1)
    pairs = []
    for i in range(n - 1):
        inter = np.prod(np.clip(np.minimum(b[i], b[i + 1:])
                                - np.maximum(a[i], a[i + 1:]), 0.0, None), axis=1)
        ratio = inter / np.minimum(volume[i], volume[i + 1:])
        pairs += [(r, i, i + 1 + j) for j, r in enumerate(ratio) if r >= overlap]
    box_lo, box_hi = {i: lo[i] for i in range(n)}, {i: hi[i] for i in range(n)}
    for _, i, j in sorted(pairs, reverse=True):
        gi, gj = group[i], group[j]
        if gi == gj:
            continue
        new_lo = np.minimum(box_lo[gi], box_lo[gj])
        new_hi = np.maximum(box_hi[gi], box_hi[gj])
        if np.max((new_hi - new_lo)[:2]) > max_extent:
            continue
        group[group == gj] = gi
        box_lo[gi], box_hi[gi] = new_lo, new_hi
    return group


def instances(xyz: np.ndarray, identity: np.ndarray, ids: np.ndarray,
              classes: np.ndarray, scores: np.ndarray, levels=None,
              min_points: int = 5, overlap: float = 0.3, pad: float = 0.1,
              max_extent: float = 8.0) -> list[dict]:
    """Instances of the classified identities (`ids` with `classes` > 0)
    from the per-point identity of the map points `xyz` (overlap 0: one
    instance per identity)."""
    names = {c.id: c.name for c in taxonomy.CLASSES}
    of = {int(i): (int(c), float(s)) for i, c, s in zip(ids, classes, scores)
          if c > 0}
    keep = identity >= 0
    order = np.argsort(identity[keep], kind="stable")
    pts, idn = xyz[keep][order], identity[keep][order]
    out = []
    if len(idn) == 0:
        return out
    starts = np.r_[0, np.nonzero(np.diff(idn))[0] + 1]
    ends = np.r_[starts[1:], len(idn)]
    parts = [(int(idn[a]), pts[a:b]) for a, b in zip(starts, ends)
             if int(idn[a]) in of and b - a >= min_points]
    for c in sorted({of[i][0] for i, _ in parts}):
        mine = [(i, p) for i, p in parts if of[i][0] == c]
        group = merge_groups(*_boxes([p for _, p in mine]), overlap, pad,
                             max_extent)
        for g in np.unique(group):
            members = [mine[k] for k in np.nonzero(group == g)[0]]
            p = np.concatenate([q for _, q in members])
            n = np.array([len(q) for _, q in members], float)
            lo, hi = _boxes([p])
            out.append({"id": min(i for i, _ in members), "class": names[c],
                        "score": round(float(np.average(
                            [of[i][1] for i, _ in members], weights=n)), 3),
                        "points": len(p),
                        "identities": sorted(i for i, _ in members),
                        "centroid": np.median(p, axis=0).round(3).tolist(),
                        "min": lo[0].round(3).tolist(),
                        "max": hi[0].round(3).tolist()})
    out.sort(key=lambda o: o["id"])
    if levels is not None and out:
        level, height, _ = levels.assign(np.array([o["centroid"] for o in out]))
        for o, lv, h in zip(out, level, height):
            o["level"], o["height"] = int(lv), round(float(h), 2)
    return out


def attach_text(instances_: list[dict], landmarks: list[dict],
                classes=("sign", "other"), max_distance: float = 0.5) -> int:
    """Sets `text` of the instances and `instance` of the landmarks for each
    landmark inside an instance of `classes` grown by `max_distance` (the
    nearest centroid wins). Returns how many landmarks were attached."""
    attached = 0
    for mark in landmarks:
        xyz = np.asarray(mark["xyz"], float)
        best, best_d = None, np.inf
        for inst in instances_:
            if inst["class"] not in classes:
                continue
            lo = np.asarray(inst["min"]) - max_distance
            hi = np.asarray(inst["max"]) + max_distance
            if np.all(xyz >= lo) and np.all(xyz <= hi):
                d = float(np.linalg.norm(np.asarray(inst["centroid"]) - xyz))
                if d < best_d:
                    best, best_d = inst, d
        if best is not None:
            mark["instance"] = best["id"]
            best.setdefault("text", []).append(mark["text"])
            attached += 1
    return attached
