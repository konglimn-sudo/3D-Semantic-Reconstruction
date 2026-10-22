"""Objects from 2D detection (stage objects2d, README 5.1 / 5.2).

Structure (floor, walls, ceiling, columns, beams) is decided by the LiDAR
geometry; zero-shot classes of whole-image masks do not help there (README
5.2). The 2D models name what geometry cannot: lights, signs, barriers, wheel
stoppers, people, two-wheelers, fire equipment.

Per image: open-vocabulary boxes (models.owlv2) -> SAM 2 masks from the boxes
(models.sam2; the boxes themselves without it) -> a class image where the
highest-scoring detection wins. The class images are lifted onto the static
map by multi-view voting (C++ LabelLifter, semantics/lifting.py), and a point
takes the lifted class where it is confident and the rule route left it as
structure or unlabelled (never over markings or rule-detected vehicles).

Geometry checks what the detector says (README 5.2, rules route): a mask
larger than `max_mask_fraction` of the image is no sign or speed bump (a box
over a bright floor strip made a floor-sized mask), and a lifted class must
lie in its `height` band above the floor of its level (speed bumps on the
floor, lights near the ceiling).
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from garage_recon import geometry
from garage_recon.models import interfaces
from garage_recon.semantics import language_field, lifting, taxonomy


IGNORE = "_ignore"


def queries(prompts: dict) -> tuple[list[str], np.ndarray]:
    """Flat text queries and the class id of each, from {class: [prompts]}.
    Queries under `_ignore` (class -1) only compete for the boxes: OWLv2
    gives each box its best query, so "a pedestrian crosswalk" takes the
    zebra crossings that would otherwise be "speed bumps", and "a car" the
    vehicles the rule route already has; their detections are dropped."""
    texts, ids = [], []
    for name, items in prompts.items():
        for text in items:
            texts.append(str(text))
            ids.append(-1 if name == IGNORE else taxonomy.cid(name))
    return texts, np.asarray(ids, np.int32)


def label_image(shape: tuple[int, int],
                detections: list[interfaces.Detection],
                masks: np.ndarray | None, query_class: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """(class image (H, W) int32, -1 = none; score image (H, W) float32).
    `masks`: (N, H, W) bool per detection, or None to use the boxes."""
    labels = np.full(shape, -1, np.int32)
    scores = np.zeros(shape, np.float32)
    order = np.argsort([d.score for d in detections])  # Best painted last.
    for k in order:
        d = detections[k]
        if query_class[d.query] < 0:  # An absorbing query.
            continue
        if masks is not None:
            region = masks[k]
        else:
            region = np.zeros(shape, bool)
            x0, y0, x1, y1 = np.round(d.box).astype(int)
            region[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        labels[region] = query_class[d.query]
        scores[region] = d.score
    return labels, scores


def plausible(classes: np.ndarray, height: np.ndarray,
              bands: dict) -> np.ndarray:
    """Whether each lifted class lies in its [min, max] height band above
    the floor (classes without a band always do; -1 = none)."""
    ok = np.ones(len(classes), bool)
    for name, (low, high) in bands.items():
        mine = classes == taxonomy.cid(name)
        ok[mine] = (height[mine] >= float(low)) & (height[mine] <= float(high))
    return ok


def keep_detections(found, masks, query_class, shape, limits: dict):
    """Detections (and masks) whose mask is at most the class's share of the
    image (`limits`: {class: fraction, "default": fraction})."""
    if masks is None or not found:
        return found, masks
    names = {c.id: c.name for c in taxonomy.CLASSES}
    area = float(shape[0] * shape[1])
    keep = []
    for k, d in enumerate(found):
        c = query_class[d.query]
        limit = limits.get(names.get(int(c), ""), limits.get("default", 1.0))
        if c < 0 or masks[k].sum() <= float(limit) * area:
            keep.append(k)
    return [found[k] for k in keep], masks[keep]


def merge(rule_labels: np.ndarray, static: np.ndarray, lifted: np.ndarray,
          confidence: np.ndarray, override: list[str],
          min_confidence: float) -> tuple[np.ndarray, int]:
    """Rule labels with the confident lifted classes of the static points
    `static` written over the `override` classes. Returns (labels, count)."""
    labels = rule_labels.copy()
    allowed = np.isin(rule_labels[static],
                      [taxonomy.cid(n) for n in override])
    take = (lifted >= 0) & (confidence >= min_confidence) & allowed
    labels[static[take]] = lifted[take].astype(labels.dtype)
    return labels, int(take.sum())


def run(pose_set, point_map, models, cfg, out_dir: pathlib.Path,
        levels=None, log=print) -> dict:
    """Detections, lifted classes and their summary (out_dir/objects2d)."""
    section = cfg.objects2d
    texts, query_class = queries(section.prompts)
    keys = language_field.frame_keys(pose_set, max(1, int(section.frame_stride)))
    static = np.nonzero(~point_map.dynamic)[0] if point_map.dynamic is not None \
        else np.arange(len(point_map.xyz))
    cameras = {}
    records, previews = [], []
    preview_every = max(1, len(keys) // 4)
    start = time.time()

    def views():
        for n, key in enumerate(keys):
            data = pose_set.datasets[key.trip]
            if key.camera not in cameras:
                cameras[key.camera] = pose_set.rig(key.trip).cameras[
                    key.camera].make_model()
            frame = interfaces.Frame(key.trip_id, key.camera, key.stamp_ns,
                                     data.load_image(key.camera, key.stamp_ns))
            found = models.detector.detect(frame, texts)
            masks = (models.segmenter.masks_from_boxes(
                frame, np.stack([d.box for d in found]))
                if found and models.segmenter is not None else None)
            shape = frame.image.shape[:2]
            found, masks = keep_detections(found, masks, query_class, shape,
                                           dict(section.max_mask_fraction))
            labels, scores = label_image(shape, found, masks, query_class)
            records.append({"trip": key.trip_id, "camera": key.camera,
                            "stamp_ns": key.stamp_ns, "detections": [
                                [texts[d.query], round(d.score, 3),
                                 [round(float(v), 1) for v in d.box]]
                                for d in found]})
            if n % preview_every == 0 and len(previews) < 4:
                previews.append((frame.image, labels))
            exposure = pose_set.camera_exposure(key.trip, key.camera,
                                                key.stamp_ns)
            camera_from_world = (geometry.inv(exposure) if exposure.ndim == 2
                                 else np.stack([geometry.inv(e) for e in exposure]))
            yield cameras[key.camera], camera_from_world, labels, scores

    lifted, confidence, _ = lifting.lift_labels(
        np.ascontiguousarray(point_map.xyz[static], dtype=np.float64), views(),
        len(taxonomy.CLASSES), float(section.spacing), float(section.max_range),
        int(section.boundary), int(section.min_views))
    implausible = 0
    if levels is not None and section.get("height"):
        _, height, _ = levels.assign(point_map.xyz[static])
        ok = plausible(lifted, height, dict(section.height))
        implausible = int(((lifted >= 0) & ~ok).sum())
        lifted = np.where(ok, lifted, -1)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "points.npz", static=static,
                        labels=lifted.astype(np.int16),
                        confidence=confidence.astype(np.float32))
    (out_dir / "detections.json").write_text(json.dumps(records))
    _write_previews(previews, out_dir / "preview.jpg")
    names = {c.id: c.name for c in taxonomy.CLASSES}
    per_query = {}
    for r in records:
        for text, _, _ in r["detections"]:
            per_query[text] = per_query.get(text, 0) + 1
    good = (lifted >= 0) & (confidence >= float(section.min_confidence))
    points = {names[int(c)]: int(n) for c, n in zip(
        *np.unique(lifted[good], return_counts=True))}
    summary = {"images": len(keys), "seconds": round(time.time() - start, 1),
               "detections": per_query, "points": points,
               "implausible_height": implausible}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"[objects2d] {len(keys)} images, "
        f"{sum(per_query.values())} detections, confident points per class: "
        f"{points}; {implausible} out of their height band "
        f"({summary['seconds']:.0f}s)")
    return {"static": static, "labels": lifted, "confidence": confidence,
            "summary": summary}


def _write_previews(previews, path: pathlib.Path) -> None:
    if not previews:
        return
    import cv2
    rng = np.random.default_rng(3)
    palette = rng.integers(60, 255, (len(taxonomy.CLASSES), 3))
    tiles = []
    for image, labels in previews:
        over = image.copy()
        hit = labels >= 0
        over[hit] = (0.45 * over[hit] + 0.55 * palette[labels[hit]]).astype(
            np.uint8)
        tiles.append(cv2.resize(np.concatenate([image, over], axis=1), None,
                                fx=0.4, fy=0.4))
    width = min(t.shape[1] for t in tiles)
    cv2.imwrite(str(path), np.concatenate([t[:, :width] for t in tiles])[..., ::-1])
