"""Slot numbers and zones read by OCR (the "numbers" stage).

Every text the OCR model finds in a camera frame is lifted onto the floor:
the ray through the text centre, from the mid-row pose of the exposure,
meets the floor plane of each level, and a hit within `max_distance` of a
slot of that level votes for its text (the slot whose entrance is nearest).
A slot takes its most voted text; each row's numbers are then repaired with
the consecutive-numbering rule (rules.fix_slot_numbers). "C-105" or "C105"
splits into zone C and number 105.
"""

from __future__ import annotations

import collections
import re

import numpy as np
import shapely

from garage_recon.models import interfaces
from garage_recon.semantics import rules
from garage_recon.semantics import slots as slots_lib

_TEXT = re.compile(r"(?:([A-Z]+)\W*)?(\d+)\s*$")


def split_text(text: str) -> tuple[str | None, str | None]:
    """(zone, number) of a slot text such as "C-105", "C105" or "105"."""
    m = _TEXT.search(text.strip().upper())
    if not m:
        return None, None
    return m.group(1), m.group(2)


def floor_hits(camera, world_from_camera: np.ndarray, uv: np.ndarray,
               floor_z: list[float], max_range: float
               ) -> list[tuple[int, np.ndarray]]:
    """(level, xyz) where the ray through pixel `uv` meets each level's
    floor plane within `max_range`."""
    rays, valid = camera.unproject(np.asarray(uv, float).reshape(1, 2))
    if not valid[0]:
        return []
    direction = world_from_camera[:3, :3] @ rays[0]
    origin = world_from_camera[:3, 3]
    hits = []
    if abs(direction[2]) < 1e-6:
        return hits
    for level, z in enumerate(floor_z):
        s = (z - origin[2]) / direction[2]
        if 0.0 < s <= max_range:
            hits.append((level, origin + s * direction))
    return hits


def _row_order(slots: list[slots_lib.Slot]) -> list[int]:
    """Indices of a row's slots in order along the row."""
    centers = np.array([np.mean(s.corners, axis=0) for s in slots])
    if len(centers) < 2:
        return list(range(len(slots)))
    _, _, vt = np.linalg.svd(centers - centers.mean(axis=0))
    return list(np.argsort(centers @ vt[0]))


def read_numbers(recon, slots: list[slots_lib.Slot], cfg, log=print) -> dict:
    """Sets `number`, `zone` and `text_votes` of the slots in place.

    Returns:
        Summary: frames read, texts, slots with a number before and after the
        row repair.
    """
    section = cfg.numbers
    ocr = recon.models.ocr
    if ocr is None or not slots:
        log("[numbers] no OCR model or no slots")
        return {"frames": 0}
    pose_set = recon.pose_set()
    floor_z = list(np.asarray(recon.levels().z, float))
    polygons = [shapely.Polygon(s.corners).buffer(section.max_distance)
                for s in slots]
    entrances = [np.mean(np.asarray(s.corners)[:2], axis=0) for s in slots]
    votes = [collections.Counter() for _ in slots]  # Text -> summed score.
    readings = [0] * len(slots)
    needs_images = getattr(ocr, "needs_images", True)
    frames = texts = 0
    for t, data in enumerate(recon.datasets):
        rig = pose_set.rig(t)
        for name, calib in rig.cameras.items():
            camera = calib.make_model()
            stamps = data.camera_stamps_ns(name)[::int(section.frame_stride)]
            for ns in stamps:
                image = data.load_image(name, int(ns)) if needs_images else None
                found = ocr.read(interfaces.Frame(data.trip_id, name, int(ns),
                                                  image))
                frames += 1
                if not found:
                    continue
                pose = pose_set.camera_center_pose(t, name, int(ns))
                for result in found:
                    texts += 1
                    uv = np.asarray(result.polygon, float).mean(axis=0)
                    for level, xyz in floor_hits(camera, pose, uv, floor_z,
                                                 section.max_range):
                        point = shapely.Point(xyz[0], xyz[1])
                        best, best_d = -1, np.inf
                        for k, slot in enumerate(slots):
                            if slot.level != level or not polygons[k].contains(
                                    point):
                                continue
                            d = np.linalg.norm(entrances[k] - xyz[:2])
                            if d < best_d:
                                best, best_d = k, d
                        if best >= 0:
                            votes[best][result.text.strip().upper()] += float(
                                result.score)
                            readings[best] += 1
    read = 0
    for slot, counter, count in zip(slots, votes, readings):
        slot.text_votes = count
        if not counter or counter.most_common(1)[0][1] < section.min_score:
            continue
        slot.zone, slot.number = split_text(counter.most_common(1)[0][0])
        read += slot.number is not None
    # Row repair with the consecutive-numbering rule.
    rows = collections.defaultdict(list)
    for k, slot in enumerate(slots):
        rows[(slot.level, slot.row)].append(k)
    for members in rows.values():
        order = [members[i] for i in _row_order([slots[k] for k in members])]
        numbers = [slots[k].number for k in order]
        fixed = rules.fix_slot_numbers([slots[k] for k in order], numbers)
        zones = collections.Counter(slots[k].zone for k in order
                                    if slots[k].zone)
        zone = zones.most_common(1)[0][0] if zones else None
        for k, number in zip(order, fixed):
            slots[k].number = number
            if zone and slots[k].number is not None:
                slots[k].zone = zone
    summary = {"frames": frames, "texts": texts, "slots": len(slots),
               "read": read,
               "numbered": int(sum(s.number is not None for s in slots))}
    log(f"[numbers] {frames} frames, {texts} texts; {read} of {len(slots)} "
        f"slots read, {summary['numbered']} numbered after the row repair")
    return summary
