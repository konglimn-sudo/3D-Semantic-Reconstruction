"""Slot numbers and zones read by OCR (the "numbers" stage).

Every text the OCR model finds in a camera frame is lifted onto the floor:
the ray through the text centre, from the mid-row pose of the exposure,
meets the floor plane of each level, and a hit within `max_distance` of a
slot of that level votes for its text (the slot whose entrance is nearest).
A slot takes its most voted text; each row's numbers are then repaired with
the consecutive-numbering rule (rules.fix_slot_numbers). "C-105" or "C105"
splits into zone C and number 105.

The floor is the level's floor grid, not one height per level (a level may
rise along an aisle). Every text is also placed in 3D where the LiDAR map
surface is at its pixel (signs hang on walls and columns, not on the floor);
readings of one text within `landmark_radius` merge into a landmark (exit,
gate, zone, level or other text) for the navigation map (README 5.2 / 6).
The OCR reads a sign's arrows as glyphs ("出口个", "《中央区", "5A》"), so
they are stripped first; landmarks of one kind on one sign (within the
radius) then merge when they are exits ("EXIT" and "出口" side by side) or
their texts are similar (a misreading, "中区" beside "中央区"), keeping
the best text and the others in `texts`.
"""

from __future__ import annotations

import collections
import difflib
import re

import cv2
import numpy as np
import shapely

from garage_recon import _core, geometry
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
               levels, max_range: float, step: float = 0.05
               ) -> list[tuple[int, np.ndarray]]:
    """(level, xyz) where the ray through pixel `uv` first goes below each
    level's floor grid (LevelModel.floor_z) within `max_range`."""
    rays, valid = camera.unproject(np.asarray(uv, float).reshape(1, 2))
    if not valid[0]:
        return []
    direction = world_from_camera[:3, :3] @ rays[0]
    origin = world_from_camera[:3, 3]
    if direction[2] >= -1e-6:  # Not looking down: no floor ahead.
        return []
    s = np.arange(step, max_range + step / 2, step)
    points = origin + s[:, None] * direction
    hits = []
    for level in range(levels.n):
        above = points[:, 2] - levels.floor_z(points[:, :2], level)
        below = np.nonzero(above <= 0.0)[0]
        if len(below) == 0 or below[0] == 0:
            continue
        k = below[0]  # Linear between the last sample above and this one.
        f = above[k - 1] / (above[k - 1] - above[k])
        hits.append((level, points[k - 1] + f * (points[k] - points[k - 1])))
    return hits


# Slot attributes and their texts (painted in the slot or on a sign).
SLOT_ATTRIBUTES = {
    "ev": r"充电|新能源|电动车|EV|CHARG",
    "accessible": r"无障碍|残疾|残障|ACCESSIBLE|DISABLED|♿",
    "reserved": r"专用|私家|固定车位|业主|VIP|RESERVED|PRIVATE",
}


def text_attributes(text: str) -> list[str]:
    """Slot attributes a text names ("充电车位" -> ["ev"])."""
    t = re.sub(r"\s+", "", str(text).upper())
    return [name for name, pattern in SLOT_ATTRIBUTES.items()
            if re.search(pattern, t)]


def text_kind(text: str) -> str:
    """no_entry | one_way | exit | gate | attribute | level | zone | text."""
    t = re.sub(r"\s+", "", text.upper())
    if re.search(r"禁止驶入|禁止进入|严禁驶入|NOENTRY|DONOTENTER", t):
        return "no_entry"
    if re.search(r"单行|ONEWAY", t):
        return "one_way"
    if re.search(r"EXIT|出口", t):
        return "exit"
    if text_attributes(t):
        return "attribute"
    if re.search(r"\d+号门|GATE", t):
        return "gate"
    if re.fullmatch(r"(B|L|F)\d{1,2}|负?[一二三四五六七八九十]+层", t):
        return "level"
    if re.fullmatch(r"\d{0,2}[A-Z]{1,2}\d{0,2}", t) or t.endswith("区"):
        return "zone"
    return "text"


# Arrow glyphs and their usual misreadings at either end of a sign text.
_ARROWS = r"^[《》<>→←↑↓↗↖↘↙^个十]+|[《》<>→←↑↓↗↖↘↙^个十]+$"


def normalize_text(text: str) -> str:
    """Upper case, no spaces, no arrow glyphs at either end."""
    return re.sub(_ARROWS, "", re.sub(r"\s+", "", text.strip().upper()))


def merge_signs(marks: list[dict], radius: float,
                similarity: float = 0.6) -> list[dict]:
    """Landmarks of one kind within `radius` of a better one merged into it
    when both are exits or their texts are `similarity` alike; the merged
    landmark keeps the better text and position, adds the views and scores,
    and lists every text in `texts`."""
    out = []
    for mark in sorted(marks, key=lambda m: -m["score"]):
        xyz = np.asarray(mark["xyz"])
        for sign in out:
            if (sign["kind"] == mark["kind"]
                    and np.linalg.norm(np.asarray(sign["xyz"]) - xyz) <= radius
                    and (mark["kind"] == "exit" or difflib.SequenceMatcher(
                        None, sign["text"], mark["text"]).ratio() >= similarity)):
                sign["views"] += mark["views"]
                sign["score"] = round(sign["score"] + mark["score"], 2)
                if mark["text"] not in sign["texts"]:
                    sign["texts"].append(mark["text"])
                break
        else:
            out.append({**mark, "texts": [mark["text"]]})
    return out


def _facing(sign: np.ndarray, cameras: list) -> list[float] | None:
    """Unit horizontal direction from a sign towards the cameras that read
    it (the side it faces), None without cameras."""
    rays = [np.asarray(c, float)[:2] - sign[:2] for c in cameras
            if c is not None]
    rays = [r / np.linalg.norm(r) for r in rays if np.linalg.norm(r) > 1e-6]
    if not rays:
        return None
    mean = np.mean(rays, axis=0)
    norm = float(np.linalg.norm(mean))
    return None if norm < 0.3 else np.round(mean / norm, 3).tolist()


def text_landmarks(observations: list[tuple], radius: float, min_views: int,
                   min_score: float, similarity: float = 0.6) -> list[dict]:
    """Readings (text, xyz, score[, camera xyz]) of one text within `radius`
    merged into landmarks (best readings first); kept with `min_views`
    readings and a summed score of `min_score`; then one landmark per sign
    (merge_signs). With the cameras, `facing` is the horizontal direction
    the sign faces (towards its readers)."""
    groups = collections.defaultdict(list)
    for text, xyz, score, *camera in observations:
        key = normalize_text(text)
        if len(key) >= 2:  # Single characters are mostly misreadings.
            groups[key].append((np.asarray(xyz, float), float(score),
                                camera[0] if camera else None))
    out = []
    for key, readings in groups.items():
        clusters = []  # [points, scores, cameras]
        for xyz, score, camera in sorted(readings, key=lambda r: -r[1]):
            for c in clusters:
                if np.linalg.norm(np.median(c[0], axis=0) - xyz) <= radius:
                    c[0].append(xyz)
                    c[1].append(score)
                    c[2].append(camera)
                    break
            else:
                clusters.append([[xyz], [score], [camera]])
        for points, scores, cameras in clusters:
            if len(points) >= min_views and sum(scores) >= min_score:
                center = np.median(points, axis=0)
                mark = {"text": key, "kind": text_kind(key),
                        "xyz": center.round(3).tolist(),
                        "views": len(points),
                        "score": round(float(sum(scores)), 2)}
                facing = _facing(center, cameras)
                if facing is not None:
                    mark["facing"] = facing
                out.append(mark)
    return sorted(merge_signs(out, radius, similarity),
                  key=lambda l: -l["score"])


def _range_image(camera, camera_from_world, local, cfg) -> np.ndarray:
    """LiDAR-map range of every pixel (z-buffer of the static points)."""
    rng, _, _ = _core.render_points(
        camera, camera_from_world, local, None,
        float(cfg.map.voxel), 1, 6, 0.3, float(cfg.numbers.text_max_range))
    return rng


def _surface_range(rng: np.ndarray, polygon) -> float:
    """Median map range inside a text polygon (NaN if none)."""
    mask = np.zeros(rng.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    values = rng[(mask > 0) & (rng > 0)]
    return float(np.median(values)) if len(values) else float("nan")


def _row_order(slots: list[slots_lib.Slot]) -> list[int]:
    """Indices of a row's slots in order along the row."""
    centers = np.array([np.mean(s.corners, axis=0) for s in slots])
    if len(centers) < 2:
        return list(range(len(slots)))
    _, _, vt = np.linalg.svd(centers - centers.mean(axis=0))
    return list(np.argsort(centers @ vt[0]))


def read_numbers(recon, slots: list[slots_lib.Slot], cfg, log=print,
                 cache=None) -> dict:
    """Sets `number`, `zone` and `text_votes` of the slots in place.

    `cache`: a JSON file of the raw OCR results per frame, read if present
    and written after (OCR is the slow part; delete it after changing the
    OCR model).

    Returns:
        Summary: frames read, texts, slots with a number before and after the
        row repair.
    """
    section = cfg.numbers
    ocr = recon.models.ocr
    if ocr is None:
        log("[numbers] no OCR model")
        return {"frames": 0, "landmark_list": []}
    pose_set = recon.pose_set()
    levels = recon.levels()
    point_map = recon.point_map()
    static = np.ascontiguousarray(
        point_map.xyz if point_map.dynamic is None
        else point_map.xyz[~point_map.dynamic], dtype=np.float64)
    index = _core.PointCloudIndex(static)
    observations = []  # (text, xyz, score) of every reading on a surface.
    polygons = [shapely.Polygon(s.corners).buffer(section.max_distance)
                for s in slots]
    entrances = [np.mean(np.asarray(s.corners)[:2], axis=0) for s in slots]
    votes = [collections.Counter() for _ in slots]  # Text -> summed score.
    attribute_votes = [collections.Counter() for _ in slots]
    readings = [0] * len(slots)
    needs_images = getattr(ocr, "needs_images", True)
    import json
    import pathlib
    cache = pathlib.Path(cache) if cache else None
    cached = json.loads(cache.read_text()) if cache and cache.exists() else {}
    fresh = {}
    frames = texts = 0
    for t, data in enumerate(recon.datasets):
        rig = pose_set.rig(t)
        for name, calib in rig.cameras.items():
            camera = calib.make_model()
            stamps = data.camera_stamps_ns(name)[::int(section.frame_stride)]
            for ns in stamps:
                key = f"{data.trip_id}/{name}/{int(ns)}"
                if key in cached:
                    found = [interfaces.TextResult(t, np.asarray(p, np.float32), s)
                             for t, p, s in cached[key]]
                else:
                    image = (data.load_image(name, int(ns)) if needs_images
                             else None)
                    found = ocr.read(interfaces.Frame(data.trip_id, name,
                                                      int(ns), image))
                fresh[key] = [[r.text, np.asarray(r.polygon).round(1).tolist(),
                               round(float(r.score), 3)] for r in found]
                frames += 1
                if cache is not None and frames % 100 == 0:  # Kept if stopped.
                    cache.write_text(json.dumps({**cached, **fresh},
                                                ensure_ascii=False))
                if not found:
                    continue
                pose = pose_set.camera_center_pose(t, name, int(ns))
                exposure = pose_set.camera_exposure(t, name, int(ns))
                camera_from_world = (
                    geometry.inv(exposure) if exposure.ndim == 2
                    else np.stack([geometry.inv(e) for e in exposure]))
                local = np.ascontiguousarray(static[index.radius_search(
                    pose[:3, 3], float(section.text_max_range))])
                # One z-buffer per frame, shared by its texts.
                rng = (_range_image(camera, camera_from_world, local, cfg)
                       if len(local) else None)
                for result in found:
                    texts += 1
                    uv = np.asarray(result.polygon, float).mean(axis=0)
                    rays, ok = camera.unproject(uv.reshape(1, 2))
                    distance = (_surface_range(rng, result.polygon)
                                if ok[0] and rng is not None else float("nan"))
                    if np.isfinite(distance):
                        observations.append((
                            result.text, pose[:3, 3] + distance * (
                                pose[:3, :3] @ rays[0]), result.score,
                            pose[:3, 3]))
                    for level, xyz in floor_hits(camera, pose, uv, levels,
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
                        if best < 0:
                            continue
                        named = text_attributes(result.text)
                        for attribute in named:  # "充电", not a number.
                            attribute_votes[best][attribute] += float(
                                result.score)
                        if not named:
                            votes[best][result.text.strip().upper()] += float(
                                result.score)
                            readings[best] += 1
    for slot, counter in zip(slots, attribute_votes):
        slot.attributes = sorted(
            set(slot.attributes) | {a for a, score in counter.items()
                                    if score >= section.min_score})
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
    if cache is not None:
        cache.write_text(json.dumps({**cached, **fresh}, ensure_ascii=False))
    landmarks = text_landmarks(observations, float(section.landmark_radius),
                               int(section.landmark_min_views),
                               float(section.landmark_min_score),
                               float(section.get("landmark_similarity", 0.6)))
    if landmarks:
        level, height, _ = levels.assign(np.array([l["xyz"] for l in landmarks]))
        for l, lv, h in zip(landmarks, level, height):
            l["level"], l["height"] = int(lv), round(float(h), 2)
    kinds = collections.Counter(l["kind"] for l in landmarks)
    summary = {"frames": frames, "texts": texts, "slots": len(slots),
               "read": read,
               "attributes": dict(collections.Counter(
                   a for s in slots for a in s.attributes)),
               "numbered": int(sum(s.number is not None for s in slots)),
               "landmarks": dict(kinds)}
    log(f"[numbers] {frames} frames, {texts} texts; {read} of {len(slots)} "
        f"slots read, {summary['numbered']} numbered after the row repair; "
        f"{len(landmarks)} text landmarks {dict(kinds)}")
    summary["landmark_list"] = landmarks
    return summary
