"""Evaluation against reference data.

For synthetic datasets the reference is exact (gt/scene.json, gt/trip_XX/,
gt/calibration.json). The metric functions only take arrays / plain records,
so the same functions score a reconstruction against a handheld-scanner
ground-truth map once it is converted to the same records (columns, slots,
marking lines, points). The pose set (IMU bias, camera calibration, camera
exposures) and the language route are scored by pose_eval.py and
semantic_eval.py.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import shapely
from scipy import spatial

from garage_recon import config as config_lib
from garage_recon.data import dataset
from garage_recon.data.synthetic import ground_truth
from garage_recon.eval import pose_eval
from garage_recon.eval import semantic_eval
from garage_recon import geometry
from garage_recon.models import registry
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.semantics import taxonomy


# ----------------------------------------------------------- trajectory
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Rigid transform (no scale) aligning src points onto dst."""
    mean_src, mean_dst = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((dst - mean_dst).T @ (src - mean_src))
    sign = np.eye(3)
    sign[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ sign @ vt
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = mean_dst - rotation @ mean_src
    return out


def eval_trajectory(est_t, est_poses, gt_t, gt_poses) -> dict:
    """ATE (raw and Umeyama-aligned) and rotation errors at the GT stamps."""
    poses = geometry.interpolate_poses(est_t, est_poses, gt_t)
    trans, rot = geometry.pose_error(poses, gt_poses)
    align = umeyama(poses[:, :3, 3], gt_poses[:, :3, 3])
    aligned = np.linalg.norm(
        (poses[:, :3, 3] @ align[:3, :3].T + align[:3, 3])
        - gt_poses[:, :3, 3], axis=1)
    return {"ate_rmse": float(np.sqrt(np.mean(trans**2))),
            "ate_max": float(trans.max()),
            "ate_aligned_rmse": float(np.sqrt(np.mean(aligned**2))),
            "rot_mean_deg": float(np.degrees(rot.mean())),
            "rot_max_deg": float(np.degrees(rot.max())),
            "frames": int(len(gt_t))}


# ------------------------------------------------------------- geometry
def box_distance(points: np.ndarray, boxes: list[dict],
                 chunk: int = 200000) -> tuple[np.ndarray, np.ndarray]:
    """Unsigned distance of every point to the nearest box surface, index."""
    centers = np.array([b["center"] for b in boxes])
    halves = np.array([b["size"] for b in boxes]) / 2.0
    rotations = np.array([b["R"] for b in boxes])
    best = np.full(len(points), np.inf)
    arg = np.full(len(points), -1)
    for start in range(0, len(points), chunk):
        part = points[start:start + chunk]
        for k in range(len(boxes)):
            # World -> box frame (the columns of R are the box axes).
            local = (part - centers[k]) @ rotations[k]
            q = np.abs(local) - halves[k]
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            distance = np.abs(outside + inside)
            better = distance < best[start:start + chunk]
            best[start:start + chunk][better] = distance[better]
            arg[start:start + chunk][better] = k
    return best, arg


def eval_map(xyz: np.ndarray, static_boxes: list[dict], sample: int = 300000,
             seed: int = 0, sigma: np.ndarray | None = None) -> dict:
    """Distance of (sampled) map points to the true surfaces.

    With per-point `sigma`, also how often the error is within 1 / 2 sigma
    (about 68% / 95% if sigma is well calibrated).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xyz), min(sample, len(xyz)), replace=False)
    distance, _ = box_distance(xyz[idx], static_boxes)
    out = {"points": int(len(idx)), "median": float(np.median(distance)),
           "p90": float(np.percentile(distance, 90)),
           "p99": float(np.percentile(distance, 99)),
           "within_5cm": float((distance < 0.05).mean())}
    if sigma is not None:
        s = np.asarray(sigma, float)[idx]
        out["sigma_median"] = float(np.median(s))
        out["within_1_sigma"] = float((distance <= s).mean())
        out["within_2_sigma"] = float((distance <= 2 * s).mean())
    return out


# -------------------------------------------------------------- columns
def eval_columns(pred: list[dict], gt: list[dict], gate: float = 1.0) -> dict:
    """Column detection precision / recall and centre errors."""
    out = {"gt": len(gt), "pred": len(pred), "tp": 0}
    errors = []
    used = set()
    for g in gt:
        candidates = [
            (np.linalg.norm(np.subtract(p["center"], g["center"])), i)
            for i, p in enumerate(pred)
            if p["level"] == g["level"] and i not in used]
        if not candidates:
            continue
        distance, i = min(candidates)
        if distance <= gate:
            used.add(i)
            out["tp"] += 1
            errors.append(distance)
    errors = np.array(errors)
    out.update(_precision_recall(out["tp"], out["pred"], out["gt"]))
    out["center_err_mean"] = float(errors.mean()) if len(errors) else None
    out["center_err_p95"] = (float(np.percentile(errors, 95)) if len(errors)
                             else None)
    return out


# ------------------------------------------------------------- markings
def eval_markings(pred: list[dict], gt_lines: list[dict], tol: float = 0.15,
                  step: float = 0.05) -> dict:
    """Length-based recall / precision of marking centre lines."""

    def samples(p0, p1):
        n = max(2, int(np.linalg.norm(np.subtract(p1, p0)) / step) + 1)
        return np.linspace(p0, p1, n)

    out = {}
    for level in sorted({g["level"] for g in gt_lines}):
        truth = [g for g in gt_lines if g["level"] == level]
        predicted = [p for p in pred if p["level"] == level]
        truth_samples = np.vstack([
            samples([(g["x0"] + g["x1"]) / 2, g["y0"]],
                    [(g["x0"] + g["x1"]) / 2, g["y1"]]) for g in truth])
        pred_samples = (np.vstack([samples(p["p0"], p["p1"])
                                   for p in predicted])
                        if predicted else np.zeros((0, 2)))
        if len(pred_samples):
            to_pred = spatial.cKDTree(pred_samples).query(truth_samples)[0]
            to_truth = spatial.cKDTree(truth_samples).query(pred_samples)[0]
            recall = float((to_pred < tol).mean())
            precision = float((to_truth < tol).mean())
        else:
            recall = precision = 0.0
        out[f"level_{level}"] = {"recall": recall, "precision": precision,
                                 "segments": len(predicted)}
    return out


# ---------------------------------------------------------------- slots
def eval_slots(pred: list[dict], gt: list[dict], iou_th: float = 0.5) -> dict:
    """Slot matching by polygon IoU; corner and entrance errors."""
    tp, corner_errors, entrance_ok = 0, [], 0
    used = set()
    for g in gt:
        truth = shapely.Polygon(g["corners"])
        best, best_index = 0.0, -1
        for i, p in enumerate(pred):
            if i in used or p["level"] != g["level"]:
                continue
            candidate = shapely.Polygon(p["corners"])
            iou = (truth.intersection(candidate).area
                   / max(truth.union(candidate).area, 1e-9))
            if iou > best:
                best, best_index = iou, i
        if best >= iou_th:
            used.add(best_index)
            tp += 1
            pc = np.asarray(pred[best_index]["corners"])
            gc = np.asarray(g["corners"])
            corner_errors += list(np.linalg.norm(pc - gc, axis=1))
            entrance_ok += np.linalg.norm(
                (pc[0] + pc[1]) / 2 - (gc[0] + gc[1]) / 2) < 0.5
    errors = np.asarray(corner_errors)
    return {"gt": len(gt), "pred": len(pred), "tp": tp,
            **_precision_recall(tp, len(pred), len(gt)),
            "corner_err_mean": float(errors.mean()) if len(errors) else None,
            "corner_err_p95": (float(np.percentile(errors, 95))
                               if len(errors) else None),
            "entrance_correct": float(entrance_ok / tp) if tp else None,
            "inferred": int(sum(p.get("inferred", False) for p in pred))}


# --------------------------------------------------------------- arrows
def eval_arrows(pred: list[dict], gt: list[dict], gate: float = 1.0) -> dict:
    """Arrows matched by centre (within `gate` m, same level); direction
    correct when within 10 deg of the painted one."""
    tp, direction_ok, centre_errors = 0, 0, []
    used = set()
    for g in gt:
        centre = (np.asarray(g["tail"]) + np.asarray(g["tip"])) / 2
        best, best_index = gate, -1
        for i, p in enumerate(pred):
            if i in used or p["level"] != g["level"]:
                continue
            d = float(np.linalg.norm(np.asarray(p["center"]) - centre))
            if d < best:
                best, best_index = d, i
        if best_index < 0:
            continue
        used.add(best_index)
        tp += 1
        centre_errors.append(best)
        direction_ok += float(np.dot(pred[best_index]["direction"],
                                     g["direction"])) > np.cos(np.radians(10))
    return {"gt": len(gt), "pred": len(pred), "tp": tp,
            **_precision_recall(tp, len(pred), len(gt)),
            "direction_correct": direction_ok / tp if tp else None,
            "centre_err_median": (float(np.median(centre_errors))
                                  if centre_errors else None)}


# -------------------------------------------------------------- objects
def eval_static_vehicles(tracks: list[dict], parked: list[dict],
                         gate: float = 1.5) -> dict:
    """Parked cars of one trip vs static vehicle tracks of that trip."""
    by_key = {}
    for box in parked:  # Body and cabin boxes share `key`.
        by_key.setdefault(box["key"], box)
    gt = list(by_key.values())
    pred = [t["box"] for t in tracks
            if t["static"] and t["cls"] == "vehicle" and t["box"] is not None]
    tp, center_errors, yaw_errors = 0, [], []
    used = set()
    for g in gt:
        candidates = [
            (np.linalg.norm(np.subtract(p["center"][:2], g["center"][:2])), i)
            for i, p in enumerate(pred) if i not in used]
        if not candidates:
            continue
        distance, i = min(candidates)
        if distance <= gate:
            used.add(i)
            tp += 1
            center_errors.append(distance)
            dyaw = (pred[i]["yaw"] - g["yaw"]) % np.pi
            yaw_errors.append(min(dyaw, np.pi - dyaw))
    return {"gt": len(gt), "pred": len(pred), "tp": tp,
            **_precision_recall(tp, len(pred), len(gt)),
            "center_err_mean": (float(np.mean(center_errors)) if center_errors
                                else None),
            "yaw_err_mean_deg": (float(np.degrees(np.mean(yaw_errors)))
                                 if yaw_errors else None)}


def eval_moving(moving_boxes: dict, kf_stamp: dict, movers: list[dict],
                gate: float = 2.0) -> dict:
    """Moving GT objects vs moving track boxes.

    A GT object counts as tracked if some track stays within `gate` of it in
    at least half of the keyframes where both exist; also reports how many
    tracks it was split into.
    """
    out = []
    for mover in movers:
        hits: dict[int, list[bool]] = {}
        for k, boxes in moving_boxes.items():
            t = kf_stamp[int(k)]
            if not mover["t0"] <= t <= mover["t1"]:
                continue
            center = (np.asarray(mover["start"])[:2]
                      + np.asarray(mover["velocity"])[:2] * (t - mover["t0"]))
            for box in boxes:
                hits.setdefault(box["track_id"], []).append(
                    np.linalg.norm(np.asarray(box["center"][:2]) - center)
                    <= gate)
        good = [track for track, v in hits.items() if np.mean(v) >= 0.5]
        out.append({"instance": mover["instance"], "tracked": bool(good),
                    "tracks": len(good)})
    return {"objects": out, "tracked": sum(o["tracked"] for o in out),
            "gt": len(movers)}


def _precision_recall(tp: int, n_pred: int, n_gt: int) -> dict:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


# ------------------------------------------------------------ top level
STATIC_KINDS = ("slab", "roof", "wall", "column", "ramp", "parapet", "duct")


def evaluate(work_dir: str | pathlib.Path, gt_dir: str | pathlib.Path,
             log=print) -> dict:
    """Scores a work directory against a synthetic dataset's gt/."""
    work, gt_dir = pathlib.Path(work_dir), pathlib.Path(gt_dir)
    scene = json.loads((gt_dir / "scene.json").read_text())
    semantics = work / "semantics"
    report: dict = {"trajectory": {}}
    for trip_gt in sorted(gt_dir.glob("trip_*")):
        estimate = work / "trips" / trip_gt.name / "poses.tum"
        if estimate.exists():
            gt_t, gt_poses = geometry.read_tum(trip_gt / "poses.tum")
            est_t, est_poses = geometry.read_tum(estimate)
            report["trajectory"][trip_gt.name] = eval_trajectory(
                est_t, est_poses, gt_t, gt_poses)
    map_path = work / "map" / "point_map.npz"
    if map_path.exists():
        with np.load(map_path) as z:
            xyz, dynamic = z["xyz"], z["dynamic"]
            sigma = z["sigma"] if "sigma" in z.files else None
        boxes = [b for b in scene["boxes"] if b["kind"] in STATIC_KINDS]
        cars = []
        for trip_gt in sorted(gt_dir.glob("trip_*")):
            objects = json.loads((trip_gt / "objects.json").read_text())
            for car in objects["parked"]:
                cars.append({"center": car["center"], "size": car["size"],
                             "R": geometry.rot_z(car["yaw"])})
        report["map_accuracy"] = eval_map(
            xyz[~dynamic], boxes + cars,
            sigma=None if sigma is None else sigma[~dynamic])
        labels_path = semantics / "labels.npz"
        if labels_path.exists():
            with np.load(labels_path) as z:
                labels = z["labels"]
            structure = (np.isin(labels, list(taxonomy.STRUCTURE_IDS))
                         & ~dynamic)
            report["map_accuracy_structure"] = eval_map(xyz[structure], boxes)
    if (semantics / "structure.json").exists():
        report["columns"] = eval_columns(
            json.loads((semantics / "structure.json").read_text())["columns"],
            scene["columns"])
    if (semantics / "markings.json").exists():
        report["markings"] = eval_markings(
            json.loads((semantics / "markings.json").read_text()),
            scene["lines"])
    if (semantics / "arrows.json").exists() and scene.get("arrows"):
        report["arrows"] = eval_arrows(
            json.loads((semantics / "arrows.json").read_text()),
            scene["arrows"])
    if (semantics / "slots.json").exists():
        report["slots"] = eval_slots(
            json.loads((semantics / "slots.json").read_text()), scene["slots"])
    tracks_path = semantics / "objects" / "tracks.json"
    if tracks_path.exists():
        tracks = json.loads(tracks_path.read_text())
        report["static_vehicles"] = {}
        report["moving_objects"] = {}
        moving_path = semantics / "objects" / "moving.json"
        moving = (json.loads(moving_path.read_text()) if moving_path.exists()
                  else {})
        with np.load(work / "graph" / "kf_poses.npz") as z:
            kf_stamp = dict(enumerate(z["stamp"]))
            kf_trip = z["trip"]
        trip_names = [p.name for p in sorted((work / "trips").iterdir())]
        for trip_gt in sorted(gt_dir.glob("trip_*")):
            objects = json.loads((trip_gt / "objects.json").read_text())
            report["static_vehicles"][trip_gt.name] = eval_static_vehicles(
                [t for t in tracks if t["trip"] == trip_gt.name],
                objects["parked"])
            if trip_gt.name in trip_names:
                index = trip_names.index(trip_gt.name)
                boxes_of_trip = {k: v for k, v in moving.items()
                                 if kf_trip[int(k)] == index}
                report["moving_objects"][trip_gt.name] = eval_moving(
                    boxes_of_trip, kf_stamp, objects["moving"])
    if pose_set_lib.PoseSet.exists(work):
        report.update(_evaluate_pose_set(work, gt_dir))
    out = work / "eval"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))
    return report


def _evaluate_pose_set(work: pathlib.Path, gt_dir: pathlib.Path) -> dict:
    """Pose-set and language-route metrics of a work directory."""
    meta = json.loads((work / "pose_set" / "pose_set.json").read_text())
    datasets = [dataset.TripDataset(t["dataset"]) for t in meta["trips"]]
    pose_set = pose_set_lib.PoseSet.load(work, datasets)
    world = ground_truth.SyntheticWorld(gt_dir)
    frame_poses = {}
    for trip in pose_set.trips:
        with np.load(work / "trips" / trip.trip_id / "poses.npz") as z:
            frame_poses[trip.trip_id] = (z["stamps"], z["vehicle"])
    out = {"poses": pose_eval.evaluate_pose_set(pose_set, world, frame_poses)}
    used = work / "config.used.yaml"
    cfg = config_lib.load_config(used if used.exists() else None)
    if (work / "semantics" / "language" / "labels.npz").exists():
        models = registry.build(cfg.models, gt_dir)
        out["language"] = semantic_eval.evaluate_language(
            work, pose_set, world, models.encoder, cfg)
    return out
