"""Evaluation of the language route against synthetic ground truth.

* Point labels: classes of the static map points against the true surface
  classes (accuracy, per-class IoU, mIoU); the rule-based labels of the
  geometric route are scored the same way for comparison.
* Identities: purity (share of an identity's points on its dominant true
  segment) and fragmentation (identities per true segment).
* Frames: labels rendered from the 3D field into sampled images against the
  true label images (mIoU), and the per-image 2D predictions of the same
  images (the single-view baseline the 3D aggregation should beat).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from garage_recon import _core
from garage_recon import geometry
from garage_recon.data.synthetic import ground_truth
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.recon import static_map
from garage_recon.semantics import language_field
from garage_recon.semantics import taxonomy

# Classes the language route is expected to produce on the synthetic garage.
EVALUATED = ("floor", "ramp", "ceiling", "wall", "column", "parking_line",
             "vehicle", "mep")


def _ious(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    out = {}
    for name in EVALUATED:
        c = taxonomy.cid(name)
        union = np.sum((predicted == c) | (truth == c))
        if np.sum(truth == c) == 0:
            continue
        out[name] = float(np.sum((predicted == c) & (truth == c)) / union)
    return out


def label_scores(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Accuracy and IoUs over points with a true class."""
    known = truth > 0
    ious = _ious(predicted[known], truth[known])
    return {
        "points": int(known.sum()),
        "accuracy": float((predicted[known] == truth[known]).mean())
        if known.any() else None,
        "labelled": float((predicted[known] > 0).mean()) if known.any()
        else None,
        "iou": ious,
        "miou": float(np.mean(list(ious.values()))) if ious else None,
    }


def identity_scores(identity: np.ndarray, segment: np.ndarray) -> dict:
    """Purity of identities and fragmentation of true segments."""
    valid = (identity >= 0) & (segment >= 0)
    if not valid.any():
        return {}
    pairs = np.stack([identity[valid], segment[valid]], axis=1)
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    dominant = {}
    for (ident, seg), count in zip(unique, counts):
        if count > dominant.get(ident, (0, -1))[0]:
            dominant[ident] = (count, seg)
    purity = sum(c for c, _ in dominant.values()) / valid.sum()
    per_segment = {}
    segment_total = {}
    for (ident, seg), count in zip(unique, counts):
        segment_total[seg] = segment_total.get(seg, 0) + count
    for (ident, seg), count in zip(unique, counts):
        if count >= 0.1 * segment_total[seg]:
            per_segment[seg] = per_segment.get(seg, 0) + 1
    fragments = np.asarray(list(per_segment.values()))
    return {
        "identities": int(len(dominant)),
        "true_segments": int(len(segment_total)),
        "purity": float(purity),
        "identities_per_segment_mean": float(fragments.mean()),
        "segments_split": float((fragments > 1).mean()),
    }


def frame_scores(pose_set: pose_set_lib.PoseSet,
                 world: ground_truth.SyntheticWorld, work: pathlib.Path,
                 encoder, min_score: float, spacing: float,
                 max_range: float, frames_per_camera: int = 6) -> dict:
    """mIoU of rendered 3D labels and of per-image 2D predictions."""
    semantics = work / "semantics"
    with np.load(semantics / "language" / "labels.npz") as z:
        labels = z["labels"]
    point_map = static_map.load_point_map(work / "map")
    static = ~point_map.dynamic
    xyz = point_map.xyz[static]
    point_labels = labels[static].astype(np.int32)
    class_ids, class_embeddings = language_field.class_text_embeddings(encoder)
    keys = language_field.frame_keys(pose_set, 1)
    by_camera: dict[tuple, list] = {}
    for key in keys:
        by_camera.setdefault((key.trip, key.camera), []).append(key)
    rendered_all, predicted_all, truth_all = [], [], []
    index = _core.PointCloudIndex(xyz)
    for frames in by_camera.values():
        pick = np.linspace(0, len(frames) - 1,
                           min(frames_per_camera, len(frames))).astype(int)
        for key in (frames[i] for i in pick):
            path = language_field.feature_path(semantics / "features", key)
            if not path.exists():
                continue
            masks, mask_ids, embeddings = language_field.load_features(path)
            predicted_class, _ = language_field.classify(
                embeddings, class_ids, class_embeddings, min_score)
            predicted = np.zeros(masks.shape, np.int64)
            for mask_id, cls in zip(mask_ids, predicted_class):
                predicted[masks == mask_id] = cls
            camera = pose_set.rig(key.trip).cameras[key.camera].make_model()
            exposure = pose_set.camera_exposure(key.trip, key.camera,
                                                key.stamp_ns)
            center = np.asarray(exposure).reshape(-1, 4, 4).mean(axis=0)[:3, 3]
            near = index.radius_search(center, max_range + 1.0)
            _, rendered, _ = _core.render_points(
                camera, geometry.inv(exposure), xyz[near], point_labels[near],
                spacing, 1, 3, 0.1, max_range)
            truth = world.render(key.trip_id, key.camera,
                                 key.stamp_ns)["label"].copy()
            truth[truth == taxonomy.cid("vehicle_moving")] = 0  # Not static.
            rendered_all.append(rendered.ravel())
            predicted_all.append(predicted.ravel())
            truth_all.append(truth.ravel())
    if not truth_all:
        return {}
    truth = np.concatenate(truth_all)
    rendered = np.concatenate(rendered_all)
    predicted = np.concatenate(predicted_all)
    # Both on every pixel with a true static class; no label counts as wrong.
    known = truth > 0
    rendered_ious = _ious(rendered[known], truth[known])
    single_ious = _ious(predicted[known], truth[known])
    return {
        "frames": len(truth_all),
        "rendered_3d": {"iou": rendered_ious,
                        "miou": float(np.mean(list(rendered_ious.values()))),
                        "labelled": float((rendered[known] > 0).mean())},
        "single_view_2d": {"iou": single_ious,
                           "miou": float(np.mean(list(single_ious.values()))),
                           "labelled": float((predicted[known] > 0).mean())},
    }


def evaluate_language(work: pathlib.Path, pose_set: pose_set_lib.PoseSet,
                      world: ground_truth.SyntheticWorld, encoder,
                      cfg) -> dict:
    """All language-route metrics (empty if the route did not run)."""
    semantics = work / "semantics"
    labels_path = semantics / "language" / "labels.npz"
    if not labels_path.exists():
        return {}
    point_map = static_map.load_point_map(work / "map")
    xyz, dynamic = point_map.xyz, point_map.dynamic
    with np.load(labels_path) as z:
        language_labels, identity = z["labels"], z["identity"]
    static = ~dynamic
    truth = world.point_truth(xyz[static])
    report = {"points": label_scores(language_labels[static], truth["cls"])}
    rules_path = semantics / "labels.npz"
    if rules_path.exists():
        with np.load(rules_path) as z:
            rule_labels = z["labels"]
        report["rules_points"] = label_scores(rule_labels[static],
                                              truth["cls"])
    report["identities"] = identity_scores(identity[static], truth["segment"])
    summary_path = semantics / "language" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        report["cross_view_consistency"] = summary.get(
            "cross_view_consistency_mean")
    if encoder is not None:
        report["frames"] = frame_scores(
            pose_set, world, work, encoder, float(cfg.language.min_score),
            float(cfg.map.voxel), float(cfg.identities.max_range))
    report["point_miou"] = report["points"]["miou"]
    if report.get("frames"):
        report["frame_miou_3d"] = report["frames"]["rendered_3d"]["miou"]
        report["frame_miou_2d"] = report["frames"]["single_view_2d"]["miou"]
    return report
