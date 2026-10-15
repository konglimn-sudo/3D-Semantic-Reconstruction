"""Language route of the semantics: ILGS adapted to garage scale.

2D models decide what a thing is; the 3D identity field makes every view
agree.

features    Per image: class-agnostic masks (Segmenter2D) and one language
            embedding per mask (LanguageEncoder).
identities  The masks of all images are associated in 3D through the shared
            poses and the LiDAR map (C++ IdentityAssociator) instead of a
            video tracker; every map point gets an identity.
language    One embedding per identity: the view-weighted mean of its masks'
            embeddings (close, unclipped views weigh more), classified
            against the taxonomy's text prompts. Map points inherit the class
            of their identity. Every image is then rendered from the 3D
            labels and compared with its own 2D prediction: the cross-view
            consistency rate.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import cv2
import numpy as np
from scipy import spatial

from garage_recon import _core
from garage_recon import geometry
from garage_recon.models import interfaces
from garage_recon.models import registry
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.semantics import taxonomy


@dataclasses.dataclass(frozen=True)
class FrameKey:
    trip: int
    trip_id: str
    camera: str
    stamp_ns: int


def frame_keys(pose_set: pose_set_lib.PoseSet, stride: int) -> list[FrameKey]:
    """Fisheye images of every trip in time order (every `stride`-th)."""
    keys = []
    for t, data in enumerate(pose_set.datasets):
        for camera in pose_set.rig(t).fisheye_cameras:
            for ns in data.camera_stamps_ns(camera)[::stride]:
                keys.append(FrameKey(t, data.trip_id, camera, int(ns)))
    return sorted(keys, key=lambda k: (k.trip, k.stamp_ns, k.camera))


def feature_path(directory: pathlib.Path, key: FrameKey) -> pathlib.Path:
    return directory / key.trip_id / key.camera / f"{key.stamp_ns}.npz"


def extract_features(pose_set: pose_set_lib.PoseSet,
                     models: registry.ModelSuite, keys: list[FrameKey],
                     directory: pathlib.Path) -> int:
    """Masks and mask embeddings of every image. Returns the mask count."""
    needs_images = (getattr(models.segmenter, "needs_images", True)
                    or getattr(models.encoder, "needs_images", True))
    total = 0
    for key in keys:
        data = pose_set.datasets[key.trip]
        frame = interfaces.Frame(
            key.trip_id, key.camera, key.stamp_ns,
            data.load_image(key.camera, key.stamp_ns) if needs_images
            else None)
        masks = models.segmenter.segment(frame)
        embeddings = models.encoder.encode_masks(frame, masks)
        ids = sorted(embeddings)
        masks = np.where(np.isin(masks, ids), masks, -1).astype(np.int32)
        path = feature_path(directory, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, masks=masks, mask_ids=np.asarray(ids, np.int32),
            embeddings=(np.stack([embeddings[i] for i in ids]) if ids
                        else np.zeros((0, models.encoder.dimension))
                        ).astype(np.float16))
        total += len(ids)
    return total


def load_features(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray,
                                                np.ndarray]:
    """(masks (H, W), mask ids (M,), embeddings (M, D) float32)."""
    with np.load(path) as z:
        return z["masks"], z["mask_ids"], z["embeddings"].astype(np.float32)


def carrier_points(xyz: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one map point per voxel (the identity carriers)."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def associate(pose_set: pose_set_lib.PoseSet, carriers: np.ndarray,
              normals: np.ndarray | None, keys: list[FrameKey],
              directory: pathlib.Path, config: _core.IdentityConfig) -> dict:
    """Associates the masks of all images through the carrier points.

    `normals` (oriented towards the sensors) cull back faces.

    Returns:
        Dict with point_identity / point_confidence per carrier, and the mask
        table (view, mask, identity, points, pixels, mean_range,
        touches_border) whose `view` indexes `keys`.
    """
    associator = _core.IdentityAssociator(carriers, normals, config)
    cameras = {}
    for key in keys:
        if key.camera not in cameras:
            cameras[key.camera] = pose_set.rig(key.trip).cameras[
                key.camera].make_model()
        masks, _, _ = load_features(feature_path(directory, key))
        exposure = pose_set.camera_exposure(key.trip, key.camera,
                                            key.stamp_ns)
        associator.add_view(cameras[key.camera], exposure, masks)
    return associator.finish()


def class_text_embeddings(encoder: interfaces.LanguageEncoder
                          ) -> tuple[np.ndarray, np.ndarray]:
    """(class ids (C,), unit embeddings (C, D)), averaged over prompts."""
    ids, embeddings = [], []
    for name, prompts in taxonomy.PROMPTS.items():
        mean = encoder.encode_text(list(prompts)).mean(axis=0)
        ids.append(taxonomy.cid(name))
        embeddings.append(mean / max(np.linalg.norm(mean), 1e-12))
    return np.asarray(ids), np.stack(embeddings)


def classify(embeddings: np.ndarray, class_ids: np.ndarray,
             class_embeddings: np.ndarray,
             min_score: float) -> tuple[np.ndarray, np.ndarray]:
    """Best class (0 if below `min_score`) and its cosine per embedding."""
    if len(embeddings) == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    scores = embeddings @ class_embeddings.T
    best = np.argmax(scores, axis=1)
    score = scores[np.arange(len(best)), best]
    labels = np.where(score >= min_score, class_ids[best], taxonomy.UNLABELED)
    return labels, score


def identity_embeddings(association: dict, keys: list[FrameKey],
                        directory: pathlib.Path, section) -> dict:
    """View-weighted mean embedding of every identity.

    Returns:
        Dict of identity ids (I,), embeddings (I, D), mask counts (I,).
    """
    masks = association["masks"]
    sums: dict[int, np.ndarray] = {}
    weights: dict[int, float] = {}
    counts: dict[int, int] = {}
    by_view: dict[int, list[int]] = {}
    for row in range(len(masks["view"])):
        if masks["identity"][row] >= 0:
            by_view.setdefault(int(masks["view"][row]), []).append(row)
    for view, rows in by_view.items():
        _, mask_ids, embeddings = load_features(feature_path(directory,
                                                             keys[view]))
        lookup = {int(m): i for i, m in enumerate(mask_ids)}
        for row in rows:
            k = lookup.get(int(masks["mask"][row]))
            if k is None:
                continue
            identity = int(masks["identity"][row])
            weight = (np.sqrt(masks["points"][row])
                      / (1.0 + masks["mean_range"][row]
                         / float(section.range_scale)))
            if masks["touches_border"][row]:
                weight *= float(section.border_weight)
            sums[identity] = sums.get(identity, 0.0) + weight * embeddings[k]
            weights[identity] = weights.get(identity, 0.0) + weight
            counts[identity] = counts.get(identity, 0) + 1
    ids = np.asarray(sorted(sums), np.int64)
    if len(ids) == 0:
        return {"ids": ids, "embeddings": np.zeros((0, 1)),
                "masks": np.zeros(0, np.int64)}
    stacked = np.stack([sums[i] / weights[i] for i in ids])
    stacked /= np.maximum(np.linalg.norm(stacked, axis=1, keepdims=True),
                          1e-12)
    return {"ids": ids, "embeddings": stacked,
            "masks": np.asarray([counts[i] for i in ids])}


def frame_consistency(pose_set: pose_set_lib.PoseSet, carriers: np.ndarray,
                      carrier_labels: np.ndarray, keys: list[FrameKey],
                      directory: pathlib.Path, class_ids: np.ndarray,
                      class_embeddings: np.ndarray, section, spacing: float,
                      max_range: float,
                      label_dir: pathlib.Path | None = None) -> dict:
    """Agreement of the rendered 3D labels with each image's 2D prediction.

    Returns:
        Dict of per-frame agreement rates and the compared pixel counts.
    """
    rates, pixels = [], []
    cameras = {}
    index = _core.PointCloudIndex(carriers)
    for key in keys:
        if key.camera not in cameras:
            cameras[key.camera] = pose_set.rig(key.trip).cameras[
                key.camera].make_model()
        camera = cameras[key.camera]
        masks, mask_ids, embeddings = load_features(feature_path(directory,
                                                                 key))
        predicted_class, _ = classify(embeddings, class_ids, class_embeddings,
                                      float(section.min_score))
        prediction = np.zeros(masks.shape, np.int64)
        for mask_id, cls in zip(mask_ids, predicted_class):
            prediction[masks == mask_id] = cls
        exposure = pose_set.camera_exposure(key.trip, key.camera,
                                            key.stamp_ns)
        center = np.asarray(exposure).reshape(-1, 4, 4).mean(axis=0)[:3, 3]
        near = index.radius_search(center, max_range + 1.0)
        _, rendered, _ = _core.render_points(
            camera, geometry.inv(exposure), carriers[near],
            carrier_labels[near].astype(np.int32), spacing, 1, 3, 0.1,
            max_range)
        both = (rendered > 0) & (prediction > 0)
        if both.sum() > 0:
            rates.append(float((rendered[both] == prediction[both]).mean()))
            pixels.append(int(both.sum()))
        if label_dir is not None:
            path = label_dir / key.trip_id / key.camera / f"{key.stamp_ns}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), np.clip(rendered, 0, 255).astype(np.uint8))
    return {"rates": rates, "pixels": pixels}


def spread_to_points(xyz: np.ndarray, carriers: np.ndarray,
                     values: np.ndarray, max_distance: float,
                     fill) -> np.ndarray:
    """Value of the nearest carrier for every point (`fill` if too far)."""
    out = np.full(len(xyz), fill, dtype=values.dtype)
    if len(carriers) == 0 or len(xyz) == 0:
        return out
    distance, nearest = spatial.cKDTree(carriers).query(
        xyz, distance_upper_bound=max_distance)
    found = np.isfinite(distance)
    out[found] = values[nearest[found]]
    return out


def write_summary(path: pathlib.Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, indent=1))
