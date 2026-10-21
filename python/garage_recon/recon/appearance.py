"""Appearance loop candidates (README 8.4): AnyLoc-style image retrieval.

Every `stride`-th keyframe gets the image of its trip's camera taken closest
to it; DINOv2 patch tokens are aggregated by VLAD (models/dino.py), and each
keyframe proposes its `top_k` most similar keyframes: of other trips, or of
its own trip more than `min_index_gap` keyframes away. The absolute
similarity barely separates true from false matches (Keble: hits 0.72-0.87,
misses 0.68-0.80), so there is no threshold: the C++ graph stage registers
each pair without a pose guess and keeps only unambiguous ones
(VerifyAppearanceLoops), then PCM.
"""

from __future__ import annotations

import numpy as np

from garage_recon.models import dino


def appearance_pairs(keyframes, datasets, features, cfg, log=print
                     ) -> np.ndarray:
    """(N, 2) keyframe pairs (later, earlier) to verify."""
    section = cfg.loops.appearance
    stride = max(1, int(section.stride))
    max_dt = float(section.max_dt)
    ids, tokens = [], []
    for k in range(0, len(keyframes.stamp), stride):
        data = datasets[int(keyframes.trip[k])]
        cameras = list(data.rig.cameras)
        if not cameras:
            continue
        camera = section.get("camera") or cameras[0]
        if camera not in data.rig.cameras:
            continue
        stamps = data.camera_stamps_ns(camera)
        if len(stamps) == 0:
            continue
        target = float(keyframes.stamp[k]) * 1e9
        nearest = int(stamps[np.argmin(np.abs(stamps - target))])
        if abs(nearest - target) > max_dt * 1e9:
            continue
        tokens.append(features.tokens(data.load_image(camera, nearest)))
        ids.append(k)
    if len(ids) < 2:
        return np.zeros((0, 2), int)
    desc = dino.vlad_descriptors(tokens, num_words=int(section.num_words))
    ids = np.asarray(ids)
    trips = np.asarray(keyframes.trip)[ids]
    sim = desc @ desc.T
    gap = int(section.min_index_gap)
    allowed = (trips[:, None] != trips[None, :]) | (
        np.abs(ids[:, None] - ids[None, :]) >= gap)
    np.fill_diagonal(allowed, False)
    sim = np.where(allowed, sim, -np.inf)
    pairs = set()
    top_k = int(section.top_k)
    for a in range(len(ids)):
        for b in np.argsort(-sim[a])[:top_k]:
            if not np.isfinite(sim[a, b]):
                continue
            i, j = int(ids[a]), int(ids[b])
            pairs.add((max(i, j), min(i, j)))
    out = np.array(sorted(pairs), int).reshape(-1, 2)
    log(f"[loops] appearance: {len(ids)} keyframe images, {len(out)} "
        f"candidate pairs")
    return out
