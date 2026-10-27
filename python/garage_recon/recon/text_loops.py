"""Text loop candidates: keyframes that read the same text.

Garages repeat their layout (identical levels, regular column grids), where
geometry alone proposes wrong loops and the ICP cannot tell the levels
apart; painted slot numbers and level / zone signs ("B2-105", "C区") can.
Every `stride`-th keyframe reads the images of its trip taken within
`max_dt` of it (models.ocr); texts of at least `min_length` characters that
at most `max_keyframes` keyframes read (a sign seen from everywhere is no
place) pair the keyframes that read them: of other trips, or of their own
trip more than `min_index_gap` keyframes apart.

Only pairs the geometric loop search cannot propose are kept: a pair of
anchored trips (or of one trip) whose odometry already puts them within
`loops.search_radius` is left to it, since the registration without a pose
guess these pairs need is less precise (on the synthetic garage, where
geometry finds every loop, adding all text pairs moved the keyframes by 15
cm). What remains are pairs with an anchorless trip (its frame is its own)
or with a large drift between them, at most `max_pairs_per_text` per text.
The pairs go to the C++ graph stage with the appearance pairs: registered
without a pose guess and kept only when unambiguous (VerifyAppearanceLoops),
then PCM.
"""

from __future__ import annotations

import collections

import numpy as np

from garage_recon.models import interfaces
from garage_recon.semantics import numbers


def text_pairs(keyframes, datasets, ocr, cfg, log=print) -> np.ndarray:
    """(N, 2) keyframe pairs (later, earlier) that read a common text."""
    section = cfg.loops.text
    stride = max(1, int(section.stride))
    max_dt = float(section.max_dt) * 1e9
    readers = collections.defaultdict(set)  # Text -> keyframes.
    read = 0
    for k in range(0, len(keyframes.stamp), stride):
        data = datasets[int(keyframes.trip[k])]
        target = float(keyframes.stamp[k]) * 1e9
        for camera in data.rig.cameras:
            stamps = data.camera_stamps_ns(camera)
            if len(stamps) == 0:
                continue
            nearest = int(stamps[np.argmin(np.abs(stamps - target))])
            if abs(nearest - target) > max_dt:
                continue
            image = (data.load_image(camera, nearest)
                     if getattr(ocr, "needs_images", True) else None)
            read += 1
            for result in ocr.read(interfaces.Frame(data.trip_id, camera,
                                                    nearest, image)):
                text = numbers.normalize_text(result.text)
                if len(text) >= int(section.min_length):
                    readers[text].add(k)
    trips = np.asarray(keyframes.trip)
    positions = np.asarray(keyframes.odom)[:, :3, 3]
    anchored = [data.initial_pose() is not None for data in datasets]
    radius = float(cfg.loops.search_radius)
    gap = int(section.min_index_gap)
    pairs = set()
    distinctive = 0
    for text, ks in readers.items():
        if len(ks) < 2 or len(ks) > int(section.max_keyframes):
            continue
        distinctive += 1
        ks = sorted(ks)
        found = 0
        for a in range(len(ks)):
            for b in range(a):
                i, j = ks[a], ks[b]
                if trips[i] == trips[j] and i - j < gap:
                    continue
                comparable = trips[i] == trips[j] or (
                    anchored[trips[i]] and anchored[trips[j]])
                if (comparable and np.linalg.norm(positions[i] - positions[j])
                        <= radius):
                    continue  # The geometric loop search covers it.
                if found < int(section.max_pairs_per_text):
                    pairs.add((i, j))
                    found += 1
    out = np.array(sorted(pairs), int).reshape(-1, 2)
    log(f"[loops] text: {read} images, {len(readers)} texts "
        f"({distinctive} distinctive), {len(out)} candidate pairs")
    return out
