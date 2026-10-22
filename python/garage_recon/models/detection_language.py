"""Detector instances as the language route's masks and embeddings
(`features.source: detector`, README 5.2).

On real garages zero-shot classes of whole-image masks fail (SigLIP 2 agreed
with the geometric labels on 12 % of the masks), while open-vocabulary
detection names the objects well. This adapter gives the language route
(features -> identities -> language) the detector's instances instead:
  segment()      the detections of the objects2d queries, as SAM 2 masks from
                 their boxes (or the boxes), one mask id per detection, the
                 best-scoring on top;
  encode_masks() the one-hot of each mask's class;
  encode_text()  the one-hot of the class whose taxonomy prompts contain the
                 text, so the language stage's prompt classes match.
The identity association then merges the views of one object into a 3D
instance with its class (the majority of its views).
"""

from __future__ import annotations

import numpy as np

from garage_recon.models import interfaces
from garage_recon.semantics import objects2d, taxonomy


class DetectionLanguage:
    """Segmenter2D and LanguageEncoder from a Detector2D (+ SAM 2)."""

    needs_images = True

    def __init__(self, detector, segmenter, prompts: dict,
                 max_mask_fraction: dict):
        self.detector = detector
        self.segmenter = segmenter  # Sam2Segmenter (box prompts) or None.
        self.texts, self.query_class = objects2d.queries(prompts)
        self.limits = dict(max_mask_fraction)
        self.classes = {}  # (trip, camera, stamp) -> {mask id: class id}
        self._text_class = {str(p).lower(): taxonomy.cid(name)
                            for name, items in taxonomy.PROMPTS.items()
                            for p in items}

    @property
    def dimension(self) -> int:
        return len(taxonomy.CLASSES)

    def _one_hot(self, class_id: int) -> np.ndarray:
        v = np.zeros(self.dimension, np.float32)
        v[int(class_id)] = 1.0
        return v

    def segment(self, frame: interfaces.Frame) -> np.ndarray:
        """(H, W) int32 mask ids of the detected objects, -1 elsewhere."""
        shape = frame.image.shape[:2]
        found = [d for d in self.detector.detect(frame, self.texts)
                 if self.query_class[d.query] >= 0]
        masks = (self.segmenter.masks_from_boxes(
            frame, np.stack([d.box for d in found]))
            if found and self.segmenter is not None else None)
        found, masks = objects2d.keep_detections(found, masks,
                                                 self.query_class, shape,
                                                 self.limits)
        ids = np.full(shape, -1, np.int32)
        classes = {}
        for k in np.argsort([d.score for d in found]):  # Best on top.
            d = found[k]
            if masks is not None:
                region = masks[k]
            else:
                region = np.zeros(shape, bool)
                x0, y0, x1, y1 = np.round(d.box).astype(int)
                region[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
            ids[region] = k
            classes[int(k)] = int(self.query_class[d.query])
        self.classes[(frame.trip_id, frame.camera, frame.stamp_ns)] = classes
        return ids

    def encode_masks(self, frame: interfaces.Frame,
                     masks: np.ndarray) -> dict[int, np.ndarray]:
        classes = self.classes.pop((frame.trip_id, frame.camera,
                                    frame.stamp_ns), {})
        return {int(i): self._one_hot(classes[int(i)])
                for i in np.unique(masks) if i >= 0 and int(i) in classes}

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """One-hot of each prompt's taxonomy class (zeros if unknown)."""
        out = np.zeros((len(prompts), self.dimension), np.float32)
        for row, text in enumerate(prompts):
            c = self._text_class.get(str(text).lower())
            if c is not None:
                out[row, c] = 1.0
        return out
