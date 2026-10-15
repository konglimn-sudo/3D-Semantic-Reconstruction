"""Integration points for learned 3D models and real 2D model adapters.

The 2D foundation-model protocols (segmentation, language embeddings, dense
features, matching, depth, OCR) and their synthetic oracles live in
garage_recon.models. This module keeps the 3D detector protocol of the
object stage and placeholders for adapters of real models; each placeholder
documents its I/O and raises until it is wired to a model, so no heavy
dependency is imported unless it is used.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from garage_recon.models import interfaces


class Detector3D(Protocol):
    """Offboard LiDAR object detector."""

    def detect(self, points: np.ndarray, intensity: np.ndarray) -> list[dict]:
        """Boxes {center, size, yaw, cls, score} in the points' frame."""


class Sam2Segmenter:
    """Class-agnostic masks from SAM 2 automatic mask generation.

    Implements interfaces.Segmenter2D. Fisheye frames should be resampled to
    perspective tiles first, segmented, and the masks mapped back.
    """

    def __init__(self, checkpoint: str, device: str = "cuda"):
        raise NotImplementedError(
            "install SAM 2 and implement Sam2Segmenter.segment()")

    def segment(self, frame: interfaces.Frame) -> np.ndarray:
        raise NotImplementedError  # pragma: no cover


class PaddleOcrReader:
    """OCR adapter (PaddleOCR or a VLM); implements interfaces.OcrReader.

    Reads numbers on the floor, columns and walls;
    semantics.rules.fix_slot_numbers repairs sequences afterwards.
    """

    def __init__(self, lang: str = "ch"):
        try:
            # Optional dependency, imported only when OCR is requested.
            import paddleocr  # pylint: disable=g-import-not-at-top
            del paddleocr
        except ImportError as error:  # pragma: no cover
            raise NotImplementedError(
                "pip install paddleocr to enable OCR") from error
        raise NotImplementedError("map PaddleOCR results to TextResult here")

    def read(self, frame: interfaces.Frame
             ) -> list[interfaces.TextResult]:  # pragma: no cover
        raise NotImplementedError


class LearnedDetector3D:
    """Slot for an offboard LiDAR detector (e.g. CenterPoint / DSVT).

    Its boxes replace ClusterDetector detections and go through the same
    tracker and track-level refinement.
    """

    def __init__(self, checkpoint: str):
        raise NotImplementedError("wire your 3D detector here")

    def detect(self, points, intensity):  # pragma: no cover
        raise NotImplementedError
