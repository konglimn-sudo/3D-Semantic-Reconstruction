"""Interfaces of the foundation models in the pipeline.

Route A of the semantics (2D models decide "what") and the geometric
correspondence use these protocols:

* Segmenter2D      class-agnostic masks (SAM 2 / SAM 3 automatic masks).
* LanguageEncoder  embeddings of masks and text in one space (SigLIP 2 /
                   CLIP); text prompts query the garage taxonomy.
* DenseFeatures    patch features (DINOv2 / DINOv3).
* Matcher          image correspondences (DINO-based dense matching, e.g.
                   RoMa), refined to sub-pixel accuracy.
* DepthPrior       pose-conditioned multi-view depth (Depth Anything 3).
* OcrReader        text on the floor, columns and signs.

Every call receives `Frame` records: real models read `image`, the oracle
models of synthetic datasets (oracle.py) read the frame reference instead.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol

import numpy as np


@dataclasses.dataclass
class Frame:
    """One camera image and where it comes from.

    Attributes:
        trip_id: Trip of the image.
        camera: Camera name.
        stamp_ns: Image stamp (ns).
        image: (H, W, 3) RGB uint8, or None when the model does not need it.
    """

    trip_id: str
    camera: str
    stamp_ns: int
    image: np.ndarray | None = None


@dataclasses.dataclass
class Matches:
    """Point correspondences between two frames (pixels).

    Attributes:
        uv_a: (N, 2) pixels in frame a.
        uv_b: (N, 2) matching pixels in frame b.
        score: (N,) confidence in [0, 1].
        index_a: (N,) index of each match in the query points, when the
            caller gave query points; None otherwise.
    """

    uv_a: np.ndarray
    uv_b: np.ndarray
    score: np.ndarray
    index_a: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.uv_a)


@dataclasses.dataclass
class DepthMap:
    """Metric depth (range along the ray, m) with a confidence per pixel."""

    depth: np.ndarray  # (H, W) float32, NaN where unknown.
    confidence: np.ndarray  # (H, W) float32 in [0, 1].


@dataclasses.dataclass
class TextResult:
    text: str
    polygon: np.ndarray  # (4, 2) pixels.
    score: float


class Segmenter2D(Protocol):
    """Class-agnostic segmentation into masks."""

    def segment(self, frame: Frame) -> np.ndarray:
        """Returns an (H, W) int32 mask-id image, -1 where no mask."""


class LanguageEncoder(Protocol):
    """Vision-language embeddings of masks and of text."""

    @property
    def dimension(self) -> int:
        """Embedding dimension."""

    def encode_masks(self, frame: Frame,
                     masks: np.ndarray) -> dict[int, np.ndarray]:
        """Unit-norm embedding of every mask id of `masks`."""

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """(P, D) unit-norm embeddings of text prompts."""


class DenseFeatures(Protocol):
    """Patch features for correspondence and grouping."""

    def features(self, frame: Frame) -> tuple[np.ndarray, int]:
        """Returns ((h, w, D) features, stride in pixels)."""


class Matcher(Protocol):
    """Dense image matching."""

    def match(self, frame_a: Frame, frame_b: Frame,
              points_a: np.ndarray | None = None) -> Matches:
        """Correspondences between two frames.

        Args:
            frame_a: First frame.
            frame_b: Second frame.
            points_a: Optional (N, 2) query pixels in frame a; the matches
                then refer to them through Matches.index_a (dense matchers
                sample their warp there), which chains matches of one anchor
                image into multi-view tracks.
        """


class DepthPrior(Protocol):
    """Multi-view depth conditioned on the shared poses."""

    def predict(self, frames: list[Frame],
                world_from_camera: list[np.ndarray]) -> list[DepthMap]:
        """Depth of every frame given its (4, 4) world <- camera pose."""


class OcrReader(Protocol):
    """Text reading: slot numbers, zone and level signs."""

    def read(self, frame: Frame) -> list[TextResult]:
        """Text found in the frame."""
