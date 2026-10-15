"""Builds the model suite of a run from the configuration.

`models.backend`:
    none     No models: calibration and the language route are skipped.
    oracle   Ground-truth oracles of a synthetic dataset (needs its gt/).
    auto     oracle when a gt/ directory is given, otherwise none.
Adapters of the real models (SAM 2, SigLIP 2, DINO / RoMa, Depth Anything 3,
OCR) implement the protocols of interfaces.py and are added here when the
GPU stages are set up.
"""

from __future__ import annotations

import dataclasses
import pathlib

from garage_recon.data.synthetic import ground_truth
from garage_recon.models import interfaces
from garage_recon.models import oracle


@dataclasses.dataclass
class ModelSuite:
    """The models available to a run (None = not available)."""

    segmenter: interfaces.Segmenter2D | None = None
    encoder: interfaces.LanguageEncoder | None = None
    matcher: interfaces.Matcher | None = None
    features: interfaces.DenseFeatures | None = None
    depth: interfaces.DepthPrior | None = None
    ocr: interfaces.OcrReader | None = None
    name: str = "none"


def build(models_cfg, gt_dir: str | pathlib.Path | None = None) -> ModelSuite:
    """Model suite for the `models` config section.

    Args:
        models_cfg: The `models` section of the configuration.
        gt_dir: Ground truth of a synthetic dataset, for the oracles.

    Returns:
        The models; an empty suite for the "none" backend.

    Raises:
        ValueError: For an unknown backend, or oracles without gt_dir.
    """
    backend = str(models_cfg.get("backend", "auto"))
    if backend == "auto":
        backend = "oracle" if gt_dir is not None else "none"
    if backend == "none":
        return ModelSuite()
    if backend != "oracle":
        raise ValueError(f"unknown model backend: {backend}")
    if gt_dir is None:
        raise ValueError("the oracle models need the dataset's gt/ directory")
    world = ground_truth.SyntheticWorld(gt_dir)
    settings = models_cfg.get("oracle") or {}
    return ModelSuite(
        segmenter=oracle.OracleSegmenter(world,
                                         **(settings.get("segmenter") or {})),
        encoder=oracle.OracleLanguageEncoder(world,
                                             **(settings.get("encoder") or {})),
        matcher=oracle.OracleMatcher(world, **(settings.get("matcher") or {})),
        features=oracle.OracleDenseFeatures(world),
        depth=oracle.OracleDepthPrior(world),
        ocr=oracle.OracleOcr(world),
        name="oracle",
    )
