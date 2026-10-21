"""Builds the model suite of a run from the configuration.

`models.backend`:
    none     No models: calibration and the language route are skipped.
    oracle   Ground-truth oracles of a synthetic dataset (needs its gt/).
    real     The GPU models of the config (models.da3, ...), imported lazily.
    auto     oracle when a gt/ directory is given, otherwise none.
Real adapters implement the protocols of interfaces.py: Depth Anything 3
(da3.py), RoMa and DINOv2 (dino.py); SAM 2, SigLIP 2 and OCR are added with
their stages.
"""

from __future__ import annotations

import dataclasses
import pathlib

from garage_recon.data.synthetic import ground_truth
from garage_recon.models import interfaces
from garage_recon.models import oracle


class _Lazy:
    """A model adapter built on first attribute access."""

    def __init__(self, module: str, cls: str, settings):
        self._spec = (module, cls, dict(settings))
        self._model = None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._model is None:
            import importlib
            module, cls, settings = self._spec
            adapter = getattr(importlib.import_module(
                f"garage_recon.models.{module}"), cls)
            self._model = adapter(**settings)
        return getattr(self._model, name)


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
    if backend == "real":
        # Each model only when configured (null disables it), and loaded on
        # first use: a stage that needs one model does not hold the others
        # in GPU memory.
        suite = ModelSuite(name="real")
        if models_cfg.get("da3"):
            suite.depth = _Lazy("da3", "Da3DepthPrior", models_cfg.get("da3"))
        if models_cfg.get("roma"):
            suite.matcher = _Lazy("dino", "RomaMatcher", models_cfg.get("roma"))
        if models_cfg.get("dino"):
            suite.features = _Lazy("dino", "Dinov2Features",
                                   models_cfg.get("dino"))
        return suite
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
