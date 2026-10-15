"""Shared fixtures: one tiny synthetic garage, reconstructed and labelled once
per test session (~40 s), with the ground-truth oracle models."""

from __future__ import annotations

import pytest

from garage_recon import config as config_lib
from garage_recon.data import synthetic
from garage_recon.eval import synthetic_eval
from garage_recon.models import registry
from garage_recon.recon import pipeline as recon_pipeline
from garage_recon.semantics import pipeline as semantics_pipeline


def _quiet(*args, **kwargs):
    del args, kwargs


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """The tiny garage generated, reconstructed, labelled and evaluated."""
    root = tmp_path_factory.mktemp("tiny")
    trips = synthetic.generate(root / "data", preset="tiny", log=_quiet)
    cfg = config_lib.load_config()
    models = registry.build(cfg.models, root / "data" / "gt")
    recon = recon_pipeline.ReconPipeline(trips, root / "work", cfg, _quiet,
                                         models)
    recon.run()
    semantics = semantics_pipeline.SemanticsPipeline(recon, cfg, _quiet)
    semantics.run()
    report = synthetic_eval.evaluate(root / "work", root / "data" / "gt",
                                     _quiet)
    return {"root": root, "recon": recon, "sem": semantics, "report": report,
            "cfg": cfg}
