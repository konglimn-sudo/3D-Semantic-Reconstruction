"""Held-out 3DGS images borrow the appearance of their neighbours in time."""

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from garage_recon.recon import gs  # noqa: E402


def _image(trip, camera, stamp, holdout):
    return types.SimpleNamespace(trip=trip, camera=camera, stamp_ns=stamp,
                                 holdout=holdout)


def test_holdout_appearance_interpolates_the_same_camera_in_time():
    images = [_image(0, "a", 0, False), _image(0, "a", 10, True),
              _image(0, "a", 40, False), _image(0, "b", 10, False),
              _image(0, "a", 50, True), _image(1, "a", 10, True)]
    trainer = gs.BlockTrainer.__new__(gs.BlockTrainer)
    trainer.torch = torch
    trainer.images = images
    trainer.train_ids = [i for i, im in enumerate(images) if not im.holdout]
    trainer.gain = torch.ones((len(images), 3))
    trainer.bias = torch.zeros((len(images), 3))
    trainer.gain[0], trainer.gain[2], trainer.gain[3] = 1.0, 2.0, 5.0
    trainer.bias[2] = 0.4
    trainer.neighbour_appearance([1, 4, 5])
    # A quarter of the way from the image at 0 to the one at 40; camera b
    # (closer in time) is another camera and does not count.
    assert np.allclose(trainer.gain[1].numpy(), 1.25)
    assert np.allclose(trainer.bias[1].numpy(), 0.1)
    assert np.allclose(trainer.gain[4].numpy(), 2.0)  # After the last: held.
    assert np.allclose(trainer.gain[5].numpy(), 1.0)  # No trip-1 neighbour.
    app = trainer.appearance()
    assert app["holdout"].tolist() == [False, True, False, False, True, True]
    assert app["gain"].shape == (6, 3)
