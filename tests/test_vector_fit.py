"""Sub-pixel line measurement on the floor image and slot rows built from
measured ends."""

import numpy as np

from garage_recon.recon import bev
from garage_recon.semantics import vector_fit

# pylint: disable=protected-access

GRID = bev.BevGrid(0, -1.0, -1.0, 0.05, 120, 140)  # x -1..5, y -1..6 m


def _floor_image(hidden: tuple[float, float] | None = None
                 ) -> vector_fit.FloorImage:
    """A 15 cm line at x = 2 from y = 1.0 to 4.3, rendered by area coverage;
    `hidden` (y0, y1) is unobserved floor (e.g. under a car)."""
    ss = 10
    xs = GRID.x0 + (np.arange(GRID.width * ss) + 0.5) * GRID.res / ss
    ys = GRID.y0 + (np.arange(GRID.height * ss) + 0.5) * GRID.res / ss
    gx, gy = np.meshgrid(xs, ys)
    paint = ((np.abs(gx - 2.0) <= 0.075) & (gy >= 1.0) & (gy <= 4.3))
    cover = paint.reshape(GRID.height, ss, GRID.width, ss).mean(axis=(1, 3))
    rng = np.random.default_rng(0)
    image = 0.2 + 0.7 * cover + rng.normal(0.0, 0.01, cover.shape)
    valid = np.ones_like(image, bool)
    if hidden is not None:
        rows = GRID.y0 + (np.arange(GRID.height) + 0.5) * GRID.res
        valid[(rows >= hidden[0]) & (rows <= hidden[1]), :] = False
    return vector_fit.FloorImage("lidar", image, valid, GRID)


def test_line_centre_and_ends_are_sub_pixel():
    # Skeleton-like start: ends eroded by 8 cm, centre 1 cm off.
    fit = vector_fit.fit_line(np.array([2.01, 1.08]), np.array([2.01, 4.22]),
                              _floor_image(), vector_fit.FitConfig())
    assert fit is not None and fit.end_ok == (True, True)
    assert abs(fit.p0[0] - 2.0) < 0.003 and abs(fit.p1[0] - 2.0) < 0.003
    assert abs(fit.p0[1] - 1.0) < 0.005 and abs(fit.p1[1] - 4.3) < 0.005
    assert abs(fit.width - 0.15) < 0.01


def test_an_end_without_observed_floor_beyond_is_unreliable():
    # The floor beyond the far end (y > 4.35) is hidden.
    fit = vector_fit.fit_line(np.array([2.0, 1.08]), np.array([2.0, 4.22]),
                              _floor_image(hidden=(4.35, 6.0)),
                              vector_fit.FitConfig())
    assert fit is not None
    assert fit.end_ok == (True, False)
    assert abs(fit.p0[1] - 1.0) < 0.005
