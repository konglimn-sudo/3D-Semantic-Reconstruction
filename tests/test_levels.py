"""Level floors that rise gently along an aisle (below the ramp slope)."""

import numpy as np

from garage_recon import config as config_lib
from garage_recon.recon import levels as levels_lib


def test_a_gently_rising_aisle_is_one_level_whose_floor_follows_the_slope():
    grade = 0.011  # FusionPortableV2: 4.5 m over 400 m.
    x = np.linspace(0.0, 300.0, 3000)
    poses = np.tile(np.eye(4), (len(x), 1, 1))
    poses[:, 0, 3], poses[:, 2, 3] = x, grade * x
    rng = np.random.default_rng(0)
    fx = rng.uniform(0, 300, 60000)
    fy = rng.uniform(-6, 6, 60000)
    floor = np.column_stack([fx, fy, grade * fx + rng.normal(0, 0.01, len(fx))])
    ceiling = floor[::3] + [0, 0, 3.0]
    xyz = np.vstack([floor, ceiling])
    normals = np.vstack([np.tile([0, 0, 1.0], (len(floor), 1)),
                         np.tile([0, 0, -1.0], (len(ceiling), 1))])
    cfg = config_lib.load_config(None)
    model = levels_lib.detect_levels([poses], xyz, normals, cfg, log=lambda *_: None)
    assert model.n == 1
    ends = model.floor_z(np.array([[5.0, 0.0], [295.0, 0.0], [150.0, 5.0]]), 0)
    assert np.allclose(ends, grade * np.array([5.0, 295.0, 150.0]), atol=0.1)
    _, height, _ = model.assign(floor)
    assert np.percentile(np.abs(height), 95) < 0.1
    assert abs(model.ceiling[0] - model.z[0] - 3.0) < 0.2
