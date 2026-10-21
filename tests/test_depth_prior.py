"""Depth prior: the C++ fusion, the fisheye <-> pinhole views of the DA3
adapter, and the depth_prior stage on the tiny garage (oracle depth)."""

import json

import numpy as np

from garage_recon import _core
from garage_recon import provenance
from garage_recon.models import da3


def test_fusion_recovers_scale_and_keeps_the_lidar():
    h, w = 60, 80
    truth = np.tile(np.linspace(20.0, 2.0, h)[:, None], (1, w)).astype(np.float32)
    lidar = np.full_like(truth, np.nan)
    lidar[30::2] = truth[30::2]
    prior = (truth - 0.2) / 1.1
    config = _core.DepthAlignConfig()
    config.hole_radius = 0
    rng, conf, source, alignment = _core.fuse_depth_prior(prior, None, lidar,
                                                          None, config)
    assert alignment["valid"]
    assert abs(alignment["scale"] - 1.1) < 1e-3
    assert abs(alignment["shift"] - 0.2) < 1e-2
    assert (source[30::2] == _core.DEPTH_LIDAR).all()
    assert (source[:30] == _core.DEPTH_PRIOR).all()
    np.testing.assert_allclose(rng[:30], truth[:30], rtol=1e-3)
    assert (conf[source == _core.DEPTH_LIDAR] == 1.0).all()


def test_fisheye_views_round_trip():
    # Kannala-Brandt camera looking at a plane 4 m ahead.
    camera = _core.make_camera("kannala_brandt", 360, 270,
                               [175.0, 175.0, 180.0, 135.0, -0.04, 0.0, 0.0, 0.0])
    views = da3.make_views(camera, [(-30, -18), (30, -18), (-30, 18), (30, 18)],
                           80.0, (126, 98))
    ranges = []
    for view in views.views:
        h, w = view.ray_norm.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        rays = np.stack([(u - view.K[0, 2]) / view.K[0, 0],
                         (v - view.K[1, 2]) / view.K[1, 1], np.ones((h, w))], -1)
        cam_rays = rays.reshape(-1, 3) @ view.R_cam_view.T
        cam_rays /= np.linalg.norm(cam_rays, axis=1, keepdims=True)
        rng = np.where(cam_rays[:, 2] > 0, 4.0 / cam_rays[:, 2], np.nan)
        ranges.append(rng.reshape(h, w).astype(np.float32))
    rng, conf = da3.warp_to_camera(views, ranges,
                                   [np.ones_like(r) for r in ranges])
    cu, cv = np.meshgrid(np.arange(360.0), np.arange(270.0))
    rays, ok = camera.unproject(np.column_stack([cu.ravel(), cv.ravel()]))
    truth = (4.0 / rays[:, 2]).reshape(270, 360)
    covered = np.isfinite(rng)
    assert covered.mean() > 0.8  # The four views cover most of the fisheye.
    rel = np.abs(rng[covered] - truth[covered]) / truth[covered]
    assert np.median(rel) < 0.01
    assert np.percentile(rel, 95) < 0.05
    assert (conf[covered] > 0.99).all()


def test_depth_prior_stage(tiny_run):
    work = tiny_run["recon"].work
    summary = json.loads((work / "depth_prior" / "summary.json").read_text())
    assert summary["cameras"]
    for stats in summary["cameras"].values():
        assert stats["valid"] >= 0.9 * stats["images"]
        # Oracle depth: exact up to the LiDAR map's own error.
        assert abs(stats["scale_p10_p50_p90"][1] - 1.0) < 0.01
        assert stats["residual_median"] < 0.02
        assert stats["lidar_fraction"] > 0.2
    assert "prior_points" in summary
    # Prior pixels only in the holes: away from every rendered LiDAR pixel.
    radius = tiny_run["cfg"].depth_prior.align.hole_radius
    for path in (work / "depth_prior").rglob("*.npz"):
        if path.name == "points.npz":
            continue
        frame = np.load(path)
        assert frame["range"].dtype == np.float16
        assert set(np.unique(frame["source"])) <= {0, 1, 2}
        rows, cols = np.nonzero(frame["source"] == _core.DEPTH_PRIOR)
        lidar = frame["source"] == _core.DEPTH_LIDAR
        for r, c in zip(rows[:50], cols[:50]):
            window = lidar[max(r - radius, 0):r + radius + 1,
                           max(c - radius, 0):c + radius + 1]
            assert not window.any()
    assert "recon.depth_prior" in provenance.entries(work)


def test_views_are_turned_upright():
    world_from_camera = np.eye(4)  # Looking along world +y, upright.
    world_from_camera[:3, :3] = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    assert da3.upright_roll(world_from_camera) == 0  # y down = world down.
    flipped = world_from_camera.copy()
    flipped[:3, :3] = world_from_camera[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
    assert da3.upright_roll(flipped) == 180
    camera = _core.make_camera("kannala_brandt", 360, 270,
                               [175.0, 175.0, 180.0, 135.0, 0.0, 0.0, 0.0, 0.0])
    views = da3.make_views(camera, [(0, 0)], 80.0, (126, 98), roll_deg=180)
    # The top-centre pixel of the rolled view looks at the bottom of the camera.
    top = views.views[0].to_camera[0, 63]
    assert top[1] > 135 and abs(top[0] - 180) < 2


def test_surface_depth_is_the_mean_of_the_first_surface():
    # A wall 4 m in front of a pinhole camera, sampled every 1 cm with 1.5 cm
    # noise along the view: the z-buffer keeps the nearest sample per pixel.
    rng = np.random.default_rng(5)
    ys, zs = np.meshgrid(np.arange(-2.0, 2.0, 0.01), np.arange(-1.5, 1.5, 0.01))
    wall = np.column_stack([ys.ravel(), zs.ravel(),
                            4.0 + rng.normal(0, 0.015, ys.size)])
    camera = _core.make_camera("pinhole", 160, 120, [100.0, 100.0, 80.0, 60.0])
    view = np.eye(4)
    nearest, _, _ = _core.render_points(camera, view, wall, None, 0.01, 1, 2,
                                        0.1, 20.0)
    surface = _core.render_surface_depth(camera, view, wall, 0.01, 1, 2, 0.1,
                                         20.0, 0.05, 0.01, 1)
    u, v = np.meshgrid(np.arange(160), np.arange(120))
    ray = np.sqrt(((u - 80) / 100.0) ** 2 + ((v - 60) / 100.0) ** 2 + 1)
    truth = 4.0 * ray
    inside = (u >= 35) & (u < 125) & (v >= 25) & (v < 95)  # The wall's image.
    ok = inside & np.isfinite(surface) & np.isfinite(nearest) & (nearest < 100)
    assert ok.sum() > 0.9 * inside.sum()
    assert np.median(nearest[ok] - truth[ok]) < -0.01  # Biased short.
    assert abs(np.median(surface[ok] - truth[ok])) < 0.003


def test_context_images_are_taken_elsewhere_in_the_trip():
    from garage_recon.recon import depth_prior as dp
    # 100 images 1 m apart along x; the batch is the first ten.
    images = [(k, "cam") for k in range(100)]
    centers = {key: np.array([float(key[0]), 0.0, 0.0]) for key in images}
    batch = images[:10]
    far = dp._context(images, centers, batch, 4, "distance", 15.0, 40.0, 0.0)
    assert len(far) == 4 and not set(far) & set(batch)
    assert all(15.0 <= abs(key[0] - 4.5) <= 40.0 for key in far)
    spread = dp._context(images, centers, batch, 5, "spread", 0, 0, 0.0)
    assert len(spread) == 5 and not set(spread) & set(batch)
    assert max(k for k, _ in spread) - min(k for k, _ in spread) > 50
    assert dp._context(images, centers, batch, 0, "distance", 15, 40, 0) == []
