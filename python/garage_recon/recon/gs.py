"""The gs stage (README 8.3): 3D Gaussians on the LiDAR skeleton.

gsplat (Apache 2.0) in 3DGUT mode renders the raw fisheye images (unscented
projection, 3D evaluation, rolling shutter from the pose set's first and last
row poses), so nothing is undistorted. The skeleton stays the LiDAR's:

* every static map point starts a flat, anchored Gaussian (C++
  InitGaussians: normal axis = the point's sigma, tangential = 0.8 x the
  local spacing); DA3 points (depth_prior/points.npz) fill the LiDAR holes;
* Gaussians seen in fewer than `min_views` usable views (C++ CameraCoverage)
  are frozen: geometry and opacity stay, colour comes from the coverage pass
  or the LiDAR intensity, and they are never pruned;
* anchored Gaussians pay for leaving their LiDAR point and are pruned beyond
  `anchor_prune`; Gaussians in carved free space (C++ FreeSpaceCarver) or
  below the floor (epoxy reflections) are pruned;
* losses: L1 + D-SSIM per image with a per-image gain / bias (exposure and
  white balance), z-depth against the LiDAR range rendered into the image and
  against the aligned DA3 prior in the holes (weighted by confidence,
  decaying), flattening of the smallest scale.

The scene is cut into blocks of `block_size` with a `block_margin`; each
block trains on the images taken in or near it, and keeps the Gaussians of
its core. Every `holdout_every`-th image is held out for evaluation. A
held-out image has no fitted gain / bias, and auto exposure changes from
frame to frame: it takes the appearance interpolated from the same camera's
neighbouring training images (holdout_psnr; holdout_psnr_raw with none).

Outputs (work/gs/): model.npz (merged Gaussians, flags, intensity),
model.ply (standard 3DGS layout), appearance.npz (per-image gain / bias),
summary.json, previews/ (held-out images
next to their renders).
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
from concurrent import futures

import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon import geometry
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.recon import static_map

FLAG_ANCHORED = 1   # From a LiDAR point (source L).
FLAG_APPEARANCE = 2  # Enough usable views to learn colour.
FLAG_FROZEN = 4     # No camera supervision: geometry and opacity fixed.
FLAG_PRIOR = 16     # From the depth prior (source P).
SH_C0 = 0.28209479177387814


@dataclasses.dataclass
class TrainImage:
    trip: int
    camera: str
    stamp_ns: int
    rgb: np.ndarray  # (H, W, 3) uint8
    viewmat: np.ndarray  # (4, 4) camera <- world, first row
    viewmat_rs: np.ndarray | None  # last row, rolling shutter
    K: np.ndarray  # (3, 3)
    radial: np.ndarray  # (4,) fisheye or (6,) pinhole coefficients
    model: str  # gsplat camera model
    mask: np.ndarray  # (H, W) bool: pixel supervises
    depth: np.ndarray  # (H, W) float16 z-depth target, 0 = none
    depth_prior: np.ndarray  # (H, W) bool: target from the prior
    depth_weight: np.ndarray  # (H, W) float16
    center: np.ndarray  # (3,) camera position
    holdout: bool
    sky: np.ndarray | None = None  # (H, W) bool: up-looking ray, no surface
    # (H, W) bool: saturated pixels (out of `mask`): the radiance there is at
    # least the saturation level, a one-sided target (gs.saturated_weight).
    saturated: np.ndarray | None = None


def _camera_params(calib):
    """gsplat camera model, K and distortion of a rig camera, or None."""
    p = [float(v) for v in calib.params]
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
    if calib.model in ("kannala_brandt", "kb", "fisheye", "opencv_fisheye"):
        return "fisheye", K, np.array((p[4:8] + [0.0] * 4)[:4])
    if calib.model == "pinhole":
        return "pinhole", K, np.zeros(6)
    return None


def _rays(camera: _core.CameraModel) -> np.ndarray:
    """(H, W, 3) unit ray of every pixel in the camera frame (NaN outside)."""
    u, v = np.meshgrid(np.arange(camera.width, dtype=float),
                       np.arange(camera.height, dtype=float))
    rays, ok = camera.unproject(np.column_stack([u.ravel(), v.ravel()]))
    rays = np.where(ok[:, None], rays, np.nan)
    return rays.reshape(camera.height, camera.width, 3).astype(np.float32)


def _load_images(pose_set, point_map, cfg, work: pathlib.Path, log):
    """Training images with masks and depth targets (threads: C++ releases
    the GIL)."""
    section = cfg.gs
    static = point_map.xyz if point_map.dynamic is None else (
        point_map.xyz[~point_map.dynamic])
    static_index = _core.PointCloudIndex(np.ascontiguousarray(static))
    moving = (point_map.xyz[point_map.dynamic]
              if point_map.dynamic is not None else None)
    mask_config = core_config.fill(_core.TrainingMaskConfig(),
                                   section.get("mask"))
    depth_dir = work / "depth_prior"
    jobs = []
    for t, data in enumerate(pose_set.datasets):
        rig = pose_set.rig(t)
        for name, calib in rig.cameras.items():
            params = _camera_params(calib)
            if params is None:
                log(f"[gs] camera {name}: model {calib.model} not supported")
                continue
            camera = calib.make_model()
            rays = _rays(camera)
            stamps = data.camera_stamps_ns(name)[::max(1, int(
                section.frame_stride))]
            for k, ns in enumerate(stamps):
                jobs.append((t, data, name, calib, camera, params, rays,
                             int(ns), k))

    def load(job):
        t, data, name, calib, camera, (model, K, radial), rays, ns, k = job
        ray_z = rays[..., 2]
        rgb = data.load_image(name, ns)
        exposure = pose_set.camera_exposure(t, name, ns)
        world_from_cam = exposure if exposure.ndim == 2 else exposure[0]
        viewmat = geometry.inv(world_from_cam)
        viewmat_rs = geometry.inv(exposure[1]) if exposure.ndim == 3 else None
        mask = ray_z > 0  # Inside the fisheye.
        if moving is not None and len(moving):
            training = _core.render_training_mask(
                camera, exposure, static_index, moving, None, [], mask_config)
            mask &= training == 0
        bright = rgb.max(axis=2) >= int(section.saturated)  # Sun, lamps.
        holdout = k % max(1, int(section.holdout_every)) == 0
        # Kept for training with saturated_weight, and always on the held-out
        # images (saturation_agreement).
        saturated = (mask & bright if holdout or float(
            section.get("saturated_weight") or 0.0) > 0 else None)
        mask &= ~bright
        mask &= rgb.max(axis=2) > int(section.dark)
        depth = np.full(ray_z.shape, np.nan, np.float32)
        weight = np.zeros(ray_z.shape, np.float32)
        prior = np.zeros(ray_z.shape, bool)
        path = depth_dir / data.trip_id / name / f"{ns}.npz"
        if path.exists():
            with np.load(path) as z:
                rng = z["range"].astype(np.float32)
                source = z["source"]
                conf = z["confidence"].astype(np.float32) / 255.0
            lidar = source == _core.DEPTH_LIDAR
            prior = source == _core.DEPTH_PRIOR
            depth = np.where(lidar | prior, rng * ray_z, np.nan)
            weight = np.where(lidar, 1.0, np.where(prior, conf, 0.0)).astype(
                np.float32)
        else:
            rng = _core.render_surface_depth(
                camera, geometry.inv(exposure), np.ascontiguousarray(static),
                float(cfg.map.voxel), 1, 4, 0.3,
                float(section.depth_max_range), 0.05, 0.01, 1)
            lidar = np.isfinite(rng)
            depth = np.where(lidar, rng * ray_z, np.nan).astype(np.float32)
            weight = lidar.astype(np.float32)
        # Sky: rays above the horizon (world z up) that meet no LiDAR or
        # prior surface; Gaussians there only explain the background.
        sky = None
        if (float(section.get("sky_weight") or 0.0) > 0
                and section.get("sky_source", "segmentation") == "depth"):
            up = np.nan_to_num(rays @ world_from_cam[:3, :3].T)[..., 2]
            horizon = np.sin(np.radians(float(section.sky_min_elevation)))
            sky = (up > horizon) & ~np.isfinite(depth) & (ray_z > 0)
        # Compact (uint8 / float16 / bool): thousands of images per run.
        return TrainImage(
            t, name, ns, rgb, viewmat, viewmat_rs, K, radial, model, mask,
            np.nan_to_num(depth, nan=0.0).astype(np.float16), prior,
            weight.astype(np.float16), world_from_cam[:3, 3].copy(),
            holdout, sky, saturated)

    with futures.ThreadPoolExecutor(max(1, int(section.workers))) as pool:
        images = list(pool.map(load, jobs))
    log(f"[gs] {len(images)} images ({sum(i.holdout for i in images)} held "
        f"out)")
    return images


def _segment_sky(images, cfg, out_dir, log):
    """Sky masks from DINOv2 + the ADE20K head (models.dino.ade20k_head): sky
    where the class probability passes `gs.sky_threshold` and no LiDAR or
    prior surface was found. Sky by "no depth" alone also takes far facades
    and trees beyond the LiDAR range (README 8.3)."""
    import cv2
    from garage_recon.models import dino
    settings = dict(cfg.models.get("dino") or {})
    if not settings.get("ade20k_head"):
        log("[gs] sky: no models.dino.ade20k_head configured, no sky masks")
        return
    model = dino.Dinov2Features(**settings)
    threshold = float(cfg.gs.get("sky_threshold", 0.5))
    bright = int(cfg.gs.saturated)
    # Small blobs (dark ceilings, glare) are not sky.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    start, share = time.time(), []
    preview = []
    for n, im in enumerate(images):
        p = model.class_probability(im.rgb, model.ADE20K_SKY)
        # Saturated pixels are out of `mask`, but most of a bright sky is
        # saturated: they count, the black outside the fisheye does not.
        usable = im.mask | (im.rgb.max(axis=2) >= bright)
        sky = (p > threshold) & (im.depth == 0) & usable
        im.sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_OPEN,
                                  kernel) > 0
        share.append(im.sky.mean())
        if n % max(1, len(images) // 4) == 0 and len(preview) < 4:
            over = im.rgb.copy()
            over[im.sky] = (0.4 * over[im.sky] + [153, 0, 0]).astype(np.uint8)
            preview.append(cv2.resize(np.concatenate([im.rgb, over], axis=1),
                                      None, fx=0.3, fy=0.3))
    del model
    import torch
    torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    if preview:
        width = min(p.shape[1] for p in preview)
        cv2.imwrite(str(out_dir / "sky_preview.jpg"),
                    np.concatenate([p[:, :width] for p in preview])[..., ::-1])
    log(f"[gs] sky: {np.mean(share) * 100:.1f}% of the pixels (median image "
        f"{np.median(share) * 100:.1f}%), {time.time() - start:.0f}s")


def _coverage(points, normals, images, pose_set, cfg):
    """Usable views and view-weighted colour per point (C++ CameraCoverage)."""
    config = core_config.fill(_core.CoverageConfig(), cfg.gs.get("coverage"))
    coverage = _core.CameraCoverage(np.ascontiguousarray(points),
                                    np.ascontiguousarray(normals), config)
    stride = max(1, int(cfg.gs.coverage_stride))
    for image in images[::stride]:
        if image.holdout:
            continue
        rig = pose_set.rig(image.trip)
        camera = rig.cameras[image.camera].make_model()
        exposure = pose_set.camera_exposure(image.trip, image.camera,
                                            image.stamp_ns)
        coverage.add_frame(camera, exposure, image.rgb,
                           (~image.mask).astype(np.uint8))
    result = coverage.result()
    return (np.asarray(result["good_views"]) * stride,
            np.asarray(result["color"], np.float32))


def _free_space(keyframes, poses, bounds, cfg):
    """Carved free space of the LiDAR rays (C++ FreeSpaceCarver)."""
    section = cfg.gs.free_space
    carver = _core.FreeSpaceCarver(bounds[0], bounds[1], float(section.voxel),
                                   float(section.endpoint_margin))
    for k in range(len(keyframes)):
        points, _ = keyframes.load(k)
        world = points @ poses[k][:3, :3].T + poses[k][:3, 3]
        carver.integrate(poses[k][:3, 3], np.ascontiguousarray(world))
    return carver


def _lidar_views(keyframes, kf_poses, point_map, cfg, log):
    """Depth targets seen from the LiDAR (gs.lidar_views, README 8.3).

    At every `stride`-th keyframe, `faces` gravity-aligned pinhole views at
    the LiDAR origin share the horizon (yaw from the keyframe heading); each
    holds the static map's first-surface z-depth (C++ RenderSurfaceDepth, as
    the camera targets). The camera images alone let training pull surfaces
    towards the cameras; these constrain them along the rays the LiDAR
    measured.

    Returns:
        dict(viewmats (V, 4, 4) camera <- world, depth (V, H, W) float16
        z-depth (0 = none), centers (V, 3), K (3, 3), faces), or None.
    """
    section = cfg.gs.lidar_views
    faces = int(section.faces)
    width = int(section.width)
    focal = 0.5 * width / math.tan(math.pi / faces)
    height = 2 * int(round(focal * math.tan(math.radians(
        0.5 * float(section.vfov)))))
    camera = _core.make_camera("pinhole", width, height,
                               [focal, focal, 0.5 * width, 0.5 * height])
    ray_z = _rays(camera)[..., 2]
    static = np.ascontiguousarray(
        point_map.xyz if point_map.dynamic is None
        else point_map.xyz[~point_map.dynamic], dtype=np.float64)
    index = _core.PointCloudIndex(static)
    max_range = float(section.max_range)
    jobs = []
    for k in range(0, len(kf_poses), max(1, int(section.stride))):
        pose = kf_poses[k]
        heading = math.atan2(pose[1, 0], pose[0, 0])  # LiDAR x axis.
        for f in range(faces):
            yaw = heading + 2 * math.pi * f / faces
            forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            down = np.array([0.0, 0.0, -1.0])
            world_from_cam = np.eye(4)
            world_from_cam[:3, :3] = np.column_stack(
                [np.cross(down, forward), down, forward])
            world_from_cam[:3, 3] = pose[:3, 3]
            jobs.append(world_from_cam)

    def render(world_from_cam):
        local = np.ascontiguousarray(static[index.radius_search(
            world_from_cam[:3, 3], max_range)])
        rng = _core.render_surface_depth(
            camera, geometry.inv(world_from_cam), local,
            float(cfg.map.voxel), 1, 4, 0.3, max_range, 0.05, 0.01, 1)
        return np.nan_to_num(rng * ray_z, nan=0.0).astype(np.float16)

    start = time.time()
    with futures.ThreadPoolExecutor(max(1, int(cfg.gs.workers))) as pool:
        depth = list(pool.map(render, jobs))
    if not depth:
        return None
    depth = np.stack(depth)
    log(f"[gs] LiDAR views: {len(jobs) // faces} keyframes x {faces} "
        f"{width}x{height} views, {np.mean(depth > 0) * 100:.0f}% of the "
        f"pixels with a surface, {time.time() - start:.0f}s")
    return {"viewmats": np.stack([geometry.inv(T) for T in jobs]).astype(
                np.float32),
            "depth": depth,
            "centers": np.stack([T[:3, 3] for T in jobs]),
            "K": np.array([[focal, 0, 0.5 * width], [0, focal, 0.5 * height],
                           [0, 0, 1]], np.float32),
            "faces": faces}


def _initial_gaussians(point_map, prior_points, cfg):
    """Means, quats (wxyz), scales, opacity, intensity and flags."""
    static = (np.ones(len(point_map), bool) if point_map.dynamic is None
              else ~point_map.dynamic)
    xyz = point_map.xyz[static]
    normals = (point_map.normals[static] if point_map.normals is not None
               else np.zeros_like(xyz))
    projection = cfg.gs.get("init_projection")
    if projection and projection.get("radius"):
        # Discs on a noisy layer of points stop rays at the front of the
        # noise (Keble: rendered LiDAR depth 8.6 cm short); the anchors start
        # from the surface of the local planes instead (README 8.3).
        xyz, _ = _core.project_to_local_planes(
            np.ascontiguousarray(xyz, dtype=np.float64),
            float(projection.radius), int(projection.min_neighbors), 64,
            float(projection.min_planarity), float(projection.max_shift))
    sigma = (point_map.sigma[static] if point_map.sigma is not None else None)
    init = core_config.fill(_core.GaussianInitConfig(), cfg.gs.get("init"))
    g = _core.init_gaussians(np.ascontiguousarray(xyz),
                             np.ascontiguousarray(normals, dtype=np.float64),
                             None if sigma is None else sigma.astype(float),
                             init)
    out = {"means": xyz, "quats": g["quats"], "scales": g["scales"],
           "opacity": g["opacity"],
           "intensity": point_map.intensity[static].astype(np.float32),
           "normals": normals.astype(np.float32),
           "flags": np.full(len(xyz), FLAG_ANCHORED, np.uint8)}
    if prior_points is not None and len(prior_points["xyz"]):
        p = prior_points["xyz"]
        s = float(cfg.gs.prior_scale)
        out["means"] = np.concatenate([out["means"], p])
        out["quats"] = np.concatenate(
            [out["quats"], np.tile([1.0, 0, 0, 0], (len(p), 1))])
        out["scales"] = np.concatenate([out["scales"], np.full((len(p), 3), s)])
        out["opacity"] = np.concatenate(
            [out["opacity"], np.full(len(p), float(cfg.gs.prior_opacity))])
        out["intensity"] = np.concatenate(
            [out["intensity"], np.full(len(p), 0.5, np.float32)])
        out["normals"] = np.concatenate(
            [out["normals"], np.zeros((len(p), 3), np.float32)])
        out["flags"] = np.concatenate(
            [out["flags"], np.full(len(p), FLAG_PRIOR, np.uint8)])
        out["prior_rgb"] = prior_points["rgb"]
    return out


def _blocks(means, size, max_gaussians):
    """(core_min, core_max) xy rectangles tiling the Gaussians; one block
    while they fit the GPU budget."""
    if len(means) <= max_gaussians:
        return [(np.full(2, -np.inf), np.full(2, np.inf))]
    lo, hi = means[:, :2].min(axis=0), means[:, :2].max(axis=0)
    extent = hi - lo
    counts = np.maximum(1, np.ceil((extent - 0.25 * size) / size)).astype(int)
    counts[extent <= 1.25 * size] = 1
    edges = [np.linspace(lo[a], hi[a], counts[a] + 1) for a in range(2)]
    blocks = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            core_min = np.array([edges[0][i], edges[1][j]])
            core_max = np.array([edges[0][i + 1], edges[1][j + 1]])
            # Outer edges reach to infinity: every Gaussian has one core.
            core_min[0] = -np.inf if i == 0 else core_min[0]
            core_min[1] = -np.inf if j == 0 else core_min[1]
            core_max[0] = np.inf if i == counts[0] - 1 else core_max[0]
            core_max[1] = np.inf if j == counts[1] - 1 else core_max[1]
            blocks.append((core_min, core_max))
    return blocks


def _ssim(x, y, torch):
    """Mean SSIM of (1, 3, H, W) images (11 px Gaussian window)."""
    import torch.nn.functional as F
    g = torch.exp(-((torch.arange(11, device=x.device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum()).float()
    w = (g[:, None] @ g[None, :])[None, None].repeat(3, 1, 1, 1)
    mu_x = F.conv2d(x, w, padding=5, groups=3)
    mu_y = F.conv2d(y, w, padding=5, groups=3)
    sxx = F.conv2d(x * x, w, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=5, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))


class BlockTrainer:
    """Trains the Gaussians of one block on its images."""

    def __init__(self, init, colors, frozen, images, carver, levels, cfg, log,
                 lidar=None):
        import torch
        self.torch = torch
        self.cfg = cfg.gs
        self.lidar = None
        if lidar is not None and len(lidar["depth"]):
            self.lidar = {
                "viewmats": torch.from_numpy(lidar["viewmats"]).cuda(),
                "depth": torch.from_numpy(lidar["depth"]).cuda(),
                "K": torch.from_numpy(lidar["K"]).cuda(),
                "faces": int(lidar["faces"])}
        self.log = log
        self.levels = levels
        self.carver = carver
        dev = self.device = torch.device("cuda")
        f32 = dict(device=dev, dtype=torch.float32)
        n = len(init["means"])
        self.means = torch.tensor(init["means"], **f32).requires_grad_()
        self.anchor = torch.tensor(init["means"], **f32)
        self.normals = torch.tensor(init["normals"], **f32)
        self.log_scales = torch.tensor(
            np.log(np.maximum(init["scales"], 1e-4)), **f32).requires_grad_()
        self.quats = torch.tensor(init["quats"], **f32).requires_grad_()
        op = np.clip(init["opacity"], 1e-3, 1 - 1e-3)
        self.logit_op = torch.tensor(np.log(op / (1 - op)),
                                     **f32).requires_grad_()
        sh0 = (np.clip(colors, 0.0, 1.0) - 0.5) / SH_C0
        self.sh0 = torch.tensor(sh0[:, None, :], **f32).requires_grad_()
        degree = int(self.cfg.sh_degree)
        self.shn = torch.zeros((n, (degree + 1) ** 2 - 1, 3),
                               **f32).requires_grad_()
        self.flags = torch.tensor(init["flags"].astype(np.int32), device=dev)
        self.frozen = torch.tensor(frozen, device=dev)
        # Upper bound of every scale (log): the initial scale times
        # max_scale_growth. Grown discs reach further across a noisy surface
        # layer and stop LiDAR / camera rays in front of it.
        growth = float(self.cfg.get("max_scale_growth") or 0.0)
        self.scale_limit = (self.log_scales.detach() + math.log(growth)
                            if growth > 0 else
                            torch.full_like(self.log_scales, float("inf")))
        self.anchor_tol = torch.tensor(
            np.maximum(2.0 * init["scales"].min(axis=1),
                       float(self.cfg.anchor_tolerance)), **f32)
        self.intensity = torch.tensor(init["intensity"], **f32)
        self.images = images
        self.train_ids = [i for i, im in enumerate(images) if not im.holdout]
        m = len(images)
        self.gain = torch.ones((m, 3), **f32).requires_grad_()
        self.bias = torch.zeros((m, 3), **f32).requires_grad_()
        # Background (sky, beyond the far plane): one learned colour.
        self.background = torch.zeros(3, **f32).requires_grad_()
        lr = self.cfg.lr
        self.optimizer = torch.optim.Adam([
            {"params": [self.means], "lr": float(lr.means), "name": "means"},
            {"params": [self.log_scales], "lr": float(lr.scales),
             "name": "log_scales"},
            {"params": [self.quats], "lr": float(lr.quats), "name": "quats"},
            {"params": [self.logit_op], "lr": float(lr.opacity),
             "name": "logit_op"},
            {"params": [self.sh0], "lr": float(lr.sh0), "name": "sh0"},
            {"params": [self.shn], "lr": float(lr.shn), "name": "shn"},
        ], eps=1e-15)
        self.app_optimizer = torch.optim.Adam(
            [self.gain, self.bias, self.background], lr=float(lr.appearance))
        self.grad_accum = torch.zeros(n, device=dev)
        self.grad_count = torch.zeros(n, device=dev)

    # ------------------------------------------------------------ helpers
    def _image(self, i):
        """Targets of image i on the GPU, moved per step from their compact
        host copy (uint8 / float16): caching thousands of float32 images on
        the GPU does not fit next to the Gaussians, nor a second pinned copy
        in the host memory limit."""
        im = self.images[i]
        d = self.device
        torch = self.torch

        def tensor(a):  # GPU-resident already (to_gpu), or moved now.
            return a if torch.is_tensor(a) else torch.from_numpy(a).to(d)
        rgb = tensor(im.rgb).float().div_(255.0)
        sky = tensor(im.sky) if im.sky is not None else None
        return (rgb, tensor(im.mask), tensor(im.depth).float(),
                tensor(im.depth_weight).float(), tensor(im.depth_prior), sky)

    def _saturated(self, i):
        """Saturated-pixel mask of image i on the GPU, or None."""
        value = self.images[i].saturated
        if value is None:
            return None
        torch = self.torch
        return value if torch.is_tensor(value) else torch.from_numpy(
            value).to(self.device)

    def saturation_agreement(self, ids, margin=0.05):
        """Share of the saturated pixels of images `ids` rendered no darker
        than `margin` below the saturation level (None without masks)."""
        torch = self.torch
        hit = total = 0
        level = float(self.cfg.saturated) / 255.0 - margin
        with torch.no_grad():
            for i in ids:
                sat = self._saturated(i)
                if sat is None or not sat.any():
                    continue
                rgb, _, _, _ = self.render(i, int(self.cfg.sh_degree))
                hit += int((rgb.max(dim=-1).values[sat] >= level).sum())
                total += int(sat.sum())
        return hit / total if total else None

    def render(self, i, degree, appearance=True):
        from gsplat import rasterization
        from gsplat.cuda._wrapper import RollingShutterType
        torch = self.torch
        im = self.images[i]
        d = self.device
        view = torch.tensor(im.viewmat, device=d, dtype=torch.float32)[None]
        K = torch.tensor(im.K, device=d, dtype=torch.float32)[None]
        kwargs = {}
        if im.model == "fisheye":
            kwargs["radial_coeffs"] = torch.tensor(
                im.radial, device=d, dtype=torch.float32)[None]
        if im.viewmat_rs is not None:
            kwargs["rolling_shutter"] = RollingShutterType.ROLLING_TOP_TO_BOTTOM
            kwargs["viewmats_rs"] = torch.tensor(
                im.viewmat_rs, device=d, dtype=torch.float32)[None]
        colors = torch.cat([self.sh0, self.shn], dim=1)
        h, w = im.rgb.shape[:2]
        out, alpha, meta = rasterization(
            self.means, self.quats, torch.exp(self.log_scales),
            torch.sigmoid(self.logit_op), colors, view, K, w, h,
            sh_degree=degree, packed=False, render_mode="RGB+ED",
            camera_model=im.model, with_ut=True, with_eval3d=True,
            near_plane=float(self.cfg.near_plane),
            far_plane=float(self.cfg.far_plane),
            backgrounds=torch.sigmoid(self.background)[None], **kwargs)
        rgb = out[0, ..., :3]
        if appearance:  # Per-image gain / bias (exposure).
            rgb = rgb * self.gain[i] + self.bias[i]
        return rgb, out[0, ..., 3], alpha[0, ..., 0], meta

    def _lidar_loss(self, rng):
        """Depth and opacity error of one keyframe's LiDAR views."""
        from gsplat import rasterization
        torch = self.torch
        c = self.cfg.lidar_views
        views = self.lidar
        faces = views["faces"]
        g = int(rng.integers(len(views["depth"]) // faces))
        target = views["depth"][g * faces:(g + 1) * faces].float()
        valid = target > 0
        if not valid.any():
            return 0.0
        _, h, w = target.shape
        out, alpha, _ = rasterization(
            self.means, self.quats, torch.exp(self.log_scales),
            torch.sigmoid(self.logit_op), self.sh0[:, 0, :],
            views["viewmats"][g * faces:(g + 1) * faces],
            views["K"][None].expand(faces, 3, 3), w, h, packed=False,
            render_mode="RGB+ED", camera_model="pinhole", with_ut=True,
            with_eval3d=True, near_plane=float(self.cfg.near_plane),
            far_plane=float(self.cfg.far_plane),
            backgrounds=torch.zeros((faces, 3), device=target.device))
        z = out[..., 3][valid]
        t = target[valid]
        return (float(c.weight) * (torch.abs(z - t) / t).mean()
                + float(c.alpha_weight) * (1 - alpha[..., 0][valid]).mean())

    def _replace(self, keep=None, append=None):
        """Applies a boolean `keep` mask and appends new Gaussians (dict of
        tensors), with the Adam states."""
        torch = self.torch
        names = ["means", "log_scales", "quats", "logit_op", "sh0", "shn"]
        tensors = {n: getattr(self, n) for n in names}
        extra = {"anchor": self.anchor, "normals": self.normals,
                 "flags": self.flags, "frozen": self.frozen,
                 "anchor_tol": self.anchor_tol, "intensity": self.intensity,
                 "scale_limit": self.scale_limit,
                 "grad_accum": self.grad_accum, "grad_count": self.grad_count}
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            state = self.optimizer.state.get(old, {})
            new = old.detach()
            if keep is not None:
                new = new[keep]
            if append is not None:
                new = torch.cat([new, append[name]])
            new = new.requires_grad_()
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    s = state[key] if keep is None else state[key][keep]
                    if append is not None:
                        s = torch.cat([s, torch.zeros_like(append[name])])
                    state[key] = s
            if old in self.optimizer.state:
                del self.optimizer.state[old]
            if state:
                self.optimizer.state[new] = state
            group["params"][0] = new
            setattr(self, name, new)
        for name, value in extra.items():
            new = value if keep is None else value[keep]
            if append is not None:
                new = torch.cat([new, append[name]])
            setattr(self, name, new)
        del tensors

    def _prune_and_densify(self, step):
        torch = self.torch
        c = self.cfg
        movable = ~self.frozen
        with torch.no_grad():
            opacity = torch.sigmoid(self.logit_op)
            prune = movable & (opacity < float(c.prune_opacity))
            drift = torch.linalg.norm(self.means - self.anchor, dim=1)
            anchored = (self.flags & FLAG_ANCHORED) > 0
            prune |= movable & anchored & (drift > float(c.anchor_prune))
            means = self.means.detach().cpu().numpy().astype(np.float64)
            if self.levels is not None and self.levels.n:
                _, height, _ = self.levels.assign(means)
                below = np.nan_to_num(height, nan=1.0) < -float(c.below_floor)
                prune |= movable & torch.tensor(below, device=self.device)
            if self.carver is not None:
                hits, misses = self.carver.query(np.ascontiguousarray(means))
                free = (misses >= int(c.free_space.min_misses)) & (hits == 0)
                prune |= movable & ~anchored & torch.tensor(
                    free, device=self.device)
            n_pruned = int(prune.sum())
            # Densify: the largest accumulated 3D gradients (3DGUT has no
            # screen-space means), movable Gaussians only.
            append = None
            score = self.grad_accum / self.grad_count.clamp(min=1)
            score[~movable | prune] = 0
            budget = int(c.max_gaussians) - len(self.means) + n_pruned
            n_new = 0
            if (step <= int(c.densify_until) and budget > 0
                    and (score > 0).any()):
                k = min(budget, int(float(c.densify_fraction) * len(score)))
                if k > 0:
                    top = torch.topk(score, k).indices
                    top = top[score[top] > float(c.densify_grad)]
                    n_new = len(top)
            if n_new:
                scales = torch.exp(self.log_scales[top])
                split = scales.max(dim=1).values > float(c.split_scale)
                offset = torch.randn_like(scales) * scales * split[:, None]
                new_scales = torch.where(split[:, None], scales / 1.6, scales)
                append = {
                    "means": self.means[top] + offset,
                    "log_scales": torch.log(new_scales),
                    "quats": self.quats[top], "logit_op": self.logit_op[top],
                    "sh0": self.sh0[top], "shn": self.shn[top],
                    "anchor": self.means[top] + offset,
                    "normals": self.normals[top],
                    "flags": self.flags[top] & ~FLAG_ANCHORED,
                    "frozen": self.frozen[top],
                    "anchor_tol": self.anchor_tol[top],
                    "scale_limit": self.scale_limit[top],
                    "intensity": self.intensity[top],
                    "grad_accum": torch.zeros(n_new, device=self.device),
                    "grad_count": torch.zeros(n_new, device=self.device)}
                # A split Gaussian is replaced by its two halves.
                with torch.no_grad():
                    self.log_scales[top[split]] = torch.log(new_scales[split])
            self._replace(~prune, append)
            self.grad_accum.zero_()
            self.grad_count.zero_()
        return n_pruned, n_new

    # -------------------------------------------------------------- train
    def train(self):
        torch = self.torch
        c = self.cfg
        iterations = int(c.iterations)
        rng = np.random.default_rng(0)
        start = time.time()
        for step in range(1, iterations + 1):
            degree = min(int(c.sh_degree), step // int(c.sh_interval))
            i = int(self.train_ids[rng.integers(len(self.train_ids))])
            target, mask, depth, depth_w, depth_prior, sky = self._image(i)
            raw, z, alpha, meta = self.render(i, degree, appearance=False)
            rgb = raw * self.gain[i] + self.bias[i]
            m = mask[..., None].float()
            l1 = (torch.abs(rgb - target) * m).sum() / (3 * m.sum() + 1)
            ssim = _ssim(rgb.permute(2, 0, 1)[None] * m.permute(2, 0, 1)[None],
                         target.permute(2, 0, 1)[None] * m.permute(2, 0, 1)[None],
                         torch)
            loss = (1 - float(c.ssim_weight)) * l1 + float(c.ssim_weight) * (
                1 - ssim.mean())
            # Depth: LiDAR targets throughout, prior ones decaying.
            decay = max(0.0, 1.0 - step / max(1.0, float(c.prior_decay_until)))
            w = depth_w * torch.where(depth_prior, float(c.prior_weight) * decay,
                                      1.0)
            valid = (depth > 0) & (w > 0)
            if valid.any():
                rel = torch.abs(z - depth)[valid] / depth[valid]
                loss = loss + float(c.depth_weight) * (rel * w[valid]).sum() / (
                    w[valid].sum() + 1e-6)
            if sky is not None and sky.any() and float(c.sky_weight) > 0:
                # Per image pixel, like the photometric loss: a mean over
                # the few sky pixels outweighs everything else.
                loss = loss + float(c.sky_weight) * (alpha * sky).mean()
            sat = self._saturated(i)
            if sat is not None and sat.any():
                # One-sided: a saturated pixel only says "at least this
                # bright"; the brightest channel is pushed up to the level.
                level = float(c.saturated) / 255.0
                bright = rgb
                if c.get("saturated_fixed_appearance", True):
                    # Through the Gaussians only: raising the image's gain
                    # would brighten every other pixel too.
                    bright = (raw * self.gain[i].detach()
                              + self.bias[i].detach())
                short = torch.relu(level - bright.max(dim=-1).values[sat])
                loss = loss + float(c.saturated_weight) * short.mean()
            scales = torch.exp(self.log_scales)
            movable = ~self.frozen
            if movable.any():
                loss = loss + float(c.flatten_weight) * scales[movable].min(
                    dim=1).values.mean()
                anchored = movable & ((self.flags & FLAG_ANCHORED) > 0)
                if anchored.any():
                    drift = torch.linalg.norm(
                        self.means[anchored] - self.anchor[anchored], dim=1)
                    excess = torch.relu(drift - self.anchor_tol[anchored])
                    loss = loss + float(c.anchor_weight) * (
                        excess ** 2).mean() / 0.01 ** 2
            loss = loss + float(c.appearance_weight) * (
                (self.gain[i] - 1).square().sum() + self.bias[i].square().sum())
            if self.lidar is not None and step % int(c.lidar_views.every) == 0:
                loss = loss + self._lidar_loss(rng)
            self.optimizer.zero_grad(set_to_none=True)
            self.app_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                frozen = self.frozen
                for p in (self.means, self.log_scales, self.quats,
                          self.logit_op):
                    if p.grad is not None:
                        p.grad[frozen] = 0
                visible = meta["radii"][0].max(dim=-1).values > 0 if (
                    meta["radii"].dim() == 3) else meta["radii"][0] > 0
                if self.means.grad is not None:
                    g = torch.linalg.norm(self.means.grad, dim=1)
                    self.grad_accum[visible] += g[visible]
                    self.grad_count[visible] += 1
            self.optimizer.step()
            with torch.no_grad():
                self.log_scales.copy_(torch.minimum(self.log_scales,
                                                    self.scale_limit))
            self.app_optimizer.step()
            if step % int(c.refine_every) == 0 and step >= int(c.densify_from):
                pruned, added = self._prune_and_densify(step)
                if step % (10 * int(c.refine_every)) == 0:
                    self.log(f"[gs]   step {step}/{iterations}: loss "
                             f"{float(loss):.4f}, {len(self.means)} Gaussians "
                             f"(-{pruned} +{added}), "
                             f"{time.time() - start:.0f}s")
        return time.time() - start

    def evaluate(self, ids, preview_dir=None, previews=4, no_sky=False):
        """PSNR / SSIM on the given images (masked pixels only; `no_sky`:
        also without the sky pixels)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for n, i in enumerate(ids):
                target, mask, _, _, _, sky = self._image(i)
                if no_sky:
                    if sky is None:
                        return []
                    mask = mask & ~sky
                rgb, _, _, _ = self.render(i, int(self.cfg.sh_degree))
                rgb = rgb.clamp(0, 1)
                m = mask[..., None].float()
                mse = ((rgb - target) ** 2 * m).sum() / (3 * m.sum() + 1)
                psnr = float(-10 * torch.log10(mse + 1e-10))
                ssim = float(_ssim(rgb.permute(2, 0, 1)[None],
                                   target.permute(2, 0, 1)[None], torch)[
                    0].mean(dim=0)[mask].mean())
                out.append((psnr, ssim))
                if preview_dir is not None and n < previews:
                    import cv2
                    preview_dir.mkdir(parents=True, exist_ok=True)
                    im = self.images[i]
                    both = np.concatenate([
                        (target.cpu().numpy() * 255).astype(np.uint8),
                        (rgb.cpu().numpy() * 255).astype(np.uint8)], axis=1)
                    cv2.imwrite(str(preview_dir / f"{im.camera}_{im.stamp_ns}.jpg"),
                                both[..., ::-1])
        return out

    def neighbour_appearance(self, ids):
        """Sets the gain / bias of images `ids` (held out: never fitted) by
        interpolating in time between the training images of the same trip
        and camera before and after them; no pixel of theirs is used."""
        torch = self.torch
        series = {}
        for j in self.train_ids:
            im = self.images[j]
            series.setdefault((im.trip, im.camera), []).append((im.stamp_ns, j))
        with torch.no_grad():
            for i in ids:
                im = self.images[i]
                ref = sorted(series.get((im.trip, im.camera), []))
                if not ref:
                    continue
                stamps = np.array([t for t, _ in ref], dtype=np.float64)
                k = int(np.searchsorted(stamps, im.stamp_ns))
                a, b = max(k - 1, 0), min(k, len(ref) - 1)
                span = stamps[b] - stamps[a]
                w = 0.0 if span <= 0 else float(
                    np.clip((im.stamp_ns - stamps[a]) / span, 0.0, 1.0))
                ja, jb = ref[a][1], ref[b][1]
                self.gain[i] = (1 - w) * self.gain[ja] + w * self.gain[jb]
                self.bias[i] = (1 - w) * self.bias[ja] + w * self.bias[jb]

    def appearance(self) -> dict:
        """Per-image gain / bias with the image keys."""
        return {
            "trip": np.array([im.trip for im in self.images], np.int32),
            "camera": np.array([im.camera for im in self.images]),
            "stamp_ns": np.array([im.stamp_ns for im in self.images], np.int64),
            "holdout": np.array([im.holdout for im in self.images], bool),
            "gain": self.gain.detach().cpu().numpy(),
            "bias": self.bias.detach().cpu().numpy()}

    def export(self):
        torch = self.torch
        with torch.no_grad():
            return {
                "means": self.means.cpu().numpy(),
                "quats": torch.nn.functional.normalize(
                    self.quats, dim=1).cpu().numpy(),
                "scales": torch.exp(self.log_scales).cpu().numpy(),
                "opacity": torch.sigmoid(self.logit_op).cpu().numpy(),
                "sh0": self.sh0.cpu().numpy(), "shn": self.shn.cpu().numpy(),
                "flags": self.flags.cpu().numpy().astype(np.uint8),
                "intensity": self.intensity.cpu().numpy(),
            }


def write_ply(path: pathlib.Path, model: dict) -> None:
    """Standard 3DGS PLY (x y z, normals, f_dc, f_rest, opacity, scale,
    rot); opacity as a logit, scales as logs."""
    n = len(model["means"])
    rest = model["shn"].transpose(0, 2, 1).reshape(n, -1)  # Channel-major.
    op = np.clip(model["opacity"], 1e-6, 1 - 1e-6)
    columns = [model["means"], np.zeros((n, 3)), model["sh0"][:, 0, :], rest,
               np.log(op / (1 - op))[:, None], np.log(model["scales"]),
               model["quats"]]
    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(rest.shape[1])]
             + ["opacity", "scale_0", "scale_1", "scale_2"]
             + [f"rot_{i}" for i in range(4)])
    data = np.concatenate([np.asarray(c, np.float32).reshape(n, -1)
                           for c in columns], axis=1)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property float {name}\n" for name in names)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.astype("<f4").tobytes())


def run_gs(pose_set: pose_set_lib.PoseSet, point_map: static_map.PointMap,
           levels, keyframes, kf_poses, cfg, work: pathlib.Path,
           log=print) -> dict:
    """Trains the Gaussians block by block and writes work/gs/."""
    out_dir = work / "gs"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = _load_images(pose_set, point_map, cfg, work, log)
    if cfg.gs.get("sky_source", "segmentation") == "segmentation" and (
            float(cfg.gs.get("sky_weight") or 0.0) > 0
            or (cfg.models.get("dino") or {}).get("ade20k_head")):
        # Also without the loss, for holdout_psnr_nosky.
        _segment_sky(images, cfg, work / "gs", log)
    prior_path = work / "depth_prior" / "points.npz"
    prior = dict(np.load(prior_path)) if (
        prior_path.exists() and cfg.gs.use_prior_points) else None
    init = _initial_gaussians(point_map, prior, cfg)
    t = time.time()
    good, colors = _coverage(init["means"], init["normals"], images,
                             pose_set, cfg)
    frozen = good < int(cfg.gs.min_views)
    # Frozen Gaussians keep the LiDAR intensity as grey; prior points their
    # pixel colour.
    grey = np.repeat(init["intensity"][:, None], 3, axis=1)
    colors = np.where(frozen[:, None], grey, colors)
    if "prior_rgb" in init:
        n_prior = len(init["prior_rgb"])
        colors[-n_prior:] = init["prior_rgb"] / 255.0
        frozen[-n_prior:] = False
    init["flags"] |= np.where(frozen, FLAG_FROZEN, FLAG_APPEARANCE).astype(
        np.uint8)
    log(f"[gs] {len(frozen)} Gaussians, {frozen.mean() * 100:.0f}% frozen "
        f"(fewer than {cfg.gs.min_views} usable views), coverage "
        f"{time.time() - t:.0f}s")
    if cfg.gs.get("images_on_gpu", True):
        # The compact arrays (uint8 / float16 / bool) move to the GPU once:
        # host memory (the container limit) then no longer bounds how many
        # runs share the machine.
        import torch
        for im in images:
            for field in ("rgb", "mask", "depth", "depth_weight",
                          "depth_prior", "sky", "saturated"):
                value = getattr(im, field)
                if value is not None:
                    setattr(im, field, torch.from_numpy(
                        np.ascontiguousarray(value)).to("cuda"))
    lo = point_map.xyz.min(axis=0) - 1.0
    hi = point_map.xyz.max(axis=0) + 1.0
    carver = None
    if cfg.gs.free_space.enabled:
        carver = _free_space(keyframes, kf_poses, (lo, hi), cfg)
    lidar = None
    if (cfg.gs.get("lidar_views") or {}).get("enabled"):
        lidar = _lidar_views(keyframes, kf_poses, point_map, cfg, log)
    blocks = _blocks(init["means"], float(cfg.gs.block_size),
                     int(cfg.gs.block_max_gaussians))
    reach = float(cfg.gs.image_reach)
    # Images taken up to `reach` outside a core see that far: the block holds
    # the Gaussians at least as far out, or stretched Gaussians fill the gap.
    margin = max(float(cfg.gs.block_margin), reach)
    merged, summary, appearance = [], {"blocks": []}, []
    for b, (core_min, core_max) in enumerate(blocks):
        xy = init["means"][:, :2]
        inside = np.all((xy >= core_min - margin) & (xy <= core_max + margin),
                        axis=1)
        ids = [i for i, im in enumerate(images)
               if np.all((im.center[:2] >= core_min - reach)
                         & (im.center[:2] <= core_max + reach))]
        if inside.sum() == 0 or not any(not images[i].holdout for i in ids):
            continue
        block_init = {k: v[inside] for k, v in init.items()
                      if k != "prior_rgb"}
        block_images = [images[i] for i in ids]
        log(f"[gs] block {b + 1}/{len(blocks)}: {inside.sum()} Gaussians, "
            f"{len(block_images)} images")
        block_lidar = None
        if lidar is not None:
            # Whole keyframes (all faces) whose origin is near the block.
            near = np.all((lidar["centers"][:, :2] >= core_min - reach)
                          & (lidar["centers"][:, :2] <= core_max + reach),
                          axis=1).reshape(-1, lidar["faces"]).all(axis=1)
            take = np.repeat(near, lidar["faces"])
            block_lidar = {**lidar, "viewmats": lidar["viewmats"][take],
                           "depth": lidar["depth"][take]}
        trainer = BlockTrainer(block_init, colors[inside], frozen[inside],
                               block_images, carver, levels, cfg, log,
                               block_lidar)
        seconds = trainer.train()
        holdout = [i for i, im in enumerate(block_images) if im.holdout]
        train = trainer.train_ids[::max(1, len(trainer.train_ids) // 50)]
        raw = trainer.evaluate(holdout)
        trainer.neighbour_appearance(holdout)
        test = trainer.evaluate(holdout, out_dir / "previews")
        bright = trainer.saturation_agreement(holdout)
        no_sky = trainer.evaluate(holdout, no_sky=True)
        fit = trainer.evaluate(train)
        appearance.append({"block": np.full(len(block_images), b, np.int32),
                           **trainer.appearance()})
        model = trainer.export()
        core = np.all((model["means"][:, :2] >= core_min)
                      & (model["means"][:, :2] < core_max), axis=1)
        merged.append({k: v[core] for k, v in model.items()})
        stats = {"block": b, "gaussians": int(core.sum()), "images": len(ids),
                 "seconds": seconds,
                 "train_psnr": float(np.mean([p for p, _ in fit])) if fit else None,
                 "holdout_psnr": float(np.mean([p for p, _ in test])) if test else None,
                 "holdout_ssim": float(np.mean([s for _, s in test])) if test else None,
                 "holdout_psnr_raw": float(np.mean([p for p, _ in raw])) if raw else None,
                 "holdout_ssim_raw": float(np.mean([s for _, s in raw])) if raw else None,
                 "holdout_saturated_bright": bright,
                 "holdout_psnr_nosky": float(np.mean([p for p, _ in no_sky])) if no_sky else None,
                 "frozen_fraction": float(np.mean(
                     (model["flags"][core] & FLAG_FROZEN) > 0))}
        summary["blocks"].append(stats)
        log(f"[gs] block {b + 1}: {stats}")
        del trainer
        import torch
        torch.cuda.empty_cache()
    model = {k: np.concatenate([m[k] for m in merged]) for k in merged[0]}
    np.savez(out_dir / "model.npz", **model)
    np.savez(out_dir / "appearance.npz", **{
        k: np.concatenate([a[k] for a in appearance]) for k in appearance[0]})
    write_ply(out_dir / "model.ply", model)
    summary["gaussians"] = int(len(model["means"]))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
