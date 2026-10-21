"""The gs_verify and ortho stages (README 8.3).

gs_verify checks the trained Gaussians against the LiDAR, which they must not
contradict:
* rendered depth along LiDAR sweeps that are not keyframes (the map is built
  from keyframes only, so these rays are held out), C++ GaussianRayRenderer:
  range error per ray;
* distance of every opaque Gaussian to the LiDAR map surface, per source
  (L anchored on a LiDAR point, G densified, P depth prior; frozen
  separately), and a top view of the mean distance per cell;
* with a registered handheld reference (acceptance), the same distance to
  the reference cloud per source.

ortho renders every level from straight above with gsplat's orthographic
camera, Gaussians clipped to the level's slab (floor to below the ceiling),
in tiles: an orthophoto with alpha that replaces the 5 cm floor texture.

Outputs: gs/verify.json, gs/verify_top.png, ortho/level_<i>.png (+ .json
with x0, y0 of the top-left pixel and the resolution).
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon.recon import gs as gs_lib

SOURCES = {"L": gs_lib.FLAG_ANCHORED, "P": gs_lib.FLAG_PRIOR}


def _sources(flags: np.ndarray) -> dict[str, np.ndarray]:
    anchored = (flags & gs_lib.FLAG_ANCHORED) > 0
    prior = (flags & gs_lib.FLAG_PRIOR) > 0
    frozen = (flags & gs_lib.FLAG_FROZEN) > 0
    return {"L": anchored & ~frozen, "G": ~anchored & ~prior & ~frozen,
            "P": prior & ~frozen, "frozen": frozen}


def _stats(errors: np.ndarray) -> dict:
    e = np.abs(errors[np.isfinite(errors)])
    if len(e) == 0:
        return {"n": 0}
    return {"n": int(len(e)), "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)),
            "within_2cm": float(np.mean(e < 0.02)),
            "within_5cm": float(np.mean(e < 0.05)),
            "within_10cm": float(np.mean(e < 0.10))}


def _held_out_rays(pose_set, keyframe_frames, cfg, rng):
    """(origins, directions, ranges) of rays of non-keyframe sweeps."""
    section = cfg.gs_verify
    origins, directions, ranges = [], [], []
    for t, data in enumerate(pose_set.datasets):
        T_vl = pose_set.rig(t).primary_lidar.T_vehicle_lidar
        trajectory = pose_set.trajectory(t)
        used = set(int(k) for k in keyframe_frames.get(data.trip_id, []))
        frames = [k for k in range(len(data)) if k not in used]
        for k in frames[::max(1, int(section.sweep_stride))]:
            f = data.load_lidar(k)
            r = np.linalg.norm(f.points, axis=1)
            keep = (r > float(section.min_range)) & (r < float(section.max_range))
            if keep.sum() < 100:
                continue
            idx = np.nonzero(keep)[0]
            idx = rng.choice(idx, min(len(idx), int(section.rays_per_sweep)),
                             replace=False)
            local = _core.deskew_sweep(
                np.ascontiguousarray(f.points[idx], dtype=np.float64),
                f.time[idx].astype(np.float64), f.stamp, trajectory, T_vl)
            pose = pose_set.world_from_lidar(t, f.stamp)[0]
            world = local @ pose[:3, :3].T + pose[:3, 3]
            d = world - pose[:3, 3]
            n = np.linalg.norm(d, axis=1)
            origins.append(np.repeat(pose[None, :3, 3], len(d), axis=0))
            directions.append(d / n[:, None])
            ranges.append(n)
    if not origins:
        return None
    return (np.concatenate(origins), np.concatenate(directions),
            np.concatenate(ranges))


def _top_view(xy, values, resolution, path, vmax=0.10):
    """Mean |value| per cell as a colour image (blue 0 .. red vmax)."""
    lo = xy.min(axis=0)
    ij = ((xy - lo) / resolution).astype(int)
    shape = ij.max(axis=0)[::-1] + 1
    total = np.zeros(shape)
    count = np.zeros(shape)
    np.add.at(total, (ij[:, 1], ij[:, 0]), np.abs(values))
    np.add.at(count, (ij[:, 1], ij[:, 0]), 1)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    norm = np.clip(np.nan_to_num(mean, nan=0.0) / vmax, 0, 1)
    image = cv2.applyColorMap((255 * norm).astype(np.uint8), cv2.COLORMAP_JET)
    image[count == 0] = 255
    cv2.imwrite(str(path), image[::-1])  # North up.


def run_gs_verify(pose_set, point_map, keyframe_frames, cfg,
                  work: pathlib.Path, reference=None, log=print) -> dict:
    """Checks gs/model.npz against the LiDAR (and a reference cloud).

    Args:
        keyframe_frames: {trip id: frame indices of the keyframes}.
        reference: optional (N, 3) reference cloud in the map frame.
    """
    section = cfg.gs_verify
    model = dict(np.load(work / "gs" / "model.npz"))
    rng = np.random.default_rng(0)
    report = {}
    # 1. Held-out LiDAR rays.
    rays = _held_out_rays(pose_set, keyframe_frames, cfg, rng)
    if rays is not None:
        config = core_config.fill(_core.RayRenderConfig(), section.get("render"))
        renderer = _core.GaussianRayRenderer(
            np.ascontiguousarray(model["means"], dtype=np.float64),
            model["quats"].astype(np.float64), model["scales"].astype(np.float64),
            model["opacity"].astype(np.float64), None, None, config)
        origins, directions, ranges = rays
        out = renderer.render(np.ascontiguousarray(origins),
                              np.ascontiguousarray(directions), -1)
        err = out["depth"] - ranges
        report["lidar_rays"] = {**_stats(err),
                                "rays": int(len(ranges)),
                                "missed": float(np.mean(~np.isfinite(err)))}
        log(f"[gs_verify] held-out LiDAR rays: {report['lidar_rays']}")
    # 2. Opaque Gaussians against the LiDAR map surface, per source.
    opaque = model["opacity"] > float(section.min_opacity)
    means = np.ascontiguousarray(model["means"][opaque], dtype=np.float64)
    surface = np.ascontiguousarray(
        point_map.xyz if point_map.dynamic is None
        else point_map.xyz[~point_map.dynamic], dtype=np.float64)
    signed, nearest, _ = _core.surface_distances(
        means, surface, float(section.surface_radius),
        float(section.max_distance))
    dist = np.where(np.isfinite(signed), np.abs(signed), nearest)
    sources = _sources(model["flags"][opaque])
    report["to_lidar_map"] = {
        name: {**_stats(dist[mask]),
               "beyond_max": float(np.mean(~np.isfinite(dist[mask])))
               if mask.any() else None}
        for name, mask in sources.items()}
    log(f"[gs_verify] opaque Gaussians to the LiDAR map: "
        f"{ {k: (v.get('median'), v.get('within_5cm')) for k, v in report['to_lidar_map'].items()} }")
    finite = np.isfinite(dist)
    _top_view(means[finite, :2], dist[finite], float(section.top_resolution),
              work / "gs" / "verify_top.png")
    # 3. Against the handheld reference.
    if reference is not None:
        signed, nearest, _ = _core.surface_distances(
            means, np.ascontiguousarray(reference, dtype=np.float64),
            float(section.surface_radius), float(section.max_distance))
        dist = np.where(np.isfinite(signed), np.abs(signed), nearest)
        report["to_reference"] = {name: _stats(dist[mask])
                                  for name, mask in sources.items()}
        log(f"[gs_verify] opaque Gaussians to the reference: "
            f"{ {k: (v.get('median'), v.get('within_5cm')) for k, v in report['to_reference'].items()} }")
    (work / "gs" / "verify.json").write_text(json.dumps(report, indent=1))
    return report


def run_ortho(levels, cfg, work: pathlib.Path, log=print) -> list[dict]:
    """Orthophoto of every level from the Gaussians."""
    import torch
    from gsplat import rasterization
    section = cfg.ortho
    model = dict(np.load(work / "gs" / "model.npz"))
    out_dir = work / "ortho"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = float(section.resolution)
    tile = int(section.tile)
    dev = torch.device("cuda")
    results = []
    for lv in range(levels.n):
        floor = float(levels.z[lv])
        top = float(levels.ceiling[lv]) - float(section.below_ceiling)
        slab = ((model["means"][:, 2] > floor - float(section.below_floor))
                & (model["means"][:, 2] < top)
                & (model["opacity"] > float(section.min_opacity)))
        if slab.sum() == 0:
            continue
        m = {k: torch.tensor(v[slab], device=dev, dtype=torch.float32)
             for k, v in model.items() if k in ("means", "quats", "scales",
                                                 "opacity", "sh0", "shn")}
        colors = torch.cat([m["sh0"], m["shn"]], dim=1)
        degree = int(round(np.sqrt(colors.shape[1]))) - 1
        xy = model["means"][slab, :2]
        lo = np.percentile(xy, 0.1, axis=0)
        hi = np.percentile(xy, 99.9, axis=0)
        width = int(np.ceil((hi[0] - lo[0]) / res))
        height = int(np.ceil((hi[1] - lo[1]) / res))
        image = np.zeros((height, width, 4), np.uint8)
        # Camera above the slab looking down: x right = world x, image rows
        # go south (world -y), view axis world -z.
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
        for r0 in range(0, height, tile):
            for c0 in range(0, width, tile):
                h = min(tile, height - r0)
                w = min(tile, width - c0)
                center = np.array([lo[0] + (c0 + w / 2) * res,
                                   hi[1] - (r0 + h / 2) * res, top + 1.0])
                view = np.eye(4)
                view[:3, :3] = R
                view[:3, 3] = -R @ center
                K = np.array([[1 / res, 0, w / 2], [0, 1 / res, h / 2],
                              [0, 0, 1]])
                with torch.no_grad():
                    rgb, alpha, _ = rasterization(
                        m["means"], m["quats"], m["scales"], m["opacity"],
                        colors, torch.tensor(view, device=dev,
                                             dtype=torch.float32)[None],
                        torch.tensor(K, device=dev, dtype=torch.float32)[None],
                        w, h, sh_degree=degree, camera_model="ortho",
                        packed=False, near_plane=0.0, far_plane=1e3)
                image[r0:r0 + h, c0:c0 + w, :3] = (
                    rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                image[r0:r0 + h, c0:c0 + w, 3] = (
                    alpha[0, ..., 0].clamp(0, 1).cpu().numpy() * 255).astype(
                        np.uint8)
        path = out_dir / f"level_{lv}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA))
        meta = {"level": lv, "x0": float(lo[0]), "y0": float(hi[1]),
                "resolution": res, "width": width, "height": height,
                "gaussians": int(slab.sum()),
                "covered": float(np.mean(image[..., 3] > 127))}
        (out_dir / f"level_{lv}.json").write_text(json.dumps(meta, indent=1))
        results.append(meta)
        log(f"[ortho] level {lv}: {width}x{height} px at {res * 100:.0f} cm, "
            f"{meta['covered'] * 100:.0f}% covered")
    return results
