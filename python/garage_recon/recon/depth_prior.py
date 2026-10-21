"""The depth_prior stage (README 8.2): DA3 depth in the LiDAR holes.

Per trip, the images of all cameras go to the depth model in time-ordered
batches with the pose set's camera poses (DA3 is pose-conditioned and
multi-view). Each predicted range image is then fitted to the LiDAR map
rendered into the same image and fused with it in C++ (FuseDepthPrior): a
robust scale / shift per image, images and cells that still disagree
rejected, the prior kept only where the LiDAR map has no surface.

Optionally (`context_frames`) each batch also carries images taken elsewhere
in the trip, whose predictions are dropped: consecutive images span a short
baseline, and far structure (the LiDAR holes beyond its range) gets low
model confidence there.

Outputs (work/depth_prior/):
    <trip>/<camera>/<stamp_ns>.npz  range (float16, NaN = none), confidence
                                    (uint8, /255), source (uint8, DEPTH_*).
    frames.csv                      per image: alignment, holes filled and the
                                    model's own pose against ours.
    points.npz                      prior points of the holes (xyz, rgb,
                                    confidence, trip): source P for the 3DGS
                                    initialisation, never measurement.
    summary.json                    per camera statistics.
"""

from __future__ import annotations

import csv
import json
import pathlib
from concurrent import futures

import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon import geometry
from garage_recon.models import interfaces
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.recon import static_map

_FIELDS = ("trip", "camera", "stamp_ns", "valid", "scale", "shift",
           "raw_ratio", "pixels",
           "residual_median", "residual_p90", "inlier_fraction",
           "cells_checked", "cells_rejected", "lidar_fraction",
           "prior_fraction", "pose_rot_deg", "pose_trans_m")


def align_config(cfg) -> _core.DepthAlignConfig:
    return core_config.fill(_core.DepthAlignConfig(), cfg.depth_prior.align)


def _pose_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = geometry.inv(a) @ b
    c = np.clip((np.trace(d[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c))), float(np.linalg.norm(d[:3, 3]))


def _prior_points(camera: _core.CameraModel, world_from_camera: np.ndarray,
                  rng: np.ndarray, confidence: np.ndarray, source: np.ndarray,
                  image: np.ndarray | None, stride: int, min_confidence: float):
    """World points of the prior pixels on a `stride` grid."""
    rows, cols = np.mgrid[0:rng.shape[0]:stride, 0:rng.shape[1]:stride]
    rows, cols = rows.ravel(), cols.ravel()
    keep = (source[rows, cols] == _core.DEPTH_PRIOR) & (
        confidence[rows, cols] >= min_confidence)
    rows, cols = rows[keep], cols[keep]
    if len(rows) == 0:
        return None
    rays, ok = camera.unproject(np.ascontiguousarray(
        np.column_stack([cols, rows]).astype(np.float64)))
    rays, rows, cols = rays[ok], rows[ok], cols[ok]
    local = rays * rng[rows, cols, None]
    xyz = local @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    rgb = (image[rows, cols] if image is not None
           else np.zeros((len(rows), 3), np.uint8))
    return xyz, rgb, confidence[rows, cols]


def _context(images, centers, batch, count, mode, near, far, offset):
    """`count` extra images for a batch: evenly over the trip (`spread`) or
    among the cameras `near`..`far` m from the batch (`distance`)."""
    if count <= 0:
        return []
    inside = set(batch)
    middle = np.mean([centers[key] for key in batch], axis=0)
    candidates = [key for key in images if key not in inside]
    if mode == "distance":
        candidates = [key for key in candidates if near <= np.linalg.norm(
            centers[key] - middle) <= far]
    elif mode != "spread":
        raise ValueError(f"depth_prior.context_mode: unknown {mode!r}")
    if len(candidates) <= count:
        return candidates
    step = len(candidates) / count
    return [candidates[int((k + 0.5 + offset) * step) % len(candidates)]
            for k in range(count)]


def run_depth_prior(pose_set: pose_set_lib.PoseSet,
                    point_map: static_map.PointMap,
                    model: interfaces.DepthPrior, cfg, out_dir: pathlib.Path,
                    log=print) -> dict:
    """Runs the depth model on every image and fuses it with the LiDAR map.

    Returns:
        The summary (also written to out_dir/summary.json).
    """
    section = cfg.depth_prior
    config = align_config(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    static = point_map.xyz if point_map.dynamic is None else (
        point_map.xyz[~point_map.dynamic])
    static = np.ascontiguousarray(static, dtype=np.float64)
    # The fit uses visible map points at their sub-pixel position (a z-buffer
    # reads short on grazing surfaces); a thinned set is enough.
    fit_points = np.ascontiguousarray(static[_core.voxel_downsample_indices(
        static, float(section.fit_voxel))])
    needs_images = getattr(model, "needs_images", True)
    stride = max(1, int(section.frame_stride))
    chunk = max(1, int(section.chunk_frames))
    static_index = _core.PointCloudIndex(static)
    fit_index = _core.PointCloudIndex(fit_points)
    rows_out, points, summary = [], [], {"cameras": {}}

    def fuse(t, data, cameras, frame, pose, pred, local, local_fit):
        """LiDAR render, fit, fusion and outputs of one image."""
        camera = cameras[frame.camera]
        camera_from_world = geometry.inv(pose_set.camera_exposure(
            t, frame.camera, frame.stamp_ns))
        lidar, _, _ = _core.render_points(
            camera, camera_from_world, local, None,
            float(section.point_spacing), 1, 6, float(config.min_range),
            float(section.max_range))
        uv, inside, visible, ranges = _core.project_visibility(
            camera, camera_from_world, local_fit, lidar,
            float(section.visibility_abs_tol),
            float(section.visibility_rel_tol))
        seen = inside & visible
        samples = np.column_stack([uv[seen], ranges[seen]])
        lidar = np.where(lidar > 0, lidar, np.nan).astype(np.float32)
        rng, confidence, source, alignment = _core.fuse_depth_prior(
            pred.depth, pred.confidence, lidar, samples, config)
        # The z-buffer decides occlusion and the holes; the stored LiDAR range
        # is the mean of the first surface (its nearest point reads short).
        surface = _core.render_surface_depth(
            camera, camera_from_world, local, float(section.point_spacing), 1,
            6, float(config.min_range), float(section.max_range),
            float(section.surface_abs_tol), float(section.surface_rel_tol), 1)
        is_lidar = source == _core.DEPTH_LIDAR
        rng = np.where(is_lidar, surface, rng)
        source = np.where(is_lidar & ~np.isfinite(surface),
                          _core.DEPTH_NONE, source).astype(np.uint8)
        confidence = np.where(source == _core.DEPTH_NONE, 0.0, confidence)
        rot, trans = (_pose_difference(pose, pred.pose_estimate)
                      if pred.pose_estimate is not None else (np.nan, np.nan))
        directory = out_dir / data.trip_id / frame.camera
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / f"{frame.stamp_ns}.npz", range=rng.astype(np.float16),
            confidence=np.round(255 * confidence).astype(np.uint8),
            source=source)
        row = {"trip": data.trip_id, "camera": frame.camera,
               "stamp_ns": frame.stamp_ns,
               **{k: alignment[k] for k in (
                   "valid", "scale", "shift", "raw_ratio", "pixels",
                   "residual_median", "residual_p90", "inlier_fraction",
                   "cells_checked", "cells_rejected")},
               "lidar_fraction": float(np.mean(source == _core.DEPTH_LIDAR)),
               "prior_fraction": float(np.mean(source == _core.DEPTH_PRIOR)),
               "pose_rot_deg": rot, "pose_trans_m": trans}
        found = _prior_points(camera, pose, rng, confidence, source,
                              frame.image, int(section.points_stride),
                              float(section.min_point_confidence))
        if found is not None:
            found = (*found, np.full(len(found[0]), t, np.int16))
        return row, found

    # The model runs on the next batch (GPU thread) while the current one is
    # fused (CPU threads: the C++ calls release the GIL).
    with futures.ThreadPoolExecutor(1) as gpu, futures.ThreadPoolExecutor(
            max(1, int(section.workers))) as cpu:
        for t, data in enumerate(pose_set.datasets):
            rig = pose_set.rig(t)
            names = list(rig.cameras)
            if not names:
                continue
            cameras = {n: rig.cameras[n].make_model() for n in names}
            # (stamp, camera) of the images, in time order across cameras.
            images = sorted((int(ns), n) for n in names
                            for ns in data.camera_stamps_ns(n)[::stride])
            size = chunk * len(names)
            batches = [images[s:s + size] for s in range(0, len(images), size)]
            count = int(section.get("context_frames") or 0)
            where = ({(ns, name): pose_set.camera_center_pose(t, name, ns)[:3, 3]
                      for ns, name in images} if count > 0 else {})
            views = getattr(model, "batch_views", None)
            if count > 0 and views and 2 * (size + count) > views:
                log(f"[depth_prior] warning: {size} images + {count} context "
                    f"need {2 * (size + count)} views per model pass, "
                    f"models.da3.batch_views is {views}")

            def predict(batch, b, t=t, data=data, cameras=cameras,
                        images=images, where=where, count=count):
                extra = _context(images, where, batch, count,
                                 str(section.get("context_mode", "distance")),
                                 float(section.get("context_near", 15.0)),
                                 float(section.get("context_far", 40.0)),
                                 (b * 0.618) % 1.0)
                frames, poses = [], []
                for ns, name in list(batch) + extra:
                    image = data.load_image(name, ns) if needs_images else None
                    frames.append(interfaces.Frame(data.trip_id, name, ns,
                                                   image))
                    poses.append(pose_set.camera_center_pose(t, name, ns))
                out = model.predict(frames, poses, cameras)
                n = len(batch)
                return frames[:n], poses[:n], out[:n]

            pending = gpu.submit(predict, batches[0], 0) if batches else None
            done = 0
            for b in range(len(batches)):
                frames, poses, predictions = pending.result()
                pending = (gpu.submit(predict, batches[b + 1], b + 1)
                           if b + 1 < len(batches) else None)
                # Map points within reach of this batch's cameras.
                centers = np.array([p[:3, 3] for p in poses])
                middle = centers.mean(axis=0)
                radius = float(section.max_range) + float(
                    np.linalg.norm(centers - middle, axis=1).max())
                local = np.ascontiguousarray(
                    static[static_index.radius_search(middle, radius)])
                local_fit = np.ascontiguousarray(
                    fit_points[fit_index.radius_search(middle, radius)])
                jobs = [cpu.submit(fuse, t, data, cameras, f, p, pred, local,
                                   local_fit)
                        for f, p, pred in zip(frames, poses, predictions)]
                for job in jobs:
                    row, found = job.result()
                    rows_out.append(row)
                    if found is not None:
                        points.append(found)
                done += len(frames)
                log(f"[depth_prior] {data.trip_id}: {done}/{len(images)} "
                    f"images")
    with open(out_dir / "frames.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows_out)
    if points:
        xyz = np.concatenate([p[0] for p in points])
        keep = _core.voxel_downsample_indices(
            np.ascontiguousarray(xyz), float(section.points_voxel))
        np.savez_compressed(
            out_dir / "points.npz", xyz=xyz[keep],
            rgb=np.concatenate([p[1] for p in points])[keep],
            confidence=np.concatenate([p[2] for p in points])[keep],
            trip=np.concatenate([p[3] for p in points])[keep])
        summary["prior_points"] = int(len(keep))
    for name in sorted({r["camera"] for r in rows_out}):
        rows = [r for r in rows_out if r["camera"] == name]
        valid = [r for r in rows if r["valid"]]
        stats = {"images": len(rows), "valid": len(valid)}
        if valid:
            def pct(key, q):
                return float(np.nanpercentile([r[key] for r in valid], q))
            stats.update({
                "scale_p10_p50_p90": [pct("scale", q) for q in (10, 50, 90)],
                "residual_median": pct("residual_median", 50),
                "lidar_fraction": pct("lidar_fraction", 50),
                "prior_fraction": pct("prior_fraction", 50)})
            rot = [r["pose_rot_deg"] for r in rows
                   if np.isfinite(r["pose_rot_deg"])]
            if rot:
                stats["model_pose_rot_deg_p50_p90"] = [
                    float(np.percentile(rot, 50)), float(np.percentile(rot, 90))]
                stats["model_pose_trans_m_p50_p90"] = [
                    float(np.percentile([r["pose_trans_m"] for r in rows
                                         if np.isfinite(r["pose_trans_m"])],
                                        q)) for q in (50, 90)]
        summary["cameras"][name] = stats
        log(f"[depth_prior] {name}: {stats}")
    summary["model"] = getattr(model, "name", type(model).__name__)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
