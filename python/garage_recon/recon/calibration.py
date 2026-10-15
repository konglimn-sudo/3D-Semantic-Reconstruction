"""Camera calibration on the shared trajectory (time offset and extrinsic).

For every camera, images of a trip are matched pairwise (a Matcher: DINO /
RoMa features on real data, the ground-truth oracle on synthetic data); each
correspondence is a two-view track. The C++ solver then estimates the
camera's time offset and vehicle_from_camera with the trajectories fixed and
the landmarks held on the LiDAR surface. Only these global parameters move,
so the cameras stay on the one pose set; the result is a new pose set
version.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon.models import interfaces
from garage_recon.recon import keyframes as keyframes_lib
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.recon import static_map


def lidar_surface(keyframes: keyframes_lib.KeyframeSet,
                  pose_set: pose_set_lib.PoseSet, voxel: float,
                  normal_radius: float,
                  max_range: float) -> tuple[np.ndarray, np.ndarray]:
    """LiDAR points and normals of all keyframes on the pose set's poses."""
    poses = keyframe_poses(keyframes, pose_set)
    point_map = static_map.accumulate(keyframes, poses, voxel, max_range)
    static_map.compute_normals(point_map, poses, normal_radius)
    return point_map.xyz, point_map.normals.astype(np.float64)


def keyframe_poses(keyframes: keyframes_lib.KeyframeSet,
                   pose_set: pose_set_lib.PoseSet) -> np.ndarray:
    """(K, 4, 4) world <- lidar of every keyframe from the pose set."""
    poses = np.zeros((len(keyframes), 4, 4))
    for t in range(len(pose_set.trips)):
        idx = keyframes.indices_of_trip(t)
        if len(idx):
            poses[idx] = pose_set.world_from_lidar(t, keyframes.stamp[idx])
    return poses


def _query_points(matcher: interfaces.Matcher, frame: interfaces.Frame,
                  calib, grid: int) -> np.ndarray:
    """Query pixels of an anchor image (the matcher's own, else a grid)."""
    if hasattr(matcher, "query_points"):
        return matcher.query_points(frame)
    vv, uu = np.mgrid[grid // 2:calib.height:grid, grid // 2:calib.width:grid]
    return np.column_stack([uu.ravel() + 0.5, vv.ravel() + 0.5])


def calibrate_cameras(pose_set: pose_set_lib.PoseSet,
                      keyframes: keyframes_lib.KeyframeSet,
                      matcher: interfaces.Matcher, cfg,
                      log=print) -> tuple[pose_set_lib.PoseSet, dict]:
    """Refines the time offset and extrinsic of every camera.

    Anchor images (every `anchor_step`-th) are matched into the next
    `track_length` - 1 images of the same camera and trip at the anchor's
    query points, which yields multi-view tracks.

    Args:
        pose_set: Current pose set (trajectories fixed).
        keyframes: Final keyframe sweeps (for the LiDAR surface).
        matcher: Image matcher.
        cfg: Pipeline configuration (`calibration` section).
        log: Logger.

    Returns:
        (new pose set, report per camera). Cameras whose solution fails the
        quality gates keep their previous calibration.
    """
    section = cfg.calibration
    surface, normals = lidar_surface(keyframes, pose_set,
                                     float(section.surface_voxel),
                                     float(section.normal_radius),
                                     float(cfg.map.max_range))
    trajectories = [pose_set.trajectory(t) for t in range(len(pose_set.trips))]
    rig = pose_set.rig(0)
    names = section.cameras or list(rig.cameras)
    rng = np.random.default_rng(0)
    needs_images = getattr(matcher, "needs_images", True)
    new_calibrations = {}
    report = {}
    for name in names:
        calib = rig.cameras[name]
        image_trip, image_ns = [], []
        for t, data in enumerate(pose_set.datasets):
            for ns in data.camera_stamps_ns(name)[::int(section.frame_stride)]:
                image_trip.append(t)
                image_ns.append(int(ns))

        def frame(k: int, camera: str = name) -> interfaces.Frame:
            data = pose_set.datasets[image_trip[k]]
            image = (data.load_image(camera, image_ns[k]) if needs_images
                     else None)
            return interfaces.Frame(data.trip_id, camera, image_ns[k], image)

        # Multi-view tracks: query points of an anchor image matched into
        # the next images of the same trip.
        tracks = []
        for a in range(0, len(image_ns), int(section.anchor_step)):
            anchor = frame(a)
            points = _query_points(matcher, anchor, calib,
                                   int(section.query_grid))
            observations = [[(a, *uv)] for uv in points]
            for b in range(a + 1, min(a + int(section.track_length),
                                      len(image_ns))):
                if image_trip[b] != image_trip[a]:
                    break
                matches = matcher.match(anchor, frame(b), points)
                for i, uv in zip(matches.index_a, matches.uv_b):
                    observations[int(i)].append((b, *uv))
            long_enough = [o for o in observations if len(o) >= 2]
            if len(long_enough) > section.tracks_per_anchor:
                pick = rng.choice(len(long_enough),
                                  int(section.tracks_per_anchor),
                                  replace=False)
                long_enough = [long_enough[i] for i in sorted(pick)]
            tracks += [np.asarray(o, float) for o in long_enough]
        if not tracks:
            log(f"[calibration] {name}: no correspondences, unchanged")
            continue
        result = _core.calibrate_camera(
            calib.make_model(), calib.T_vehicle_cam, calib.time_offset,
            calib.rolling_shutter_line_time, trajectories,
            np.asarray(image_trip), np.asarray(image_ns, float) * 1e-9,
            tracks, surface, normals,
            core_config.camera_calibration_config(cfg))
        accepted = (result["landmarks"] >= section.min_landmarks
                    and result["rms_after"] <= section.max_rms_after)
        entry = {
            "accepted": bool(accepted),
            "images": len(image_ns),
            "tracks": len(tracks),
            "landmarks": int(result["landmarks"]),
            "rms_before_px": float(result["rms_before"]),
            "rms_after_px": float(result["rms_after"]),
            "time_offset_s": float(result["time_offset"]),
            "time_offset_sigma_s": float(result["time_offset_sigma"]),
            "rotation_sigma_deg": np.degrees(
                result["rotation_sigma"]).tolist(),
            "translation_sigma_m": np.asarray(
                result["translation_sigma"]).tolist(),
            "T_vehicle_cam": np.asarray(result["T_vehicle_cam"]).tolist(),
        }
        report[name] = entry
        log(f"[calibration] {name}: {entry['landmarks']} landmarks, rms "
            f"{entry['rms_before_px']:.2f} -> {entry['rms_after_px']:.2f} px, "
            f"time offset {1e3 * entry['time_offset_s']:.1f} ms "
            f"(+-{1e3 * entry['time_offset_sigma_s']:.2f})"
            f"{'' if accepted else ' -- rejected, keeping the previous'}")
        if accepted:
            new_calibrations[name] = dataclasses.replace(
                calib, T_vehicle_cam=np.asarray(result["T_vehicle_cam"]),
                time_offset=float(result["time_offset"]))
    return pose_set.with_cameras(new_calibrations), report
