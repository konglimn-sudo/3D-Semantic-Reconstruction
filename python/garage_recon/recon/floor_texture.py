"""Floor orthophoto from the surround-view fisheyes (section 4.2).

Every floor cell keeps its best views (close, top-down, not occluded by
parked cars) and fuses them with a median, which removes specular highlights
of epoxy floors and transient occluders. Each image row is projected with
its own exposure pose from the pose set (rolling shutter, time offset).
"""

from __future__ import annotations

import numpy as np

from garage_recon import _core
from garage_recon import geometry
from garage_recon import parallel
from garage_recon.recon import bev
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.recon import static_map


def fuse_floor_texture(pose_set: pose_set_lib.PoseSet, grid: bev.BevGrid,
                       floor_z: np.ndarray, level_z: float,
                       point_map: static_map.PointMap, cfg, log=print):
    """Fuses the floor texture of one level.

    Returns:
        (rgb (H, W, 3), best score (H, W), observations (H, W)).
    """
    section = cfg.floor_texture
    fuser = _core.FloorTextureFuser(grid.x0, grid.y0, grid.res, grid.width,
                                    grid.height, floor_z, int(section.top_k))
    occluders = None
    if section.occlusion_test:  # Parked cars and columns hide the floor.
        static = (point_map.xyz if point_map.dynamic is None
                  else point_map.xyz[~point_map.dynamic])
        occluders = _core.PointCloudIndex(static)
    frames = []
    for t, data in enumerate(pose_set.datasets):
        rig = pose_set.rig(t)
        for name in rig.cameras:  # Fisheye and pinhole alike.
            camera = rig.cameras[name].make_model()
            for ns in data.camera_stamps_ns(name):
                center = pose_set.camera_center_pose(t, name, int(ns))
                vehicle = center @ geometry.inv(rig.cameras[name].T_vehicle_cam)
                # Against the floor under the vehicle, not one level height
                # (a level may rise along an aisle).
                i = int((vehicle[0, 3] - grid.x0) / grid.res)
                j = int((vehicle[1, 3] - grid.y0) / grid.res)
                inside = 0 <= i < grid.width and 0 <= j < grid.height
                floor = floor_z[j, i] if inside else np.nan
                if not np.isfinite(floor):
                    floor = level_z
                if abs(vehicle[2, 3] - floor) > 1.0:
                    continue  # The vehicle is on another level or a ramp.
                camera_from_world = geometry.inv(
                    pose_set.camera_exposure(t, name, int(ns)))
                frames.append((data, name, int(ns), camera, camera_from_world))

    def prepare(frame):
        """Image and occlusion depth of a frame (in parallel)."""
        data, name, ns, camera, camera_from_world = frame
        image = data.load_image(name, ns)
        depth = None
        if occluders is not None:
            depth = _core.FloorTextureFuser.occlusion_depth(
                camera, camera_from_world, occluders, float(cfg.map.voxel),
                float(section.max_range))
        return image, depth

    # Images and depths in parallel; the fuser takes them in order.
    threads = parallel.workers(section.get("workers"))
    for frame, (image, depth) in zip(
            frames, parallel.ordered_map(prepare, frames, threads)):
        fuser.add_frame(frame[3], frame[4], image, depth,
                        float(section.depth_tol), float(section.max_range),
                        float(section.min_cos))
    used = len(frames)
    rgb, score, observations = fuser.result()
    log(f"[texture] level {grid.level}: fused {used} images, "
        f"{100.0 * (observations > 0).mean():.0f}% of cells observed")
    return rgb, score, observations
