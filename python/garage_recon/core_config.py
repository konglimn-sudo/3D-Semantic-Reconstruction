"""YAML configuration -> the config structs of the C++ core.

Struct fields carry the YAML key names, so most sections are copied key by
key; keys a struct does not have (and null values, which keep the C++
default) are skipped.
"""

from __future__ import annotations

import numpy as np

from garage_recon import _core


def fill(struct, section, **renames):
    """Copies the matching keys of a config section onto a C++ struct."""
    for key, value in (section or {}).items():
        name = renames.get(key, key)
        if value is None or not hasattr(struct, name):
            continue
        setattr(struct, name,
                list(value) if isinstance(value, (list, tuple)) else value)
    return struct


def odometry_config(cfg) -> _core.OdometryConfig:
    return fill(_core.OdometryConfig(), cfg.odometry)


def keyframe_config(cfg) -> _core.KeyframeConfig:
    return fill(_core.KeyframeConfig(), cfg.keyframes)


def graph_config(cfg) -> _core.GraphStageConfig:
    config = _core.GraphStageConfig()
    config.loops = fill(_core.LoopConfig(), cfg.loops)
    config.backend = fill(_core.BackendConfig(), cfg.backend)
    config.ba = fill(_core.LidarBaConfig(), cfg.get("ba"))
    config.refine = fill(_core.RefineConfig(), cfg.refine)
    return config


def graph_voxels(cfg) -> list[float]:
    """Resolutions at which the back-end keeps keyframe sweeps."""
    return list(graph_config(cfg).required_voxels())


def fusion_config(cfg) -> _core.MapFusionConfig:
    return fill(_core.MapFusionConfig(), cfg.map.get("fusion"))


def dynamic_config(cfg) -> _core.DynamicConfig:
    return fill(_core.DynamicConfig(), cfg.map.dynamic)


def occupancy_config(cfg) -> _core.OccupancyConfig:
    """The occupancy section, with the grid range as bounds."""
    section = cfg.occupancy
    x0, x1, y0, y1, z0, z1 = section.range
    config = _core.OccupancyConfig()
    config.min_bound = np.array([x0, y0, z0], float)
    config.max_bound = np.array([x1, y1, z1], float)
    config.voxel = float(section.voxel)
    config.free_label = 0
    config.camera_pixel_stride = int(section.camera_stride)
    config.camera_max_range = float(section.camera_max_range)
    return config


def frame_label_config(cfg) -> _core.FrameLabelConfig:
    return fill(_core.FrameLabelConfig(), cfg.frames)


def camera_calibration_config(cfg) -> _core.CameraCalibrationConfig:
    return fill(_core.CameraCalibrationConfig(), cfg.calibration.solver)


def identity_config(cfg) -> _core.IdentityConfig:
    return fill(_core.IdentityConfig(), cfg.identities)
