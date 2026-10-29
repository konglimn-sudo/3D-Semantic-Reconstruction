"""4D object labels (section 5, module 6): parked objects from the
aggregated map, per-keyframe detections, multi-object tracking and
track-level boxes.

The algorithms are C++ (gr/semantics/objects4d.h, `run_objects`): static
tracks (parked cars) aggregate the map points of an object into one box,
far more complete than any single sweep; moving tracks get smoothed
centres, velocity-aligned yaw and an object-frame size. The detector is the
geometric baseline; a learned 3D detector can be plugged in through
`semantics.adapters.Detector3D`. This module holds the box type and feeds
the stage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from garage_recon import _core
from garage_recon.recon import levels as levels_lib
from garage_recon.semantics import taxonomy


@dataclass
class Box3D:
    """An object box (C++ ObjectBox) with its class name."""

    track_id: int
    cls: str
    center: list[float]
    size: list[float]  # Length, width, height.
    yaw: float
    static: bool
    velocity: list[float]
    num_points: int
    score: float = 1.0

    def corners(self) -> np.ndarray:
        """(8, 3) corners."""
        l, w, h = self.size
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        offsets = np.array([[sx * l / 2, sy * w / 2, sz * h / 2]
                            for sx in (-1, 1) for sy in (-1, 1)
                            for sz in (-1, 1)])
        return np.asarray(self.center) + offsets @ rotation.T

    def contains(self, pts: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Whether each point lies in the box grown by `margin`."""
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        d = pts - np.asarray(self.center)
        lx = c * d[:, 0] - s * d[:, 1]
        ly = s * d[:, 0] + c * d[:, 1]
        half = np.asarray(self.size) / 2 + margin
        return ((np.abs(lx) <= half[0]) & (np.abs(ly) <= half[1])
                & (np.abs(d[:, 2]) <= half[2]))

    def to_json(self) -> dict:
        return asdict(self)

    def to_core(self) -> _core.ObjectBox:
        return _core.ObjectBox(
            int(self.track_id), taxonomy.cid(self.cls),
            np.asarray(self.center, float), np.asarray(self.size, float),
            float(self.yaw), bool(self.static),
            np.asarray(self.velocity, float), int(self.num_points),
            float(self.score))

    @classmethod
    def from_core(cls, b: _core.ObjectBox) -> Box3D:
        return cls(int(b.track_id), taxonomy.BY_ID[int(b.label)].name,
                   [float(v) for v in b.center], [float(v) for v in b.size],
                   float(b.yaw), bool(b.is_static),
                   [float(v) for v in b.velocity], int(b.num_points),
                   float(b.score))


def objects_config(cfg) -> _core.ObjectsConfig:
    """The C++ ObjectsConfig of the objects section."""
    section = cfg.objects
    config = _core.ObjectsConfig()
    for key in ("frame_stride", "cluster_res", "min_points",
                "detect_max_range", "match_gate", "max_missed", "min_hits",
                "static_displacement", "moving_fraction", "streak_min_moved",
                "parked_min_points", "parked_min_extent",
                "vehicle_max_height", "max_flat_height_span", "nms_radius",
                "structure_dilation"):
        if section.get(key) is not None:
            setattr(config, key, section[key])
    config.height_min, config.height_max = (
        float(v) for v in section.height_range)
    config.typical_length, config.typical_width = (
        float(v) for v in section.typical_vehicle_size[:2])
    for name in ("vehicle", "vehicle_moving", "pedestrian", "two_wheeler",
                 "other"):
        setattr(config, f"{name}_id", taxonomy.cid(name))
    return config


def run_objects(kfs, kf_poses: np.ndarray, levels: levels_lib.LevelModel,
                structure_xyz: np.ndarray, pm, labels: np.ndarray, yaw: float,
                paths_by_level: dict, cfg, slots=(), log=print):
    """The objects stage (C++ run_objects).

    Args:
        kfs: Keyframes (sweeps in the LiDAR frame).
        kf_poses: (K, 4, 4) world <- LiDAR.
        levels: Level model.
        structure_xyz: Static map points labelled as structure (detections
            next to them are not objects).
        pm: The point map.
        labels: Semantic class per map point (structure labels).
        yaw: Garage grid orientation (rad).
        paths_by_level: Level -> driven path (N, 2).
        cfg: Configuration.
        slots: Parking slots (semantics.slots.Slot).
        log: Logger.

    Returns:
        (tracks_json, parked boxes per trip index, moving boxes per
        keyframe, mask of map points that belong to moving objects).
    """
    kind = (pm.dynamic_kind if pm.dynamic_kind is not None
            else pm.dynamic.astype(np.uint8) * 2)
    view = {"xyz": pm.xyz, "height": pm.height, "level": pm.level,
            "dynamic_kind": kind, "trip": pm.trip,
            "normal_z": np.ascontiguousarray(pm.normals[:, 2]),
            "labels": np.asarray(labels, np.uint8),
            "floor_id": taxonomy.cid("floor"), "ramp_id": taxonomy.cid("ramp")}
    paths = {int(lv): np.column_stack([np.asarray(p, float).reshape(-1, 2),
                                       np.zeros(len(p))])
             for lv, p in paths_by_level.items()}
    outlines = [(int(s.level), np.column_stack([np.asarray(s.corners, float),
                                                np.zeros(4)]),
                 float(s.width)) for s in slots]
    out = _core.run_objects(
        view, *levels.core_parts(),
        np.ascontiguousarray(structure_xyz, dtype=np.float64), float(yaw),
        paths, outlines, np.ascontiguousarray(kf_poses, dtype=np.float64),
        np.asarray(kfs.trip, np.int32), np.asarray(kfs.stamp, np.float64),
        [str(t) for t in kfs.trip_ids], lambda k: kfs.load(int(k))[0],
        objects_config(cfg), log)
    parked = {t: [Box3D.from_core(b) for b in boxes]
              for t, boxes in enumerate(out["parked"])}
    moving = {int(k): [Box3D.from_core(b) for b in boxes]
              for k, boxes in out["moving"].items()}
    tracks = [{"trip": kfs.trip_ids[t["trip"]], "track_id": t["track_id"],
               "static": t["static"],
               "cls": taxonomy.BY_ID[int(t["label"])].name,
               "frames": list(t["frames"]),
               "box": (Box3D.from_core(t["box"]).to_json() if t["static"]
                       else None)}
              for t in out["tracks"]]
    return tracks, parked, moving, out["moving_points"]
