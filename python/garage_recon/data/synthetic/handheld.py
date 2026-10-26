"""Synthetic handheld-scanner ground truth (plan sections 3.5 and 7).

A surveyor walks the aisles of the garage with a handheld LiDAR scanner at a
different time than the vehicle drives (other parked cars). The result is
what such a scanner exports: a dense, gravity-aligned cloud with intensity in
the scanner's own frame, a few millimetres of noise and a slight deformation
(the scanner's own SLAM error). The vector features (levels, columns,
parking slots, marking lines) are given in the same frame, as an annotator
would digitise them on the handheld cloud.

The transform from the synthetic world to the handheld frame is written to
gt/handheld.json only: the acceptance tool has to find it by registration.

Layout:
    <data>/handheld/cloud.ply      binary PLY: float x, y, z, uchar intensity
    <data>/handheld/features.json  levels, columns, slots, lines
    <data>/gt/handheld.json        world -> handheld transform and settings
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from garage_recon import geometry
from garage_recon.data.synthetic import scene as scene_lib
from garage_recon.data.synthetic import sensors

# Scanner, walk and error model.
HANDHELD = {
    "step": 2.0,             # m between scan positions along the walk
    "height": 1.5,           # scanner above the floor (m)
    "azimuth_step": 0.5,     # deg
    "elevation": (-70.0, 70.0, 1.0),  # deg: min, max, step
    "min_range": 0.5,
    "max_range": 25.0,
    "noise": 0.003,          # m along the ray
    "voxel": 0.04,           # output spacing (m), points averaged per voxel
    "scale_error": 1e-4,     # SLAM scale error of the handheld scanner
    "level_shift": 0.002,    # m vertical shift per level (SLAM drift)
    "session": 7,            # parked-car draw of the handheld session
    "frame_yaw_deg": 23.7,   # handheld frame relative to the world
    "frame_offset": (152.3, -41.8, 11.5),
}


def frame_transform(settings: dict) -> np.ndarray:
    """(4, 4) handheld <- world (rigid part of the handheld result)."""
    return geometry.make_T(
        geometry.rot_z(np.radians(settings["frame_yaw_deg"])),
        settings["frame_offset"])


def deform(points: np.ndarray, level_height: float,
           settings: dict) -> np.ndarray:
    """World points as the handheld SLAM places them (before its frame).

    Args:
        points: (N, 3) world points.
        level_height: Storey height (m).
        settings: The HANDHELD dictionary.

    Returns:
        (N, 3) deformed points: a small scale error about the origin and a
        vertical shift growing by `level_shift` per level below ground.
    """
    out = np.asarray(points, float) * (1.0 + settings["scale_error"])
    levels_down = np.maximum(-out[:, 2] / level_height, 0.0)
    out[:, 2] += settings["level_shift"] * np.round(levels_down)
    return out


def _directions(settings: dict) -> np.ndarray:
    azimuth = np.radians(np.arange(0.0, 360.0, settings["azimuth_step"]))
    e0, e1, de = settings["elevation"]
    elevation = np.radians(np.arange(e0, e1 + 1e-9, de))
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el)], axis=-1).reshape(-1, 3)


def _walk(trajectory: np.ndarray, step: float) -> np.ndarray:
    """Positions every `step` m along a (K, 4, 4) trajectory."""
    positions = trajectory[:, :3, 3]
    travelled = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))])
    wanted = np.arange(0.0, travelled[-1], step)
    return np.stack([np.interp(wanted, travelled, positions[:, a])
                     for a in range(3)], axis=1)


class _VoxelSums:
    """Per-voxel sums of points and values, merged batch by batch."""

    _OFFSET = 1 << 20

    def __init__(self, voxel: float):
        self.voxel = voxel
        self.keys = np.zeros(0, np.int64)
        self.sums = np.zeros((0, 4))  # x, y, z, value
        self.counts = np.zeros(0)

    def add(self, points: np.ndarray, values: np.ndarray) -> None:
        """Adds (N, 3) points with one value each."""
        cells = np.floor(points / self.voxel).astype(np.int64) + self._OFFSET
        keys = (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]
        keys = np.concatenate([self.keys, keys])
        sums = np.concatenate([self.sums,
                               np.column_stack([points, values])])
        counts = np.concatenate([self.counts, np.ones(len(points))])
        self.keys, inverse = np.unique(keys, return_inverse=True)
        self.sums = np.stack([np.bincount(inverse, sums[:, a],
                                          minlength=len(self.keys))
                              for a in range(4)], axis=1)
        self.counts = np.bincount(inverse, counts, minlength=len(self.keys))

    def means(self) -> tuple[np.ndarray, np.ndarray]:
        """(points (M, 3), values (M,)): the mean of every voxel."""
        mean = self.sums / self.counts[:, None]
        return mean[:, :3], mean[:, 3]


def write_ply(path: pathlib.Path, xyz: np.ndarray,
              intensity: np.ndarray) -> None:
    """Binary little-endian PLY with float x, y, z and uchar intensity."""
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"),
                                       ("z", "<f4"), ("intensity", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz.T.astype(np.float32)
    vertex["intensity"] = np.clip(np.round(intensity * 255.0), 0, 255)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\nproperty float x\n"
              "property float y\nproperty float z\n"
              "property uchar intensity\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def _features(garage: scene_lib.GarageScene, transform: np.ndarray,
              level_height: float, settings: dict) -> dict:
    """Vector ground truth in the handheld frame."""

    def to_frame(points: np.ndarray) -> np.ndarray:
        points = deform(np.atleast_2d(points), level_height, settings)
        return points @ transform[:3, :3].T + transform[:3, 3]

    yaw = float(np.degrees(np.arctan2(transform[1, 0], transform[0, 0])))
    scene = garage.to_json()
    levels = [{"id": lv["id"],
               "z": float(to_frame([[0.0, 0.0, lv["z"]]])[0, 2])}
              for lv in scene["levels"]]
    columns = []
    for column in scene["columns"]:
        center = to_frame([[*column["center"],
                            0.5 * (column["z0"] + column["z1"])]])[0]
        columns.append({"id": column["id"], "level": column["level"],
                        "center": center.tolist(),
                        "size": list(column["size"]), "yaw_deg": yaw,
                        "height": column["z1"] - column["z0"]})
    slots = []
    for slot in scene["slots"]:
        corners = np.column_stack([np.asarray(slot["corners"], float),
                                   np.full(4, slot["z"])])
        slots.append({"id": slot["slot_id"], "level": slot["level"],
                      "corners": to_frame(corners).tolist()})
    lines = []
    for k, line in enumerate(scene["lines"]):
        x = 0.5 * (line["x0"] + line["x1"])
        z = garage.layout.level_z(line["level"])
        ends = to_frame([[x, line["y0"], z], [x, line["y1"], z]])
        lines.append({"id": f"line-{k}", "level": line["level"],
                      "cls": line["cls"], "p0": ends[0].tolist(),
                      "p1": ends[1].tolist(),
                      "width": line["x1"] - line["x0"]})
    return {"frame": "handheld", "levels": levels, "columns": columns,
            "slots": slots, "lines": lines}


def simulate_handheld(data_dir: str | pathlib.Path, seed: int = 0,
                      settings: dict | None = None, log=print) -> pathlib.Path:
    """Writes the handheld ground truth of a synthetic dataset.

    Args:
        data_dir: Synthetic dataset root (with gt/ and trip_XX/).
        seed: Seed of the dataset (parked cars of the handheld session).
        settings: Overrides of HANDHELD.
        log: Progress output.

    Returns:
        The handheld/ directory.
    """
    data_dir = pathlib.Path(data_dir)
    settings = {**HANDHELD, **(settings or {})}
    rng = np.random.default_rng(seed + 777)
    scene_json = json.loads((data_dir / "gt" / "scene.json").read_text())
    layout = scene_lib.Layout(**scene_json["layout"])
    garage = scene_lib.build_static_scene(layout)
    cars = scene_lib.parked_cars(garage, settings["session"], seed)
    caster = sensors.SceneRaycaster(garage, cars, [])
    trip = sorted((data_dir / "gt").glob("trip_*"))[0]
    with np.load(trip / "trajectory.npz") as z:
        walk = _walk(z["poses"], settings["step"])
    walk[:, 2] += settings["height"]
    directions = _directions(settings)
    sums = _VoxelSums(settings["voxel"])
    batch_points, batch_values = [], []
    for k, position in enumerate(walk):
        hit = caster.cast(position[None, :], directions, 0.0,
                          settings["max_range"])
        ranges = hit["t"] + rng.normal(0.0, settings["noise"], len(hit["t"]))
        ok = (hit["hit"] & (ranges >= settings["min_range"])
              & (ranges <= settings["max_range"]))
        batch_points.append(position + directions[ok] * ranges[ok, None])
        batch_values.append(caster.intensity(
            {key: value[ok] for key, value in hit.items()}, rng))
        if len(batch_points) == 20 or k + 1 == len(walk):
            sums.add(np.concatenate(batch_points),
                     np.concatenate(batch_values))
            batch_points, batch_values = [], []
    xyz, value = sums.means()
    transform = frame_transform(settings)
    xyz = deform(xyz, layout.level_height, settings)
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    out = data_dir / "handheld"
    out.mkdir(parents=True, exist_ok=True)
    write_ply(out / "cloud.ply", xyz, value)
    (out / "features.json").write_text(json.dumps(
        _features(garage, transform, layout.level_height, settings), indent=1))
    truth = {"handheld_from_world": transform.tolist(),
             "settings": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in settings.items()},
             "positions": len(walk), "points": int(len(xyz))}
    (data_dir / "gt" / "handheld.json").write_text(json.dumps(truth, indent=1))
    log(f"[handheld] {len(walk)} scan positions, {len(xyz)} points at "
        f"{settings['voxel']} m -> {out}")
    return out
