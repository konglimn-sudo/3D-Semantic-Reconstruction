"""Sensor simulation for the synthetic garage.

Rotating LiDAR with motion distortion, fisheye / pinhole cameras with
rolling shutter, IMU with bias and wheel speed. The ray caster also returns
ground-truth segment ids (one per object or surface) for the oracle models.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import transform

from garage_recon import _core
from garage_recon.data.synthetic import scene as scene_lib
from garage_recon.data.synthetic import trajectory as trajectory_lib
from garage_recon.semantics import taxonomy

_KINDS = ["slab", "roof", "wall", "column", "ramp", "parapet", "duct", "car",
          "moving_car"]
_KIND_ID = {k: i for i, k in enumerate(_KINDS)}

# Ground-truth segment ids: static boxes use 8 * box index + face code (slab
# and ramp faces differ), cars and markings get their own ranges.
PARKED_CAR_SEGMENT = 1_000_000
MOVING_CAR_SEGMENT = 2_000_000
MARKING_SEGMENT = 3_000_000
_FACE_TOP = 1
_FACE_BOTTOM = 2
_FACE_SIDE = 3


def _hash01(ix, iy, iz) -> np.ndarray:
    h = ((ix.astype(np.uint64) * np.uint64(73856093))
         ^ (iy.astype(np.uint64) * np.uint64(19349663))
         ^ (iz.astype(np.uint64) * np.uint64(83492791)))
    h ^= h >> np.uint64(13)
    h *= np.uint64(0x5BD1E995)
    h ^= h >> np.uint64(15)
    return (h & np.uint64(0xFFFF)).astype(np.float64) / 65535.0


def _bounds(corners) -> list[float]:
    """[x0, x1, y0, y1] of axis-aligned corners."""
    c = np.asarray(corners, float)
    return [c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max()]


def value_noise(points: np.ndarray, scale: float) -> np.ndarray:
    """Smooth 3D value noise in [0, 1]."""
    grid = points / scale
    base = np.floor(grid)
    frac = grid - base
    frac = frac * frac * (3.0 - 2.0 * frac)
    base = base.astype(np.int64)
    acc = np.zeros(len(points))
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                acc += wx * wy * wz * _hash01(base[:, 0] + dx,
                                              base[:, 1] + dy,
                                              base[:, 2] + dz)
    return acc


@dataclasses.dataclass
class LidarModel:
    """Rotating multi-beam LiDAR: beams, field of view, resolution, noise."""

    beams: int = 32
    vfov: tuple[float, float] = (-25.0, 15.0)
    h_res_deg: float = 0.6
    max_range: float = 60.0
    min_range: float = 0.6
    range_noise: float = 0.01
    sweep: float = 0.1

    def angles(self) -> tuple[np.ndarray, np.ndarray]:
        elevation = np.deg2rad(
            np.linspace(self.vfov[0], self.vfov[1], self.beams))
        azimuth = np.deg2rad(np.arange(0.0, 360.0, self.h_res_deg))
        return elevation, azimuth


class SceneRaycaster:
    """Static scene plus time-dependent moving cars.

    Resolves the class, intensity, colour and ground-truth segment of every
    hit.
    """

    def __init__(self, garage: scene_lib.GarageScene,
                 cars: list[scene_lib.Box],
                 movers: list[scene_lib.MovingCar]):
        self.scene = garage
        self.boxes = list(garage.boxes) + list(cars)
        self.movers = movers
        self.static = _core.BoxScene()
        for box in self.boxes:
            self.static.add_box(box.center, box.size, box.R)
        self.lines = garage.lines_array()
        self.paint_cls = garage.paint_classes()
        # Green floors of the charging slots: [level, x0, x1, y0, y1].
        self.green = np.array(
            [[s["level"], *_bounds(s["corners"])] for s in garage.slots
             if "ev" in s.get("attributes", [])], float).reshape(-1, 5)

    def _attributes(self, boxes: list[scene_lib.Box]):
        return (np.array([b.cls for b in boxes], int),
                np.array([_KIND_ID[b.kind] for b in boxes], int),
                np.array([b.level for b in boxes], int),
                np.array([b.instance for b in boxes], int))

    def cast(self, origins: np.ndarray, dirs: np.ndarray, t_query: float,
             max_t: float) -> dict:
        """Casts rays (one origin, or one per ray) at time `t_query`."""
        t, ids, normals = self.static.raycast(origins, dirs, max_t)
        boxes = list(self.boxes)
        dynamic = [m.box_at(t_query) for m in self.movers]
        dynamic = [b for b in dynamic if b is not None]
        if dynamic:
            dynamic_scene = _core.BoxScene()
            for box in dynamic:
                dynamic_scene.add_box(box.center, box.size, box.R)
            t_dyn, ids_dyn, normals_dyn = dynamic_scene.raycast(
                origins, dirs, max_t)
            closer = t_dyn < t
            t = np.where(closer, t_dyn, t)
            ids = np.where(closer, ids_dyn + len(boxes), ids)
            normals = np.where(closer[:, None], normals_dyn, normals)
            boxes = boxes + dynamic
        hit = ids >= 0
        cls_b, kind_b, level_b, inst_b = self._attributes(boxes)
        o = (origins if len(origins) == len(dirs)
             else np.broadcast_to(origins, dirs.shape))
        points = o + dirs * np.where(hit, t, 0.0)[:, None]
        kind = np.full(len(t), -1)
        kind[hit] = kind_b[ids[hit]]
        cls = np.zeros(len(t), int)
        cls[hit] = cls_b[ids[hit]]
        level = np.full(len(t), -1)
        level[hit] = level_b[ids[hit]]
        instance = np.full(len(t), -1)
        instance[hit] = inst_b[ids[hit]]
        segment = np.full(len(t), -1, np.int64)
        segment[hit] = ids[hit].astype(np.int64) * 8
        nz = normals[:, 2]
        cid = taxonomy.cid
        slab = np.isin(kind, [_KIND_ID["slab"], _KIND_ID["roof"]])
        cls[slab & (nz > 0.7)] = cid("floor")
        cls[slab & (nz < -0.7)] = cid("ceiling")
        cls[slab & (np.abs(nz) <= 0.7)] = cid("wall")
        ramp = kind == _KIND_ID["ramp"]
        cls[ramp & (nz > 0.5)] = cid("ramp")
        cls[ramp & (nz < -0.5)] = cid("ceiling")
        cls[ramp & (np.abs(nz) <= 0.5)] = cid("wall")
        faced = slab | ramp
        segment[faced & (nz > 0.5)] += _FACE_TOP
        segment[faced & (nz < -0.5)] += _FACE_BOTTOM
        segment[faced & (np.abs(nz) <= 0.5)] += _FACE_SIDE
        car = kind == _KIND_ID["car"]
        segment[car] = PARKED_CAR_SEGMENT + instance[car]
        mover = kind == _KIND_ID["moving_car"]
        segment[mover] = MOVING_CAR_SEGMENT + instance[mover]
        floor_hit = np.where(cls == cid("floor"))[0]
        if len(floor_hit) and len(self.lines):
            line = self.marking_index(points[floor_hit], level[floor_hit])
            on_line = line >= 0
            cls[floor_hit[on_line]] = self.paint_cls[line[on_line]]
            segment[floor_hit[on_line]] = MARKING_SEGMENT + line[on_line]
        return {"t": t, "hit": hit, "points": points, "normals": normals,
                "cls": cls, "kind": kind, "instance": instance,
                "segment": segment}

    def marking_index(self, points: np.ndarray,
                      level: np.ndarray) -> np.ndarray:
        """Index of the painted line under each floor point, -1 if none."""
        out = np.full(len(points), -1, np.int64)
        for lv in np.unique(level):
            selected = np.where(level == lv)[0]
            line_ids = np.where(self.lines[:, 0] == lv)[0]
            if len(line_ids) == 0:
                continue
            lines = self.lines[line_ids]
            for k in range(0, len(selected), 20000):
                idx = selected[k:k + 20000]
                x, y = points[idx, 0:1], points[idx, 1:2]
                inside = ((x >= lines[:, 1]) & (x <= lines[:, 2])
                          & (y >= lines[:, 3]) & (y <= lines[:, 4]))
                any_line = inside.any(axis=1)
                out[idx[any_line]] = line_ids[
                    np.argmax(inside[any_line], axis=1)]
        return out

    def on_marking(self, points: np.ndarray, level: np.ndarray) -> np.ndarray:
        return self.marking_index(points, level) >= 0

    def height_above_level(self, z: np.ndarray) -> np.ndarray:
        level_height = self.scene.layout.level_height
        return z + level_height * np.ceil(-z / level_height - 1e-9)

    def intensity(self, hit: dict, rng: np.random.Generator) -> np.ndarray:
        """LiDAR intensity of the hits (paint and column stripes are bright)."""
        base = np.full(len(hit["t"]), 0.3)
        table = {"floor": 0.22, "ramp": 0.25, "ceiling": 0.30, "wall": 0.40,
                 "column": 0.45, "mep": 0.65, "parking_line": 0.85, "arrow": 0.85,
                 "vehicle": 0.35, "vehicle_moving": 0.35}
        for name, value in table.items():
            base[hit["cls"] == taxonomy.cid(name)] = value
        column = hit["cls"] == taxonomy.cid("column")
        if column.any():
            p = hit["points"][column]
            stripe = ((np.floor((p[:, 0] + p[:, 1] + p[:, 2]) / 0.15) % 2 == 0)
                      & (self.height_above_level(p[:, 2]) < 1.0))
            base[np.where(column)[0][stripe]] = 0.7
        out = (base * (1.0 - 0.004 * np.nan_to_num(hit["t"], posinf=0.0))
               + rng.normal(0.0, 0.03, len(base)))
        return np.clip(out, 0.0, 1.0)

    def colour(self, hit: dict, rng: np.random.Generator) -> np.ndarray:
        """RGB of the hits: class palette, texture noise and shading."""
        n = len(hit["t"])
        points, cls = hit["points"], hit["cls"]
        rgb = np.zeros((n, 3))
        fine = value_noise(points, 0.06) - 0.5
        coarse = value_noise(points, 0.7) - 0.5
        palette = {
            "floor": (108, 108, 112), "ramp": (96, 92, 88),
            "ceiling": (84, 84, 90), "wall": (198, 188, 164),
            "column": (192, 192, 192), "mep": (172, 176, 182),
            "parking_line": (232, 232, 228), "arrow": (232, 232, 228),
            "vehicle_moving": (196, 36, 36),
        }
        for name, colour in palette.items():
            rgb[cls == taxonomy.cid(name)] = colour
        floor = np.nonzero(cls == taxonomy.cid("floor"))[0]
        if len(self.green) and len(floor):
            p = points[floor]
            level = np.rint(-p[:, 2] / self.scene.layout.level_height)
            for lv, x0, x1, y0, y1 in self.green:
                inside = ((level == lv) & (p[:, 0] > x0) & (p[:, 0] < x1)
                          & (p[:, 1] > y0) & (p[:, 1] < y1))
                rgb[floor[inside]] = (70, 150, 90)
        vehicle = cls == taxonomy.cid("vehicle")
        if vehicle.any():
            instance = hit["instance"][vehicle]
            colours = np.array([(40, 70, 160), (230, 230, 230), (20, 20, 20),
                                (150, 150, 155), (160, 30, 30), (40, 120, 70)],
                               float)
            rgb[vehicle] = colours[np.abs(instance) % len(colours)]
        column = cls == taxonomy.cid("column")
        if column.any():
            p = points[column]
            local_z = self.height_above_level(p[:, 2])
            stripe = np.floor((p[:, 0] + p[:, 1] + p[:, 2]) / 0.15) % 2 == 0
            low = local_z < 1.0
            c = rgb[column]
            c[low & stripe] = (230, 188, 30)
            c[low & ~stripe] = (30, 30, 30)
            rgb[column] = c
        texture = 1.0 + 0.35 * fine + 0.2 * coarse
        line = np.isin(cls, [taxonomy.cid("parking_line"), taxonomy.cid("arrow")])
        texture[line] = 1.0 + 0.08 * fine[line]
        nz = hit["normals"][:, 2]
        shade = ((0.62 + 0.38 * (0.5 + 0.5 * nz))
                 / (1.0 + 0.015 * np.nan_to_num(hit["t"], posinf=0.0)))
        rgb = rgb * (texture * shade)[:, None] + rng.normal(0.0, 3.0, (n, 3))
        return np.clip(rgb, 0, 255).astype(np.uint8)


def simulate_sweep(caster: SceneRaycaster,
                   drive: trajectory_lib.Trajectory, stamp: float,
                   T_vehicle_lidar: np.ndarray, model: LidarModel,
                   rng: np.random.Generator) -> dict:
    """One sweep ending at `stamp`.

    Points are expressed in the sensor frame at the time of each firing
    (raw, motion-distorted).
    """
    elevation, azimuth = model.angles()
    tau = -model.sweep + model.sweep * azimuth / (2.0 * np.pi)
    T_wl = drive.poses_at(stamp + tau) @ T_vehicle_lidar
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    d_local = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                        np.sin(el)], axis=-1)  # (B, A, 3)
    d_world = np.einsum("aij,baj->bai", T_wl[:, :3, :3],
                        d_local).reshape(-1, 3)
    origins = np.broadcast_to(T_wl[None, :, :3, 3],
                              d_local.shape).reshape(-1, 3)
    hit = caster.cast(np.ascontiguousarray(origins),
                      np.ascontiguousarray(d_world), stamp, model.max_range)
    t = hit["t"] + rng.normal(0.0, model.range_noise, len(hit["t"]))
    ok = hit["hit"] & (t >= model.min_range) & (t <= model.max_range)
    times = np.broadcast_to(tau[None, :], el.shape).reshape(-1)
    intensity = caster.intensity(hit, rng)
    return {
        "points": d_local.reshape(-1, 3)[ok] * t[ok, None],
        "intensity": intensity[ok],
        "time": times[ok],
        "cls": hit["cls"][ok],
        "instance": hit["instance"][ok],
    }


class CameraRenderer:
    """Ray-casting renderer for one camera, with rolling shutter."""

    def __init__(self, calib):
        self.calib = calib
        self.model = calib.make_model()
        w, h = calib.width, calib.height
        uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        rays, valid = self.model.unproject(
            np.stack([uu.ravel(), vv.ravel()], axis=1))
        self.rays = rays
        self.valid = valid
        self.rows = np.repeat(np.arange(h), w)

    def render(self, caster: SceneRaycaster, world_from_camera: np.ndarray,
               t_query: float, rng: np.random.Generator) -> dict:
        """Renders an image.

        Args:
            caster: The scene.
            world_from_camera: (4, 4), or (H, 4, 4) with one pose per row.
            t_query: Time of the moving objects.
            rng: Colour noise.

        Returns:
            Dict of rgb (H, W, 3) uint8, label (H, W) int32 class ids, depth
            (H, W) float32 range (inf where empty) and segment (H, W) int64
            ground-truth segment ids (-1 where empty).
        """
        h, w = self.calib.height, self.calib.width
        poses = np.asarray(world_from_camera, float)
        idx = np.where(self.valid)[0]
        if poses.ndim == 2:
            origins = poses[None, :3, 3].copy()
            dirs = self.rays[self.valid] @ poses[:3, :3].T
        else:
            row_poses = poses[self.rows[idx]]
            origins = np.ascontiguousarray(row_poses[:, :3, 3])
            dirs = np.einsum("nij,nj->ni", row_poses[:, :3, :3],
                             self.rays[self.valid])
        hit = caster.cast(origins, np.ascontiguousarray(dirs), t_query, 80.0)
        rgb = np.zeros((h * w, 3), np.uint8)
        label = np.zeros(h * w, np.int32)
        depth = np.full(h * w, np.inf, np.float32)
        segment = np.full(h * w, -1, np.int64)
        rgb[idx] = caster.colour(hit, rng)
        rgb[idx[~hit["hit"]]] = 0
        label[idx] = hit["cls"]
        depth[idx] = np.where(hit["hit"], hit["t"], np.inf)
        segment[idx] = np.where(hit["hit"], hit["segment"], -1)
        return {"rgb": rgb.reshape(h, w, 3), "label": label.reshape(h, w),
                "depth": depth.reshape(h, w),
                "segment": segment.reshape(h, w)}


def proprioception(drive: trajectory_lib.Trajectory, t0: float, t1: float,
                   rng: np.random.Generator, rate: float = 100.0,
                   imu_bias: np.ndarray | None = None):
    """IMU (vehicle frame) and wheel-speed samples of a drive.

    Args:
        drive: The ground-truth trajectory.
        t0: Start time.
        t1: End time.
        rng: White noise of the samples.
        rate: Sample rate (Hz).
        imu_bias: Optional (6,) [gyro (rad/s), accel (m/s^2)] added to every
            IMU sample.

    Returns:
        (imu rows [t, gx, gy, gz, ax, ay, az], wheel rows [t, speed]).
    """
    t = np.arange(t0, t1, 1.0 / rate)
    poses = drive.poses_at(t)
    dt = 1.0 / rate
    rotations = poses[:, :3, :3]
    delta = np.einsum("kji,kjl->kil", rotations[:-1], rotations[1:])
    gyro = transform.Rotation.from_matrix(delta).as_rotvec() / dt
    gyro = np.vstack([gyro, gyro[-1:]])
    velocity = np.gradient(poses[:, :3, 3], dt, axis=0)
    accel = np.gradient(velocity, dt, axis=0)
    force = np.einsum("kji,kj->ki", rotations,
                      accel + np.array([0.0, 0.0, 9.81]))
    speed = np.einsum("ki,ki->k", velocity, rotations[:, :, 0])
    bias = np.zeros(6) if imu_bias is None else np.asarray(imu_bias, float)
    imu = np.column_stack([
        t, gyro + rng.normal(0, 0.002, gyro.shape) + bias[:3],
        force + rng.normal(0, 0.02, force.shape) + bias[3:]])
    wheel = np.column_stack([t, speed + rng.normal(0, 0.01, len(t))])
    return imu, wheel
