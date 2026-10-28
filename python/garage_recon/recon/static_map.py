"""Global point map (sections 4.2 / 4.3): accumulation, normals and dynamic
points. The algorithms are C++ (`MapAccumulator`, `map_normals`,
`DynamicDetector`, and `build_tiled_map` for the map stage, in one piece or
tile by tile); this module feeds them the keyframe sweeps and keeps the
per-point attributes in `PointMap`.

A map built in tiles (`map.tile_size`) is stored as map/tiles/*.npz with an
index map/tiles.json instead of map/point_map.npz; load_point_map reads
either, the whole map in the order of one piece or only the tiles that
touch a region."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
import shutil

import numpy as np

from .. import _core
from ..core_config import dynamic_config, fusion_config
from . import levels as levels_lib
from .keyframes import KeyframeSet


@dataclass
class PointMap:
    xyz: np.ndarray  # (N, 3) float64 world
    intensity: np.ndarray  # (N,) float32
    kf: np.ndarray  # (N,) int32 keyframe that contributed the point
    trip: np.ndarray  # (N,) int16
    normals: np.ndarray | None = None  # (N, 3) float32, oriented towards the sensor
    planarity: np.ndarray | None = None  # (N,) float32
    sigma: np.ndarray | None = None  # (N,) float32 (m) along the normal, fused maps
    observations: np.ndarray | None = None  # (N,) uint16 observations averaged per point
    dynamic: np.ndarray | None = None  # (N,) bool
    dynamic_kind: np.ndarray | None = None  # (N,) uint8: 0 static, 1 moved within its session, 2 changed between sessions
    level: np.ndarray | None = None  # (N,) int16, -1 unknown
    height: np.ndarray | None = None  # (N,) float32 above the floor of `level` (or ramp)
    on_ramp: np.ndarray | None = None  # (N,) bool
    label: np.ndarray | None = None  # (N,) uint8 semantic class
    instance: np.ndarray | None = None  # (N,) int32

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask: np.ndarray) -> "PointMap":
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = None if v is None else v[mask]
        return PointMap(**kw)

    def save(self, path: Path) -> None:
        arrays = {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}
        arrays["xyz"] = self.xyz.astype(np.float64)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "PointMap":
        with np.load(path) as z:
            return cls(**{f.name: (z[f.name] if f.name in z else None) for f in fields(cls)})


def accumulate(kfs: KeyframeSet, poses: np.ndarray, voxel: float, max_range: float = 20.0,
               fusion: _core.MapFusionConfig | None = None) -> PointMap:
    """Merges all keyframes into one voxel-downsampled cloud of the returns
    within `max_range` (C++ `MapAccumulator`).

    With `fusion.fuse` (the default), the observations of a voxel taken from
    comparable ranges are averaged (two passes over the sweeps) and the point
    gets a sigma once compute_normals() knows its surface normal; otherwise
    the observation from the closest sensor position wins.
    """
    fusion = fusion if fusion is not None else _core.MapFusionConfig()
    acc = _core.MapAccumulator(voxel, max_range, fusion)
    if fusion.fuse:
        for k in range(len(kfs)):
            acc.add_ranges(poses[k], kfs.load(k)[0])
    for k in range(len(kfs)):
        pts, intensity = kfs.load(k)
        acc.add(k, poses[k], pts, intensity)
    result = acc.result()
    kf = result["keyframe"]
    pm = PointMap(result["xyz"], result["intensity"].astype(np.float32), kf.astype(np.int32),
                  kfs.trip[kf].astype(np.int16),
                  observations=np.minimum(result["observations"], 65535).astype(np.uint16))
    # Kept until compute_normals() turns them into sigma (not saved).
    pm._fusion = (result["spread"], result["model_sigma"], fusion.sigma_floor)  # pylint: disable=protected-access
    return pm


def compute_normals(pm: PointMap, poses: np.ndarray, radius: float) -> None:
    """Oriented normals and planarity; the sigma of fused points along them."""
    normals, planarity = _core.map_normals(pm.xyz, pm.kf, poses[:, :3, 3], radius, 6, 48)
    pm.normals = normals.astype(np.float32)
    pm.planarity = planarity.astype(np.float32)
    fusion = getattr(pm, "_fusion", None)
    if fusion is not None:
        spread, model_sigma, floor = fusion
        pm.sigma = _core.point_sigmas(normals, pm.observations.astype(np.int32), spread.astype(np.float64),
                                      model_sigma.astype(np.float64), floor).astype(np.float32)


def carve_dynamic(pm: PointMap, kfs: KeyframeSet, poses: np.ndarray, cfg, log=print,
                  ego_paths: list[np.ndarray] | None = None) -> np.ndarray:
    """Per-session range-image visibility voting plus the ego vehicle's swept
    volume (C++ `DynamicDetector`). Sets ``pm.dynamic_kind`` (0 static, 1 moved
    within its session, 2 changed between sessions) and returns the dynamic mask."""
    if pm.normals is None:
        raise ValueError("compute_normals() must run before carve_dynamic()")
    det = _core.DynamicDetector(pm.xyz, pm.normals.astype(np.float64), pm.trip.astype(np.int32), dynamic_config(cfg))
    if not cfg.map.dynamic.el_res:  # range-image rows at the measured beam spacing
        sample = np.linspace(0, len(kfs) - 1, min(3, len(kfs))).astype(int)
        det.set_elevation_resolution(_core.estimate_beam_spacing([kfs.load(int(k))[0] for k in sample]))
    for k in range(len(kfs)):
        det.add_sweep(int(kfs.trip[k]), poses[k], kfs.load(k)[0])
    for t, path in enumerate(ego_paths or []):
        det.add_ego_path(t, path)
    if pm.height is not None:
        det.set_heights(pm.height.astype(np.float64))
    pm.dynamic_kind = det.classify()
    dyn = pm.dynamic_kind > 0
    log(f"[map] dynamic points: {dyn.sum()} / {len(dyn)} ({100.0 * dyn.mean():.1f}%), "
        f"moved {(pm.dynamic_kind == 1).sum()}, changed between sessions {(pm.dynamic_kind == 2).sum()}")
    return dyn


def tiled_map_config(cfg) -> _core.TiledMapConfig:
    """The C++ map stage configuration (map section, levels)."""
    config = _core.TiledMapConfig()
    config.voxel = float(cfg.map.voxel)
    config.max_range = float(cfg.map.max_range)
    config.normal_radius = float(cfg.map.normal_radius)
    config.tile_size = float(cfg.map.get("tile_size") or 0.0)
    config.fusion = fusion_config(cfg)
    config.levels = levels_lib.level_config(cfg)
    config.dynamic = bool(cfg.map.dynamic.enabled)
    config.dynamic_config = dynamic_config(cfg)
    return config


def build_map(kfs: KeyframeSet, poses: np.ndarray, paths: list[np.ndarray],
              cfg, out: Path, log=print) -> tuple[dict, dict]:
    """The map stage (C++ `build_tiled_map`): accumulation, normals and
    sigmas, levels and dynamic points, written to `out` (map/) as one
    point_map.npz or, with `map.tile_size`, as tiles.

    Args:
        kfs: Keyframes (sweeps in the LiDAR frame).
        poses: (K, 4, 4) world <- LiDAR per keyframe.
        paths: Vehicle origins on the floor, (N, 3) per trip.
        cfg: Configuration.
        out: The map directory.
        log: Logger.

    Returns:
        (levels as C++ detect_levels returns them, summary).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "tiles", ignore_errors=True)
    for name in ("tiles.json", "point_map.npz"):
        (out / name).unlink(missing_ok=True)
    config = tiled_map_config(cfg)
    tiles = []

    def load(k):
        points, intensity = kfs.load(k)
        return points, intensity

    def sink(tile, arrays):
        if tile["whole"]:
            del arrays["range"], arrays["sequence"]
            np.savez(out / "point_map.npz", **arrays)
            return
        name = f"tiles/tile_{tile['ix']}_{tile['iy']}.npz"
        (out / "tiles").mkdir(exist_ok=True)
        np.savez(out / name, **arrays)
        tiles.append({"ix": tile["ix"], "iy": tile["iy"], "file": name,
                      "points": len(arrays["xyz"])})

    result = _core.build_tiled_map(
        np.ascontiguousarray(poses, dtype=np.float64),
        np.asarray(kfs.trip, np.int32),
        [np.ascontiguousarray(p, dtype=np.float64) for p in paths], load,
        config, sink, log)
    if config.tile_size > 0.0:
        (out / "tiles.json").write_text(json.dumps({
            "tile_size": config.tile_size, "voxel": config.voxel,
            "points": result["points"], "tiles": tiles}, indent=1))
    return result.pop("levels"), result


def load_point_map(map_dir: Path, bounds=None) -> PointMap:
    """The point map of a map directory, built in one piece or in tiles.

    Args:
        map_dir: work/map.
        bounds: (x0, y0, x1, y1): only the tiles that touch this box (a
            tiled map; the points of those tiles, not cut to the box). None:
            the whole map, in the order of a map built in one piece.
    """
    map_dir = Path(map_dir)
    index = map_dir / "tiles.json"
    if not index.exists():
        return PointMap.load(map_dir / "point_map.npz")
    meta = json.loads(index.read_text())
    size = float(meta["tile_size"])
    parts = []
    for tile in meta["tiles"]:
        if bounds is not None:
            x0, y0 = tile["ix"] * size, tile["iy"] * size
            if (x0 > bounds[2] or x0 + size < bounds[0]
                    or y0 > bounds[3] or y0 + size < bounds[1]):
                continue
        with np.load(map_dir / tile["file"]) as z:
            parts.append({name: z[name] for name in z.files})
    names = [f.name for f in fields(PointMap)]
    if not parts:
        return PointMap(np.zeros((0, 3)), np.zeros(0, np.float32),
                        np.zeros(0, np.int32), np.zeros(0, np.int16))
    arrays = {key: np.concatenate([p[key] for p in parts])
              for key in parts[0]}
    order = np.lexsort((arrays["sequence"], arrays["range"]))
    return PointMap(**{name: arrays[name][order] if name in arrays else None
                       for name in names})
