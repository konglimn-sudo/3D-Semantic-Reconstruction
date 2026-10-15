"""Global point map (sections 4.2 / 4.3): accumulation, normals and dynamic
points. The algorithms are C++ (`MapAccumulator`, `map_normals`,
`DynamicDetector`); this module feeds them the keyframe sweeps and keeps the
per-point attributes in `PointMap`."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .. import _core
from ..core_config import dynamic_config, fusion_config
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
