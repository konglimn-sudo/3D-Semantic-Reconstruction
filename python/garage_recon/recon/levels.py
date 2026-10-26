"""Level (B1/B2/...) and ramp segmentation (section 4.3 / 5 module 1).

The algorithm is the C++ DetectLevels (gr/mapping/levels.h); this module holds
the level model (save / load, floor heights, assignment of points).

The vehicle drives on the floor, so its trajectory is the most robust source
of floor heights: flat stretches give the levels, sloped stretches give the
ramps. Map points then refine a per-level floor height grid (drainage slopes).
The grid takes the floor points near the level's own trajectory height at
their position, not near one constant: a level may rise gently along a long
aisle (FusionPortableV2: 4.5 m over 400 m at 1.1 %, below the ramp slope).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import _core


@dataclass
class FloorGrid:
    x0: float
    y0: float
    res: float
    z: np.ndarray  # (H, W) floor height

    def sample(self, xy: np.ndarray) -> np.ndarray:
        i = np.clip(((xy[:, 0] - self.x0) / self.res).astype(int), 0, self.z.shape[1] - 1)
        j = np.clip(((xy[:, 1] - self.y0) / self.res).astype(int), 0, self.z.shape[0] - 1)
        return self.z[j, i]


@dataclass
class Ramp:
    level_from: int  # upper level
    level_to: int  # lower level
    plane: np.ndarray  # z = a x + b y + c
    polygon: np.ndarray  # (M, 2) xy outline
    slope: float

    def contains(self, xy: np.ndarray) -> np.ndarray:
        import shapely

        return shapely.contains_xy(shapely.Polygon(self.polygon), xy[:, 0], xy[:, 1])

    def z_at(self, xy: np.ndarray) -> np.ndarray:
        return self.plane[0] * xy[:, 0] + self.plane[1] * xy[:, 1] + self.plane[2]


@dataclass
class LevelModel:
    z: np.ndarray  # (L,) floor height per level, level 0 = top
    ceiling: np.ndarray  # (L,) ceiling height per level
    grids: list[FloorGrid] = field(default_factory=list)
    ramps: list[Ramp] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.z)

    def floor_z(self, xy: np.ndarray, level: int) -> np.ndarray:
        if level < len(self.grids) and self.grids[level] is not None:
            return self.grids[level].sample(xy)
        return np.full(len(xy), self.z[level])

    def assign(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Level index, height above that floor and on-ramp flag per point
        (C++ AssignLevels).

        A point belongs to the highest floor that is not above it; points over
        a ramp outline that sit closer to the ramp surface use the ramp instead.
        """
        grids = [(g.x0, g.y0, g.res, np.asarray(g.z, np.float32))
                 for g in self.grids]
        ramps = [(r.level_from, r.level_to, np.asarray(r.plane, float),
                  np.asarray(r.polygon, float), r.slope) for r in self.ramps]
        return _core.assign_levels(
            np.asarray(self.z, float).tolist(), grids, ramps,
            np.ascontiguousarray(xyz, dtype=np.float64))

    def to_json(self) -> dict:
        return {
            "levels": [{"id": i, "z": float(z), "ceiling": float(c)} for i, (z, c) in enumerate(zip(self.z, self.ceiling))],
            "ramps": [{"level_from": r.level_from, "level_to": r.level_to, "plane": r.plane.tolist(),
                       "polygon": r.polygon.tolist(), "slope": r.slope} for r in self.ramps],
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "levels.json").write_text(json.dumps(self.to_json(), indent=2))
        np.savez(directory / "floor_grids.npz",
                 **{f"g{i}": g.z for i, g in enumerate(self.grids)},
                 meta=np.array([[g.x0, g.y0, g.res] for g in self.grids]))

    @classmethod
    def load(cls, directory: Path) -> "LevelModel":
        d = json.loads((directory / "levels.json").read_text())
        grids = []
        with np.load(directory / "floor_grids.npz") as z:
            for i, m in enumerate(z["meta"]):
                grids.append(FloorGrid(float(m[0]), float(m[1]), float(m[2]), z[f"g{i}"]))
        ramps = [Ramp(r["level_from"], r["level_to"], np.asarray(r["plane"]), np.asarray(r["polygon"]), r["slope"])
                 for r in d["ramps"]]
        return cls(np.array([l["z"] for l in d["levels"]]), np.array([l["ceiling"] for l in d["levels"]]), grids,
                   ramps)


def level_config(cfg) -> _core.LevelConfig:
    config = _core.LevelConfig()
    for key in ("min_separation", "flat_slope", "min_samples", "grid_res",
                "floor_band"):
        if cfg.levels.get(key) is not None:
            setattr(config, key, cfg.levels[key])
    return config


def detect_levels(vehicle_paths: list[np.ndarray], xyz: np.ndarray,
                  normals: np.ndarray, cfg, log=print) -> LevelModel:
    """Levels and ramps (C++ DetectLevels) from the vehicle poses ((N, 4, 4)
    per trip, vehicle origin on the floor) and the map points."""
    out = _core.detect_levels(
        [np.ascontiguousarray(P[:, :3, 3]) for P in vehicle_paths],
        np.ascontiguousarray(xyz, dtype=np.float64),
        np.ascontiguousarray(normals, dtype=np.float64), level_config(cfg),
        log)
    grids = [FloorGrid(g["x0"], g["y0"], g["res"], np.asarray(g["z"]))
             for g in out["grids"]]
    ramps = [Ramp(r["level_from"], r["level_to"], np.asarray(r["plane"]),
                  np.asarray(r["polygon"]), r["slope"]) for r in out["ramps"]]
    return LevelModel(np.asarray(out["z"]), np.asarray(out["ceiling"]), grids,
                      ramps)
