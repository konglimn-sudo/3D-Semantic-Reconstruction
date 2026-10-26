"""Structure by geometric rules (section 5, route C): floor / ramp / ceiling /
MEP point labels, columns and walls from multi-height slicing, column-grid
regularisation and axis labels.

The algorithm is the C++ ExtractStructure (gr/semantics/structure.h); this
module holds the result types. Columns are found as cells that stay occupied
across all horizontal slices up to the top one (cars stop below it, ducts
start above it). Their faces are then
measured on the map points (median position of the points whose normals face
that way), which removes the raster quantisation; columns with an unseen side
are completed with the typical column size away from the seen face (or away
from the driven path). Grid lines are clustered from the centres for the axis
labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .. import _core
from ..recon.levels import LevelModel
from ..recon.static_map import PointMap
from .taxonomy import cid


@dataclass
class Column:
    id: str
    level: int
    center: list[float]  # world xy
    size: list[float]  # along grid x / y
    yaw: float  # grid orientation
    z0: float
    z1: float
    completed: bool  # extent completed from the typical column size
    axis_label: str = ""


@dataclass
class Wall:
    level: int
    p0: list[float]
    p1: list[float]
    thickness: float


@dataclass
class StructureResult:
    labels: np.ndarray  # (N,) uint8 per map point
    columns: list[Column]
    walls: list[Wall]
    yaw: float  # dominant grid orientation (rad)

    def to_json(self) -> dict:
        return {"yaw": self.yaw, "columns": [asdict(c) for c in self.columns], "walls": [asdict(w) for w in self.walls]}


def rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def structure_config(cfg) -> _core.StructureConfig:
    """C++ StructureConfig from the structure / bev / levels sections."""
    sc = cfg.structure
    config = _core.StructureConfig()
    for key in ("slice_thickness", "raster_res", "column_min", "column_max",
                "wall_min_length", "wall_max_thickness", "parapet_min_length",
                "grid_snap_tol", "measure_faces"):
        if sc.get(key) is not None:
            setattr(config, key, sc[key])
    config.slice_heights = [float(v) for v in sc.slice_heights]
    config.mep_band_low, config.mep_band_high = (float(v) for v in sc.mep_band)
    config.floor_tolerance = float(cfg.bev.floor_tolerance)
    config.ceiling_band = float(cfg.levels.ceiling_band)
    for name in ("floor", "ramp", "ceiling", "column", "wall", "mep"):
        setattr(config, f"{name}_id", cid(name))
    return config


def extract_structure(pm: PointMap, levels: LevelModel,
                      paths_xy: list[np.ndarray], cfg,
                      log=print) -> StructureResult:
    """Columns, walls and structure labels (C++ ExtractStructure)."""
    path = np.vstack(paths_xy) if paths_xy else np.zeros((0, 2))
    path = np.column_stack([path, np.zeros(len(path))])
    dynamic = (None if pm.dynamic is None
               else np.asarray(pm.dynamic, np.int32))
    out = _core.extract_structure(
        np.ascontiguousarray(pm.xyz, np.float64),
        np.ascontiguousarray(pm.normals, np.float64),
        np.asarray(pm.height, np.float32), np.asarray(pm.level, np.int32),
        np.asarray(pm.on_ramp, np.int32), dynamic,
        np.asarray(levels.z, float).tolist(),
        np.asarray(levels.ceiling, float).tolist(), path,
        structure_config(cfg), log)
    columns = [Column(c["id"], c["level"], np.asarray(c["center"]).tolist(),
                      np.asarray(c["size"]).tolist(), c["yaw"], c["z0"],
                      c["z1"], c["completed"], c["axis_label"])
               for c in out["columns"]]
    walls = [Wall(w["level"], np.asarray(w["p0"]).tolist(),
                  np.asarray(w["p1"]).tolist(), w["thickness"])
             for w in out["walls"]]
    return StructureResult(np.asarray(out["labels"], np.uint8), columns, walls,
                           float(out["yaw"]))
