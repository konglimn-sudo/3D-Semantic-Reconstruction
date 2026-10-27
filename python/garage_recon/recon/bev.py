"""Per-level bird's-eye-view rasters: LiDAR intensity, height above floor,
floor height and observation counts (section 4.3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .. import _core
from .levels import LevelModel
from .static_map import PointMap


@dataclass
class BevGrid:
    level: int
    x0: float
    y0: float
    res: float
    width: int
    height: int

    def centers(self) -> np.ndarray:
        """(H*W, 2) cell centres, row-major (row j <-> y0 + (j + .5) res)."""
        xs = self.x0 + (np.arange(self.width) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.height) + 0.5) * self.res
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X.ravel(), Y.ravel()], axis=1)

    def world_to_px(self, xy: np.ndarray) -> np.ndarray:
        """Continuous (col, row) pixel coordinates."""
        return np.stack([(xy[:, 0] - self.x0) / self.res, (xy[:, 1] - self.y0) / self.res], axis=1)

    def px_to_world(self, cr: np.ndarray) -> np.ndarray:
        return np.stack([self.x0 + cr[:, 0] * self.res, self.y0 + cr[:, 1] * self.res], axis=1)

    def to_json(self) -> dict:
        return asdict(self)


def level_grid(pm: PointMap, level: int, res: float, margin: float) -> BevGrid:
    """Bounds of the static points of `level` plus `margin` (C++)."""
    x0, y0, width, height = _core.level_grid(pm.xyz, pm.level, pm.dynamic,
                                             level, res, margin)
    return BevGrid(level, x0, y0, res, width, height)


def floor_height_raster(levels: LevelModel, grid: BevGrid) -> np.ndarray:
    """(H, W) floor height: the level's floor grid, ramps leaving it
    downwards by their plane (C++)."""
    return _core.floor_height_raster(*levels.core_parts(), grid.level, grid.x0,
                                     grid.y0, grid.res, grid.width,
                                     grid.height)


def level_rasters(pm: PointMap, levels: LevelModel, grid: BevGrid, cfg) -> dict[str, np.ndarray]:
    """Floor intensity / observations, highest object on the floor and the
    floor height of a level (C++ LevelRasters). Objects stand between the
    floor tolerance and `object_max_height`: the ceiling and the services
    below it must not count, or every floor cell under an observed ceiling
    looks covered."""
    bc = cfg.bev
    out = _core.bev_rasters(
        pm.xyz, pm.intensity, pm.level, pm.dynamic, pm.height, pm.normals,
        *levels.core_parts(), grid.level, grid.x0, grid.y0, grid.res,
        grid.width, grid.height, float(bc.floor_tolerance),
        float(bc.get("object_max_height", 2.2)))
    return {k: out[k] for k in ("intensity", "floor_count", "height_max",
                                 "floor_z")}


def save_bev(path: Path, grid: BevGrid, rasters: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, grid=json.dumps(grid.to_json()), **rasters)


def load_bev(path: Path) -> tuple[BevGrid, dict[str, np.ndarray]]:
    with np.load(path) as z:
        grid = BevGrid(**json.loads(str(z["grid"])))
        return grid, {k: z[k] for k in z.files if k != "grid"}


def to_png(path: Path, raster: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> None:
    """Writes a raster with north (+y) up."""
    img = raster.astype(float)
    if img.ndim == 2:
        finite = np.isfinite(img)
        lo = np.nanpercentile(img[finite], 1) if vmin is None and finite.any() else (vmin or 0.0)
        hi = np.nanpercentile(img[finite], 99) if vmax is None and finite.any() else (vmax or 1.0)
        img = np.where(finite, (img - lo) / max(hi - lo, 1e-9), 0.0)
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    else:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), np.flipud(img))
