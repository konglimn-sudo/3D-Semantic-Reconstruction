"""Per-level navigation layers on a grid.

Obstacles come from the geometry: every static map point between
`obstacle_min_height` and the vehicle height plus `top_margin` above the
floor blocks its cell whatever its class, so a mislabelled wall still blocks.
Unclassified points must add up to `min_cell_observations` fused
observations per cell: a static object is seen in many sweeps, remnants of
moving objects and noise once or twice.
The semantic classes then split the obstacles into layers: permanent
structure (walls, columns, barriers, unclassified objects), hanging structure
inside the vehicle's height band (pipes, beams, signs), parked vehicles at
capture time, and low objects the vehicle may cross (speed bumps, wheel
stoppers). The layers are computed in C++ (gr/nav/layers.h).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from garage_recon import _core
from garage_recon.recon import bev
from garage_recon.semantics import taxonomy

VEHICLE_CLASSES = ("vehicle", "vehicle_moving", "two_wheeler", "pedestrian")
LOW_CLASSES = ("wheel_stopper", "speed_bump")
OVERHEAD_CLASSES = ("ceiling", "mep", "beam", "sign", "light")


@dataclasses.dataclass
class LevelLayers:
    """Navigation layers of one level; rasters are (height, width), row j at
    y0 + (j + 0.5) res.

    Attributes:
        grid: The grid.
        floor: Floor observed (gaps between LiDAR rings and small enclosed
            unobserved patches filled).
        driven: Swept by the vehicle during the capture.
        structure: Permanent obstacle within the vehicle's height band.
        overhead: The part of `structure` that hangs from above.
        parked: Parked vehicles (and people) at capture time.
        low: Speed bumps and wheel stoppers.
        slot: Slot index + 1, 0 outside the slots.
        ramp: Inside a ramp outline that touches this level.
        floor_z: Floor height (m).
        clearance: Distance to the nearest obstacle or unknown cell (m).
        headroom: Height above the floor of the lowest static object
            (vehicles and low objects excluded; inf where none, e.g. open
            sky).
    """

    grid: bev.BevGrid
    floor: np.ndarray
    driven: np.ndarray
    structure: np.ndarray
    overhead: np.ndarray
    parked: np.ndarray
    low: np.ndarray
    slot: np.ndarray
    ramp: np.ndarray
    floor_z: np.ndarray
    clearance: np.ndarray
    headroom: np.ndarray | None = None

    @property
    def known(self) -> np.ndarray:
        return self.floor | self.driven | self.ramp

    @property
    def drivable(self) -> np.ndarray:
        return self.known & ~(self.structure | self.parked)

    @property
    def unknown(self) -> np.ndarray:
        return ~(self.known | self.structure | self.parked)

    def cells(self, xy: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(col, row, inside) of world points."""
        g = self.grid
        col = np.floor((xy[:, 0] - g.x0) / g.res).astype(int)
        row = np.floor((xy[:, 1] - g.y0) / g.res).astype(int)
        inside = (col >= 0) & (row >= 0) & (col < g.width) & (row < g.height)
        return (np.clip(col, 0, g.width - 1), np.clip(row, 0, g.height - 1),
                inside)

    def sample(self, name: str, xy: np.ndarray, outside=0):
        """Values of a layer at world points (`outside` beyond the grid)."""
        col, row, inside = self.cells(xy)
        values = getattr(self, name)[row, col]
        return np.where(inside, values, outside)

    @classmethod
    def from_npz(cls, data) -> "LevelLayers":
        """Inverse of to_npz (an open npz file or a dict)."""
        g = np.asarray(data["grid"])
        grid = bev.BevGrid(int(g[5]), float(g[0]), float(g[1]), float(g[2]),
                           int(g[3]), int(g[4]))
        return cls(grid=grid, **{f.name: np.asarray(data[f.name])
                                 for f in dataclasses.fields(cls)
                                 if f.name != "grid" and f.name in data})

    def to_npz(self) -> dict:
        out = {f.name: getattr(self, f.name)
               for f in dataclasses.fields(self)
               if f.name != "grid" and getattr(self, f.name) is not None}
        out["grid"] = np.asarray([self.grid.x0, self.grid.y0, self.grid.res,
                                  self.grid.width, self.grid.height,
                                  self.grid.level])
        return out


def _ids(names) -> list[int]:
    return [taxonomy.cid(n) for n in names]


def fill_polygon(shape: tuple[int, int], grid: bev.BevGrid, polygon,
                 image: np.ndarray | None = None, value: int = 1
                 ) -> np.ndarray:
    """Rasterises a world polygon into `image` (int32, filled in place) as
    cv2.fillPoly does with its cell coordinates (C++)."""
    image = np.zeros(shape, np.int32) if image is None else image
    if image.dtype != np.int32 or not image.flags.c_contiguous:
        raise ValueError("fill_polygon fills a C-contiguous int32 image")
    _core.fill_polygon(image, np.asarray(polygon, float)[:, :2], grid.x0,
                       grid.y0, grid.res, int(value))
    return image


def fill_holes(known: np.ndarray, obstacles: np.ndarray, grid: bev.BevGrid,
               max_area: float) -> np.ndarray:
    """`known` with enclosed unobserved patches smaller than `max_area` (m2)
    added: floor hidden behind parked cars or between sparse scan lines. An
    obstacle there would have left points in the height band; larger or
    open unknown areas stay unknown (C++)."""
    return _core.fill_holes(known, obstacles, grid.res, max_area)


def layer_config(cfg) -> _core.LayerConfig:
    """The `nav` settings of the layers."""
    nav, vehicle = cfg.nav, cfg.nav.vehicle
    config = _core.LayerConfig()
    config.floor_tolerance = float(nav.floor_tolerance)
    config.obstacle_min_height = float(nav.obstacle_min_height)
    config.obstacle_min_points = int(nav.obstacle_min_points)
    config.min_cell_observations = float(nav.min_cell_observations)
    config.max_headroom = float(nav.get("max_headroom", 8.0))
    config.floor_closing = int(nav.floor_closing)
    config.max_hole_area = float(nav.max_hole_area)
    config.vehicle_height = float(vehicle.height)
    config.vehicle_top_margin = float(vehicle.top_margin)
    config.vehicle_width = float(vehicle.width)
    return config


def level_layers(point_map, labels: np.ndarray, levels, slots, path_xy,
                 grid: bev.BevGrid, cfg) -> LevelLayers:
    """Layers of one level (C++ ComputeLevelLayers).

    Args:
        point_map: Point map with height, level, dynamic and normals.
        labels: Class id per map point.
        levels: The LevelModel.
        slots: All slots (those of other levels are skipped).
        path_xy: (N, 2) vehicle positions on this level during the capture.
        grid: The level grid.
        cfg: Configuration (`nav` section).
    """
    level_slots = [(index, np.asarray(s.corners, float))
                   for index, s in enumerate(slots) if s.level == grid.level]
    path = (None if path_xy is None or not len(path_xy)
            else np.asarray(path_xy, float)[:, :2])
    out = _core.level_layers(
        point_map.xyz, point_map.level, point_map.dynamic, point_map.height,
        point_map.normals, np.asarray(labels, np.uint8),
        point_map.observations, _ids(VEHICLE_CLASSES), _ids(LOW_CLASSES),
        _ids(OVERHEAD_CLASSES), taxonomy.UNLABELED, *levels.core_parts(),
        grid.level, level_slots, path, grid.x0, grid.y0, grid.res,
        grid.width, grid.height, layer_config(cfg))
    return LevelLayers(grid, out["floor"], out["driven"], out["structure"],
                       out["overhead"], out["parked"], out["low"],
                       out["slot"], out["ramp"], out["floor_z"],
                       out["clearance"], out["headroom"])
