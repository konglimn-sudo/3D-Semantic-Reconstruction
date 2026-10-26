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
stoppers). The rasterisation and distance transform are C++.
"""

from __future__ import annotations

import dataclasses

import cv2
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
    """Rasterises a world polygon (cells whose centre is inside)."""
    image = np.zeros(shape, np.int32) if image is None else image
    pixels = (np.asarray(polygon, float)[:, :2] - [grid.x0, grid.y0]) / (
        grid.res) - 0.5
    cv2.fillPoly(image, [np.round(pixels * 16).astype(np.int32)], int(value),
                 lineType=cv2.LINE_8, shift=4)
    return image


def fill_holes(known: np.ndarray, obstacles: np.ndarray, grid: bev.BevGrid,
               max_area: float) -> np.ndarray:
    """`known` with enclosed unobserved patches smaller than `max_area` (m2)
    added: floor hidden behind parked cars or between sparse scan lines. An
    obstacle there would have left points in the height band; larger or
    open unknown areas stay unknown."""
    unknown = (~(known | obstacles)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(unknown, 8)
    h, w = known.shape
    small = np.zeros(count, bool)
    for k in range(1, count):
        x, y, bw, bh, area = stats[k]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        small[k] = not touches and area * grid.res ** 2 < max_area
    return known | small[labels]


def level_layers(point_map, labels: np.ndarray, levels, slots, path_xy,
                 grid: bev.BevGrid, cfg) -> LevelLayers:
    """Layers of one level.

    Args:
        point_map: Point map with height, level, dynamic and normals.
        labels: Class id per map point.
        levels: The LevelModel.
        slots: All slots (those of other levels are skipped).
        path_xy: (N, 2) vehicle positions on this level during the capture.
        grid: The level grid.
        cfg: Configuration (`nav` section).
    """
    nav = cfg.nav
    vehicle = nav.vehicle
    shape = (grid.height, grid.width)
    args = (grid.x0, grid.y0, grid.res, grid.width, grid.height)
    xy = np.ascontiguousarray(point_map.xyz[:, :2])

    def count(mask: np.ndarray) -> np.ndarray:
        return _core.rasterize(xy[mask], np.ones(int(mask.sum())), *args,
                               "count", 0.0)

    selected = (point_map.level == grid.level) & ~np.asarray(
        point_map.dynamic, bool)
    height = np.nan_to_num(point_map.height, nan=-10.0)
    nz = (np.abs(point_map.normals[:, 2]) if point_map.normals is not None
          else np.ones(len(height)))
    floor_points = (selected & (np.abs(height) < nav.floor_tolerance)
                    & (nz > 0.7))
    top = vehicle.height + vehicle.top_margin
    band = selected & (height > nav.obstacle_min_height) & (height < top)
    vehicles = np.isin(labels, _ids(VEHICLE_CLASSES))
    low_points = selected & np.isin(labels, _ids(LOW_CLASSES))
    structure_points = band & ~vehicles & ~low_points
    min_points = int(nav.obstacle_min_points)
    unclassified = structure_points & (labels == taxonomy.UNLABELED)
    structure = count(structure_points & ~unclassified) >= min_points
    # Unclassified points must add up to `min_cell_observations` fused
    # observations per cell: a static object is seen in many sweeps,
    # remnants of moving objects and noise in one or two.
    observations = (np.asarray(point_map.observations, float)
                    if point_map.observations is not None
                    else np.full(len(labels), float(nav.min_cell_observations)))
    structure |= _core.rasterize(
        xy[unclassified], observations[unclassified], *args, "sum",
        0.0) >= nav.min_cell_observations
    overhead = structure & (count(
        structure_points & np.isin(labels, _ids(OVERHEAD_CLASSES))
        & (height > 0.5)) >= min_points)
    parked = (count(band & vehicles) >= min_points) & ~structure
    low = count(low_points) >= min_points
    # Headroom: what hangs over the band the vehicle occupies (lower
    # objects are obstacles already), unclassified points filtered by their
    # observations as above.
    above = (selected & (height >= top)
             & (height < nav.get("max_headroom", 8.0)) & ~vehicles)
    above_unclassified = above & (labels == taxonomy.UNLABELED)

    def lowest(mask: np.ndarray) -> np.ndarray:
        return _core.rasterize(xy[mask], height[mask].astype(np.float64),
                               *args, "min", np.inf)

    headroom = lowest(above & ~above_unclassified)
    seen = _core.rasterize(
        xy[above_unclassified], observations[above_unclassified], *args,
        "sum", 0.0) >= nav.min_cell_observations
    headroom = np.where(seen, np.minimum(headroom, lowest(above_unclassified)),
                        headroom).astype(np.float32)

    k = int(nav.floor_closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1,) * 2)
    floor = cv2.morphologyEx((count(floor_points) > 0).astype(np.uint8),
                             cv2.MORPH_CLOSE, kernel) > 0

    driven = np.zeros(shape, bool)
    if path_xy is not None and len(path_xy):
        hits = _core.rasterize(np.ascontiguousarray(path_xy[:, :2]),
                               np.ones(len(path_xy)), *args, "count", 0.0) > 0
        r = max(1, int(np.ceil(0.5 * vehicle.width / grid.res)))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
        driven = cv2.dilate(hits.astype(np.uint8), disc) > 0

    slot = np.zeros(shape, np.int32)
    for index, s in enumerate(slots):
        if s.level == grid.level:
            fill_polygon(shape, grid, s.corners, slot, index + 1)
    ramp = np.zeros(shape, np.int32)
    for r in levels.ramps:
        if grid.level in (r.level_from, r.level_to):
            fill_polygon(shape, grid, r.polygon, ramp)
    ramp = ramp > 0

    known = fill_holes(floor | driven | ramp, structure | parked, grid,
                       nav.max_hole_area)
    blocked = structure | parked | ~known
    clearance = _core.distance_transform(blocked.astype(np.uint8)) * grid.res
    return LevelLayers(grid, known & ~(driven | ramp), driven, structure,
                       overhead, parked, low, slot, ramp,
                       bev.floor_height_raster(levels, grid),
                       clearance.astype(np.float32), headroom)
