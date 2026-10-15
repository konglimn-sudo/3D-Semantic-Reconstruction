"""Structure by geometric rules (section 5, route C): floor / ramp / ceiling /
MEP point labels, columns and walls from multi-height slicing, column-grid
regularisation and axis labels.

Columns are found as cells that stay occupied across all horizontal slices up
to the top one (cars stop below it, ducts start above it). Their faces are then
measured on the map points (median position of the points whose normals face
that way), which removes the raster quantisation; columns with an unseen side
are completed with the typical column size away from the seen face (or away
from the driven path). Grid lines are clustered from the centres for the axis
labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy import ndimage
from scipy import spatial

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


def dominant_yaw(normals: np.ndarray) -> float:
    """Orientation of the column / wall grid from vertical-surface normals."""
    vert = np.abs(normals[:, 2]) < 0.2
    a = np.arctan2(normals[vert, 1], normals[vert, 0])
    if len(a) < 10:
        return 0.0
    m = np.angle(np.mean(np.exp(4j * a)))  # 90-degree symmetric circular mean
    return float(m / 4.0)


def _components(mask: np.ndarray):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    return lab, n


def _snap(values: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """1D clustering; returns snapped values and the sorted line positions."""
    if len(values) == 0:
        return values, values
    order = np.argsort(values)
    groups, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if values[b] - values[a] > tol:
            groups.append(cur)
            cur = []
        cur.append(b)
    groups.append(cur)
    lines = np.array([np.median(values[g]) for g in groups])
    out = values.copy()
    for g, v in zip(groups, lines):
        out[g] = v
    return out, lines


def _measure_faces(xy: np.ndarray, normals: np.ndarray, box, band: float = 0.15,
                   min_points: int = 15) -> dict:
    """Face positions of a column box (grid frame) from the map points on its
    faces: {(axis, side): position}, side -1 for the low face.

    Points count for a face if they lie within `band` of the raster edge, away
    from the corners, and their normal points out of that face (map normals
    face the sensor, i.e. away from the column).
    """
    x0, x1, y0, y1 = box
    faces = {}
    for axis, (lo, hi), (c0, c1) in ((0, (x0, x1), (y0, y1)), (1, (y0, y1), (x0, x1))):
        coord, across = xy[:, axis], xy[:, 1 - axis]
        inside = (across > c0 + 0.05) & (across < c1 - 0.05)
        for side, edge in ((-1, lo), (1, hi)):
            on = inside & (np.abs(coord - edge) < band) & (side * normals[:, axis] > 0.7)
            if on.sum() >= min_points:
                faces[(axis, side)] = float(np.median(coord[on]))
    return faces


def _axis_name(k: int) -> str:
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(65 + r) + s
    return s


def extract_structure(pm: PointMap, levels: LevelModel, paths_xy: list[np.ndarray], cfg, log=print) -> StructureResult:
    sc = cfg.structure
    n = len(pm)
    labels = np.zeros(n, np.uint8)
    static = ~pm.dynamic if pm.dynamic is not None else np.ones(n, bool)
    nz = pm.normals[:, 2]
    h = pm.height
    floor_like = static & (np.abs(h) < cfg.bev.floor_tolerance) & (nz > 0.8)
    labels[floor_like & ~pm.on_ramp] = cid("floor")
    labels[floor_like & pm.on_ramp] = cid("ramp")

    yaw0 = dominant_yaw(pm.normals[static])
    to_grid, to_world = rot2(-yaw0), rot2(yaw0)
    path_g = np.vstack(paths_xy) @ to_grid.T if paths_xy else np.zeros((0, 2))
    res = float(sc.raster_res)
    columns: list[Column] = []
    walls: list[Wall] = []

    for lv in range(levels.n):
        ceil_h = float(levels.ceiling[lv] - levels.z[lv])
        sel = np.where(static & (pm.level == lv))[0]
        if len(sel) == 0:
            continue
        hs = h[sel]
        labels[sel[(hs > ceil_h - cfg.levels.ceiling_band) & (nz[sel] < -0.5)]] = cid("ceiling")
        xy = pm.xyz[sel, :2] @ to_grid.T
        lo = xy.min(axis=0) - 1.0
        W, H = (np.ceil((xy.max(axis=0) + 1.0 - lo) / res)).astype(int)
        ij = ((xy - lo) / res).astype(int)
        slices = [s for s in sc.slice_heights if s < ceil_h - 0.3]
        occ = []
        for s in slices:
            m = np.abs(hs - s) < sc.slice_thickness / 2
            g = np.zeros((H, W), bool)
            g[ij[m, 1], ij[m, 0]] = True
            occ.append(ndimage.binary_dilation(g, iterations=1))
        occ = np.asarray(occ)
        # Height reached per cell, ignoring the ceiling itself (its points would
        # make every cell look "tall").
        below = (hs < ceil_h - 0.3) & (labels[sel] != cid("ceiling"))
        hmax_cells = np.full((H, W), -np.inf)
        np.maximum.at(hmax_cells, (ij[below, 1], ij[below, 0]), hs[below])
        hmax_cells = ndimage.grey_dilation(hmax_cells, size=3)  # cells are dilated like the slices
        tall_cells = hmax_cells >= ceil_h - 0.8
        count = occ.sum(axis=0)
        persistent = (count >= len(slices) - 1) & occ[-1]
        any_occ = occ.any(axis=0)

        # Walls: straight runs (>= 1 m along a grid axis) of any slice, judged
        # per run: thin, and either reaching the ceiling or longer than a car.
        # Splitting into runs keeps the perimeter ring / L-junctions apart.
        wall_cells = np.zeros((H, W), bool)
        bulge_cols = []
        run_len = max(3, int(round(1.0 / res)))
        any_u8 = any_occ.astype(np.uint8)
        for axis in (0, 1):  # 0: runs along grid x, 1: along grid y
            kernel = np.ones((1, run_len), np.uint8) if axis == 0 else np.ones((run_len, 1), np.uint8)
            runs = cv2.morphologyEx(any_u8, cv2.MORPH_OPEN, kernel).astype(bool)
            lab_r, n_r = _components(runs)
            for c in range(1, n_r + 1):
                cells = lab_r == c
                jj, ii = np.where(cells)
                along = ii if axis == 0 else jj
                across = jj if axis == 0 else ii
                length = (along.max() - along.min() + 1) * res
                prof = np.bincount(along - along.min())
                thick = np.median(prof[prof > 0]) * res
                if length < sc.wall_min_length or thick > sc.wall_max_thickness:
                    continue
                tall = tall_cells[cells].mean() >= 0.5  # robust to a few stray high points
                if not tall and length < sc.parapet_min_length:
                    continue  # e.g. the side of a parked car
                wall_cells |= cells
                mid = (np.median(across) + 0.5) * res + (lo[1] if axis == 0 else lo[0])
                a0 = along.min() * res + (lo[0] if axis == 0 else lo[1])
                a1 = (along.max() + 1) * res + (lo[0] if axis == 0 else lo[1])
                p0 = np.array([a0, mid]) if axis == 0 else np.array([mid, a0])
                p1 = np.array([a1, mid]) if axis == 0 else np.array([mid, a1])
                walls.append(Wall(lv, (p0 @ to_world.T).tolist(), (p1 @ to_world.T).tolist(), float(thick)))
                # Columns embedded in the wall show up as persistent thickness bulges.
                pj, pi = np.where(persistent & cells)
                if len(pj) == 0:
                    continue
                p_along = (pi if axis == 0 else pj) - along.min()
                prof_p = np.bincount(p_along, minlength=len(prof))
                bulge = prof_p * res >= thick + 0.2
                for a, b in _runs(bulge):
                    span = (b - a) * res
                    if sc.column_min <= span <= sc.column_max:
                        sel_b = (p_along >= a) & (p_along < b)
                        bulge_cols.append((pi[sel_b], pj[sel_b]))

        # Columns: compact persistent components (+ bulges found in walls).
        lab_p, n_p = _components(persistent & ~wall_cells)
        cands = []
        for c in range(1, n_p + 1):
            jj, ii = np.where(lab_p == c)
            cands.append((ii, jj))
        cands += bulge_cols
        boxes = []
        for ii, jj in cands:
            # Slices were dilated by one cell: shrink the extent back.
            x0, x1 = lo[0] + (ii.min() + 1) * res, lo[0] + ii.max() * res
            y0, y1 = lo[1] + (jj.min() + 1) * res, lo[1] + jj.max() * res
            if max(x1 - x0, y1 - y0) > sc.column_max or min(x1 - x0, y1 - y0) < sc.column_min * 0.5:
                continue
            boxes.append([x0, x1, y0, y1])
        if not boxes:
            continue
        boxes = _merge_boxes(np.asarray(boxes), gap=0.3, max_size=sc.column_max)
        # Faces measured on the map points (column height band only).
        face_band = (hs > 0.3) & (hs < min(2.0, ceil_h - 0.5))
        face_xy = xy[face_band]
        face_normals = pm.normals[sel[face_band], :2] @ to_grid.T
        tree = spatial.cKDTree(face_xy) if len(face_xy) else None
        measured = []
        for box in boxes:
            if not sc.get("measure_faces", True) or tree is None:
                measured.append({})
                continue
            c = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])
            near = tree.query_ball_point(c, 0.5 * np.hypot(box[1] - box[0], box[3] - box[2]) + 0.3)
            measured.append(_measure_faces(face_xy[near], face_normals[near], box))
        both = [faces[(axis, 1)] - faces[(axis, -1)] for faces in measured for axis in (0, 1)
                if (axis, -1) in faces and (axis, 1) in faces]
        raster_typ = np.median(np.maximum(boxes[:, 1] - boxes[:, 0], boxes[:, 3] - boxes[:, 2]))
        typ = float(np.median(both)) if both else float(raster_typ)
        centers, sizes, completed = [], [], []
        for (x0, x1, y0, y1), faces in zip(boxes, measured):
            c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
            done = False
            near = path_g[np.argmin(np.linalg.norm(path_g - c, axis=1))] if len(path_g) else c
            extent = [[x0, x1], [y0, y1]]
            for axis in (0, 1):
                lo_face, hi_face = faces.get((axis, -1)), faces.get((axis, 1))
                if lo_face is not None and hi_face is not None:
                    extent[axis] = [lo_face, hi_face]
                elif lo_face is not None:  # Far face unseen: the typical size.
                    done = True
                    extent[axis] = [lo_face, lo_face + typ]
                elif hi_face is not None:
                    done = True
                    extent[axis] = [hi_face - typ, hi_face]
                elif extent[axis][1] - extent[axis][0] < typ - 0.08:
                    # No face measured and too small: extend away from the path.
                    done = True
                    a0, a1 = extent[axis]
                    extent[axis] = [a0, a0 + typ] if near[axis] < c[axis] else [a1 - typ, a1]
            (x0, x1), (y0, y1) = extent
            centers.append([(x0 + x1) / 2, (y0 + y1) / 2])
            sizes.append([x1 - x0, y1 - y0])
            completed.append(done)
        centers = np.asarray(centers)
        # Grid lines for the axis labels. Measured centres stay as they are;
        # raster centres are snapped to the grid lines.
        sx, xlines = _snap(centers[:, 0], sc.grid_snap_tol)
        sy, ylines = _snap(centers[:, 1], sc.grid_snap_tol)
        if sc.get("measure_faces", True):
            sx, sy = centers[:, 0], centers[:, 1]
        for k, (cx, cy) in enumerate(zip(sx, sy)):
            ax = int(np.argmin(np.abs(xlines - cx)))
            ay = int(np.argmin(np.abs(ylines - cy)))
            label = f"{_axis_name(ay)}{ax + 1}"
            world = np.array([cx, cy]) @ to_world.T
            columns.append(Column(f"L{lv}-{label}", lv, world.tolist(), list(map(float, sizes[k])), yaw0,
                                  float(levels.z[lv]), float(levels.ceiling[lv]), bool(completed[k]), label))
        # Point labels: columns first, then walls, then MEP below the ceiling.
        g_xy = xy
        free = labels[sel] == 0
        for col in [c for c in columns if c.level == lv]:
            cg = np.asarray(col.center) @ to_grid.T
            half = np.asarray(col.size) / 2 + 0.05
            inside = (np.abs(g_xy - cg) <= half).all(axis=1) & (hs > 0.02) & free
            labels[sel[inside]] = cid("column")
        free = labels[sel] == 0
        in_wall = wall_cells[ij[:, 1], ij[:, 0]] & (hs > 0.02) & (np.abs(nz[sel]) < 0.5) & free
        labels[sel[in_wall]] = cid("wall")
        free = labels[sel] == 0
        mep_lo = max(2.2, ceil_h - sc.mep_band[1])
        labels[sel[free & (hs > mep_lo) & (hs < ceil_h - sc.mep_band[0])]] = cid("mep")
    log(f"[structure] grid yaw {np.degrees(yaw0):.1f} deg, {len(columns)} columns, {len(walls)} wall segments")
    return StructureResult(labels, columns, walls, yaw0)


def _runs(mask: np.ndarray):
    out, start = [], None
    for k, v in enumerate(mask):
        if v and start is None:
            start = k
        elif not v and start is not None:
            out.append((start, k))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _merge_boxes(boxes: np.ndarray, gap: float, max_size: float) -> np.ndarray:
    """Unions axis-aligned boxes closer than `gap` (fragments of one column
    seen from different sides) as long as the union stays column-sized."""
    boxes = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                A, B = boxes[a], boxes[b]
                dx = max(A[0], B[0]) - min(A[1], B[1])
                dy = max(A[2], B[2]) - min(A[3], B[3])
                if dx > gap or dy > gap:
                    continue
                U = [min(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), max(A[3], B[3])]
                if max(U[1] - U[0], U[3] - U[2]) > max_size:
                    continue
                boxes[a] = U
                boxes.pop(b)
                merged = True
                break
            if merged:
                break
    return np.asarray(boxes)
