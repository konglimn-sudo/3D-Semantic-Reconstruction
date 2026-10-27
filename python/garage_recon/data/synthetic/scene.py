"""Procedural multi-level underground garage with ground truth.

Geometry (per level, plan view, y up):

    row D  bays   (entrance on the low-y side)
    aisle 2       (on upper levels the east part is a ramp opening down to the next level)
    row C  bays   (entrance on the high-y side)
    row B  bays   (entrance on the low-y side)
    aisle 1
    row A  bays   (entrance on the high-y side)

Cross aisles run along the west and east ends. Columns sit at the aisle-side
corners of every bay group; painted separators bound each bay. Arrows painted
on the aisles give the traffic direction: one arrow per lane of a two-way
aisle (right-hand traffic), centred arrows on a one-way aisle.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ...geometry import rot_z
from ...semantics.taxonomy import cid


@dataclass
class Layout:
    n_levels: int = 2
    level_height: float = 3.6
    slab: float = 0.3
    groups: int = 4
    bays_per_group: int = 3
    bay_width: float = 2.5
    bay_depth: float = 5.3
    aisle_width: float = 6.0
    cross_width: float = 6.0
    column_size: float = 0.5
    line_width: float = 0.15
    wall_thickness: float = 0.3
    ramp_length: float = 24.0
    duct_drop: float = 0.6
    duct_height: float = 0.4
    duct_width: float = 0.8
    parapet_height: float = 1.1
    arrows: bool = True  # Paint the floor arrows.
    arrow_length: float = 3.5
    arrow_shaft: float = 0.2
    arrow_head_length: float = 1.2
    arrow_head_width: float = 0.7
    # One-way aisles, "L<level>-<aisle><sign>": the sign is the direction of
    # travel along the aisle axis (aisles "1", "2" run along x, the cross
    # aisles "W", "E" along y). None by default: the aisles of a level form
    # a tree (the ramp opening cuts aisle 2), so any one-way aisle would cut
    # part of the level off from the entrance.
    one_way: tuple = ()
    # Slot attributes: charging slots have a green floor and 充电 painted in
    # them; the others carry their text (无障碍, 专用) on the floor. Ids
    # missing in a smaller layout are ignored.
    ev_slots: tuple = ("L0-A-002", "L0-A-003", "L1-B-005")
    accessible_slots: tuple = ("L0-B-001",)
    reserved_slots: tuple = ("L0-D-002",)

    def slot_attributes(self, slot_id: str) -> list[str]:
        named = {"ev": self.ev_slots, "accessible": self.accessible_slots,
                 "reserved": self.reserved_slots}
        return [name for name, ids in named.items() if slot_id in ids]

    @property
    def group_span(self) -> float:
        return self.bays_per_group * self.bay_width + self.column_size

    @property
    def x_lo(self) -> float:
        return -self.column_size / 2.0

    @property
    def x_hi(self) -> float:
        return self.groups * self.group_span + self.column_size / 2.0

    @property
    def y_hi(self) -> float:
        return 4.0 * self.bay_depth + 2.0 * self.aisle_width

    def rows(self) -> dict[str, tuple[float, float, str]]:
        d, a = self.bay_depth, self.aisle_width
        return {
            "A": (0.0, d, "high"),
            "B": (d + a, 2 * d + a, "low"),
            "C": (2 * d + a, 3 * d + a, "high"),
            "D": (3 * d + 2 * a, 4 * d + 2 * a, "low"),
        }

    def aisles(self) -> dict[str, tuple[float, float]]:
        d, a = self.bay_depth, self.aisle_width
        return {"1": (d, d + a), "2": (3 * d + a, 3 * d + 2 * a)}

    def footprint(self) -> tuple[float, float, float, float]:
        return self.x_lo - self.cross_width, self.x_hi + self.cross_width, 0.0, self.y_hi

    def level_z(self, i: int) -> float:
        return -i * self.level_height

    def ceiling_z(self, i: int) -> float:
        return self.level_z(i) + self.level_height - self.slab

    @property
    def has_ramp(self) -> bool:
        return self.n_levels > 1

    def ramp_x(self) -> tuple[float, float]:
        x1 = self.x_hi
        return max(self.x_lo, x1 - self.ramp_length), x1


@dataclass
class Box:
    center: np.ndarray
    size: np.ndarray
    R: np.ndarray
    cls: int  # taxonomy id (slabs are resolved per face at hit time)
    kind: str  # slab | roof | wall | column | ramp | parapet | duct | car | moving_car
    level: int
    instance: int = -1
    key: str = ""  # stable object identity across trips (parked cars)


@dataclass
class GarageScene:
    layout: Layout
    boxes: list[Box] = field(default_factory=list)
    slots: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    columns: list[dict] = field(default_factory=list)
    ramps: list[dict] = field(default_factory=list)
    # Floor arrows: id, level, aisle, tail / tip (xy), direction, one_way and
    # the painted rectangles [x0, x1, y0, y1].
    arrows: list[dict] = field(default_factory=list)

    def levels(self) -> list[dict]:
        L = self.layout
        return [{"id": i, "z": L.level_z(i), "ceiling": L.ceiling_z(i)} for i in range(L.n_levels)]

    def lines_array(self) -> np.ndarray:
        """(M, 5) [level, x0, x1, y0, y1] painted rectangles: the lines, then
        the rectangles of the arrows."""
        rows = [[l["level"], l["x0"], l["x1"], l["y0"], l["y1"]] for l in self.lines]
        rows += [[a["level"], *r] for a in self.arrows for r in a["rects"]]
        return np.array(rows, float) if rows else np.zeros((0, 5))

    def paint_classes(self) -> np.ndarray:
        """Taxonomy class of every row of lines_array()."""
        n_arrow = sum(len(a["rects"]) for a in self.arrows)
        return np.array([cid("parking_line")] * len(self.lines) + [cid("arrow")] * n_arrow, np.int64)

    def to_json(self) -> dict:
        return {
            "layout": self.layout.__dict__,
            "levels": self.levels(),
            "slots": self.slots,
            "lines": self.lines,
            "columns": self.columns,
            "ramps": self.ramps,
            "arrows": self.arrows,
            "boxes": [
                {
                    "center": b.center.tolist(),
                    "size": b.size.tolist(),
                    "R": b.R.tolist(),
                    "cls": b.cls,
                    "kind": b.kind,
                    "level": b.level,
                    "instance": b.instance,
                }
                for b in self.boxes
            ],
        }


def _box(center, size, cls, kind, level, R=None, instance=-1) -> Box:
    return Box(np.asarray(center, float), np.asarray(size, float), np.eye(3) if R is None else R, cls, kind, level,
               instance)


def _xy_box(x0, x1, y0, y1, z0, z1, cls, kind, level) -> Box:
    return _box([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], [x1 - x0, y1 - y0, z1 - z0], cls, kind, level)


def build_static_scene(L: Layout) -> GarageScene:
    sc = GarageScene(L)
    fx0, fx1, fy0, fy1 = L.footprint()
    a2 = L.aisles()["2"]
    rx0, rx1 = L.ramp_x()
    wt = L.wall_thickness
    floor, wall, column, mep = cid("floor"), cid("wall"), cid("column"), cid("mep")

    for i in range(L.n_levels):
        z, zc = L.level_z(i), L.ceiling_z(i)
        has_opening = L.has_ramp and i < L.n_levels - 1
        # Floor slab (minus the ramp opening on upper levels).
        if has_opening:
            sc.boxes += [
                _xy_box(fx0, fx1, fy0, a2[0], z - L.slab, z, floor, "slab", i),
                _xy_box(fx0, fx1, a2[1], fy1, z - L.slab, z, floor, "slab", i),
                _xy_box(fx0, rx0, a2[0], a2[1], z - L.slab, z, floor, "slab", i),
                _xy_box(rx1, fx1, a2[0], a2[1], z - L.slab, z, floor, "slab", i),
            ]
        else:
            sc.boxes.append(_xy_box(fx0, fx1, fy0, fy1, z - L.slab, z, floor, "slab", i))
        if i == 0:
            sc.boxes.append(_xy_box(fx0, fx1, fy0, fy1, zc, zc + L.slab, floor, "roof", i))
        # Perimeter walls.
        sc.boxes += [
            _xy_box(fx0 - wt, fx0, fy0 - wt, fy1 + wt, z, zc, wall, "wall", i),
            _xy_box(fx1, fx1 + wt, fy0 - wt, fy1 + wt, z, zc, wall, "wall", i),
            _xy_box(fx0, fx1, fy0 - wt, fy0, z, zc, wall, "wall", i),
            _xy_box(fx0, fx1, fy1, fy1 + wt, z, zc, wall, "wall", i),
        ]
        # Columns at the aisle-side corners of every bay group.
        cs = L.column_size
        col_rows = {}
        for name, (y0, y1, ent) in L.rows().items():
            col_rows[name] = (y1 - cs, y1) if ent == "high" else (y0, y0 + cs)
        for name, (cy0, cy1) in col_rows.items():
            for k in range(L.groups + 1):
                x = k * L.group_span
                sc.boxes.append(_xy_box(x - cs / 2, x + cs / 2, cy0, cy1, z, zc, column, "column", i))
                sc.columns.append({
                    "id": f"L{i}-{name}{k + 1}",
                    "level": i,
                    "center": [x, (cy0 + cy1) / 2],
                    "size": [cs, cs],
                    "z0": z,
                    "z1": zc,
                })
        # Ducts along both aisles (skipped above the ramp corridor).
        for name, (ya, yb) in L.aisles().items():
            yc = (ya + yb) / 2
            dz0, dz1 = zc - L.duct_drop, zc - L.duct_drop + L.duct_height
            hw = L.duct_width / 2
            if name == "2" and L.has_ramp:
                if rx0 - 0.5 > 0.0:
                    sc.boxes.append(_xy_box(0.0, rx0 - 0.5, yc - hw, yc + hw, dz0, dz1, mep, "duct", i))
            else:
                sc.boxes.append(_xy_box(0.0, L.x_hi - 0.5, yc - hw, yc + hw, dz0, dz1, mep, "duct", i))

    # Ramps between consecutive levels in aisle 2 (top at x = rx1).
    for i in range(L.n_levels - 1):
        z_top, z_bot = L.level_z(i), L.level_z(i + 1)
        p_bot, p_top = np.array([rx0, z_bot]), np.array([rx1, z_top])
        u = (p_top - p_bot) / np.linalg.norm(p_top - p_bot)
        length = float(np.linalg.norm(p_top - p_bot))
        ux, uz = u
        n = np.array([-uz, 0.0, ux])
        R = np.stack([np.array([ux, 0.0, uz]), np.array([0.0, 1.0, 0.0]), n], axis=1)
        mid = np.array([(rx0 + rx1) / 2, (a2[0] + a2[1]) / 2, (z_top + z_bot) / 2]) - n * L.slab / 2
        sc.boxes.append(_box(mid, [length, a2[1] - a2[0], L.slab], cid("ramp"), "ramp", i, R=R))
        sc.ramps.append({"level_from": i, "level_to": i + 1, "x0": rx0, "x1": rx1, "y0": a2[0], "y1": a2[1],
                         "z_bottom": z_bot, "z_top": z_top})
        # Parapets around the opening on the upper level, enclosure walls below.
        ph = L.parapet_height
        sc.boxes += [
            _xy_box(rx0, rx1, a2[0] - 0.2, a2[0], z_top, z_top + ph, wall, "parapet", i),
            _xy_box(rx0, rx1, a2[1], a2[1] + 0.2, z_top, z_top + ph, wall, "parapet", i),
            _xy_box(rx0 - 0.2, rx0, a2[0], a2[1], z_top, z_top + ph, wall, "parapet", i),
            _xy_box(rx0, rx1, a2[0] - 0.2, a2[0], z_bot, L.ceiling_z(i + 1), wall, "wall", i + 1),
            _xy_box(rx0, rx1, a2[1], a2[1] + 0.2, z_bot, L.ceiling_z(i + 1), wall, "wall", i + 1),
        ]

    # Bays and painted separators (none next to the ramp corridor).
    lw = L.line_width
    for i in range(L.n_levels):
        z = L.level_z(i)
        for name, (y0, y1, ent) in L.rows().items():
            for g in range(L.groups):
                gx0 = g * L.group_span + L.column_size / 2
                gx1 = gx0 + L.bays_per_group * L.bay_width
                if L.has_ramp and name in ("C", "D") and gx1 > rx0 + 1e-6:
                    continue
                for b in range(L.bays_per_group + 1):
                    x = gx0 + b * L.bay_width
                    sc.lines.append({"level": i, "x0": x - lw / 2, "x1": x + lw / 2, "y0": y0, "y1": y1,
                                     "cls": "parking_line"})
                for b in range(L.bays_per_group):
                    x0, x1 = gx0 + b * L.bay_width, gx0 + (b + 1) * L.bay_width
                    if ent == "high":  # entrance at y1, looking -y into the bay
                        corners = [[x1, y1], [x0, y1], [x0, y0], [x1, y0]]
                    else:  # entrance at y0, looking +y into the bay
                        corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                    sc.slots.append({
                        "slot_id": f"L{i}-{name}-{g * L.bays_per_group + b + 1:03d}",
                        "level": i,
                        "row": name,
                        "corners": corners,
                        "z": z,
                        "entrance": ent,
                        "width": L.bay_width,
                        "depth": y1 - y0,
                        "attributes": L.slot_attributes(
                            f"L{i}-{name}-{g * L.bays_per_group + b + 1:03d}"),
                    })
    _add_arrows(sc)
    return sc


def _arrow_rects(tail: np.ndarray, direction: np.ndarray, L: Layout) -> list[list[float]]:
    """Axis-aligned rectangles of one arrow: the shaft, then the head as a
    staircase narrowing towards the tip."""
    normal = np.array([-direction[1], direction[0]])
    shaft = L.arrow_length - L.arrow_head_length

    def rect(s0: float, s1: float, half: float) -> list[float]:
        p0 = tail + direction * s0 - normal * half
        p1 = tail + direction * s1 + normal * half
        return [min(p0[0], p1[0]), max(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[1], p1[1])]

    rects = [rect(0.0, shaft, L.arrow_shaft / 2)]
    steps = 6
    step = L.arrow_head_length / steps
    for k in range(steps):
        rects.append(rect(shaft + k * step, shaft + (k + 1) * step, L.arrow_head_width / 2 * (1.0 - k / steps)))
    return rects


def _add_arrows(sc: GarageScene) -> None:
    """Arrows on the aisles (see the module docstring)."""
    L = sc.layout
    if not L.arrows:
        return
    rx0, _ = L.ramp_x()
    half_length = L.arrow_length / 2
    along_x = {name: (ya + yb) / 2 for name, (ya, yb) in L.aisles().items()}
    along_y = {"W": L.x_lo - L.cross_width / 2, "E": L.x_hi + L.cross_width / 2}
    d, a = L.bay_depth, L.aisle_width
    for i in range(L.n_levels):
        for name, centre in list(along_x.items()) + list(along_y.items()):
            if name in along_x:
                stations = [g * L.group_span + L.group_span / 2 for g in range(L.groups)]
                if name == "2" and L.has_ramp:  # Clear of the ramp corridor.
                    stations = [s for s in stations if s + half_length < rx0 - 1.0]
                axis = np.array([1.0, 0.0])
            else:
                stations = [d / 2, 2 * d + a, 4 * d + 2 * a - d / 2]
                axis = np.array([0.0, 1.0])
            right = np.array([axis[1], -axis[0]])  # Right of the +axis heading.
            key = f"L{i}-{name}"
            one_way = [w for w in L.one_way if w[:-1] == key]
            if one_way:
                lanes = [(0.0, 1.0 if one_way[0][-1] == "+" else -1.0)]
            else:  # Right-hand traffic: each direction keeps to its right.
                lanes = [(1.5, 1.0), (-1.5, -1.0)]  # (offset along `right`, sign)
            for k, station in enumerate(stations):
                for offset, sign in lanes:
                    middle = np.array([station, centre] if name in along_x else [centre, station])
                    middle = middle + right * offset
                    direction = axis * sign
                    tail = middle - direction * half_length
                    sc.arrows.append({
                        "id": f"{key}-{k}{'+' if sign > 0 else '-'}",
                        "level": i,
                        "aisle": name,
                        "tail": tail.tolist(),
                        "tip": (tail + direction * L.arrow_length).tolist(),
                        "direction": direction.tolist(),
                        "one_way": bool(one_way),
                        "rects": _arrow_rects(tail, direction, L),
                    })


def parked_cars(scene: GarageScene, trip_index: int, seed: int, occupancy: float = 0.45,
                persistent_fraction: float = 0.5) -> list[Box]:
    """Cars in bays: a persistent subset shared by all trips plus trip-specific ones."""
    L = scene.layout
    shared = np.random.default_rng(seed + 1000)
    trip = np.random.default_rng(seed + 17 * (trip_index + 1))
    cars: list[Box] = []
    inst = 1000
    vehicle = cid("vehicle")
    for slot in scene.slots:
        persistent = shared.random() < occupancy * persistent_fraction
        shared_dims = shared.uniform([4.2, 1.75, 1.40], [4.9, 1.95, 1.70])
        shared_jitter = shared.uniform(-0.12, 0.12, size=2)
        transient = trip.random() < occupancy * (1.0 - persistent_fraction)
        trip_dims = trip.uniform([4.2, 1.75, 1.40], [4.9, 1.95, 1.70])
        trip_jitter = trip.uniform(-0.12, 0.12, size=2)
        if not (persistent or transient):
            continue
        dims, jitter = (shared_dims, shared_jitter) if persistent else (trip_dims, trip_jitter)
        c = np.asarray(slot["corners"], float)
        xc = c[:, 0].mean() + jitter[0]
        y0, y1 = c[:, 1].min(), c[:, 1].max()
        length, width, height = dims
        if slot["entrance"] == "high":  # back at y0
            yc = y0 + 0.35 + length / 2
        else:
            yc = y1 - 0.35 - length / 2
        yc += jitter[1] * 0.3
        R = rot_z(np.pi / 2)
        z = slot["z"]
        body_h = 0.55 * height
        key = slot["slot_id"] if persistent else f"{slot['slot_id']}-t{trip_index}"
        inst += 1
        body = _box([xc, yc, z + 0.15 + body_h / 2], [length, width, body_h], vehicle, "car", slot["level"], R, inst)
        cabin = _box([xc, yc, z + 0.15 + body_h + (height - 0.15 - body_h) / 2],
                     [0.55 * length, 0.92 * width, height - 0.15 - body_h], vehicle, "car", slot["level"], R, inst)
        body.key = cabin.key = key
        cars += [body, cabin]
    return cars


@dataclass
class MovingCar:
    """Constant-speed car along a straight lane during [t0, t1]."""

    start: np.ndarray
    velocity: np.ndarray
    t0: float
    t1: float
    size: np.ndarray
    level: int
    instance: int

    def box_at(self, t: float) -> Box | None:
        if t < self.t0 or t > self.t1:
            return None
        c = self.start + self.velocity * (t - self.t0)
        yaw = float(np.arctan2(self.velocity[1], self.velocity[0]))
        return _box(c, self.size, cid("vehicle_moving"), "moving_car", self.level, rot_z(yaw), self.instance)


def moving_cars(scene: GarageScene, trip_index: int, seed: int) -> list[MovingCar]:
    L = scene.layout
    rng = np.random.default_rng(seed + 333 + trip_index)
    a1 = L.aisles()["1"]
    y = (a1[0] + a1[1]) / 2 + 1.5
    x_start = L.x_hi + L.cross_width / 2
    speed = rng.uniform(1.2, 1.8)
    t0 = rng.uniform(2.0, 6.0)
    length = (L.x_hi - L.x_lo) + L.cross_width
    h = 1.5
    return [MovingCar(np.array([x_start, y, L.level_z(0) + h / 2]), np.array([-speed, 0.0, 0.0]), t0,
                      t0 + length / speed, np.array([4.5, 1.85, h]), 0, 900 + trip_index)]


def apply_change(scene: GarageScene, change: dict) -> GarageScene:
    """A copy of the scene with boxes added and removed (one trip's change:
    construction, maintenance).

    `change`: {"add": [{"center", "size", "level", "cls" (taxonomy name),
    "kind"}], "remove": [{"kind", "level", "near": [x, y]}]}: the box of that
    kind and level nearest to `near` is removed.
    """
    out = dataclasses.replace(scene, boxes=list(scene.boxes))
    for spec in change.get("remove", []):
        candidates = [k for k, b in enumerate(out.boxes)
                      if b.kind == spec["kind"] and b.level == spec["level"]]
        if not candidates:
            raise ValueError(f"no {spec['kind']} box on level {spec['level']}")
        near = np.asarray(spec["near"], float)
        k = min(candidates, key=lambda i: float(
            np.linalg.norm(out.boxes[i].center[:2] - near)))
        del out.boxes[k]
    for spec in change.get("add", []):
        out.boxes.append(_box(spec["center"], spec["size"],
                              cid(spec.get("cls", "other")),
                              spec.get("kind", "wall"), int(spec["level"])))
    return out
