"""Garage label taxonomy shared by point clouds, occupancy, rasters, exports.

Class ids are uint8 and stable; LAS exports use 64 + id because LAS 1.4
reserves classification codes 0-63 for ASPRS classes. PROMPTS holds the text
queries (English and Chinese) with which the language route recognises each
class; the open vocabulary can add more classes later.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class SemanticClass:
    id: int
    name: str
    group: str
    color: tuple[int, int, int]


UNLABELED = 0  # Also "free" in occupancy grids.

CLASSES: tuple[SemanticClass, ...] = (
    SemanticClass(0, "unlabeled", "none", (0, 0, 0)),
    SemanticClass(1, "floor", "structure", (128, 128, 128)),
    SemanticClass(2, "ramp", "structure", (160, 120, 90)),
    SemanticClass(3, "ceiling", "structure", (70, 70, 90)),
    SemanticClass(4, "wall", "structure", (220, 200, 160)),
    SemanticClass(5, "column", "structure", (250, 220, 40)),
    SemanticClass(6, "beam", "structure", (190, 170, 120)),
    SemanticClass(7, "parking_line", "marking", (255, 255, 255)),
    SemanticClass(8, "lane_marking", "marking", (255, 240, 120)),
    SemanticClass(9, "arrow", "marking", (120, 255, 255)),
    SemanticClass(10, "speed_bump", "traffic", (255, 140, 0)),
    SemanticClass(11, "wheel_stopper", "traffic", (255, 90, 90)),
    SemanticClass(12, "vehicle", "object", (0, 120, 255)),
    SemanticClass(13, "vehicle_moving", "object", (255, 0, 120)),
    SemanticClass(14, "pedestrian", "object", (255, 0, 0)),
    SemanticClass(15, "two_wheeler", "object", (180, 0, 255)),
    SemanticClass(16, "mep", "mep", (0, 200, 120)),
    SemanticClass(17, "sign", "mep", (0, 255, 0)),
    SemanticClass(18, "light", "mep", (255, 255, 180)),
    SemanticClass(19, "barrier", "traffic", (255, 60, 200)),
    SemanticClass(20, "other", "object", (90, 90, 255)),
)

BY_ID = {c.id: c for c in CLASSES}
BY_NAME = {c.name: c for c in CLASSES}
NUM_CLASSES = len(CLASSES)

STRUCTURE_IDS = frozenset(
    c.id for c in CLASSES if c.group == "structure") | {BY_NAME["mep"].id}
OBJECT_IDS = frozenset(c.id for c in CLASSES if c.group == "object")
MARKING_IDS = frozenset(c.id for c in CLASSES if c.group == "marking")

# Text queries of the language route. Moving and parked vehicles look alike;
# motion is decided geometrically, so only "vehicle" is queried.
PROMPTS: dict[str, tuple[str, ...]] = {
    "floor": ("concrete floor", "garage floor", "地面"),
    "ramp": ("ramp", "sloped driveway", "坡道"),
    "ceiling": ("ceiling", "concrete ceiling", "顶板"),
    "wall": ("wall", "concrete wall", "墙"),
    "column": ("column", "pillar", "柱子"),
    "beam": ("beam", "ceiling beam", "梁"),
    "parking_line": ("parking space line", "white painted line", "车位线"),
    "lane_marking": ("lane marking", "车道线"),
    "arrow": ("arrow marking on the floor", "地面箭头"),
    "speed_bump": ("speed bump", "减速带"),
    "wheel_stopper": ("wheel stopper", "挡轮器"),
    "vehicle": ("car", "parked car", "汽车"),
    "pedestrian": ("person", "pedestrian", "行人"),
    "two_wheeler": ("bicycle", "motorcycle", "两轮车"),
    "mep": ("duct", "pipe", "管道"),
    "sign": ("sign", "exit sign", "标志牌"),
    "light": ("ceiling light", "lamp", "灯"),
    "barrier": ("barrier gate", "道闸"),
}


def cid(name: str) -> int:
    """Class id by name (raises KeyError for unknown names)."""
    return BY_NAME[name].id


def las_code(class_id: int) -> int:
    return 64 + int(class_id)


def palette() -> np.ndarray:
    """(256, 3) uint8 colour table indexed by class id."""
    table = np.zeros((256, 3), np.uint8)
    for semantic_class in CLASSES:
        table[semantic_class.id] = semantic_class.color
    return table
