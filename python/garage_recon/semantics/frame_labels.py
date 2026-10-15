"""Per-frame labels by projection (section 5, module "frame labels").

The 3D map is labelled once and every camera frame inherits consistent
labels. For each image the C++ FrameLabeler renders a z-buffer of the
labelled static map plus this frame's object boxes (semantic label image and
occlusion test), samples parking slots, markings and boxes densely in 3D,
projects them with the camera model and the image's exposure poses (rolling
shutter, time offset, all from the pose set) and keeps what is visible. A
projection-consistency score checks that projected marking edges land on
image edges; low scores reveal pose, extrinsic or time-sync errors and are
routed to review. This module builds the labeller and writes the records.
"""

from __future__ import annotations

import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon.semantics import markings as markings_lib
from garage_recon.semantics import objects4d
from garage_recon.semantics import slots as slots_lib


def frame_labeler(static_xyz: np.ndarray, static_labels: np.ndarray,
                  slots: list[slots_lib.Slot],
                  markings: list[markings_lib.MarkingSegment],
                  cfg) -> _core.FrameLabeler:
    """FrameLabeler of the labelled static map, slots and markings."""
    return _core.FrameLabeler(
        static_xyz, static_labels.astype(np.int32),
        [(np.asarray(s.corners, float), float(s.z)) for s in slots],
        [(np.asarray(m.p0, float), np.asarray(m.p1, float), float(m.width))
         for m in markings],
        core_config.frame_label_config(cfg))


def _polylines(lines) -> list[list[list[float]]]:
    return [np.round(line, 2).tolist() for line in lines]


def label_camera_frame(labeler: _core.FrameLabeler, calib, camera,
                       world_from_camera: np.ndarray,
                       image: np.ndarray | None,
                       slots: list[slots_lib.Slot],
                       markings: list[markings_lib.MarkingSegment],
                       boxes: list[_core.ObjectBox]) -> tuple[dict, np.ndarray]:
    """Labels one image.

    Args:
        labeler: The FrameLabeler of the static map.
        calib: The camera's calibration.
        camera: Its C++ model.
        world_from_camera: (4, 4), or (2, 4, 4) for the first and last row.
        image: The RGB image (for the consistency score) or None.
        slots: Slots referred to by index in the record.
        markings: Markings referred to by index in the record.
        boxes: Object boxes of the frame.

    Returns:
        (record, label image (H, W) uint8).
    """
    raw, label_image = labeler.label(camera, world_from_camera, boxes, image)
    record: dict = {"camera": calib.name, "slots": [], "markings": [],
                    "boxes": []}
    for r in raw["slots"]:
        slot = slots[r["index"]]
        record["slots"].append({
            "slot_id": slot.slot_id,
            "corners_uv": np.round(r["corners_uv"], 2).tolist(),
            "corner_visible": r["corner_visible"],
            "entrance_visible": r["entrance_visible"],
            "visible_fraction": r["visible_fraction"],
            "polylines": _polylines(r["polylines"]),
            "occupied": slot.occupied,
            "inferred": slot.inferred,
        })
    for r in raw["markings"]:
        marking = markings[r["index"]]
        record["markings"].append({"index": r["index"], "cls": marking.cls,
                                   "width": marking.width,
                                   "polylines": _polylines(r["polylines"])})
    for r in raw["boxes"]:
        box = objects4d.Box3D.from_core(boxes[r["index"]])
        record["boxes"].append({"track_id": box.track_id, "cls": box.cls,
                                "static": box.static, "bbox": r["bbox"],
                                "visible_fraction": r["visible_fraction"]})
    record["consistency_score"] = raw["consistency_score"]
    return record, label_image
