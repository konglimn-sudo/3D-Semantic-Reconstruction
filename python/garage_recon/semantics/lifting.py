"""2D -> 3D label lifting by multi-view voting (section 5, route A).

Any per-pixel 2D result (SAM 3 masks, a fine-tuned segmenter, OCR boxes
rasterised as labels...) is lifted onto map points by the C++ `LabelLifter`:
each view renders a z-buffer of the points, the winning point of every pixel
receives that pixel's label, pixels on label boundaries are skipped, and
votes are accumulated over views. Points need at least `min_views` views.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .. import _core


def lift_labels(points: np.ndarray, views: Iterable[tuple], num_classes: int, spacing: float = 0.05,
                max_range: float = 15.0, boundary: int = 1, min_views: int = 2):
    """views: iterable of (camera_model, T_cam_world, label_image[H,W] int (-1 = none), weight_image or None).

    Returns (labels (N,), confidence (N,), votes (N, C)); unlabelled points get -1.
    """
    lifter = _core.LabelLifter(points, num_classes, spacing, max_range, boundary, min_views)
    for cam, T_cw, lab, weight in views:
        lifter.add_view(cam, T_cw, np.asarray(lab, np.int32), weight)
    return lifter.result()
