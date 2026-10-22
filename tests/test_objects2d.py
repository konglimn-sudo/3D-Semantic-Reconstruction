"""Objects of a 2D detector: class images and the merge into the rule labels."""

import numpy as np

from garage_recon.models import interfaces
from garage_recon.semantics import objects2d, taxonomy


def test_queries_map_prompts_to_classes_and_absorbing_queries_to_minus_one():
    texts, ids = objects2d.queries({"light": ["a ceiling light"],
                                    "sign": ["a sign", "an exit sign"],
                                    "_ignore": ["a car"]})
    assert texts == ["a ceiling light", "a sign", "an exit sign", "a car"]
    assert ids.tolist() == [taxonomy.cid("light"), taxonomy.cid("sign"),
                            taxonomy.cid("sign"), -1]


def test_label_image_paints_the_best_detection_last_and_drops_absorbing_ones():
    query_class = np.array([taxonomy.cid("light"), taxonomy.cid("sign"), -1])
    found = [interfaces.Detection(np.array([0, 0, 6, 6.0]), 0, 0.3),
             interfaces.Detection(np.array([3, 3, 9, 9.0]), 1, 0.6),
             interfaces.Detection(np.array([0, 0, 10, 10.0]), 2, 0.9)]
    labels, scores = objects2d.label_image((10, 10), found, None, query_class)
    assert labels[1, 1] == taxonomy.cid("light") and scores[1, 1] == np.float32(0.3)
    assert labels[4, 4] == taxonomy.cid("sign")  # Overlap: the better score.
    assert labels[9, 9] == -1  # Only the absorbing query covers it.
    masks = np.zeros((3, 10, 10), bool)
    masks[0, 0, 0] = True
    labels, _ = objects2d.label_image((10, 10), found, masks, query_class)
    assert (labels >= 0).sum() == 1  # Masks replace the boxes.


def test_merge_replaces_only_allowed_rule_classes_above_the_confidence():
    floor, wall, line = (taxonomy.cid(n) for n in ("floor", "wall", "parking_line"))
    rule = np.array([floor, wall, line, floor, 0], np.uint8)
    static = np.array([0, 1, 2, 3])
    lifted = np.array([taxonomy.cid("wheel_stopper"), taxonomy.cid("sign"),
                       taxonomy.cid("sign"), taxonomy.cid("light")])
    confidence = np.array([0.9, 0.8, 0.9, 0.2])
    labels, count = objects2d.merge(rule, static, lifted, confidence,
                                    ["floor", "wall"], 0.5)
    assert count == 2
    assert labels.tolist() == [taxonomy.cid("wheel_stopper"), taxonomy.cid("sign"),
                               line, floor, 0]


def test_geometry_rejects_implausible_heights_and_oversized_masks():
    bump, light = taxonomy.cid("speed_bump"), taxonomy.cid("light")
    classes = np.array([bump, bump, light, light, -1])
    height = np.array([0.05, 1.5, 2.6, 0.4, 9.0])
    ok = objects2d.plausible(classes, height, {"speed_bump": [-0.2, 0.25],
                                               "light": [1.8, 100.0]})
    assert ok.tolist() == [True, False, True, False, True]
    query_class = np.array([bump, light])
    found = [interfaces.Detection(np.zeros(4), 0, 0.5),
             interfaces.Detection(np.zeros(4), 1, 0.5)]
    masks = np.zeros((2, 10, 10), bool)
    masks[0, :5] = True  # Half the image: a floor, not a speed bump.
    masks[1, 0, 0] = True
    kept, kept_masks = objects2d.keep_detections(found, masks, query_class, (10, 10),
                                                 {"default": 0.08})
    assert [d.query for d in kept] == [1] and len(kept_masks) == 1
