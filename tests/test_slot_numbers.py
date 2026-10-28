"""Slot numbers read from rectified floor patches (semantics/numbers.py)."""

import types

import numpy as np

from garage_recon.data.synthetic import sensors
from garage_recon.eval import synthetic_eval
from garage_recon.models import interfaces
from garage_recon.semantics import numbers


def test_split_text_keeps_letter_digit_zones():
    assert numbers.split_text("B2-017") == ("B2", "017")
    assert numbers.split_text("C105") == ("C", "105")
    assert numbers.split_text("-001") == (None, "001")


def test_slot_number_pattern_rejects_upside_down_readings():
    for text in ("A-001", "B2-017", "105", "C 105"):
        assert numbers.is_slot_number(text), text
    for text in ("100-V", "出口", "A", "A-0O1"):
        assert not numbers.is_slot_number(text), text


def _reading(text, left, right, top, bottom, score=0.9):
    polygon = np.array([[left, top], [right, top], [right, bottom],
                        [left, bottom]], np.float32)
    return interfaces.TextResult(text, polygon, score)


def test_join_line_joins_a_reading_split_at_the_hyphen():
    lines = numbers.join_line([_reading("-001", 50, 100, 10, 30, 0.8),
                               _reading("A", 20, 45, 12, 31, 1.0),
                               _reading("EV", 30, 60, 60, 80)])
    texts = [text for text, _, _ in lines]
    assert texts == ["A-001", "EV"]
    assert np.isclose(lines[0][2], 0.9)


def test_slot_patch_is_upright_for_a_driver_in_the_aisle():
    # Entrance at x = 0, back at x = 5: the driver faces +x, +y is left.
    slot = types.SimpleNamespace(
        corners=[[0.0, 2.5], [0.0, 0.0], [5.0, 0.0], [5.0, 2.5]])
    res, x0, y0 = 0.1, -1.0, -1.0
    xs = x0 + (np.arange(80) + 0.5) * res
    ys = y0 + (np.arange(50) + 0.5) * res
    x, y = np.meshgrid(xs, ys)
    texture = np.full((50, 80, 3), 100, np.uint8)
    texture[x < 0.5] = (255, 0, 0)  # Entrance: red.
    texture[x > 4.5] = (0, 0, 255)  # Back: blue.
    texture[(y > 2.0) & (x > 1.0) & (x < 4.0)] = (0, 255, 0)  # Left: green.
    patch = numbers.slot_patch(texture, np.ones((50, 80), bool), x0, y0,
                               res, slot, margin=0.2)
    h, w = patch.shape[:2]
    assert (h, w) == (54, 29)
    assert tuple(patch[-3, w // 2]) == (255, 0, 0)
    assert tuple(patch[2, w // 2]) == (0, 0, 255)
    assert tuple(patch[h // 2, 2]) == (0, 255, 0)
    assert tuple(patch[h // 2, w - 3]) == (100, 100, 100)


def test_slot_patch_fills_unobserved_floor():
    slot = types.SimpleNamespace(
        corners=[[0.0, 2.5], [0.0, 0.0], [5.0, 0.0], [5.0, 2.5]])
    texture = np.full((50, 80, 3), 90, np.uint8)
    observed = np.zeros((50, 80), bool)
    observed[:, :30] = True  # Only the entrance half was seen.
    texture[~observed] = 0
    patch = numbers.slot_patch(texture, observed, -1.0, -1.0, 0.1, slot,
                               margin=0.2)
    assert (patch == 90).all()


def test_floor_text_reads_upright_along_its_up_vector():
    # "T": its bar is at the top of the character.
    raster = sensors.FloorTextRaster([{
        "level": 0, "text": "T", "center": [0.0, 0.0], "right": [1.0, 0.0],
        "up": [0.0, 1.0], "height": 0.4}])
    x0, y0, mask = raster.levels[0]
    j, i = np.nonzero(mask)
    x, y = x0 + i * raster.res, y0 + j * raster.res
    assert abs(np.median(x)) < 0.05 and abs(np.median(y)) < 0.15
    top, bottom = y > 0.1, y < -0.1
    assert np.ptp(x[top]) > 2.0 * np.ptp(x[bottom])
    points = np.array([[0.0, 0.18, 0.0], [0.0, -0.15, 0.0], [1.0, 0.0, 0.0]])
    assert raster.painted(points, np.zeros(3)).tolist() == [True, True, False]


def test_eval_slots_scores_the_numbers_read():
    def slot(slot_id, x):
        return {"slot_id": slot_id, "level": 0,
                "corners": [[x, 0.0], [x + 2.5, 0.0], [x + 2.5, 5.0],
                            [x, 5.0]]}

    gt = [slot("L0-A-001", 0.0), slot("L0-A-002", 2.5), slot("L0-A-003", 5.0)]
    pred = [dict(slot("L0-R01-001", 0.0), zone="A", number="001"),
            dict(slot("L0-R01-002", 2.5), zone=None, number="002"),
            dict(slot("L0-R01-003", 5.0), zone=None, number=None)]
    result = synthetic_eval.eval_slots(pred, gt)
    assert result["numbers"] == {"read": 2, "correct": 1, "number_only": 1,
                                 "unread": 1}
