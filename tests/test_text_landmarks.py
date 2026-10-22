"""OCR text in 3D: floor hits on sloped floors, landmarks, zones from signs."""

import types

import numpy as np

from garage_recon import _core
from garage_recon.nav import address
from garage_recon.semantics import numbers


class _Sloped:
    n = 1
    grade = 0.011

    def floor_z(self, xy, level):
        return self.grade * xy[:, 0]


def test_floor_hits_follow_a_sloped_floor():
    camera = _core.make_camera("pinhole", 101, 101, [50.0, 50.0, 50.0, 50.0])
    # Camera 1.5 m above the floor at x = 100, looking along +x and 20 deg down.
    pitch = np.radians(20.0)
    forward = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    down = -np.cross(forward, [0.0, 1.0, 0.0])  # Image y points down.
    right = np.cross(down, forward)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = [100.0, 0.0, 1.1 + 1.5]
    hits = numbers.floor_hits(camera, pose, np.array([50.0, 50.0]), _Sloped(), 20.0)
    assert len(hits) == 1
    level, xyz = hits[0]
    assert level == 0 and abs(xyz[2] - 0.011 * xyz[0]) < 0.01
    assert 100.0 < xyz[0] < 106.0


def test_text_kinds_and_landmarks():
    assert numbers.text_kind("EXIT") == "exit" and numbers.text_kind("2号门出口") == "exit"
    assert numbers.text_kind("3号门") == "gate" and numbers.text_kind("B2") == "level"
    assert numbers.text_kind("5A") == "zone" and numbers.text_kind("中央区") == "zone"
    assert numbers.text_kind("更多车位") == "text"
    readings = [("5A", [0, 0, 2], 0.9), ("5a", [0.5, 0, 2], 0.8),
                ("5A", [40, 0, 2], 0.9), ("EXIT", [10, 2, 2.5], 0.95),
                ("EXIT", [10.4, 2, 2.5], 0.9), ("T", [1, 1, 1], 0.99)]
    marks = numbers.text_landmarks([(t, np.array(x, float), s) for t, x, s in readings],
                                   radius=3.0, min_views=2, min_score=1.0)
    assert sorted((m["text"], m["views"]) for m in marks) == [("5A", 2), ("EXIT", 2)]
    assert {m["kind"] for m in marks} == {"zone", "exit"}


def test_slots_without_a_zone_take_the_nearest_zone_sign_of_their_level():
    cfg = types.SimpleNamespace(nav={"zone_sign_radius": 30.0})
    goals = [{"park": {"xyz": [0, 0, 0]}, "level": 0},
             {"park": {"xyz": [100, 0, 0]}, "level": 0},
             {"park": {"xyz": [0, 0, 0]}, "level": 0, "zone": "C"}]
    signs = [{"text": "5A", "kind": "zone", "xyz": [5, 0, 2], "level": 0},
             {"text": "6B", "kind": "zone", "xyz": [2, 0, 2], "level": 1},
             {"text": "EXIT", "kind": "exit", "xyz": [1, 0, 2], "level": 0}]
    address.assign_zones(goals, cfg, signs)
    assert goals[0]["zone"] == "5A" and goals[0]["zone_source"] == "sign"
    assert "zone" not in goals[1]  # Beyond the radius.
    assert goals[2]["zone"] == "C" and goals[2]["zone_source"] == "ocr"


def test_arrow_glyphs_are_stripped_and_one_sign_is_one_landmark():
    assert numbers.normalize_text(" 出口个") == "出口" and numbers.normalize_text("《中央区") == "中央区"
    assert numbers.normalize_text("5A》") == "5A" and numbers.normalize_text("2号门出口") == "2号门出口"
    readings = [("EXIT", [0, 0, 3], 0.9), ("EXIT", [0.1, 0, 3], 0.9), ("出口个", [0.2, 0, 2.9], 0.8),
                ("出口十", [0.1, 0, 2.9], 0.7), ("中央区", [20, 0, 3], 0.9), ("《中央区", [20.2, 0, 3], 0.9),
                ("中区", [20.1, 0, 3], 0.6), ("中区", [20.1, 0, 3], 0.6),
                ("5A", [40, 0, 3], 0.9), ("5A", [40, 0, 3], 0.9), ("5B", [40.5, 0, 3], 0.9),
                ("5B", [40.5, 0, 3], 0.9)]
    marks = numbers.text_landmarks([(t, np.array(x, float), s) for t, x, s in readings],
                                   radius=3.0, min_views=2, min_score=1.0)
    by_text = {m["text"]: m for m in marks}
    assert sorted(by_text) == ["5A", "5B", "EXIT", "中央区"]  # 5A / 5B differ: two signs.
    assert by_text["EXIT"]["texts"] == ["EXIT", "出口"] and by_text["EXIT"]["views"] == 4
    assert by_text["中央区"]["texts"] == ["中央区", "中区"] and by_text["中央区"]["views"] == 4
