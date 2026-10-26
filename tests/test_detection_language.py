"""Detector instances as the language route's masks and class embeddings."""

import numpy as np

from garage_recon.models import detection_language, interfaces
from garage_recon.semantics import language_field, taxonomy


class _Detector:
    def detect(self, frame, queries):
        light, car = queries.index("a ceiling light"), queries.index("a car")
        return [interfaces.Detection(np.array([0, 0, 4, 4.0]), light, 0.6),
                interfaces.Detection(np.array([5, 5, 9, 9.0]), car, 0.9)]


def test_instances_carry_their_class_and_prompts_map_to_it():
    adapter = detection_language.DetectionLanguage(
        _Detector(), None, {"light": ["a ceiling light"], "_ignore": ["a car"]},
        {"default": 1.0})
    frame = interfaces.Frame("t", "cam", 1, np.zeros((10, 10, 3), np.uint8))
    masks = adapter.segment(frame)
    assert set(np.unique(masks)) == {-1, 0}  # The car only competed for boxes.
    emb = adapter.encode_masks(frame, masks)
    assert list(emb) == [0] and emb[0].argmax() == taxonomy.cid("light")
    ids, class_emb = language_field.class_text_embeddings(adapter)
    best, score = language_field.classify(np.stack([emb[0]]), ids, class_emb, 0.3)
    assert best[0] == taxonomy.cid("light") and score[0] > 0.99


def test_instances_and_their_text():
    from garage_recon.semantics import instances
    xyz = np.array([[0, 0, 2], [0.2, 0, 2.1], [0.1, 0.1, 2], [0.1, 0, 2.2], [0, 0.1, 2.1],
                    [5, 5, 0], [5.1, 5, 0]], float)
    identity = np.array([3, 3, 3, 3, 3, 7, 7])
    objs = instances.instances(xyz, identity, np.array([3, 7]),
                               np.array([taxonomy.cid("sign"), taxonomy.cid("light")]),
                               np.array([0.9, 0.8]), min_points=2)
    assert [(o["id"], o["class"], o["points"]) for o in objs] == [(3, "sign", 5), (7, "light", 2)]
    marks = [{"text": "5A", "xyz": [0.1, 0.05, 2.05]}, {"text": "EXIT", "xyz": [9, 9, 9]}]
    assert instances.attach_text(objs, marks) == 1
    assert objs[0]["text"] == ["5A"] and marks[0]["instance"] == 3 and "instance" not in marks[1]


def test_split_identities_of_one_object_merge_into_one_instance():
    from garage_recon.semantics import instances
    rng = np.random.default_rng(0)
    tube = lambda x0: np.c_[rng.uniform(x0, x0 + 1.2, 40), rng.uniform(0, 0.2, 40),  # noqa: E731
                            rng.uniform(4, 4.1, 40)]
    # Light A seen as identities 1 and 2 (overlapping halves), light B (3)
    # 3 m along the row, and a sign (4) overlapping light A.
    xyz = np.concatenate([tube(0)[:, :] * [0.6, 1, 1], tube(0) * [0.6, 1, 1] + [0.5, 0, 0],
                          tube(3), tube(0)])
    identity = np.repeat([1, 2, 3, 4], 40)
    light, sign = taxonomy.cid("light"), taxonomy.cid("sign")
    objs = instances.instances(xyz, identity, np.array([1, 2, 3, 4]),
                               np.array([light, light, light, sign]),
                               np.array([0.9, 0.7, 0.8, 0.9]))
    assert [(o["class"], o["identities"]) for o in objs] == [
        ("light", [1, 2]), ("light", [3]), ("sign", [4])]
    assert objs[0]["points"] == 80
    # The extent cap stops a row of touching lights from chaining.
    row = np.concatenate([tube(0.5 * k) for k in range(10)])
    args = (row, np.repeat(np.arange(10), 40), np.arange(10), np.full(10, light),
            np.full(10, 0.9))
    assert len(instances.instances(*args, max_extent=100.0)) == 1
    objs = instances.instances(*args, max_extent=3.0)
    assert len(objs) >= 3 and all(o["max"][0] - o["min"][0] <= 3.0 for o in objs)
