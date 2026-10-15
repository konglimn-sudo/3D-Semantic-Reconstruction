"""Oracle models of the synthetic dataset and the model registry."""

import numpy as np
import pytest

from garage_recon.data.synthetic import ground_truth
from garage_recon.models import interfaces
from garage_recon.models import registry
from garage_recon.semantics import taxonomy


@pytest.fixture(scope="module")
def suite(tiny_run):
    return registry.build({"backend": "oracle"},
                          tiny_run["root"] / "data" / "gt")


def _frames(tiny_run, count=2, camera="fisheye_left"):
    data = tiny_run["recon"].datasets[0]
    stamps = data.camera_stamps_ns(camera)
    return [interfaces.Frame(data.trip_id, camera, int(stamps[4 + k]))
            for k in range(count)]


def test_registry_backends(tmp_path):
    assert registry.build({"backend": "auto"}).segmenter is None
    with pytest.raises(ValueError):
        registry.build({"backend": "oracle"})
    with pytest.raises(ValueError):
        registry.build({"backend": "sam9"}, tmp_path)


def test_segmenter_and_encoder(tiny_run, suite):
    frame = _frames(tiny_run, 1)[0]
    masks = suite.segmenter.segment(frame)
    ids = np.unique(masks[masks >= 0])
    assert len(ids) >= 5
    embeddings = suite.encoder.encode_masks(frame, masks)
    assert embeddings and all(abs(np.linalg.norm(e) - 1) < 1e-6
                              for e in embeddings.values())
    text = suite.encoder.encode_text(["concrete floor", "地面", "unicorn"])
    assert text[0] @ text[1] > 0.8  # Same class, two languages.
    assert abs(text[0] @ text[2]) < 0.5  # Outside the vocabulary.
    # Mask classes follow the ground truth up to the configured confusion.
    world = suite.encoder.world
    label = world.render(frame.trip_id, frame.camera, frame.stamp_ns)["label"]
    agree = [suite.encoder.mask_class(frame, masks, int(m))
             == np.bincount(label[masks == m]).argmax() for m in ids]
    assert np.mean(agree) > 0.7


def test_matcher_tracks_query_points(tiny_run, suite):
    frame_a, frame_b = _frames(tiny_run, 2)
    points = suite.matcher.query_points(frame_a)
    matches = suite.matcher.match(frame_a, frame_b, points)
    assert len(matches) > 20 and matches.index_a is not None
    assert np.allclose(matches.uv_a, points[matches.index_a])
    # Every match lands near the true projection of the query point.
    world = suite.matcher.world
    render = world.render(frame_a.trip_id, frame_a.camera, frame_a.stamp_ns)
    uv = matches.uv_a
    rays, _ = world.camera(frame_a.camera).make_model().unproject(uv)
    rows = world.row_poses(frame_a.trip_id, frame_a.camera, frame_a.stamp_ns)
    poses = rows[uv[:, 1].astype(int)]
    depth = render["depth"][uv[:, 1].astype(int), uv[:, 0].astype(int)]
    points = (np.einsum("nij,nj->ni", poses[:, :3, :3], rays * depth[:, None])
              + poses[:, :3, 3])
    truth = world.project(frame_b.trip_id, frame_b.camera, frame_b.stamp_ns,
                          points)
    assert np.median(np.linalg.norm(matches.uv_b - truth, axis=1)) < 1.0


def test_depth_features_and_ocr(tiny_run, suite):
    frame = _frames(tiny_run, 1)[0]
    depth = suite.depth.predict([frame], [np.eye(4)])[0]
    assert np.nanmedian(depth.depth) > 0.5
    features, stride = suite.features.features(frame)
    assert features.shape[-1] == 32 and stride == 8
    texts = suite.ocr.read(frame)
    assert all(t.text.startswith("L0-") for t in texts)


def test_point_truth_classes(tiny_run):
    world = ground_truth.SyntheticWorld(tiny_run["root"] / "data" / "gt")
    truth = world.point_truth(np.array([[5.0, 3.0, 0.0], [5.0, 3.0, 3.29]]))
    assert truth["cls"][0] in (taxonomy.cid("floor"),
                               taxonomy.cid("parking_line"))
    assert truth["cls"][1] == taxonomy.cid("ceiling")
