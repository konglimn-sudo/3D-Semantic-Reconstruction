"""Route A machinery: per-pixel 2D labels lifted onto map points by
multi-view voting. Ground-truth label renders of the synthetic garage serve
as a perfect 2D model, so the test isolates projection, occlusion and voting.
"""

import numpy as np

from garage_recon import geometry
from garage_recon.data.synthetic import ground_truth
from garage_recon.semantics import lifting
from garage_recon.semantics import taxonomy


def test_lift_ground_truth_labels(tiny_run):
    root, recon = tiny_run["root"], tiny_run["recon"]
    world = ground_truth.SyntheticWorld(root / "data" / "gt")
    pose_set = recon.pose_set()
    data = recon.datasets[0]
    point_map = tiny_run["sem"].point_map()  # Dynamic mask refined by objects.
    point_map = point_map.subset(~point_map.dynamic)
    views = []
    for name in data.rig.fisheye_cameras:
        model = pose_set.rig(0).cameras[name].make_model()
        for ns in data.camera_stamps_ns(name)[::4][:10]:
            label = world.render(data.trip_id, name, int(ns))["label"]
            label = np.where(label == taxonomy.cid("vehicle_moving"), -1,
                             label)
            camera_from_world = geometry.inv(
                pose_set.camera_exposure(0, name, int(ns)))
            views.append((model, camera_from_world, label.astype(np.int64),
                          None))
    labels, confidence, _ = lifting.lift_labels(
        point_map.xyz, views, taxonomy.NUM_CLASSES, spacing=0.05,
        max_range=12.0, min_views=2)
    lifted = labels >= 0
    assert lifted.sum() > 3000
    # Points lifted as floor lie on the floor; columns are near-vertical.
    floor = lifted & (labels == taxonomy.cid("floor"))
    assert np.nanmedian(np.abs(point_map.height[floor])) < 0.05
    column = lifted & (labels == taxonomy.cid("column"))
    if column.any():
        assert np.median(np.abs(point_map.normals[column][:, 2])) < 0.3
    assert np.median(confidence[lifted]) > 0.8
