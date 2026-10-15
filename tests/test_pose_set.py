"""The pose set: one source of poses with content ids, and provenance."""

import dataclasses

import numpy as np

from garage_recon import geometry
from garage_recon import provenance
from garage_recon.recon import pose_set as pose_set_lib


def test_round_trip_and_ids(tiny_run, tmp_path):
    recon = tiny_run["recon"]
    pose_set = recon.pose_set()
    pose_set.save(tmp_path)
    loaded = pose_set_lib.PoseSet.load(tmp_path, recon.datasets)
    assert loaded.pose_set_id == pose_set.pose_set_id
    assert loaded.trajectory_id == pose_set.trajectory_id
    # A camera change keeps the trajectory id and changes the pose set id.
    calib = pose_set.rig(0).cameras["fisheye_front"]
    moved = dataclasses.replace(calib, time_offset=calib.time_offset + 0.01)
    changed = pose_set.with_cameras({"fisheye_front": moved})
    assert changed.trajectory_id == pose_set.trajectory_id
    assert changed.pose_set_id != pose_set.pose_set_id
    provenance.record(tmp_path, "test.lidar", pose_set, uses_cameras=False)
    provenance.record(tmp_path, "test.camera", pose_set, uses_cameras=True)
    assert provenance.stale_stages(tmp_path, changed) == ["test.camera"]


def test_knots_and_exposures(tiny_run):
    recon = tiny_run["recon"]
    pose_set = recon.pose_set()
    trip = pose_set.trips[0]
    poses = pose_set.world_from_vehicle(0, trip.knot_stamps[::7])
    error, _ = geometry.pose_error(poses, trip.knot_poses[::7])
    assert error.max() < 1e-9  # The trajectory passes through its knots.
    data = recon.datasets[0]
    name = data.rig.fisheye_cameras[0]
    ns = int(data.camera_stamps_ns(name)[3])
    exposure = pose_set.camera_exposure(0, name, ns)
    first, last = pose_set.row_times(0, name, ns)
    calib = pose_set.rig(0).cameras[name]
    assert np.isclose(last - first,
                      (calib.height - 1) * calib.rolling_shutter_line_time)
    expected = pose_set.world_from_vehicle(0, [first, last]) @ (
        calib.T_vehicle_cam)
    assert np.allclose(exposure, expected)
