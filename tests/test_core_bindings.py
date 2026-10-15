"""Python-side checks of the C++ core bindings."""

import numpy as np

from garage_recon import _core
from garage_recon.geometry import make_T, rot_z


def test_fisheye_project_unproject_roundtrip():
    cam = _core.make_camera("kannala_brandt", 400, 400, [120, 120, 200, 200, -0.01, 0.002, 0, 0, 1.7])
    rng = np.random.default_rng(0)
    d = rng.normal(size=(500, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[np.arccos(d[:, 2]) < 1.6]
    uv, valid, _ = cam.project(d * 5.0)
    rays, ok = cam.unproject(uv[valid])
    assert ok.all()
    assert np.max(np.linalg.norm(rays - d[valid], axis=1)) < 1e-6


def test_icp_point_to_plane_recovers_offset():
    rng = np.random.default_rng(1)
    floor = np.column_stack([rng.uniform(0, 10, 4000), rng.uniform(-5, 5, 4000), np.zeros(4000)])
    wall = np.column_stack([np.full(2000, 10.0), rng.uniform(-5, 5, 2000), rng.uniform(0, 3, 2000)])
    side = np.column_stack([rng.uniform(0, 10, 2000), np.full(2000, 5.0), rng.uniform(0, 3, 2000)])
    box = np.column_stack([rng.uniform(4, 5, 1000), rng.uniform(-1, 0, 1000), rng.uniform(0, 1, 1000)])
    world = np.vstack([floor, wall, side, box])
    m = _core.VoxelHashMap(0.3, 1e9, 40)
    m.add_points(world)
    T_true = make_T(rot_z(0.05), [0.3, -0.2, 0.05])
    src = (world - T_true[:3, 3]) @ T_true[:3, :3]  # points expressed in the moved frame
    r = _core.icp(src, m, np.eye(4), max_corr_dist=1.0, kernel=0.3, max_iters=60)
    assert np.linalg.norm(r["T"][:3, 3] - T_true[:3, 3]) < 0.02


def test_pose_graph_loop_and_gnss():
    g = _core.PoseGraph()
    poses = [make_T(rot_z(0.0), [float(i), 0.0, 0.0]) for i in range(5)]
    for i, P in enumerate(poses):
        Q = P.copy()
        Q[:3, 3] += [0.2 * i, 0.1, -0.05]  # perturbed initial guess
        g.add_node(Q)
    info = np.eye(6) * 100
    for i in range(4):
        g.add_between(i, i + 1, np.linalg.inv(poses[i]) @ poses[i + 1], info)
    g.add_position(0, np.zeros(3), np.eye(3) * 1e4)
    g.add_position(4, np.array([4.0, 0.0, 0.0]), np.eye(3) * 1e4)
    s = g.optimize()
    out = np.asarray(g.poses())
    assert s["final_cost"] < s["initial_cost"]
    assert np.allclose(out[:, :3, 3], np.asarray(poses)[:, :3, 3], atol=1e-3)


def test_occupancy_and_visibility_shapes():
    floor = np.column_stack([np.repeat(np.arange(-5, 5, 0.1), 100), np.tile(np.arange(-5, 5, 0.1), 100),
                             np.full(10000, -0.02)])
    sem, ml, mc = _core.build_occupancy(np.array([-5, -5, -0.5]), np.array([5, 5, 2.5]), 0.1, 0, 4, 10.0,
                                        np.eye(4), floor, np.ones(len(floor), np.int32),
                                        np.array([[3, 0, 0.75, 4, 1.8, 1.5, 0, 13]], float), np.array([0, 0, 1.9]),
                                        floor, [])
    assert sem.shape == (100, 100, 30)
    assert (sem == 13).sum() > 1000 and (sem == 1).sum() > 5000
    assert ml.any() and not mc.any()


def test_motion_prior_and_imu_deskew_bindings():
    t = np.arange(0.0, 5.0, 0.005)
    imu = np.column_stack([t, np.zeros(len(t)), np.zeros(len(t)), np.full(len(t), 0.5),
                           np.zeros(len(t)), np.full(len(t), 2.0), np.full(len(t), 9.81)])
    wheel = np.column_stack([np.arange(0.0, 5.0, 0.01), np.full(500, 4.0)])
    prior = _core.MotionPrior(_core.ImuSamples(imu), _core.WheelSamples(wheel))
    assert prior.valid and prior.delta(-1.0, 0.5) is None
    d = prior.delta(1.0, 2.0)
    assert abs(np.arctan2(d[1, 0], d[0, 0]) - 0.5) < 1e-9
    pts = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    out = _core.deskew_with_prior(pts, np.array([-0.1, 0.0]), 2.0, prior, np.eye(4))
    assert np.allclose(out[1], pts[1]) and not np.allclose(out[0], pts[0])
    g = _core.gravity_measurements(_core.ImuSamples(imu), _core.WheelSamples(wheel), np.eye(3), [1.0])
    assert g == []  # 0.5 rad/s exceeds the default yaw-rate gate


def test_odometry_pipeline_keyframes_with_external_poses():
    pipe = _core.OdometryPipeline(_core.OdometryConfig(), _core.KeyframeConfig(), np.eye(4))
    pts = np.array([[5.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    kfs = []
    for k in range(6):
        pose, is_kf, kp, ki = pipe.process(pts, np.array([0.5, 0.5]), None, 0.1 * k, k == 5,
                                           make_T(None, [0.6 * k, 0.0, 0.0]))
        if is_kf:
            kfs.append(k)
            assert kp.shape == (1, 3) and ki.dtype == np.float32
    assert kfs == [0, 2, 4, 5] and list(pipe.keyframes()) == kfs
    assert pipe.poses().shape == (6, 4, 4)


def test_loop_detection_and_graph_stage_bindings():
    rng = np.random.default_rng(3)
    # A corridor with irregular boxes. Drive 0 is anchored at its start; drive 1
    # has no anchor, a wrong start and drift, so inter-drive loops must place it.
    world = np.vstack([np.column_stack([rng.uniform(-5, 45, 30000), rng.uniform(-4, 4, 30000), np.zeros(30000)]),
                       np.column_stack([rng.uniform(-5, 45, 8000), np.full(8000, 4.0), rng.uniform(0, 3, 8000)]),
                       np.column_stack([rng.uniform(-5, 45, 8000), np.full(8000, -4.0), rng.uniform(0, 3, 8000)])]
                      + [np.column_stack([rng.uniform(x, x + 0.8, 600), rng.uniform(1.5, 3.5, 600), rng.uniform(0, 2, 600)])
                         for x in range(0, 40, 7)])
    drives = [np.arange(0.0, 20.0, 1.0), np.arange(0.5, 15.0, 1.0)]
    start_error = make_T(rot_z(0.05), [0.6, -0.4, 0.0])
    stage = _core.GraphStageConfig()
    kfs = _core.KeyframeSet(list(stage.required_voxels()))
    true, trips = [], []
    for d, xs in enumerate(drives):
        odo = []
        for k, x in enumerate(xs):
            T = make_T(None, [x, 0.0, 1.8])
            local = world[np.linalg.norm(world[:, :2] - T[:2, 3], axis=1) < 15.0] - T[:3, 3]
            drift = make_T(rot_z(0.0005 * k), [0.003 * k, 0.0, 0.0])  # 0.3 % / 0.03 deg per metre
            O = start_error @ drift @ T if d == 1 else T  # drive 0: drift-free (the reference)
            kfs.add(d, k, 100.0 * d + k, O, local)
            true.append(T)
            odo.append(O)
        ti = _core.TripInputs()
        if d == 0:
            ti.initial_pose = true[0]  # INS fix at the entrance (vehicle = lidar frame here)
        ti.frame_stamps = list(100.0 * d + np.arange(len(xs), dtype=float))
        ti.frame_odometry = np.asarray(odo)
        trips.append(ti)
    true = np.asarray(true)
    poses = kfs.odom_poses()
    cfg = _core.LoopConfig()
    loops, stats = _core.detect_loops(kfs, poses, cfg)
    assert len(loops) >= 5 and stats["tested"] >= len(loops)
    for lc in loops:
        ref = np.linalg.inv(true[lc.j]) @ true[lc.i]
        assert np.linalg.norm(lc.T_ji[:3, 3] - ref[:3, 3]) < 0.05
    assert len(_core.pcm_filter(loops, kfs, poses, cfg)) == len(loops)
    lines = []
    res = _core.run_graph_stage(kfs, trips, stage, lines.append)
    assert res["poses"].shape == (len(true), 4, 4) and len(res["loops"]) >= 5
    assert [a.type for a in res["anchors"]] == ["pose", "pose"]  # INS for drive 0, weak prior for drive 1
    assert any(line.startswith("[loops]") for line in lines)
    assert len(res["ba_rounds"]) == stage.ba.rounds  # LiDAR bundle adjustment.
    second = kfs.indices_of_trip(1)
    before = np.linalg.norm(poses[second, :3, 3] - true[second, :3, 3], axis=1)
    after = np.linalg.norm(res["poses"][second, :3, 3] - true[second, :3, 3], axis=1)
    assert before.min() > 0.4 and after.max() < 0.1


def test_map_dynamic_occupancy_and_labeler_bindings():
    nearest_only = _core.MapFusionConfig()
    nearest_only.fuse = False
    acc = _core.MapAccumulator(0.1, 20.0, nearest_only)
    acc.add(0, make_T(None, [-10.0, 0, 0]), np.array([[10.01, 0.02, 0.0]]), np.array([0.1]))
    acc.add(1, make_T(None, [-2.0, 0, 0]), np.array([[2.03, 0.01, 0.0]]), np.array([0.9]))
    result = acc.result()
    xyz, inten, kf = result["xyz"], result["intensity"], result["keyframe"]
    assert len(xyz) == 1 and kf[0] == 1 and abs(inten[0] - 0.9) < 1e-6
    det = _core.DynamicDetector(xyz, np.array([[1.0, 0, 0]]), np.array([0], np.int32), _core.DynamicConfig())
    det.add_ego_path(0, np.array([[0.0, 0.0, -0.5]]))
    assert list(det.classify()) == [1]  # inside the ego vehicle's swept volume
    box = _core.ObjectBox(track_id=3, label=13, center=np.array([4.0, 0, 0.75]), size=np.array([4.6, 1.9, 1.5]))
    tl = _core.BoxTimeline()
    tl.add_moving(0, 1.0, box)
    assert [b.track_id for b in tl.at(0, 1.2)] == [3] and tl.at(1, 1.0) == []
    floor = np.column_stack([np.repeat(np.arange(-5, 5, 0.05), 200), np.tile(np.arange(-5, 5, 0.05), 200), np.zeros(40000)])
    labels = np.ones(len(floor), np.int32)
    oc = _core.OccupancyConfig()
    oc.min_bound, oc.max_bound = np.array([-5.0, -5, -0.5]), np.array([5.0, 5, 2.5])
    gen = _core.OccupancyGenerator(floor, labels, oc, make_T(None, [1.0, 0, 1.8]), [])
    sem, ml, mc = gen.generate(np.eye(4), [box], np.array([[2.0, 0.0, -1.8]]))
    assert sem.shape == (100, 100, 30) and (sem == 13).sum() > 1000 and ml.any() and not mc.any()
    cam = _core.make_camera("pinhole", 320, 240, [160, 160, 160, 120])
    T_wc = make_T(np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float) @ np.eye(3), [0.0, 0, 1.5])
    labeler = _core.FrameLabeler(floor, labels, [(np.array([[2, -1.2], [2, 1.2], [4, 1.2], [4, -1.2]], float), 0.0)],
                                 [], _core.FrameLabelConfig())
    rec, img = labeler.label(cam, T_wc, [], np.full((240, 320, 3), 80, np.uint8))
    assert img.shape == (240, 320) and (img == 1).sum() > 1000
    assert rec["slots"] and rec["slots"][0]["index"] == 0 and rec["consistency_score"] is None
