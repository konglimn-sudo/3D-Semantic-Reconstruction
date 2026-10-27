"""Synthetic garage dataset generator (trips + ground truth).

Output:
    <out>/trip_00, trip_01, ...      TripDataset directories.
    <out>/gt/scene.json              Static geometry, slots, lines, columns.
    <out>/gt/calibration.json        True camera extrinsics / time offsets /
                                     line times and the IMU bias of each trip.
    <out>/gt/trip_XX/poses.tum       Ground-truth vehicle poses (LiDAR stamps).
    <out>/gt/trip_XX/trajectory.npz  Dense ground-truth trajectory knots.
    <out>/gt/trip_XX/objects.json    Parked and moving cars of the trip.

Sensor errors (enabled by default, `ideal=True` removes them): cameras have
a rolling shutter, a time offset between stamp and exposure and an extrinsic
error in rig.yaml; the IMU has a constant bias per trip. The pipeline must
estimate them (IMU factors, camera calibration).
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from scipy.spatial import transform

from garage_recon import geometry
from garage_recon.data import dataset
from garage_recon.data import schema
from garage_recon.data.synthetic import scene as scene_lib
from garage_recon.data.synthetic import sensors
from garage_recon.data.synthetic import trajectory as trajectory_lib

PRESETS: dict[str, dict] = {
    # Unit tests: one level, short loop.
    "tiny": {"layout": {"n_levels": 1, "groups": 2},
             "lidar": {"beams": 16, "h_res_deg": 1.0},
             "frame_spacing": 0.6, "image_size": 128, "image_every": 3,
             "trips": 1},
    # Two levels and a ramp; two trips with different parked cars.
    "small": {"layout": {"n_levels": 2, "groups": 4},
              "lidar": {"beams": 32, "h_res_deg": 0.6},
              "frame_spacing": 0.6, "image_size": 256, "image_every": 3,
              "trips": 2},
    # Scale tests: three levels, 8 bay groups, LiDAR-only trips.
    "large": {"layout": {"n_levels": 3, "groups": 8},
              "lidar": {"beams": 32, "h_res_deg": 0.6},
              "frame_spacing": 0.6, "image_size": 128, "image_every": 6,
              "trips": 3},
    "demo": {"layout": {"n_levels": 2, "groups": 6},
             "lidar": {"beams": 32, "h_res_deg": 0.4},
             "frame_spacing": 0.5, "image_size": 320, "image_every": 2,
             "trips": 2},
}

# Sensor errors of the synthetic rig.
SENSOR_ERRORS = {
    "readout": 0.025,           # s, rolling-shutter readout of an image.
    "time_offset": 0.012,       # s, mean camera time offset (spread +-4 ms).
    "extrinsic_rot_deg": 0.3,   # Rotation error of rig.yaml, per camera.
    "extrinsic_trans": 0.015,   # m, translation error of rig.yaml.
    "gyro_bias_deg_s": 0.1,     # Per axis and trip, standard deviation.
    "accel_bias": 0.04,         # m/s^2, per axis and trip.
}

SPEED = 1.4  # m/s, ~5 km/h.


def make_rig(image_size: int, stereo: bool = False,
             line_time: float = 0.0) -> schema.Rig:
    """Roof LiDAR, four surround-view fisheyes and an optional front stereo."""
    lidar = schema.LidarCalib("lidar_top", geometry.make_T(t=[0.0, 0.0, 1.9]),
                              sweep_duration=0.1)
    theta_max = np.deg2rad(95.0)
    f = (image_size / 2.0) / theta_max
    kb = [f, f, image_size / 2.0, image_size / 2.0, -0.01, 0.002, 0.0, 0.0,
          1.70]
    pitch = {"front": 20.0, "rear": 20.0, "left": 40.0, "right": 40.0}
    mounts = {
        "front": ([2.2, 0.0, 0.75], 0.0),
        "rear": ([-0.9, 0.0, 0.90], np.pi),
        "left": ([1.0, 0.95, 1.0], np.pi / 2),
        "right": ([1.0, -0.95, 1.0], -np.pi / 2),
    }
    cameras = {}
    for name, (position, yaw) in mounts.items():
        p = np.deg2rad(pitch[name])
        forward = np.array([np.cos(p) * np.cos(yaw), np.cos(p) * np.sin(yaw),
                            -np.sin(p)])
        cameras[f"fisheye_{name}"] = schema.CameraCalib(
            f"fisheye_{name}", "kannala_brandt", image_size, image_size, kb,
            geometry.make_T(geometry.look_rotation(forward), position),
            rolling_shutter_line_time=line_time)
    pairs = []
    if stereo:
        w, h, fs = 320, 240, 220.0
        forward = np.array([np.cos(np.deg2rad(8.0)), 0.0,
                            -np.sin(np.deg2rad(8.0))])
        for side, dy in (("left", 0.06), ("right", -0.06)):
            cameras[f"stereo_{side}"] = schema.CameraCalib(
                f"stereo_{side}", "pinhole", w, h, [fs, fs, w / 2, h / 2],
                geometry.make_T(geometry.look_rotation(forward),
                                [1.6, dy, 1.35]))
        pairs.append(("stereo_left", "stereo_right"))
    return schema.Rig(lidars={"lidar_top": lidar}, cameras=cameras,
                      stereo_pairs=pairs)


def _perturbed(T: np.ndarray, rng: np.random.Generator, rot_deg: float,
               trans: float) -> np.ndarray:
    """T with a random rotation of `rot_deg` and translation of `trans`."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    delta = geometry.make_T(
        transform.Rotation.from_rotvec(np.deg2rad(rot_deg) * axis).as_matrix(),
        trans * direction)
    return T @ delta


def sensor_truth(rig: schema.Rig, errors: dict | None, seed: int,
                 trips: int) -> tuple[schema.Rig, schema.Rig, dict]:
    """True rig, nominal rig (as written to rig.yaml) and IMU biases.

    Args:
        rig: The rig built by make_rig (true extrinsics).
        errors: SENSOR_ERRORS-like dict, or None for an ideal rig.
        seed: Seed of the error draws (independent of the scene noise).
        trips: Number of trips (one IMU bias each).

    Returns:
        (true rig, nominal rig, {trip index: (6,) IMU bias}).
    """
    rng = np.random.default_rng(seed + 4242)
    true_rig = schema.Rig.from_dict(rig.to_dict())
    nominal = schema.Rig.from_dict(rig.to_dict())
    biases = {k: np.zeros(6) for k in range(trips)}
    if not errors:
        for calib in list(true_rig.cameras.values()) + list(
                nominal.cameras.values()):
            calib.rolling_shutter_line_time = 0.0
        return true_rig, nominal, biases
    for name, calib in true_rig.cameras.items():
        calib.time_offset = errors["time_offset"] + rng.uniform(-0.004, 0.004)
        if calib.model != "pinhole":
            calib.rolling_shutter_line_time = errors["readout"] / calib.height
        else:
            calib.rolling_shutter_line_time = 0.0
        nominal_calib = nominal.cameras[name]
        nominal_calib.rolling_shutter_line_time = (
            calib.rolling_shutter_line_time)  # From the datasheet.
        nominal_calib.time_offset = 0.0
        nominal_calib.T_vehicle_cam = _perturbed(
            calib.T_vehicle_cam, rng, errors["extrinsic_rot_deg"],
            errors["extrinsic_trans"])
    for k in range(trips):
        biases[k] = np.concatenate([
            rng.normal(0.0, np.deg2rad(errors["gyro_bias_deg_s"]), 3),
            rng.normal(0.0, errors["accel_bias"], 3)])
    return true_rig, nominal, biases


def row_poses(drive: trajectory_lib.Trajectory, calib: schema.CameraCalib,
              first_row_time: float) -> np.ndarray:
    """world_from_camera of every image row (one pose for global shutter)."""
    if calib.rolling_shutter_line_time <= 0.0:
        return drive.poses_at(first_row_time)[0] @ calib.T_vehicle_cam
    times = first_row_time + calib.rolling_shutter_line_time * np.arange(
        calib.height)
    return drive.poses_at(times) @ calib.T_vehicle_cam


def generate(out_dir: str | pathlib.Path, preset: str = "small",
             trips: int | None = None, seed: int = 0, stereo: bool = False,
             images: bool = True, ideal: bool = False, arrows: bool = True,
             changes: dict | None = None,
             log=print) -> list[pathlib.Path]:
    """Writes a synthetic dataset and returns the trip directories.

    `changes` maps a trip index to a scene change of that trip
    (scene_lib.apply_change: boxes added or removed, e.g. a construction
    fence or a removed duct)."""
    params = PRESETS[preset]
    layout = scene_lib.Layout(**params["layout"], arrows=arrows)
    garage = scene_lib.build_static_scene(layout)
    n_trips = params["trips"] if trips is None else trips
    rigs = sensor_truth(make_rig(params["image_size"], stereo),
                        None if ideal else SENSOR_ERRORS, seed, n_trips)
    true_rig, nominal_rig, imu_biases = rigs
    out = pathlib.Path(out_dir)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "gt" / "scene.json").write_text(json.dumps(garage.to_json()))
    calibration = {
        "cameras": {n: c.to_dict() for n, c in true_rig.cameras.items()},
        "nominal_cameras": {n: c.to_dict()
                            for n, c in nominal_rig.cameras.items()},
        "imu_bias": {f"trip_{k:02d}": imu_biases[k].tolist()
                     for k in range(n_trips)},
        "ideal": bool(ideal),
    }
    (out / "gt" / "calibration.json").write_text(json.dumps(calibration))
    return [_write_trip(out, k, preset, garage, rigs, seed, images,
                        (changes or {}).get(k), log)
            for k in range(n_trips)]


def add_trip(out_dir: str | pathlib.Path, trip_index: int,
             preset: str = "small", seed: int = 0, stereo: bool = False,
             images: bool = True, ideal: bool = False, arrows: bool = True,
             change: dict | None = None, log=print) -> pathlib.Path:
    """Writes trip `trip_index` of a dataset made by generate() with the
    same preset and seed (same garage, rig and route; its own parked cars),
    with an optional scene change."""
    params = PRESETS[preset]
    layout = scene_lib.Layout(**params["layout"], arrows=arrows)
    garage = scene_lib.build_static_scene(layout)
    rigs = sensor_truth(make_rig(params["image_size"], stereo),
                        None if ideal else SENSOR_ERRORS, seed,
                        trip_index + 1)
    return _write_trip(pathlib.Path(out_dir), trip_index, preset, garage,
                       rigs, seed, images, change, log)


def _write_trip(out: pathlib.Path, k: int, preset: str,
                garage: scene_lib.GarageScene, rigs, seed: int, images: bool,
                change: dict | None, log) -> pathlib.Path:
    """Simulates and writes trip k (and its ground truth)."""
    params = PRESETS[preset]
    true_rig, nominal_rig, imu_biases = rigs
    route = trajectory_lib.build_route(garage.layout)
    model = sensors.LidarModel(**params["lidar"])
    renderers = ({n: sensors.CameraRenderer(c)
                  for n, c in true_rig.cameras.items()} if images else {})
    T_vl = true_rig.primary_lidar.T_vehicle_lidar
    t_start = time.time()
    rng = np.random.default_rng(seed * 100 + k)
    speed = SPEED * (1.0 + 0.04 * k)
    drive = trajectory_lib.Trajectory(route, speed=speed,
                                      t0=1000.0 + 600.0 * k)
    cars = scene_lib.parked_cars(garage, k, seed)
    movers = scene_lib.moving_cars(garage, k, seed)
    for m in movers:  # The moving car crosses our path early in the drive.
        m.t0 += drive.t0
        m.t1 += drive.t0
    scene = garage if not change else scene_lib.apply_change(garage, change)
    caster = sensors.SceneRaycaster(scene, cars, movers)

    # INS fix at the entrance: the true start pose with small errors.
    start = drive.poses_at(drive.t0)[0]
    noise = geometry.make_T(geometry.rot_z(rng.normal(0, np.deg2rad(0.2))),
                            rng.normal(0, 0.02, 3))
    meta = {"trip_id": f"trip_{k:02d}", "garage_id": f"synthetic_{preset}",
            "synthetic": True, "initial_pose": (start @ noise).tolist(),
            "initial_pose_time": float(drive.t0)}
    trip_dir = out / f"trip_{k:02d}"
    writer = dataset.TripWriter(trip_dir, nominal_rig, meta)
    stamps = np.arange(drive.t0 + model.sweep, drive.t_end,
                       params["frame_spacing"] / speed)
    gt_poses = drive.poses_at(stamps)
    for j, stamp in enumerate(stamps):
        sweep = sensors.simulate_sweep(caster, drive, stamp, T_vl, model, rng)
        ns = int(round(stamp * 1e9))
        writer.write_lidar("lidar_top", ns, sweep["points"],
                           sweep["intensity"], sweep["time"], quantize=0.005)
        if images and j % params["image_every"] == 0:
            for name, renderer in renderers.items():
                calib = true_rig.cameras[name]
                # Row 0 is exposed at the LiDAR stamp; the image is stamped
                # time_offset earlier.
                poses = row_poses(drive, calib, stamp)
                mid = stamp + 0.5 * calib.height * (
                    calib.rolling_shutter_line_time)
                frame = renderer.render(caster, poses, mid, rng)
                image_ns = int(round((stamp - calib.time_offset) * 1e9))
                writer.write_image(name, image_ns, frame["rgb"])
    imu, wheel = sensors.proprioception(drive, drive.t0, drive.t_end, rng,
                                        imu_bias=imu_biases[k])
    writer.write_csv("imu.csv", "t,gx,gy,gz,ax,ay,az", imu)
    writer.write_csv("wheel.csv", "t,speed", wheel)
    g_t = np.arange(drive.t0, drive.t0 + 3.0, 1.0)
    g_p = drive.poses_at(g_t)[:, :3, 3] + rng.normal(0, 0.02, (len(g_t), 3))
    writer.write_csv("gnss.csv", "t,x,y,z,std",
                     np.column_stack([g_t, g_p, np.full(len(g_t), 0.02)]))

    gt_dir = out / "gt" / f"trip_{k:02d}"
    gt_dir.mkdir(parents=True, exist_ok=True)
    geometry.write_tum(gt_dir / "poses.tum", stamps, gt_poses)
    np.savez_compressed(gt_dir / "trajectory.npz", times=drive.times,
                        poses=drive.poses)
    objects = {
        "parked": [{"key": b.key, "instance": b.instance,
                    "center": b.center.tolist(), "size": b.size.tolist(),
                    "yaw": float(np.arctan2(b.R[1, 0], b.R[0, 0])),
                    "level": b.level} for b in cars],
        "moving": [{"instance": m.instance, "start": m.start.tolist(),
                    "velocity": m.velocity.tolist(), "t0": m.t0,
                    "t1": m.t1, "size": m.size.tolist(),
                    "level": m.level} for m in movers],
    }
    if change:
        objects["change"] = change
    (gt_dir / "objects.json").write_text(json.dumps(objects))
    log(f"[synth] {trip_dir.name}: {len(stamps)} sweeps, "
        f"{drive.length:.0f} m route, {len(cars) // 2} parked cars, "
        f"{time.time() - t_start:.1f}s")
    return trip_dir
