"""Localization of a vehicle in the navigation map.

The localization map is the measurement point cloud reduced to the
permanent structure: points labelled as vehicles, two-wheelers or
pedestrians, and the moving / changed points, are dropped, since parked cars
change between visits. The nav stage writes it next to the navigation map
(nav/localization_map.npz).

A trip is localized by the reconstruction's odometry front-end (C++
OdometryPipeline, started in its own frame) and the C++ Localizer
(gr/nav/localization.h): with a prior pose (the garage entrance) or, without
one, by placing the odometry submap on the map. `where` turns a pose into what
navigation needs: the level, the position on the route graph and the slot
the vehicle stands in.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon import geometry
from garage_recon.nav import layers as layers_lib
from garage_recon.recon import odometry as odometry_lib
from garage_recon.semantics import taxonomy

MAP_NAME = "localization_map.npz"
# Classes that are not part of the localization map.
MOVABLE_CLASSES = layers_lib.VEHICLE_CLASSES


def structure_points(point_map, labels: np.ndarray, voxel: float = 0.1,
                     keep_parked: bool = False) -> np.ndarray:
    """Static points of the permanent structure, voxel-downsampled.

    Args:
        point_map: Point map with xyz and the dynamic mask.
        labels: Taxonomy class of every map point.
        voxel: Downsampling (m).
        keep_parked: Also keep vehicles that stayed in place during the
            capture (open lots have little other vertical structure).

    Returns:
        (N, 3) float64 points.
    """
    movable = np.isin(labels, [taxonomy.cid(c) for c in MOVABLE_CLASSES])
    if keep_parked:
        movable &= labels != taxonomy.cid("vehicle")
    keep = ~movable
    if point_map.dynamic is not None:
        keep &= ~np.asarray(point_map.dynamic, bool)
    xyz = np.asarray(point_map.xyz, np.float64)[keep]
    return xyz[_core.voxel_downsample_indices(xyz, voxel)]


def write_map(directory: pathlib.Path, point_map, labels: np.ndarray,
              voxel: float = 0.1, keep_parked: bool = False) -> int:
    """Writes nav/localization_map.npz; returns the number of points."""
    xyz = structure_points(point_map, labels, voxel, keep_parked)
    np.savez_compressed(directory / MAP_NAME, xyz=xyz.astype(np.float32),
                        voxel=voxel)
    return len(xyz)


def load_map(directory: pathlib.Path) -> np.ndarray:
    with np.load(directory / MAP_NAME) as z:
        return z["xyz"].astype(np.float64)


def localizer_config(cfg) -> _core.LocalizerConfig:
    section = dict(cfg.get("localization") or {})
    section.pop("level_window", None)
    placement = section.pop("placement", None)
    config = core_config.fill(_core.LocalizerConfig(), section)
    config.placement = core_config.fill(_core.PlacementConfig(), placement)
    return config


def trip_localizer_config(cfg) -> _core.TripLocalizerConfig:
    """C++ TripLocalizerConfig (as gr/io/config.cc reads it for
    gr_localize)."""
    config = _core.TripLocalizerConfig()
    config.odometry = core_config.odometry_config(cfg)
    config.keyframes = core_config.keyframe_config(cfg)
    config.localizer = localizer_config(cfg)
    window = (cfg.get("localization") or {}).get("level_window")
    if window is not None:
        config.level_window = float(window)
    return config


def levelled_start(dataset, stamp: float, window: float) -> np.ndarray:
    """world_from_vehicle at `stamp` with the roll and pitch of gravity
    (C++ LevelledStart: the mean accelerometer reading over `window` s; the
    yaw stays arbitrary). Identity without an IMU."""
    imu = dataset.imu()
    if imu is None or len(imu) == 0:
        return np.eye(4)
    samples = _core.ImuSamples(imu, dataset.rig.T_vehicle_imu[:3, :3])
    return _core.levelled_start(samples, stamp, window)


def localize_trip(dataset, map_xyz: np.ndarray, cfg,
                  prior: np.ndarray | None = None, stride: int = 1,
                  start: float = 0.0, log=print) -> dict:
    """Localizes every `stride`-th sweep of a trip in the map (C++
    TripLocalizer, the loop gr_localize runs).

    Args:
        dataset: TripDataset of the trip.
        map_xyz: Localization map points (map frame).
        cfg: Configuration (odometry, keyframes, localization sections).
        prior: Optional map_from_vehicle at the first sweep.
        stride: Sweeps localized (the odometry sees all of them).
        start: Seconds of the trip skipped (a vehicle switched on anywhere).
        log: Logger.

    Returns:
        Dict of per-sweep arrays (stamp, T, odom, status, registered,
        fitness, rmse, degenerate_t, sigma_t, sigma_yaw_deg, ms) and a
        summary.
    """
    T_vl = dataset.rig.primary_lidar.T_vehicle_lidar
    n = len(dataset.lidar_stamps)
    first = int(np.searchsorted(dataset.lidar_stamps,
                                dataset.lidar_stamps[0] + start))
    config = trip_localizer_config(cfg)
    trip = _core.TripLocalizer(
        map_xyz, config, T_vl,
        levelled_start(dataset, dataset.lidar_stamps[first],
                       config.level_window),
        odometry_lib.motion_prior(dataset), prior, log)
    rows = []
    for k in range(first, n):
        sweep = dataset.load_lidar(k)
        r = trip.process(sweep.points, sweep.intensity, sweep.time,
                         sweep.stamp, k, k == n - 1, (k - first) % stride == 0)
        if r is not None:
            rows.append(r)
    return _arrays(rows, trip.summary())


def _arrays(rows: list[dict], summary: dict) -> dict:
    out = {key: np.array([r[key] for r in rows]) for key in
           ("stamp", "status", "registered", "fitness", "rmse",
            "degenerate_t", "step_t", "step_r", "sigma_t", "sigma_yaw_deg",
            "ms")}
    out["eigen_ratio_t"] = np.array([r["min_eigen_ratio_t"] for r in rows])
    out["T"] = np.stack([r["T_map_vehicle"] for r in rows])
    out["odom"] = np.stack([r["T_odom_vehicle"] for r in rows])

    def rounded(value, digits):
        return None if value is None else round(float(value), digits)

    out["summary"] = {
        "sweeps": summary["sweeps"],
        "first_tracked": summary["first_tracked"],
        "init_time_s": rounded(summary["init_time_s"], 2),
        "init_distance_m": rounded(summary["init_distance_m"], 2),
        "placement_attempts": summary["placement_attempts"],
        "tracked": summary["tracked"],
        "registered": summary["registered"],
        "registered_share_of_tracked": rounded(
            summary["registered_share_of_tracked"], 4),
        "lost_sweeps": summary["lost_sweeps"],
        "degenerate_share": rounded(summary["degenerate_share"], 4),
        "ms_median": rounded(summary["ms_median"], 1),
        "ms_p95": rounded(summary["ms_p95"], 1),
    }
    return out


def evaluate(result: dict, gt_stamps: np.ndarray,
             gt_poses: np.ndarray) -> dict:
    """Errors of the tracked sweeps against reference vehicle poses.

    Returns:
        Dict of position (3D, horizontal, vertical) and yaw error statistics
        in m and deg (median, p95, max).
    """
    tracking = result["status"] == "tracking"
    stamps = result["stamp"]
    inside = (stamps >= gt_stamps[0]) & (stamps <= gt_stamps[-1]) & tracking
    if not inside.any():
        return {"sweeps": 0}
    truth = _core.interpolate_poses(gt_stamps, gt_poses, stamps[inside])
    est = result["T"][inside]
    d = est[:, :3, 3] - truth[:, :3, 3]
    yaw = np.degrees(np.abs(geometry.wrap_angle(
        np.arctan2(est[:, 1, 0], est[:, 0, 0])
        - np.arctan2(truth[:, 1, 0], truth[:, 0, 0]))))

    def stats(values: np.ndarray, digits: int = 4) -> dict:
        return {"median": round(float(np.median(values)), digits),
                "p95": round(float(np.percentile(values, 95)), digits),
                "max": round(float(np.max(values)), digits)}

    # Relative to the map: after the rigid alignment of the two trajectories
    # (removes the map's own anchoring error).
    # pylint: disable-next=import-outside-toplevel
    from garage_recon.eval import synthetic_eval
    T_align = synthetic_eval.umeyama(est[:, :3, 3], truth[:, :3, 3])
    aligned = geometry.transform(T_align, est[:, :3, 3]) - truth[:, :3, 3]
    return {"sweeps": int(inside.sum()),
            "position_m": stats(np.linalg.norm(d, axis=1)),
            "position_aligned_m": stats(np.linalg.norm(aligned, axis=1)),
            "horizontal_m": stats(np.linalg.norm(d[:, :2], axis=1)),
            "vertical_m": stats(np.abs(d[:, 2])),
            "yaw_deg": stats(yaw, 3)}


def write_result(directory: pathlib.Path, result: dict,
                 report: dict) -> None:
    """poses.tum (map <- vehicle of the tracked sweeps), sweeps.csv,
    localization.json."""
    directory.mkdir(parents=True, exist_ok=True)
    tracking = result["status"] == "tracking"
    geometry.write_tum(directory / "poses.tum", result["stamp"][tracking],
                       result["T"][tracking])
    with open(directory / "sweeps.csv", "w") as f:
        f.write("stamp,status,registered,fitness,rmse,degenerate_t,step_t,"
                "step_r,eigen_ratio_t,sigma_t,sigma_yaw_deg,ms,x,y,z\n")
        for k in range(len(result["stamp"])):
            x, y, z = result["T"][k, :3, 3]
            f.write(f"{result['stamp'][k]:.6f},{result['status'][k]},"
                    f"{int(result['registered'][k])},"
                    f"{result['fitness'][k]:.4f},{result['rmse'][k]:.4f},"
                    f"{result['degenerate_t'][k]},{result['step_t'][k]:.4f},"
                    f"{result['step_r'][k]:.3f},"
                    f"{result['eigen_ratio_t'][k]:.2e},"
                    f"{result['sigma_t'][k]:.4f},"
                    f"{result['sigma_yaw_deg'][k]:.4f},{result['ms'][k]:.1f},"
                    f"{x:.3f},{y:.3f},{z:.3f}\n")
    (directory / "localization.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False))


def where(T_map_vehicle: np.ndarray, graph, goals: list[dict],
          layers_by_level: dict) -> dict:
    """Where a localized vehicle is, in navigation terms (C++ Where).

    Args:
        T_map_vehicle: (4, 4) vehicle pose in the map.
        graph: NavGraph.
        goals: Slot goals (goals.json).
        layers_by_level: LevelLayers per level (for the floor heights).

    Returns:
        Dict of level, the nearest drivable graph edge (index, distance
        along it, lateral offset, heading relative to the edge direction),
        and the slot whose footprint contains the vehicle (None if none).
    """
    edges = [(int(e.level), e.kind, np.asarray(e.points, float)[:, :2])
             for e in graph.edges]
    slots = [(int(g["level"]), g["slot_id"],
              np.asarray(g["corners"], float))
             for g in goals if "corners" in g]
    floors = [(int(level), float(layers.grid.x0), float(layers.grid.y0),
               float(layers.grid.res), np.asarray(layers.floor_z, float))
              for level, layers in layers_by_level.items()]
    p = _core.nav_position(np.asarray(T_map_vehicle, float), edges, slots,
                           floors)
    edge = None
    if p["edge"] is not None:
        e = p["edge"]
        edge = {"edge": e["edge"], "kind": e["kind"],
                "distance_m": round(e["distance"], 3),
                "along_m": round(e["along"], 2),
                "lateral_m": round(e["lateral"], 3),
                "heading_rel_deg": round(e["heading_rel_deg"], 1)}
    return {"xyz": np.round(np.asarray(p["xyz"]), 3).tolist(),
            "yaw_deg": round(p["yaw_deg"], 2), "level": p["level"],
            "edge": edge, "slot": p["slot"]}


def run(work: pathlib.Path, trip: pathlib.Path, out: pathlib.Path, cfg,
        prior: str = "none", reference: pathlib.Path | None = None,
        stride: int = 1, start: float = 0.0, log=print) -> dict:
    """Localizes a trip in the navigation map of `work` (command localize).

    Args:
        work: Reconstruction with the nav stage (semantics/nav/).
        trip: Trip directory to localize.
        out: Output directory.
        cfg: Configuration.
        prior: "none" (global initialization), "entrance" (the trip's
            entrance pose, in the map frame when the map has the same
            anchor) or "reference" (the reference trajectory at the first
            sweep; a diagnostic for tracking alone).
        reference: TUM trajectory (world <- vehicle) or a directory with
            poses.tum, for the errors.
        stride: Sweeps localized.
        start: Seconds of the trip skipped (the entrance prior then does
            not apply).
        log: Logger.

    Returns:
        The report written to localization.json.
    """
    # pylint: disable-next=import-outside-toplevel
    from garage_recon.data import dataset as dataset_lib
    # pylint: disable-next=import-outside-toplevel
    from garage_recon.nav import graph as graph_lib
    # pylint: disable-next=import-outside-toplevel
    from garage_recon.nav import navmap
    nav_dir = pathlib.Path(work) / "semantics" / "nav"
    map_xyz = load_map(nav_dir)
    data = dataset_lib.TripDataset(trip)
    T_prior = None
    if prior == "entrance" and start > 0.0:
        raise ValueError("the entrance prior needs start = 0")
    if prior == "entrance":
        T_prior = odometry_lib.entrance_pose(data)
        if T_prior is None:
            raise ValueError(f"{trip} has no entrance pose")
    reference_path = None
    if reference is not None:
        reference_path = pathlib.Path(reference)
        if reference_path.is_dir():
            reference_path = reference_path / "poses.tum"
    if prior == "reference":
        if reference_path is None:
            raise ValueError("prior 'reference' needs --reference")
        stamps, poses = geometry.read_tum(reference_path)
        first = data.lidar_stamps[int(np.searchsorted(
            data.lidar_stamps, data.lidar_stamps[0] + start))]
        T_prior = _core.interpolate_poses(stamps, poses,
                                          np.array([first]))[0]
    result = localize_trip(data, map_xyz, cfg, T_prior, stride, start, log)
    report = {"trip": data.trip_id, "prior": prior, "start_s": start,
              "map_points": int(len(map_xyz)), **result["summary"]}
    if reference_path is not None:
        stamps, poses = geometry.read_tum(reference_path)
        report["errors"] = evaluate(result, stamps, poses)
    tracking = np.where(result["status"] == "tracking")[0]
    if len(tracking):
        goals = json.loads((nav_dir / "goals.json").read_text())["goals"]
        graph = graph_lib.NavGraph.from_json(json.loads(
            (nav_dir / "graph.json").read_text()))
        report["last_position"] = where(result["T"][tracking[-1]], graph,
                                        goals, navmap.load_layers(nav_dir))
    write_result(pathlib.Path(out), result, report)
    errors = report.get("errors", {})
    log(f"[localize] {data.trip_id}: tracked {report['tracked']}/"
        f"{report['sweeps']} sweeps from {report['init_distance_m']} m, "
        f"registered {report['registered_share_of_tracked']}, degenerate "
        f"{report['degenerate_share']}, {report['ms_median']} ms/sweep"
        + (f"; error median {errors['position_m']['median']} m, p95 "
           f"{errors['position_m']['p95']} m, yaw p95 "
           f"{errors['yaw_deg']['p95']} deg" if errors.get("sweeps") else ""))
    return report
