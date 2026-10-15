"""Ground-truth-free quality signals (the production gate on garages without
handheld ground truth). Calibrate their thresholds on ground-truth garages.

* pose set           id of the one pose set, and stage outputs built with
                     another one (stale: must be re-run, never shipped)
* plane thickness    crispness of floors / walls / columns (registration)
* bundle adjustment  RMS point-to-plane distance of the planes shared by
                     several keyframe sweeps, before / after each round
* map sigma          per-point sigma of the fused map (multi-observation
                     spread along the normal)
* loop residuals     chi2 of accepted loop closures after optimisation
* IMU                IMU factor residuals and the estimated bias per trip
* calibration        camera reprojection RMS and time offsets
* repeatability      distance between structure seen in different sessions
* projection score   image-edge agreement of projected markings
* cross-view         agreement of the 3D language labels with every image's
  consistency        own 2D prediction
* rule violations    counts from semantics/issues.json
"""

from __future__ import annotations

import json

import numpy as np
from scipy import spatial

from garage_recon import _core
from garage_recon import provenance
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.semantics import taxonomy


def _pose_set_report(recon) -> dict:
    if not pose_set_lib.PoseSet.exists(recon.work):
        return {"pose_set_id": None}
    pose_set = recon.pose_set()
    stale = provenance.stale_stages(recon.work, pose_set)
    return {"pose_set_id": pose_set.pose_set_id,
            "trajectory_id": pose_set.trajectory_id,
            "stages_recorded": len(provenance.entries(recon.work)),
            "stale_stages": stale}


def run_selfcheck(recon, semantics, log=print) -> dict:
    """Writes work/qa/selfcheck.json and returns the report."""
    report: dict = {"pose_set": _pose_set_report(recon)}
    have_labels = (semantics.out / "labels.npz").exists()
    point_map = semantics.point_map() if have_labels else recon.point_map()
    labels = semantics.labels() if have_labels else None
    static = ~point_map.dynamic
    structure = static & (np.isin(labels, list(taxonomy.STRUCTURE_IDS))
                          if labels is not None else True)
    points = point_map.xyz[structure]
    if len(points) > 2_000_000:
        points = points[np.random.default_rng(0).choice(len(points), 2_000_000,
                                                        replace=False)]
    report["plane_thickness"] = _core.plane_thickness(points, 0.5, 20, 0.7)

    summary_path = recon.work / "graph" / "summary.json"
    if summary_path.exists():
        graph = json.loads(summary_path.read_text())
        report["graph"] = {k: graph.get(k) for k in (
            "loops", "loop_chi2_median", "loop_chi2_max", "final_cost",
            "imu_factors", "imu_chi2_median")}
        report["imu_bias"] = {k: v for k, v in graph.items()
                              if k.startswith("imu_bias_")}
        if graph.get("ba_rounds"):
            last = graph["ba_rounds"][-1]
            report["bundle_adjustment"] = {
                "rounds": len(graph["ba_rounds"]),
                "planes": last["planes"],
                "plane_rms_median_m": last["rms_median_after"],
                "plane_rms_p90_m": last["rms_p90_after"],
                "last_round_shift_median_m": last["shift_median"],
                "last_round_shift_max_m": last["shift_max"]}
    if point_map.sigma is not None:
        sigma = point_map.sigma[structure]
        report["map_sigma"] = {
            "median_m": float(np.median(sigma)),
            "p90_m": float(np.percentile(sigma, 90)),
            "observations_median": float(np.median(
                point_map.observations[structure]))}

    calibration_path = recon.work / "calibration.json"
    if calibration_path.exists():
        cameras = json.loads(calibration_path.read_text())["cameras"]
        report["calibration"] = {
            name: {k: entry[k] for k in ("accepted", "rms_before_px",
                                          "rms_after_px", "time_offset_s",
                                          "time_offset_sigma_s")}
            for name, entry in cameras.items()}

    if len(recon.datasets) > 1:
        repeatability = {}
        trips = np.unique(point_map.trip[structure])
        for a in trips:
            for b in trips:
                if a >= b:
                    continue
                points_a = point_map.xyz[structure & (point_map.trip == a)]
                points_b = point_map.xyz[structure & (point_map.trip == b)]
                if len(points_a) < 100 or len(points_b) < 100:
                    continue
                distance, _ = spatial.cKDTree(points_b).query(
                    points_a[::max(1, len(points_a) // 200000)],
                    distance_upper_bound=0.5)
                distance = distance[np.isfinite(distance)]
                repeatability[f"{a}-{b}"] = {
                    "median": float(np.median(distance)),
                    "p90": float(np.percentile(distance, 90))}
        report["session_repeatability"] = repeatability

    frames = semantics.out / "frames" / "summary.json"
    if frames.exists():
        report["projection"] = json.loads(frames.read_text())
    language = semantics.out / "language" / "summary.json"
    if language.exists():
        summary = json.loads(language.read_text())
        report["language"] = {k: summary.get(k) for k in (
            "identities", "classified", "cross_view_consistency_mean",
            "cross_view_consistency_p10", "models")}
    issues = semantics.out / "issues.json"
    if issues.exists():
        items = json.loads(issues.read_text())
        report["issues"] = {severity: sum(i["severity"] == severity
                                          for i in items)
                            for severity in ("error", "warning", "info")}
    out = recon.work / "qa"
    out.mkdir(exist_ok=True)
    (out / "selfcheck.json").write_text(json.dumps(report, indent=1))
    log("[qa] " + json.dumps(report))
    return report
