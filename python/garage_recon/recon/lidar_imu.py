"""LiDAR-IMU calibration check of a reconstruction (command
calibrate-lidar-imu).

The rig's IMU rotation (rig.yaml T_vehicle_imu against T_vehicle_lidar) is
estimated again from every trip, with the gyro and accelerometer biases and
the accelerometer scales (C++ CalibrateLidarImu: the IMU against the
kinematics of the trip's LiDAR odometry). The report gives each estimate,
its standard deviation (large: the trip did not excite that direction) and
the angle to the rig's value; a few tenths of a degree of disagreement with
a small sigma means rig.yaml should be corrected (`rig_suggestion`).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon import geometry
from garage_recon.data import dataset as dataset_lib


def config(cfg) -> _core.LidarImuConfig:
    return core_config.fill(_core.LidarImuConfig(),
                            cfg.calibration.get("lidar_imu"))


def calibrate_trip(work: pathlib.Path, trip: pathlib.Path, cfg) -> dict:
    """The estimate of one trip from work/trips/<trip>/odometry.tum."""
    data = dataset_lib.TripDataset(trip)
    stamps, world_from_vehicle = geometry.read_tum(
        pathlib.Path(work) / "trips" / data.trip_id / "odometry.tum")
    rig = data.rig
    T_vl = rig.primary_lidar.T_vehicle_lidar
    imu_rows = data.imu()
    if imu_rows is None or len(imu_rows) < 10:
        return {"trip": data.trip_id, "error": "no IMU"}
    imu = _core.ImuSamples(imu_rows, np.eye(3))  # The IMU's own axes.
    rig_T = geometry.inv(rig.T_vehicle_imu) @ T_vl  # imu <- lidar
    rig_R = rig_T[:3, :3]
    r = _core.calibrate_lidar_imu(stamps, world_from_vehicle @ T_vl, imu,
                                  rig_T, config(cfg))
    R = np.asarray(r["R_imu_lidar"])
    angle = float(np.degrees(np.linalg.norm(
        _core.so3_log(rig_R.T @ R))))
    # The vehicle frame keeps the LiDAR's mounting: the IMU rotation that
    # would make rig.yaml agree, T_vehicle_imu = T_vehicle_lidar R^T.
    suggested = rig.T_vehicle_imu.copy()
    suggested[:3, :3] = T_vl[:3, :3] @ R.T

    def rounded(v, digits=5):
        return np.round(np.asarray(v, float), digits).tolist()

    return {"trip": data.trip_id, "converged": r["converged"],
            "samples": r["samples"],
            "angle_to_rig_deg": round(angle, 3),
            "rotation_change_deg": rounded(np.degrees(_core.so3_log(
                rig_R.T @ R)), 3),
            "sigma_rotation_deg": rounded(r["sigma_rotation_deg"], 3),
            "gyro_bias": rounded(r["gyro_bias"]),
            "accel_bias": rounded(r["accel_bias"], 4),
            "sigma_accel_bias": rounded(r["sigma_accel_bias"], 4),
            "time_offset_s": round(r["time_offset"], 4),
            "lever_arm_m": rounded(r["lever_arm"], 3),
            "rig_lever_arm_m": rounded(-rig_R.T @ rig_T[:3, 3], 3),
            "sigma_lever_arm_m": rounded(r["sigma_lever_arm"], 3),
            "sigma_time_offset_s": round(r["sigma_time_offset"], 4),
            "accel_scale": rounded(r["accel_scale"], 5),
            "sigma_accel_scale": rounded(r["sigma_accel_scale"], 5),
            "gyro_rms": round(r["gyro_rms"], 5),
            "accel_rms": round(r["accel_rms"], 4),
            "rig_suggestion_T_vehicle_imu": rounded(suggested, 9)}


def run(work: pathlib.Path, trips: list[pathlib.Path], cfg,
        log=print) -> dict:
    """Estimates of every trip into work/lidar_imu.json."""
    report = {"trips": [calibrate_trip(work, t, cfg) for t in trips]}
    for t in report["trips"]:
        if "error" in t:
            log(f"[lidar-imu] {t['trip']}: {t['error']}")
            continue
        log(f"[lidar-imu] {t['trip']}: {t['samples']} windows, rotation "
            f"{t['angle_to_rig_deg']} deg from rig.yaml (sigma "
            f"{t['sigma_rotation_deg']} deg), IMU time offset "
            f"{t['time_offset_s']} s, accel scale "
            f"{t['accel_scale']} (sigma {t['sigma_accel_scale']}), residual "
            f"gyro {t['gyro_rms']} rad/s, accel {t['accel_rms']} m/s^2")
    (pathlib.Path(work) / "lidar_imu.json").write_text(
        json.dumps(report, indent=1))
    return report
