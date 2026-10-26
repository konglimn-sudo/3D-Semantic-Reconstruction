"""Sensor rig description (intrinsics, extrinsics and timing) in rig.yaml.

Camera timing follows one convention everywhere: row r of an image stamped t
is exposed at t + time_offset + r * rolling_shutter_line_time.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import yaml

from garage_recon import _core


def _matrix(value) -> np.ndarray:
    return np.asarray(value, float).reshape(4, 4)


@dataclasses.dataclass
class CameraCalib:
    """Intrinsics, extrinsics and timing of one camera.

    Attributes:
        name: Camera name (directory under cameras/).
        model: "pinhole", "kannala_brandt" or "double_sphere".
        width: Image width in pixels.
        height: Image height in pixels.
        params: Intrinsic parameters of the model.
        T_vehicle_cam: (4, 4) vehicle <- camera (OpenCV axes).
        rolling_shutter_line_time: Seconds per row; 0 for a global shutter.
        time_offset: Seconds from the image stamp to the exposure of row 0.
    """

    name: str
    model: str
    width: int
    height: int
    params: list[float]
    T_vehicle_cam: np.ndarray
    rolling_shutter_line_time: float = 0.0
    time_offset: float = 0.0

    def make_model(self):
        """Returns the C++ camera model."""
        return _core.make_camera(self.model, int(self.width), int(self.height),
                                 [float(p) for p in self.params])

    def to_dict(self) -> dict:
        """The rig.yaml entry of the camera."""
        return {
            "model": self.model,
            "width": int(self.width),
            "height": int(self.height),
            "params": [float(p) for p in self.params],
            "T_vehicle_cam": np.asarray(self.T_vehicle_cam).tolist(),
            "rolling_shutter_line_time": float(self.rolling_shutter_line_time),
            "time_offset": float(self.time_offset),
        }


@dataclasses.dataclass
class LidarCalib:
    """Extrinsics of one LiDAR; its stamps mark the end of each sweep."""

    name: str
    T_vehicle_lidar: np.ndarray
    sweep_duration: float = 0.1

    def to_dict(self) -> dict:
        return {"T_vehicle_lidar": np.asarray(self.T_vehicle_lidar).tolist(),
                "sweep_duration": self.sweep_duration}


@dataclasses.dataclass
class Rig:
    """All sensors of the vehicle."""

    lidars: dict[str, LidarCalib]
    cameras: dict[str, CameraCalib] = dataclasses.field(default_factory=dict)
    stereo_pairs: list[tuple[str, str]] = dataclasses.field(
        default_factory=list)
    T_vehicle_imu: np.ndarray = dataclasses.field(
        default_factory=lambda: np.eye(4))
    wheel_speed_scale: float = 1.0

    @property
    def primary_lidar(self) -> LidarCalib:
        return next(iter(self.lidars.values()))

    @property
    def fisheye_cameras(self) -> list[str]:
        return [n for n, c in self.cameras.items() if c.model != "pinhole"]

    @classmethod
    def from_dict(cls, data: dict) -> Rig:
        """Rig from the content of rig.yaml."""
        lidars = {
            name: LidarCalib(name, _matrix(v["T_vehicle_lidar"]),
                             float(v.get("sweep_duration", 0.1)))
            for name, v in data["lidars"].items()
        }
        cameras = {
            name: CameraCalib(
                name,
                v["model"],
                int(v["width"]),
                int(v["height"]),
                list(v["params"]),
                _matrix(v["T_vehicle_cam"]),
                float(v.get("rolling_shutter_line_time", 0.0)),
                float(v.get("time_offset", 0.0)),
            )
            for name, v in (data.get("cameras") or {}).items()
        }
        return cls(
            lidars=lidars,
            cameras=cameras,
            stereo_pairs=[tuple(p) for p in data.get("stereo_pairs", [])],
            T_vehicle_imu=_matrix(data.get("T_vehicle_imu", np.eye(4))),
            wheel_speed_scale=float(data.get("wheel_speed_scale", 1.0)),
        )

    def to_dict(self) -> dict:
        """The content of rig.yaml."""
        return {
            "lidars": {n: l.to_dict() for n, l in self.lidars.items()},
            "cameras": {n: c.to_dict() for n, c in self.cameras.items()},
            "stereo_pairs": [list(p) for p in self.stereo_pairs],
            "T_vehicle_imu": np.asarray(self.T_vehicle_imu).tolist(),
            "wheel_speed_scale": self.wheel_speed_scale,
        }

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Rig:
        return cls.from_dict(yaml.safe_load(pathlib.Path(path).read_text()))

    def save(self, path: str | pathlib.Path) -> None:
        pathlib.Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False))
