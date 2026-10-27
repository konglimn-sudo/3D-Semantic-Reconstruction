"""On-disk trip layout (one directory per drive through the garage).

    <trip>/
      meta.yaml                         trip_id, garage_id, optional initial_pose (4x4,
                                        world<-vehicle from the INS fix at the entrance)
                                        and initial_pose_time (s; default: the first
                                        LiDAR stamp)
      rig.yaml                          see data.schema.Rig
      lidar/<name>/<stamp_ns>.npz       points (N,3) sensor frame, intensity (N,),
                                        time (N,) seconds relative to the stamp (<= 0;
                                        the stamp is the end of the sweep)
      cameras/<name>/<stamp_ns>.jpg     RGB images (any cv2-readable format)
      imu.csv                           t, gx, gy, gz, ax, ay, az   (body = vehicle frame)
      wheel.csv                         t, speed
      gnss.csv                          t, x, y, z, std             (local ENU metres)
      control_points.csv                t, px, py, pz, x, y, z, std (optional: a surveyed
                                        target at p in the LiDAR frame of time t, world
                                        position x, y, z)

Converters from ROS bags or vendor logs only need to produce this layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .. import _core
from .schema import Rig


@dataclass
class LidarFrame:
    index: int
    stamp: float  # seconds
    points: np.ndarray  # (N, 3) float64, sensor frame (raw, not deskewed)
    intensity: np.ndarray  # (N,) float32 in [0, 1]
    time: np.ndarray  # (N,) float64, seconds relative to stamp


def _stamps_in(directory: Path, suffixes: tuple[str, ...]) -> list[int]:
    if not directory.is_dir():
        return []
    return sorted(int(p.stem) for p in directory.iterdir() if p.suffix.lower() in suffixes and p.stem.isdigit())


def _read_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return data if data.size else None


class TripDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not (self.root / "rig.yaml").exists():
            raise FileNotFoundError(f"{self.root} is not a trip directory (rig.yaml missing)")
        self.meta = yaml.safe_load((self.root / "meta.yaml").read_text()) if (self.root / "meta.yaml").exists() else {}
        self.rig = Rig.load(self.root / "rig.yaml")
        self.lidar_name = self.rig.primary_lidar.name
        self._lidar_ns = _stamps_in(self.root / "lidar" / self.lidar_name, (".npz",))
        self._cam_cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ meta
    @property
    def trip_id(self) -> str:
        return str(self.meta.get("trip_id", self.root.name))

    @property
    def garage_id(self) -> str:
        return str(self.meta.get("garage_id", "garage"))

    def initial_pose(self) -> np.ndarray | None:
        T = self.meta.get("initial_pose")
        return None if T is None else np.asarray(T, float).reshape(4, 4)

    def initial_pose_time(self) -> float | None:
        """Stamp of the INS pose; None means the first LiDAR stamp."""
        t = self.meta.get("initial_pose_time")
        return None if t is None else float(t)

    # ----------------------------------------------------------------- lidar
    def __len__(self) -> int:
        return len(self._lidar_ns)

    @property
    def lidar_stamps_ns(self) -> np.ndarray:
        return np.asarray(self._lidar_ns, np.int64)

    @property
    def lidar_stamps(self) -> np.ndarray:
        return self.lidar_stamps_ns.astype(np.float64) * 1e-9

    def load_lidar(self, index: int) -> LidarFrame:
        ns = self._lidar_ns[index]
        with np.load(self.root / "lidar" / self.lidar_name / f"{ns}.npz") as z:
            if "points_q" in z:  # quantised storage: int16 * scale
                pts = z["points_q"].astype(np.float64) * float(z["scale"])
            else:
                pts = z["points"].astype(np.float64)
            inten = z["intensity"].astype(np.float32) if "intensity" in z else np.zeros(len(pts), np.float32)
            t = z["time"].astype(np.float64) if "time" in z else np.zeros(len(pts))
        ok = np.isfinite(pts).all(axis=1)
        return LidarFrame(index, ns * 1e-9, pts[ok], inten[ok], t[ok])

    # --------------------------------------------------------------- cameras
    def camera_stamps_ns(self, cam: str) -> np.ndarray:
        if cam not in self._cam_cache:
            self._cam_cache[cam] = _stamps_in(self.root / "cameras" / cam, (".jpg", ".jpeg", ".png"))
        return np.asarray(self._cam_cache[cam], np.int64)

    def load_image(self, cam: str, stamp_ns: int) -> np.ndarray:
        d = self.root / "cameras" / cam
        for ext in (".jpg", ".png", ".jpeg"):
            p = d / f"{int(stamp_ns)}{ext}"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"no image for {cam} at {stamp_ns}")

    # --------------------------------------------------------- proprioception
    def imu(self) -> np.ndarray | None:
        return _read_csv(self.root / "imu.csv")

    def wheel(self) -> np.ndarray | None:
        return _read_csv(self.root / "wheel.csv")

    def gnss(self) -> np.ndarray | None:
        return _read_csv(self.root / "gnss.csv")

    def control_points(self) -> np.ndarray | None:
        """Surveyed targets seen in the sweeps (control_points.csv)."""
        return _read_csv(self.root / "control_points.csv")

    def imu_samples(self) -> _core.ImuSamples:
        """IMU in the vehicle frame for the C++ core (empty if absent)."""
        imu = self.imu()
        return _core.ImuSamples() if imu is None else _core.ImuSamples(imu, self.rig.T_vehicle_imu[:3, :3])

    def wheel_samples(self) -> _core.WheelSamples:
        wheel = self.wheel()
        return _core.WheelSamples() if wheel is None else _core.WheelSamples(wheel, self.rig.wheel_speed_scale)


class TripWriter:
    """Writes the layout above (used by the synthetic generator and converters)."""

    def __init__(self, root: str | Path, rig: Rig, meta: dict):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        rig.save(self.root / "rig.yaml")
        (self.root / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
        self.rig = rig

    def write_lidar(self, name: str, stamp_ns: int, points, intensity, time, quantize: float = 0.0) -> None:
        """`quantize` > 0 stores coordinates as int16 multiples of that step
        (e.g. 0.005 m covers +-163 m), which keeps synthetic datasets small."""
        d = self.root / "lidar" / name
        d.mkdir(parents=True, exist_ok=True)
        arrays = {"intensity": np.asarray(intensity, np.float16), "time": np.asarray(time, np.float32)}
        if quantize > 0:
            arrays["points_q"] = np.clip(np.round(np.asarray(points) / quantize), -32767, 32767).astype(np.int16)
            arrays["scale"] = np.float64(quantize)
        else:
            arrays["points"] = np.asarray(points, np.float32)
        np.savez_compressed(d / f"{int(stamp_ns)}.npz", **arrays)

    def write_image(self, cam: str, stamp_ns: int, rgb: np.ndarray, quality: int = 92) -> None:
        d = self.root / "cameras" / cam
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{int(stamp_ns)}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, quality])

    def write_csv(self, name: str, header: str, rows: np.ndarray) -> None:
        np.savetxt(self.root / name, rows, delimiter=",", header=header, comments="", fmt="%.9f")
