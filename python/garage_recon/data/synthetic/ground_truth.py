"""Ground truth of a synthetic dataset, for oracle models and evaluation.

Rebuilds the scene of every trip from gt/ (static geometry, parked and moving
cars, dense trajectory, true camera calibration) and renders exact label,
segment and depth images of any recorded camera frame, with the true rolling
shutter and time offset.
"""

from __future__ import annotations

import collections
import json
import pathlib

import numpy as np

from garage_recon import _core
from garage_recon import geometry
from garage_recon.data import schema
from garage_recon.data.synthetic import scene as scene_lib
from garage_recon.data.synthetic import sensors
from garage_recon.semantics import taxonomy

# GT class of a static box surface by box kind, for the slab-like kinds by
# face (top, bottom, side).
_FACE_CLASSES = {
    "slab": ("floor", "ceiling", "wall"),
    "roof": ("floor", "ceiling", "wall"),
    "ramp": ("ramp", "ceiling", "wall"),
}
_KIND_CLASSES = {"wall": "wall", "parapet": "wall", "column": "column",
                 "duct": "mep", "car": "vehicle"}


class _ExactTrajectory:
    """The simulator's interpolation (lerp + slerp) of dense knots."""

    def __init__(self, times: np.ndarray, poses: np.ndarray):
        self.times = times
        self.poses = poses

    def poses_at(self, t) -> np.ndarray:
        return geometry.interpolate_poses(self.times, self.poses,
                                          np.atleast_1d(t))


class SyntheticWorld:
    """Ground truth of one synthetic dataset (the gt/ directory)."""

    def __init__(self, gt_dir: str | pathlib.Path, render_cache: int = 64):
        self.gt_dir = pathlib.Path(gt_dir)
        scene_json = json.loads((self.gt_dir / "scene.json").read_text())
        self.layout = scene_lib.Layout(**scene_json["layout"])
        self.scene = scene_lib.build_static_scene(self.layout)
        calibration_path = self.gt_dir / "calibration.json"
        self.calibration = (json.loads(calibration_path.read_text())
                            if calibration_path.exists() else None)
        self._casters: dict[str, sensors.SceneRaycaster] = {}
        self._knot_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._core_trajectories: dict[str, _core.ContinuousTrajectory] = {}
        self._renderers: dict[str, sensors.CameraRenderer] = {}
        self._render_cache: collections.OrderedDict = collections.OrderedDict()
        self._render_cache_size = render_cache

    # ------------------------------------------------------------- trips
    def trip_names(self) -> list[str]:
        return sorted(p.name for p in self.gt_dir.glob("trip_*") if p.is_dir())

    def objects(self, trip: str) -> dict:
        return json.loads((self.gt_dir / trip / "objects.json").read_text())

    def _knots(self, trip: str) -> tuple[np.ndarray, np.ndarray]:
        if trip not in self._knot_cache:
            with np.load(self.gt_dir / trip / "trajectory.npz") as z:
                self._knot_cache[trip] = (z["times"], z["poses"])
        return self._knot_cache[trip]

    def trajectory(self, trip: str) -> _ExactTrajectory:
        """World <- vehicle as the simulator interpolates it."""
        return _ExactTrajectory(*self._knots(trip))

    def core_trajectory(self, trip: str) -> _core.ContinuousTrajectory:
        """The same trajectory for the C++ projection functions."""
        if trip not in self._core_trajectories:
            times, poses = self._knots(trip)
            keep = np.concatenate([[True], np.diff(times) > 1e-9])
            self._core_trajectories[trip] = _core.ContinuousTrajectory(
                times[keep], poses[keep])
        return self._core_trajectories[trip]

    def imu_bias(self, trip: str) -> np.ndarray:
        if self.calibration is None:
            return np.zeros(6)
        return np.asarray(self.calibration["imu_bias"][trip], float)

    # ------------------------------------------------------------ cameras
    def camera(self, name: str) -> schema.CameraCalib:
        """True calibration of a camera."""
        if self.calibration is None:
            raise ValueError("gt/calibration.json is missing")
        data = self.calibration["cameras"][name]
        return schema.CameraCalib(
            name, data["model"], int(data["width"]), int(data["height"]),
            list(data["params"]), np.asarray(data["T_vehicle_cam"], float),
            float(data.get("rolling_shutter_line_time", 0.0)),
            float(data.get("time_offset", 0.0)))

    def exposure(self, trip: str, camera: str, stamp_ns: int) -> np.ndarray:
        """True world <- camera of the first and last rows, (2, 4, 4)."""
        calib = self.camera(camera)
        first = stamp_ns * 1e-9 + calib.time_offset
        last = first + (calib.height - 1) * calib.rolling_shutter_line_time
        vehicle = self.trajectory(trip).poses_at([first, last])
        return vehicle @ calib.T_vehicle_cam

    def row_poses(self, trip: str, camera: str, stamp_ns: int) -> np.ndarray:
        """True world <- camera of every row, (H, 4, 4)."""
        calib = self.camera(camera)
        first = stamp_ns * 1e-9 + calib.time_offset
        times = first + calib.rolling_shutter_line_time * np.arange(
            calib.height)
        return self.trajectory(trip).poses_at(times) @ calib.T_vehicle_cam

    def project(self, trip: str, camera: str, stamp_ns: int,
                points: np.ndarray) -> np.ndarray:
        """True pixels (N, 2) of world points (NaN where none)."""
        calib = self.camera(camera)
        return _core.project_with_calibration(
            calib.make_model(), calib.T_vehicle_cam, calib.time_offset,
            calib.rolling_shutter_line_time, self.core_trajectory(trip),
            stamp_ns * 1e-9, np.asarray(points, float))

    def caster(self, trip: str) -> sensors.SceneRaycaster:
        """The scene of a trip: static garage, its parked and moving cars."""
        if trip not in self._casters:
            objects = self.objects(trip)
            cars = []
            for car in objects["parked"]:
                box = scene_lib.Box(
                    np.asarray(car["center"], float),
                    np.asarray(car["size"], float),
                    geometry.rot_z(car["yaw"]), taxonomy.cid("vehicle"), "car",
                    int(car["level"]), int(car["instance"]), car["key"])
                cars.append(box)
            movers = [scene_lib.MovingCar(
                np.asarray(m["start"], float), np.asarray(m["velocity"], float),
                float(m["t0"]), float(m["t1"]), np.asarray(m["size"], float),
                int(m["level"]), int(m["instance"])) for m in objects["moving"]]
            self._casters[trip] = sensors.SceneRaycaster(self.scene, cars,
                                                         movers)
        return self._casters[trip]

    def render(self, trip: str, camera: str, stamp_ns: int) -> dict:
        """Exact label, segment and depth images of a recorded frame.

        Returns:
            Dict of label (H, W) class ids, segment (H, W) GT segment ids
            (-1 empty) and depth (H, W) range (inf empty).
        """
        key = (trip, camera, int(stamp_ns))
        if key in self._render_cache:
            self._render_cache.move_to_end(key)
            return self._render_cache[key]
        calib = self.camera(camera)
        if camera not in self._renderers:
            self._renderers[camera] = sensors.CameraRenderer(calib)
        poses = self.row_poses(trip, camera, stamp_ns)
        mid = (stamp_ns * 1e-9 + calib.time_offset
               + 0.5 * calib.height * calib.rolling_shutter_line_time)
        frame = self._renderers[camera].render(self.caster(trip), poses, mid,
                                               np.random.default_rng(0))
        out = {"label": frame["label"], "segment": frame["segment"],
               "depth": frame["depth"]}
        self._render_cache[key] = out
        if len(self._render_cache) > self._render_cache_size:
            self._render_cache.popitem(last=False)
        return out

    # ------------------------------------------------------------- points
    def point_truth(self, xyz: np.ndarray,
                    max_distance: float = 0.15) -> dict:
        """GT class and segment of world points (static scene, parked cars).

        A point takes the surface of the nearest box (any trip's parked cars
        count); slab and ramp faces are told apart, floor points on a painted
        line or arrow take its class. Points farther than `max_distance` from any
        surface get class 0 and segment -1.

        Returns:
            Dict of cls (N,) and segment (N,) arrays.
        """
        boxes = [{"center": b.center, "size": b.size, "R": b.R, "kind": b.kind,
                  "segment": 8 * i, "level": b.level}
                 for i, b in enumerate(self.scene.boxes)]
        keys: dict[str, int] = {}
        for trip in self.trip_names():
            for car in self.objects(trip)["parked"]:
                key_id = keys.setdefault(car["key"], len(keys))
                boxes.append({"center": np.asarray(car["center"], float),
                              "size": np.asarray(car["size"], float),
                              "R": geometry.rot_z(car["yaw"]), "kind": "car",
                              "segment": sensors.PARKED_CAR_SEGMENT + key_id,
                              "level": int(car["level"])})
        xyz = np.asarray(xyz, float)
        distance, nearest, local = _nearest_box(xyz, boxes)
        cls = np.zeros(len(xyz), np.int64)
        segment = np.full(len(xyz), -1, np.int64)
        level = np.full(len(xyz), -1)
        for k, box in enumerate(boxes):
            sel = (nearest == k) & (distance <= max_distance)
            if not sel.any():
                continue
            level[sel] = box["level"]
            if box["kind"] in _FACE_CLASSES:
                top, bottom, side = _FACE_CLASSES[box["kind"]]
                half = np.asarray(box["size"]) / 2.0
                ratio = np.abs(local[sel]) / half
                vertical = np.argmax(ratio, axis=1) == 2
                face = np.where(vertical & (local[sel][:, 2] > 0), 1,
                                np.where(vertical, 2, 3))
                names = np.array([top, bottom, side])[face - 1]
                cls[sel] = [taxonomy.cid(n) for n in names]
                segment[sel] = box["segment"] + face
            else:
                cls[sel] = taxonomy.cid(_KIND_CLASSES[box["kind"]])
                segment[sel] = box["segment"]
        floor = np.where(cls == taxonomy.cid("floor"))[0]
        if len(floor):
            caster = sensors.SceneRaycaster(self.scene, [], [])
            line = caster.marking_index(xyz[floor], level[floor])
            on = line >= 0
            cls[floor[on]] = caster.paint_cls[line[on]]
            segment[floor[on]] = sensors.MARKING_SEGMENT + line[on]
        return {"cls": cls, "segment": segment}


def _nearest_box(points: np.ndarray, boxes: list[dict],
                 chunk: int = 200000):
    """Distance to, index of and local coordinates in the nearest box."""
    best = np.full(len(points), np.inf)
    arg = np.full(len(points), -1)
    local_best = np.zeros((len(points), 3))
    for start in range(0, len(points), chunk):
        p = points[start:start + chunk]
        part = slice(start, start + len(p))
        for k, box in enumerate(boxes):
            rotation = np.asarray(box["R"], float)
            local = (p - np.asarray(box["center"], float)) @ rotation
            q = np.abs(local) - np.asarray(box["size"], float) / 2.0
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            d = np.abs(outside + inside)
            better = d < best[part]
            best[part][better] = d[better]
            arg[part][better] = k
            local_best[part][better] = local[better]
    return best, arg, local_best
