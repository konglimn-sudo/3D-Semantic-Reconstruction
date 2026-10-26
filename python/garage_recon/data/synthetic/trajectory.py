"""Smooth vehicle trajectories through the synthetic garage."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ...geometry import rot_y, rot_z
from .scene import Layout


def _chamfer(points: np.ndarray, cut: float) -> np.ndarray:
    out = [points[0]]
    for k in range(1, len(points) - 1):
        a, p, b = points[k - 1], points[k], points[k + 1]
        da, db = a - p, b - p
        la, lb = np.linalg.norm(da), np.linalg.norm(db)
        if la < 1e-6 or lb < 1e-6:
            continue
        c = min(cut, 0.45 * la, 0.45 * lb)
        out += [p + da / la * c, p + db / lb * c]
    out.append(points[-1])
    return np.asarray(out)


class Trajectory:
    """Constant-speed drive along a smoothed 3D polyline (vehicle origin on
    the floor, x forward, y left, z up)."""

    def __init__(self, waypoints: np.ndarray, speed: float, t0: float = 0.0, ds: float = 0.05):
        p = np.asarray(waypoints, float)
        for cut in (2.5, 1.2, 0.6):
            p = _chamfer(p, cut)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        si = np.arange(0.0, s[-1], ds)
        P = np.stack([np.interp(si, s, p[:, a]) for a in range(3)], axis=1)
        w = 10
        tangent = np.zeros_like(P)
        for k in range(len(P)):
            a, b = max(0, k - w), min(len(P) - 1, k + w)
            tangent[k] = P[b] - P[a]
        yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        pitch = np.arctan2(tangent[:, 2], np.linalg.norm(tangent[:, :2], axis=1))
        self.length = float(s[-1])
        self.speed = speed
        self.t0 = t0
        self.times = t0 + si / speed
        self.poses = np.zeros((len(P), 4, 4))
        for k in range(len(P)):
            self.poses[k, :3, :3] = rot_z(yaw[k]) @ rot_y(-pitch[k])
        self.poses[:, :3, 3] = P
        self.poses[:, 3, 3] = 1.0
        self._slerp = Slerp(self.times, Rotation.from_matrix(self.poses[:, :3, :3]))

    @property
    def t_end(self) -> float:
        return float(self.times[-1])

    def poses_at(self, t) -> np.ndarray:
        t = np.clip(np.atleast_1d(np.asarray(t, float)), self.times[0], self.times[-1])
        out = np.zeros((len(t), 4, 4))
        out[:, :3, :3] = self._slerp(t).as_matrix()
        for a in range(3):
            out[:, a, 3] = np.interp(t, self.times, self.poses[:, a, 3])
        out[:, 3, 3] = 1.0
        return out


def build_route(L: Layout) -> np.ndarray:
    """Waypoints covering every aisle in both directions, both ramp directions
    and repeated passes (loop closures within and across levels)."""
    a1, a2 = L.aisles()["1"], L.aisles()["2"]
    y1, y2 = (a1[0] + a1[1]) / 2, (a2[0] + a2[1]) / 2
    xw = L.x_lo - L.cross_width / 2
    xe = L.x_hi + L.cross_width / 2
    z0 = L.level_z(0)
    if not L.has_ramp:
        return np.array([
            [xw + 1.0, y1, z0], [xe, y1, z0], [xe, y2, z0], [xw, y2, z0], [xw, y1, z0], [xe - 4.0, y1, z0],
        ])
    z1 = L.level_z(1)
    rx0, rx1 = L.ramp_x()
    uturn = [[xe + 2.0, y1 + 1.5, z1], [xe + 1.0, y1 + 4.0, z1], [xe - 1.5, y1 + 3.0, z1], [xe - 2.0, y1, z1]]
    return np.array([
        [xw + 1.0, y1, z0], [xe, y1, z0],          # level 0, aisle 1 eastwards
        [xe, y2, z0], [rx1, y2, z0],               # east cross aisle, to the ramp top
        [rx0, y2, z1], [xw, y2, z1],               # ramp down, level 1 aisle 2 westwards
        [xw, y1, z1], [xe, y1, z1],                # level 1 aisle 1 eastwards
        *uturn,                                    # turn around in the east cross aisle
        [xw, y1, z1], [xw, y2, z1], [rx0, y2, z1],  # aisle 1 westwards, back to the ramp bottom
        [rx1, y2, z0], [xe, y2, z0], [xe, y1, z0],  # ramp up, east cross aisle south
        [xw, y1, z0], [xw, y2, z0], [rx0 - 3.0, y2, z0],  # aisle 1 westwards, aisle 2 west part
    ])
