"""Small SE(3) helpers on 4x4 numpy matrices and TUM trajectory I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def make_T(R: np.ndarray | None = None, t=None) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv(T: np.ndarray) -> np.ndarray:
    """Inverse of one (4,4) or a stack (N,4,4) of rigid transforms."""
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    R = T[..., :3, :3]
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -(Rt @ T[..., :3, 3, None])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    return pts @ T[:3, :3].T + T[:3, 3]


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(pitch: float) -> np.ndarray:
    c, s = np.cos(pitch), np.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def yaw_of(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    return np.arctan2(T[..., 1, 0], T[..., 0, 0])


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def pose_error(Ta: np.ndarray, Tb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translation (m) and rotation (rad) differences between pose stacks."""
    D = inv(Ta) @ Tb
    trans = np.linalg.norm(D[..., :3, 3], axis=-1)
    cos = np.clip((np.trace(D[..., :3, :3], axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    return trans, np.arccos(cos)


def interpolate_poses(times: np.ndarray, poses: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Slerp / lerp interpolation of a pose stack at query times (clamped)."""
    times = np.asarray(times, float)
    poses = np.asarray(poses, float)
    query = np.clip(np.asarray(query, float), times[0], times[-1])
    if len(times) == 1:
        return np.repeat(poses[:1], len(np.atleast_1d(query)), axis=0)
    slerp = Slerp(times, Rotation.from_matrix(poses[:, :3, :3]))
    out = np.zeros((len(np.atleast_1d(query)), 4, 4))
    out[:, :3, :3] = slerp(np.atleast_1d(query)).as_matrix()
    for a in range(3):
        out[:, a, 3] = np.interp(query, times, poses[:, a, 3])
    out[:, 3, 3] = 1.0
    return out


def look_rotation(forward, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose columns are the OpenCV camera axes (x right, y down,
    z forward) expressed in the reference frame."""
    f = np.asarray(forward, float)
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.asarray(up, float))
    if np.linalg.norm(r) < 1e-9:
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], axis=1)


def write_tum(path, stamps, poses) -> None:
    rows = []
    for t, T in zip(stamps, poses):
        q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
        rows.append(f"{t:.9f} {T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}")
    Path(path).write_text("\n".join(rows) + "\n")


def read_tum(path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, ndmin=2)
    poses = np.zeros((len(data), 4, 4))
    poses[:, :3, :3] = Rotation.from_quat(data[:, 4:8]).as_matrix()
    poses[:, :3, 3] = data[:, 1:4]
    poses[:, 3, 3] = 1.0
    return data[:, 0], poses
