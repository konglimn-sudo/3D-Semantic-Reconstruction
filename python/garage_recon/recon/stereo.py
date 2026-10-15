"""Near-field depth from the stereo pair (section 4.2): fills what the roof
LiDAR cannot see (wheel stoppers, speed bumps, kerbs, column bases).

`SGBMMatcher` is a dependency-light baseline; production systems plug a
learned matcher (e.g. FoundationStereo, IGEV) behind the same interface.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from ..data.schema import CameraCalib


class StereoMatcher(Protocol):
    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Disparity in pixels (float32, <= 0 where invalid) for rectified RGB images."""


class SGBMMatcher:
    def __init__(self, num_disparities: int = 96, block_size: int = 5):
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disparities, blockSize=block_size,
            P1=8 * 3 * block_size**2, P2=32 * 3 * block_size**2, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def disparity(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        gl = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        return self.sgbm.compute(gl, gr).astype(np.float32) / 16.0


class LearnedStereoMatcher:
    """Adapter slot for a learned matcher (FoundationStereo / IGEV / RAFT-Stereo).

    Load the model in `__init__` and return float disparities in `disparity`.
    Kept as a stub so that the pipeline has no hard deep-learning dependency.
    """

    def __init__(self, checkpoint: str):
        raise NotImplementedError("wire your stereo network here (see recon/stereo.py)")

    def disparity(self, left, right):  # pragma: no cover
        raise NotImplementedError


def stereo_points(left: np.ndarray, right: np.ndarray, cal_l: CameraCalib, cal_r: CameraCalib,
                  matcher: StereoMatcher, max_depth: float = 8.0, stride: int = 2) -> np.ndarray:
    """3D points in the left camera frame from a rectified pinhole pair."""
    if cal_l.model != "pinhole" or cal_r.model != "pinhole":
        raise ValueError("stereo_points expects rectified pinhole cameras")
    fx, fy, cx, cy = cal_l.params[:4]
    baseline = float(np.linalg.norm(cal_l.T_vehicle_cam[:3, 3] - cal_r.T_vehicle_cam[:3, 3]))
    disp = matcher.disparity(left, right)
    v, u = np.mgrid[0:disp.shape[0]:stride, 0:disp.shape[1]:stride]
    d = disp[v, u]
    ok = d > 0.5
    z = fx * baseline / np.where(ok, d, 1.0)
    ok &= z < max_depth
    x = (u[ok] + 0.5 - cx) / fx * z[ok]
    y = (v[ok] + 0.5 - cy) / fy * z[ok]
    return np.stack([x, y, z[ok]], axis=1)
