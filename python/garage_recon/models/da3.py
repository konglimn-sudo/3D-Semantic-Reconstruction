"""Depth Anything 3 as the DepthPrior (README 8.2).

DA3 runs in pose-conditioned mode: the images of a batch go in with our
world <- camera poses and intrinsics, and with align_to_input_ext_scale its
depth takes the scale of our poses. DA3 only takes pinhole intrinsics, so a
fisheye image is cut into virtual pinhole views (yaw / pitch about the
camera), and the view depths are warped back into the fisheye image as range
along the ray. DA3's own pose estimate is kept, aligned to ours by a
similarity, as a diagnostic: a large difference points to a calibration or
synchronisation problem.

Model licences: DA3-SMALL / DA3-BASE / DA3METRIC-LARGE / DA3MONO-LARGE are
Apache 2.0; DA3-LARGE / GIANT / NESTED are CC BY-NC 4.0 (research only).
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import types

import cv2
import numpy as np

from garage_recon import _core
from garage_recon.models import interfaces

# The DA3 checkout used on the GPU server, when it is not installed.
_SOURCE_DIRS = ("/root/autodl-tmp/third_party/Depth-Anything-3/src",)


def _rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """camera <- view: yaw about the camera y axis (right is +), then pitch
    about the view x axis (down is +), OpenCV axes."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    r_yaw = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0],
                      [-np.sin(y), 0, np.cos(y)]])
    r_pitch = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)],
                        [0, np.sin(p), np.cos(p)]])
    return r_yaw @ r_pitch


@dataclasses.dataclass
class VirtualView:
    """A pinhole view of a camera: K, camera <- view rotation and the maps
    between the two images."""

    K: np.ndarray  # (3, 3)
    R_cam_view: np.ndarray  # (3, 3)
    to_camera: np.ndarray  # (h, w, 2) float32 camera pixel of each view pixel
    ray_norm: np.ndarray  # (h, w) |K^-1 [u v 1]|: z-depth -> range


@dataclasses.dataclass
class CameraViews:
    views: list[VirtualView]
    # Per camera pixel: the view it takes its range from (-1: none) and the
    # view pixel (float32) there.
    view_index: np.ndarray  # (H, W) int8
    view_uv: np.ndarray  # (H, W, 2) float32


def _roll(deg: float) -> np.ndarray:
    """Rotation about the optical axis (camera z)."""
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                     [0, 0, 1.0]])


def upright_roll(world_from_camera: np.ndarray) -> int:
    """Roll (deg, a multiple of 90) that turns the image upright: world up
    towards the top of the view. DA3 is trained on upright photos; the
    Hilti Alphasense, mounted upside down, delivers some cameras inverted."""
    up = world_from_camera[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    angle = np.degrees(np.arctan2(up[0], -up[1]))  # View -y along up.
    return int(90 * round(angle / 90.0)) % 360


def make_views(camera: _core.CameraModel, angles, hfov_deg: float,
               size: tuple[int, int], roll_deg: int = 0) -> CameraViews:
    """Virtual pinhole views of `camera` at (yaw, pitch) `angles` (deg),
    rolled by `roll_deg` about the optical axis first.

    An unrolled pinhole camera is used as it is (one view, identity maps).
    """
    if camera.type == "pinhole" and roll_deg % 360 == 0:
        fx, fy, cx, cy = camera.params[:4]
        h, w = camera.height, camera.width
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        norm = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1)
        view = VirtualView(K, np.eye(3), np.stack([u, v], -1), norm)
        return CameraViews([view], np.zeros((h, w), np.int8),
                           np.stack([u, v], -1))
    w, h = size
    f = 0.5 * w / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    rays = np.stack([(u - K[0, 2]) / f, (v - K[1, 2]) / f,
                     np.ones((h, w))], -1).reshape(-1, 3)
    norm = np.linalg.norm(rays, axis=1).reshape(h, w)
    views = []
    for yaw, pitch in angles:
        R = _roll(roll_deg) @ _rotation(yaw, pitch)
        uv, valid, inside = camera.project(np.ascontiguousarray(rays @ R.T))
        uv = np.where((valid & inside)[:, None], uv, -1.0)
        views.append(VirtualView(K, R, uv.astype(np.float32).reshape(h, w, 2),
                                 norm.astype(np.float32)))
    # Back: every camera pixel takes the view whose axis is closest to it.
    H, W = camera.height, camera.width
    cu, cv = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    cam_rays, ok = camera.unproject(
        np.ascontiguousarray(np.column_stack([cu.ravel(), cv.ravel()])))
    best = np.full(H * W, -1, np.int8)
    best_cos = np.full(H * W, -np.inf)
    best_uv = np.full((H * W, 2), -1.0)
    for k, view in enumerate(views):
        r = cam_rays @ view.R_cam_view  # view <- camera, row vectors
        z = r[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pu = K[0, 0] * r[:, 0] / z + K[0, 2]
            pv = K[1, 1] * r[:, 1] / z + K[1, 2]
        inside = ok & (z > 0) & (pu >= 0) & (pu <= w - 1) & (pv >= 0) & (
            pv <= h - 1)
        cos = np.where(inside, z / np.linalg.norm(r, axis=1), -np.inf)
        take = cos > best_cos
        best[take], best_cos[take] = k, cos[take]
        best_uv[take] = np.column_stack([pu, pv])[take]
    return CameraViews(views, best.reshape(H, W),
                       best_uv.astype(np.float32).reshape(H, W, 2))


def warp_to_camera(cv: CameraViews, ranges: list[np.ndarray],
                   confidences: list[np.ndarray]) -> tuple[np.ndarray,
                                                           np.ndarray]:
    """Range and confidence of the views, resampled into the camera image
    (each camera pixel from the view whose axis is closest to its ray)."""
    H, W = cv.view_index.shape
    rng = np.full((H, W), np.nan, np.float32)
    con = np.zeros((H, W), np.float32)
    uv = cv.view_uv
    for k, (view_range, view_conf) in enumerate(zip(ranges, confidences)):
        mask = cv.view_index == k
        if not mask.any():
            continue
        # Nearest for the range (no blending across depth edges).
        sampled = cv2.remap(view_range, uv[..., 0], uv[..., 1],
                            cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=np.nan)
        sampled_conf = cv2.remap(view_conf, uv[..., 0], uv[..., 1],
                                 cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rng[mask] = sampled[mask]
        con[mask] = sampled_conf[mask]
    return rng, con


def _import_da3():
    try:
        from depth_anything_3 import api  # noqa: F401
    except ImportError:
        for d in _SOURCE_DIRS:
            if pathlib.Path(d).is_dir() and d not in sys.path:
                sys.path.insert(0, d)
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.pose_align import align_poses_umeyama
    return DepthAnything3, align_poses_umeyama


def _keep_estimate(align_poses_umeyama):
    """Replacement of DA3's alignment to the input poses that also keeps the
    network's own extrinsics, mapped into the input frame, in prediction.aux.

    DA3 fits the similarity that maps the *input* poses onto its predicted
    ones (R -> r R, c -> s r c + t for camera centres c); its prediction in
    the input frame is therefore R = r^T R_pred, c = r^T (c_pred - t) / s.
    """

    def align(self, extrinsics, intrinsics, prediction,
              align_to_input_ext_scale=True, ransac_view_thresh=10):
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        r, t, scale = align_poses_umeyama(
            prediction.extrinsics, extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh, return_aligned=False,
            random_state=42)
        predicted = np.tile(np.eye(4), (len(prediction.extrinsics), 1, 1))
        predicted[:, :3, :] = np.asarray(prediction.extrinsics)[:, :3, :]
        c2w = np.linalg.inv(predicted)
        mine = np.tile(np.eye(4), (len(c2w), 1, 1))
        mine[:, :3, :3] = r.T @ c2w[:, :3, :3]
        mine[:, :3, 3] = ((c2w[:, :3, 3] - t) @ r) / scale
        prediction.aux = dict(prediction.aux or {},
                              estimated_extrinsics=np.linalg.inv(mine),
                              pose_scale=float(scale))
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        return prediction

    return align


class Da3DepthPrior:
    """DepthPrior on Depth Anything 3 (any-view model, pose-conditioned)."""

    needs_images = True

    def __init__(self, model: str = "depth-anything/DA3-BASE",
                 device: str = "cuda",
                 views=((-28.0, 0.0), (28.0, 0.0)),
                 view_hfov: float = 80.0, view_size=(504, 378),
                 batch_views: int = 96, process_res: int = 504,
                 conf_scale: float = 3.0, log=print):
        import torch
        DepthAnything3, align_poses_umeyama = _import_da3()
        self.torch = torch
        self.device = device
        self.model = DepthAnything3.from_pretrained(model).to(device).eval()
        self.model._align_to_input_extrinsics_intrinsics = types.MethodType(
            _keep_estimate(align_poses_umeyama), self.model)
        self.name = str(model)
        self.angles = [tuple(a) for a in views]
        self.view_hfov = float(view_hfov)
        self.view_size = tuple(int(s) for s in view_size)
        self.batch_views = int(batch_views)
        self.process_res = int(process_res)
        self.conf_scale = float(conf_scale)
        self.log = log
        self._views: dict[tuple, CameraViews] = {}

    def camera_views(self, name: str, camera: _core.CameraModel,
                     roll: int = 0) -> CameraViews:
        key = (name, camera.type, camera.width, camera.height,
               tuple(camera.params), roll)
        if key not in self._views:
            self._views[key] = make_views(camera, self.angles, self.view_hfov,
                                          self.view_size, roll)
        return self._views[key]

    def predict(self, frames: list[interfaces.Frame],
                world_from_camera: list[np.ndarray],
                cameras: dict[str, _core.CameraModel] | None = None
                ) -> list[interfaces.DepthMap]:
        """Range along the rays of every frame's own camera.

        Args:
            frames: Frames with images, in time order (a batch shares
                multi-view attention, so neighbouring frames go together).
            world_from_camera: (4, 4) pose per frame.
            cameras: Camera model per camera name (required).
        """
        if cameras is None:
            raise ValueError("Da3DepthPrior needs the camera models")
        # One entry per virtual view.
        entries = []
        rolls = [upright_roll(pose) for pose in world_from_camera]
        for i, (frame, pose) in enumerate(zip(frames, world_from_camera)):
            cv = self.camera_views(frame.camera, cameras[frame.camera],
                                   rolls[i])
            for k, view in enumerate(cv.views):
                image = cv2.remap(frame.image, view.to_camera[..., 0],
                                  view.to_camera[..., 1], cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                T = np.eye(4)
                T[:3, :3] = view.R_cam_view
                entries.append((i, k, image, np.linalg.inv(pose @ T), view.K))
        depth = [None] * len(entries)
        conf = [None] * len(entries)
        estimate = [None] * len(entries)
        for start in range(0, len(entries), self.batch_views):
            batch = entries[start:start + self.batch_views]
            with self.torch.inference_mode():
                pred = self.model.inference(
                    [e[2] for e in batch],
                    extrinsics=np.stack([e[3] for e in batch]),
                    intrinsics=np.stack([e[4] for e in batch]),
                    align_to_input_ext_scale=True,
                    process_res=self.process_res)
            est = (pred.aux or {}).get("estimated_extrinsics")
            for j in range(len(batch)):
                depth[start + j] = pred.depth[j]
                conf[start + j] = pred.conf[j] if pred.conf is not None else None
                if est is not None:
                    w2c = np.eye(4)
                    w2c[:3, :] = np.asarray(est[j])[:3, :]
                    estimate[start + j] = w2c
        out = []
        for i, (frame, pose) in enumerate(zip(frames, world_from_camera)):
            cv = self.camera_views(frame.camera, cameras[frame.camera],
                                   rolls[i])
            mine = [e for e, entry in enumerate(entries) if entry[0] == i]
            ranges, confs, poses = [], [], []
            for e in mine:
                view = cv.views[entries[e][1]]
                h, w = view.ray_norm.shape
                z = depth[e]
                c = conf[e] if conf[e] is not None else np.full_like(z, 1e3)
                if z.shape != (h, w):
                    z = cv2.resize(z, (w, h), interpolation=cv2.INTER_LINEAR)
                    c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)
                ranges.append((z * view.ray_norm).astype(np.float32))
                confs.append((1.0 - np.exp(-np.maximum(c - 1.0, 0.0)
                                           / self.conf_scale)).astype(np.float32))
                if estimate[e] is not None:
                    # The network's world <- camera, from this view.
                    T = np.eye(4)
                    T[:3, :3] = view.R_cam_view
                    poses.append(np.linalg.inv(estimate[e]) @ np.linalg.inv(T))
            rng, con = warp_to_camera(cv, ranges, confs)
            out.append(interfaces.DepthMap(
                rng, con, pose_estimate=_mean_pose(poses) if poses else None))
        return out


def _mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    """Chordal mean of rotations, mean of translations."""
    R = sum(p[:3, :3] for p in poses)
    u, _, vt = np.linalg.svd(R)
    out = np.eye(4)
    out[:3, :3] = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    out[:3, 3] = np.mean([p[:3, 3] for p in poses], axis=0)
    return out
