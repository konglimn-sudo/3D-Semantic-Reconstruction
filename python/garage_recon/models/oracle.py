"""Oracle models for synthetic datasets.

They derive their outputs from the ground truth (SyntheticWorld) and stand
in for SAM, SigLIP / CLIP, DINO, RoMa, Depth Anything 3 and OCR, so that the
machinery around the models (calibration, identity association, language
aggregation, label rendering) runs end to end without GPUs or weights. Noise
knobs reproduce the failure modes of real models: class confusions, noisy
embeddings, over-segmentation and pixel noise. All noise is deterministic
per frame and mask.
"""

from __future__ import annotations

import hashlib

import numpy as np

from garage_recon.data.synthetic import ground_truth
from garage_recon.data.synthetic import sensors
from garage_recon.models import interfaces
from garage_recon.semantics import taxonomy

# Classes a vision-language model plausibly mixes up.
CONFUSABLE = {
    "floor": "ramp", "ramp": "floor", "column": "wall", "wall": "column",
    "ceiling": "mep", "mep": "ceiling", "parking_line": "floor",
    "vehicle": "wall",
}
# Classes that look alike (motion is decided geometrically).
SAME_LOOK = {"vehicle_moving": "vehicle"}


def _seed(*parts) -> int:
    text = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(np.linalg.norm(vector), 1e-12)


class OracleSegmenter:
    """Masks of the ground-truth segments (one per object or surface)."""

    needs_images = False  # Reads the ground truth, not the image.

    def __init__(self, world: ground_truth.SyntheticWorld,
                 min_pixels: int = 20, split_probability: float = 0.0,
                 seed: int = 0):
        """Creates the segmenter.

        Args:
            world: Ground truth of the dataset.
            min_pixels: Smaller segments get no mask.
            split_probability: Chance that a mask is over-segmented into a
                left and a right part.
            seed: Noise seed.
        """
        self.world = world
        self.min_pixels = min_pixels
        self.split_probability = split_probability
        self.seed = seed

    def segment(self, frame: interfaces.Frame) -> np.ndarray:
        """Mask-id image of the frame's ground-truth segments (-1: none)."""
        segment = self.world.render(frame.trip_id, frame.camera,
                                    frame.stamp_ns)["segment"]
        rng = np.random.default_rng(
            _seed(self.seed, frame.trip_id, frame.camera, frame.stamp_ns))
        masks = np.full(segment.shape, -1, np.int32)
        ids, inverse, counts = np.unique(segment, return_inverse=True,
                                         return_counts=True)
        inverse = inverse.reshape(segment.shape)
        order = rng.permutation(len(ids))  # Arbitrary mask numbering.
        next_id = len(ids)
        for k, segment_id in enumerate(ids):
            if segment_id < 0 or counts[k] < self.min_pixels:
                continue
            inside = inverse == k
            masks[inside] = order[k]
            if rng.random() < self.split_probability:
                columns = np.nonzero(inside)[1]
                right = inside & (np.arange(segment.shape[1])[None, :]
                                  > np.median(columns))
                masks[right] = next_id
                next_id += 1
        return masks


class OracleLanguageEncoder:
    """A simulated vision-language encoder.

    Every class has a random prototype embedding; text prompts of a class
    (taxonomy.PROMPTS) embed near it, and a mask embeds near the prototype of
    its majority ground-truth class, or of a confusable class with
    probability `confusion`.
    """

    needs_images = False  # Reads the ground truth, not the image.

    def __init__(self, world: ground_truth.SyntheticWorld,
                 dimension: int = 64, mask_noise: float = 0.35,
                 text_noise: float = 0.1, confusion: float = 0.1,
                 seed: int = 0):
        self.world = world
        self._dimension = dimension
        self.mask_noise = mask_noise
        self.text_noise = text_noise
        self.confusion = confusion
        self.seed = seed
        rng = np.random.default_rng(_seed("prototypes", seed))
        self.prototypes = np.stack([
            _unit(rng.normal(size=dimension))
            for _ in range(taxonomy.NUM_CLASSES)])
        self.prompt_class = {prompt: taxonomy.cid(name)
                             for name, prompts in taxonomy.PROMPTS.items()
                             for prompt in prompts}

    @property
    def dimension(self) -> int:
        return self._dimension

    def _noise(self, scale: float, *parts) -> np.ndarray:
        rng = np.random.default_rng(_seed(self.seed, *parts))
        return rng.normal(0.0, scale / np.sqrt(self._dimension),
                          self._dimension)

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """(P, D) embeddings: near the prototype of the prompt's class."""
        out = []
        for prompt in prompts:
            if prompt in self.prompt_class:
                base = self.prototypes[self.prompt_class[prompt]]
            else:  # Unknown vocabulary: unrelated to every class.
                base = _unit(self._noise(1.0, "unknown", prompt))
            out.append(_unit(base + self._noise(self.text_noise, "text",
                                                prompt)))
        return np.stack(out) if out else np.zeros((0, self._dimension))

    def mask_class(self, frame: interfaces.Frame, masks: np.ndarray,
                   mask_id: int) -> int:
        """Ground-truth class the oracle sees for a mask (after confusion)."""
        label = self.world.render(frame.trip_id, frame.camera,
                                  frame.stamp_ns)["label"]
        values = label[masks == mask_id]
        values = values[values > 0]
        if len(values) == 0:
            return taxonomy.UNLABELED
        name = taxonomy.BY_ID[int(np.bincount(values).argmax())].name
        name = SAME_LOOK.get(name, name)
        rng = np.random.default_rng(_seed(self.seed, "confusion",
                                          frame.trip_id, frame.camera,
                                          frame.stamp_ns, mask_id))
        if name in CONFUSABLE and rng.random() < self.confusion:
            name = CONFUSABLE[name]
        return taxonomy.cid(name)

    def encode_masks(self, frame: interfaces.Frame,
                     masks: np.ndarray) -> dict[int, np.ndarray]:
        """Embedding of every mask near the prototype of its class."""
        out = {}
        for mask_id in np.unique(masks[masks >= 0]):
            cls = self.mask_class(frame, masks, int(mask_id))
            if cls == taxonomy.UNLABELED:
                continue
            noise = self._noise(self.mask_noise, "mask", frame.trip_id,
                                frame.camera, frame.stamp_ns, int(mask_id))
            out[int(mask_id)] = _unit(self.prototypes[cls] + noise)
        return out


class OracleMatcher:
    """Exact correspondences from ground-truth depth and poses, plus noise."""

    needs_images = False  # Reads the ground truth, not the image.

    def __init__(self, world: ground_truth.SyntheticWorld, grid_step: int = 8,
                 pixel_noise: float = 0.3, max_matches: int = 400,
                 depth_tolerance: float = 0.02, seed: int = 0):
        self.world = world
        self.grid_step = grid_step
        self.pixel_noise = pixel_noise
        self.max_matches = max_matches
        self.depth_tolerance = depth_tolerance
        self.seed = seed

    def _world_points(self, frame: interfaces.Frame, uv: np.ndarray,
                      depth: np.ndarray) -> np.ndarray:
        calib = self.world.camera(frame.camera)
        rays, ok = calib.make_model().unproject(uv)
        rows = self.world.row_poses(frame.trip_id, frame.camera,
                                    frame.stamp_ns)
        poses = rows[np.clip(uv[:, 1].astype(int), 0, len(rows) - 1)]
        local = rays * depth[:, None]
        points = (np.einsum("nij,nj->ni", poses[:, :3, :3], local)
                  + poses[:, :3, 3])
        points[~ok] = np.nan
        return points

    def query_points(self, frame: interfaces.Frame) -> np.ndarray:
        """Grid pixels of a frame where the ground truth sees a surface."""
        depth = self.world.render(frame.trip_id, frame.camera,
                                  frame.stamp_ns)["depth"]
        h, w = depth.shape
        step = self.grid_step
        vv, uu = np.mgrid[step // 2:h:step, step // 2:w:step]
        keep = np.isfinite(depth[vv.ravel(), uu.ravel()])
        return np.column_stack([uu.ravel() + 0.5, vv.ravel() + 0.5])[keep]

    def match(self, frame_a: interfaces.Frame, frame_b: interfaces.Frame,
              points_a: np.ndarray | None = None) -> interfaces.Matches:
        """True correspondences of visible points plus pixel noise.

        Raises:
            ValueError: For frames of different trips.
        """
        if frame_a.trip_id != frame_b.trip_id:
            raise ValueError("the oracle matches frames of one trip only")
        render_a = self.world.render(frame_a.trip_id, frame_a.camera,
                                     frame_a.stamp_ns)
        render_b = self.world.render(frame_b.trip_id, frame_b.camera,
                                     frame_b.stamp_ns)
        queried = points_a is not None
        uv_a = (np.asarray(points_a, float) if queried
                else self.query_points(frame_a))
        h, w = render_a["depth"].shape
        cols = np.clip(uv_a[:, 0].astype(int), 0, w - 1)
        rows = np.clip(uv_a[:, 1].astype(int), 0, h - 1)
        depth_a = render_a["depth"][rows, cols].astype(float)
        index_a = np.nonzero(np.isfinite(depth_a))[0]
        uv_a, depth_a = uv_a[index_a], depth_a[index_a]
        points = self._world_points(frame_a, uv_a, depth_a)
        uv_b = self.world.project(frame_b.trip_id, frame_b.camera,
                                  frame_b.stamp_ns, points)
        hb, wb = render_b["depth"].shape
        inside = (np.isfinite(uv_b).all(axis=1) & (uv_b[:, 0] >= 0)
                  & (uv_b[:, 1] >= 0) & (uv_b[:, 0] < wb) & (uv_b[:, 1] < hb))
        uv_a, uv_b, points = uv_a[inside], uv_b[inside], points[inside]
        index_a = index_a[inside]
        rows_b = self.world.row_poses(frame_b.trip_id, frame_b.camera,
                                      frame_b.stamp_ns)
        row = np.clip(uv_b[:, 1].astype(int), 0, hb - 1)
        expected = np.linalg.norm(points - rows_b[row, :3, 3], axis=1)
        seen = render_b["depth"][row, np.clip(uv_b[:, 0].astype(int), 0,
                                              wb - 1)]
        visible = np.abs(seen - expected) <= self.depth_tolerance * expected
        uv_a, uv_b, index_a = uv_a[visible], uv_b[visible], index_a[visible]
        rng = np.random.default_rng(
            _seed(self.seed, frame_a.trip_id, frame_a.camera,
                  frame_a.stamp_ns, frame_b.camera, frame_b.stamp_ns))
        if len(uv_a) > self.max_matches:
            pick = np.sort(rng.choice(len(uv_a), self.max_matches,
                                      replace=False))
            uv_a, uv_b, index_a = uv_a[pick], uv_b[pick], index_a[pick]
        if not queried:  # Query points are exact by definition.
            uv_a = uv_a + rng.normal(0.0, self.pixel_noise, uv_a.shape)
        uv_b = uv_b + rng.normal(0.0, self.pixel_noise, uv_b.shape)
        return interfaces.Matches(uv_a, uv_b, np.ones(len(uv_a)),
                                  index_a if queried else None)


class OracleDenseFeatures:
    """Patch features: a random unit vector per ground-truth segment."""

    needs_images = False  # Reads the ground truth, not the image.

    def __init__(self, world: ground_truth.SyntheticWorld, dimension: int = 32,
                 stride: int = 8, seed: int = 0):
        self.world = world
        self.dimension = dimension
        self.stride = stride
        self.seed = seed

    def features(self, frame: interfaces.Frame) -> tuple[np.ndarray, int]:
        """Patch features: the vector of the segment at each patch centre."""
        segment = self.world.render(frame.trip_id, frame.camera,
                                    frame.stamp_ns)["segment"]
        centers = segment[self.stride // 2::self.stride,
                          self.stride // 2::self.stride]
        out = np.zeros(centers.shape + (self.dimension,), np.float32)
        for segment_id in np.unique(centers):
            rng = np.random.default_rng(_seed(self.seed, "segment",
                                              int(segment_id)))
            out[centers == segment_id] = _unit(rng.normal(size=self.dimension))
        return out, self.stride


class OracleDepthPrior:
    """Ground-truth range with an optional scale error and noise."""

    needs_images = False  # Reads the ground truth, not the image.

    def __init__(self, world: ground_truth.SyntheticWorld,
                 scale_error: float = 0.0, noise: float = 0.0, seed: int = 0):
        self.world = world
        self.scale_error = scale_error
        self.noise = noise
        self.seed = seed

    def predict(self, frames: list[interfaces.Frame],
                world_from_camera: list[np.ndarray]) -> list[
                    interfaces.DepthMap]:
        """Ground-truth range per frame (poses unused: they are known)."""
        del world_from_camera
        out = []
        for frame in frames:
            depth = self.world.render(frame.trip_id, frame.camera,
                                      frame.stamp_ns)["depth"].astype(float)
            rng = np.random.default_rng(
                _seed(self.seed, frame.trip_id, frame.camera, frame.stamp_ns))
            depth = depth * (1.0 + self.scale_error) * (
                1.0 + rng.normal(0.0, self.noise, depth.shape))
            finite = np.isfinite(depth)
            out.append(interfaces.DepthMap(
                np.where(finite, depth, np.nan).astype(np.float32),
                finite.astype(np.float32)))
        return out


class OracleOcr:
    """Slot numbers of the ground-truth slots whose entrance is in view."""

    needs_images = False  # Reads the ground truth, not the image.

    def __init__(self, world: ground_truth.SyntheticWorld,
                 max_range: float = 8.0):
        self.world = world
        self.max_range = max_range

    def read(self, frame: interfaces.Frame) -> list[interfaces.TextResult]:
        """Slot ids of the slots whose entrance is visible on the floor."""
        slots = self.world.scene.slots
        if not slots:
            return []
        centers = np.array([
            [(s["corners"][0][0] + s["corners"][1][0]) / 2.0,
             (s["corners"][0][1] + s["corners"][1][1]) / 2.0, s["z"] + 0.01]
            for s in slots])
        pose = self.world.row_poses(frame.trip_id, frame.camera,
                                    frame.stamp_ns)
        center = pose[len(pose) // 2, :3, 3]
        near = np.linalg.norm(centers - center, axis=1) < self.max_range
        uv = np.full((len(slots), 2), np.nan)
        if near.any():
            uv[near] = self.world.project(frame.trip_id, frame.camera,
                                          frame.stamp_ns, centers[near])
        render = self.world.render(frame.trip_id, frame.camera, frame.stamp_ns)
        h, w = render["label"].shape
        out = []
        for k in np.nonzero(near)[0]:
            u, v = uv[k]
            if not (np.isfinite(u) and 0 <= u < w and 0 <= v < h):
                continue
            if render["segment"][int(v), int(u)] < 0:
                continue
            floor_classes = (taxonomy.cid("floor"),
                             taxonomy.cid("parking_line"))
            on_floor = render["label"][int(v), int(u)] in floor_classes
            if not on_floor:  # Hidden behind a car or a column.
                continue
            box = np.array([[u - 5, v - 3], [u + 5, v - 3], [u + 5, v + 3],
                            [u - 5, v + 3]])
            out.append(interfaces.TextResult(slots[k]["slot_id"], box, 1.0))
        return out


def segment_is_moving(segment_id: int) -> bool:
    """Whether a ground-truth segment id belongs to a moving car."""
    return (sensors.MOVING_CAR_SEGMENT <= segment_id
            < sensors.MARKING_SEGMENT)
