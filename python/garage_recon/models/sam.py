"""SAM 2.1 automatic masks as the Segmenter2D (README 5.2; Apache 2.0).

The automatic mask generator prompts a grid of points and keeps the masks
that are confident and stable. Its masks overlap (an object and its parts);
the interface wants one mask id per pixel, so the masks are painted from the
largest to the smallest: parts and small objects (a sign on a column, a slot
number on the floor) stay on top of what contains them.
"""

from __future__ import annotations

import cv2
import numpy as np

from garage_recon.models import interfaces


class Sam2Segmenter:
    """Class-agnostic masks of SAM 2.1."""

    needs_images = True

    def __init__(self, checkpoint: str,
                 config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 device: str = "cuda", points_per_side: int = 32,
                 points_per_batch: int = 128, pred_iou_thresh: float = 0.8,
                 stability_score_thresh: float = 0.92,
                 min_mask_region_area: int = 100, max_side: int = 1024,
                 min_area: int = 64):
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        self.torch = torch
        self.device = device
        model = build_sam2(config, checkpoint, device=device,
                           apply_postprocessing=False)
        self.generator = SAM2AutomaticMaskGenerator(
            model, points_per_side=int(points_per_side),
            points_per_batch=int(points_per_batch),
            pred_iou_thresh=float(pred_iou_thresh),
            stability_score_thresh=float(stability_score_thresh),
            min_mask_region_area=int(min_mask_region_area))
        self.max_side = int(max_side)
        self.min_area = int(min_area)

    def segment(self, frame: interfaces.Frame) -> np.ndarray:
        """(H, W) int32 mask ids, -1 where no mask."""
        image = frame.image
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = (cv2.resize(image, (round(w * scale), round(h * scale)),
                            interpolation=cv2.INTER_AREA)
                 if scale < 1.0 else image)
        with self.torch.inference_mode(), self.torch.autocast(
                "cuda", dtype=self.torch.bfloat16):
            masks = self.generator.generate(np.ascontiguousarray(small))
        ids = np.full(small.shape[:2], -1, np.int32)
        for k, m in enumerate(sorted(masks, key=lambda m: -m["area"])):
            ids[m["segmentation"]] = k
        # Compact ids; masks mostly covered by smaller ones are dropped.
        kept, counts = np.unique(ids[ids >= 0], return_counts=True)
        kept = kept[counts * (1.0 / scale) ** 2 >= self.min_area]
        lut = np.full(len(masks) + 1, -1, np.int32)
        lut[kept] = np.arange(len(kept), dtype=np.int32)
        ids = lut[ids]  # -1 maps to lut[-1] = -1.
        if scale < 1.0:  # Nearest neighbour back to the image size.
            sh, sw = ids.shape
            rows = np.minimum(((np.arange(h) + 0.5) * sh / h).astype(int), sh - 1)
            cols = np.minimum(((np.arange(w) + 0.5) * sw / w).astype(int), sw - 1)
            ids = ids[rows[:, None], cols[None, :]]
        return ids
