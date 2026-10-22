"""SigLIP 2 as the LanguageEncoder (README 5.2; Apache 2.0, multilingual).

A mask is encoded from its bounding box (with a margin) cut out of the image,
the pixels outside the mask greyed out, as in LangSplat: the box alone would
mostly encode the surroundings of a thin or small object. Text prompts are
encoded as the model was trained (padded to 64 tokens).
"""

from __future__ import annotations

import numpy as np

from garage_recon.models import interfaces


class Siglip2Encoder:
    """Mask and text embeddings of SigLIP 2."""

    needs_images = True

    def __init__(self, model: str = "google/siglip2-so400m-patch14-384",
                 device: str = "cuda", batch: int = 64, padding: float = 0.15,
                 background: int = 128):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(
            model, dtype=torch.float16).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self.batch = int(batch)
        self.padding = float(padding)
        self.background = int(background)
        self._dimension = int(self.encode_text(["a"]).shape[1])

    @property
    def dimension(self) -> int:
        return self._dimension

    def _unit(self, out) -> np.ndarray:
        x = out.pooler_output if hasattr(out, "pooler_output") else out
        x = self.torch.nn.functional.normalize(x.float(), dim=-1)
        return x.cpu().numpy().astype(np.float32)

    def _crop(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rows, cols = np.nonzero(mask)
        y0, y1, x0, x1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        pad = int(round(self.padding * max(y1 - y0, x1 - x0))) + 2
        h, w = mask.shape
        y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
        x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
        crop = image[y0:y1, x0:x1].copy()
        crop[~mask[y0:y1, x0:x1]] = self.background
        return crop

    def encode_masks(self, frame: interfaces.Frame,
                     masks: np.ndarray) -> dict[int, np.ndarray]:
        """Unit-norm embedding of every mask id of `masks`."""
        torch = self.torch
        ids = [int(i) for i in np.unique(masks) if i >= 0]
        out = {}
        for start in range(0, len(ids), self.batch):
            chunk = ids[start:start + self.batch]
            crops = [self._crop(frame.image, masks == i) for i in chunk]
            inputs = self.processor(images=crops, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device, torch.float16)
            with torch.inference_mode():
                emb = self._unit(self.model.get_image_features(
                    pixel_values=pixels))
            out.update(zip(chunk, emb))
        return out

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        """(P, D) unit-norm embeddings of text prompts."""
        torch = self.torch
        inputs = self.processor(text=list(prompts), padding="max_length",
                                max_length=64, return_tensors="pt")
        with torch.inference_mode():
            return self._unit(self.model.get_text_features(
                **{k: v.to(self.device) for k, v in inputs.items()}))
