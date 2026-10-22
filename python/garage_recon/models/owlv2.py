"""OWLv2 as the Detector2D (README 5.2; Apache 2.0).

Open-vocabulary detection: boxes of what the text queries name. On garage
images it finds lights, columns, cars, people, signs and fire equipment at
about 0.05 s per image (FusionPortableV2), where zero-shot classes of whole
image masks fail (README 5.2). The image is padded to a square by the model,
so boxes come back relative to the padded side.
"""

from __future__ import annotations

import numpy as np

from garage_recon.models import interfaces


class Owlv2Detector:
    """Boxes, query index and score per detection; per-query NMS."""

    needs_images = True

    def __init__(self, model: str = "google/owlv2-large-patch14-ensemble",
                 device: str = "cuda", threshold: float = 0.2,
                 nms: float = 0.5):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.torch = torch
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model, dtype=torch.float16).to(device).eval()
        self.threshold = float(threshold)
        self.nms = float(nms)

    def detect(self, frame: interfaces.Frame,
               queries: list[str]) -> list[interfaces.Detection]:
        torch = self.torch
        from torchvision.ops import batched_nms
        h, w = frame.image.shape[:2]
        inputs = self.processor(text=[list(queries)], images=frame.image,
                                return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        side = max(h, w)
        result = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=self.threshold,
            target_sizes=torch.tensor([[side, side]], device=self.device))[0]
        boxes = result["boxes"].float()
        scores = result["scores"].float()
        labels = result["labels"]
        if len(boxes) == 0:
            return []
        keep = batched_nms(boxes, scores, labels, self.nms).cpu().numpy()
        boxes = boxes.cpu().numpy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return [interfaces.Detection(boxes[k], int(labels[k]),
                                     float(scores[k])) for k in keep]
