"""RapidOCR as the OcrReader (README 5.1 numbers; Apache 2.0).

PaddleOCR's PP-OCR detection and recognition models run through ONNX
Runtime (rapidocr_onnxruntime ships the models): Chinese and Latin text,
the slot numbers ("B2-017"), zone and level signs of a garage.
"""

from __future__ import annotations

import numpy as np

from garage_recon.models import interfaces


class RapidOcrReader:
    """Text with its quadrilateral and recognition score."""

    needs_images = True

    def __init__(self, min_score: float = 0.5, max_side: int = 2000):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.min_score = float(min_score)
        self.max_side = int(max_side)

    def read(self, frame: interfaces.Frame) -> list[interfaces.TextResult]:
        """Text found in the frame (RGB image)."""
        import cv2
        image = cv2.cvtColor(frame.image, cv2.COLOR_RGB2BGR)
        h, w = image.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (round(w * scale), round(h * scale)),
                               interpolation=cv2.INTER_AREA)
        result, _ = self.engine(image)
        out = []
        for box, text, score in result or []:
            if float(score) < self.min_score or not str(text).strip():
                continue
            polygon = np.asarray(box, np.float32).reshape(4, 2) / scale
            out.append(interfaces.TextResult(str(text).strip(), polygon,
                                             float(score)))
        return out
