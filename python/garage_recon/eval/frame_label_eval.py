"""Hand-labelled real frames against model labels (LabelMe JSON polygons).

Per class: object precision / recall (a prediction matches an unmatched
label of its class at IoU >= `match_iou`, best IoU first) and pixel IoU over
all frames. Used for the objects2d stage on real garage images (README 5.2),
where the labels are corrected model pre-annotations.

usage: python -m garage_recon.eval.frame_label_eval LABEL_DIR PRED_DIR [--iou 0.5]
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import cv2
import numpy as np


def load_labelme(path: pathlib.Path) -> tuple[tuple[int, int], list[tuple[str, np.ndarray]]]:
    """((H, W), [(label, (N, 2) polygon)])."""
    d = json.loads(path.read_text())
    shapes = [(s["label"], np.asarray(s["points"], float)) for s in d.get("shapes", [])
              if s.get("shape_type", "polygon") in ("polygon", "rectangle")
              and len(s["points"]) >= 2]
    out = []
    for label, points in shapes:
        if len(points) == 2:  # Rectangle: two corners.
            (x0, y0), (x1, y1) = points
            points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        out.append((label, points))
    return (int(d["imageHeight"]), int(d["imageWidth"])), out


def _mask(shape, polygon) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(polygon).astype(np.int32)], 1)
    return m > 0


def evaluate(label_dir, pred_dir, match_iou: float = 0.5) -> dict:
    """{class: {labels, predictions, matched, precision, recall, pixel_iou}}
    over the frames that have both a label and a prediction file."""
    label_dir, pred_dir = pathlib.Path(label_dir), pathlib.Path(pred_dir)
    counts = collections.defaultdict(lambda: collections.Counter())
    for label_path in sorted(label_dir.glob("*.json")):
        pred_path = pred_dir / label_path.name
        if not pred_path.exists():
            continue
        shape, labels = load_labelme(label_path)
        _, preds = load_labelme(pred_path)
        for name in {n for n, _ in labels} | {n for n, _ in preds}:
            gt = [_mask(shape, p) for n, p in labels if n == name]
            pr = [_mask(shape, p) for n, p in preds if n == name]
            c = counts[name]
            c["labels"] += len(gt)
            c["predictions"] += len(pr)
            pairs = sorted(((np.logical_and(g, p).sum() / max(1, np.logical_or(g, p).sum()), i, j)
                            for i, g in enumerate(gt) for j, p in enumerate(pr)), reverse=True)
            used_g, used_p = set(), set()
            for iou, i, j in pairs:
                if iou < match_iou:
                    break
                if i in used_g or j in used_p:
                    continue
                used_g.add(i)
                used_p.add(j)
            c["matched"] += len(used_g)
            g_all = np.any(gt, axis=0) if gt else np.zeros(shape, bool)
            p_all = np.any(pr, axis=0) if pr else np.zeros(shape, bool)
            c["intersection"] += int(np.logical_and(g_all, p_all).sum())
            c["union"] += int(np.logical_or(g_all, p_all).sum())
    out = {}
    for name, c in sorted(counts.items()):
        out[name] = {"labels": c["labels"], "predictions": c["predictions"],
                     "matched": c["matched"],
                     "precision": c["matched"] / c["predictions"] if c["predictions"] else None,
                     "recall": c["matched"] / c["labels"] if c["labels"] else None,
                     "pixel_iou": c["intersection"] / c["union"] if c["union"] else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels")
    parser.add_argument("predictions")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    result = evaluate(args.labels, args.predictions, args.iou)
    print(f"{'class':14s} {'labels':>6s} {'preds':>6s} {'match':>6s} {'prec':>6s} {'recall':>6s} {'pix IoU':>7s}")
    for name, r in result.items():
        f = lambda v: f"{v:.2f}" if v is not None else "  -"  # noqa: E731
        print(f"{name:14s} {r['labels']:6d} {r['predictions']:6d} {r['matched']:6d} "
              f"{f(r['precision']):>6s} {f(r['recall']):>6s} {f(r['pixel_iou']):>7s}")


if __name__ == "__main__":
    main()
