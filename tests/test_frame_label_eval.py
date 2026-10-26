"""Object precision / recall and pixel IoU of LabelMe files."""

import json

from garage_recon.eval import frame_label_eval


def _write(path, shapes):
    path.write_text(json.dumps({"imageHeight": 100, "imageWidth": 100, "shapes": [
        {"label": l, "points": p, "shape_type": "polygon"} for l, p in shapes]}))


def test_matches_by_iou_per_class(tmp_path):
    (tmp_path / "gt").mkdir(); (tmp_path / "pr").mkdir()
    square = lambda x, y, s: [[x, y], [x + s, y], [x + s, y + s], [x, y + s]]  # noqa: E731
    _write(tmp_path / "gt" / "a.json", [("light", square(0, 0, 10)), ("light", square(50, 50, 10)),
                                         ("sign", square(20, 20, 10))])
    _write(tmp_path / "pr" / "a.json", [("light", square(1, 1, 10)), ("sign", square(70, 0, 10)),
                                         ("light", square(80, 80, 5))])
    r = frame_label_eval.evaluate(tmp_path / "gt", tmp_path / "pr")
    assert r["light"]["labels"] == 2 and r["light"]["predictions"] == 2 and r["light"]["matched"] == 1
    assert r["light"]["precision"] == 0.5 and r["light"]["recall"] == 0.5
    assert r["sign"]["matched"] == 0 and r["sign"]["pixel_iou"] == 0.0
