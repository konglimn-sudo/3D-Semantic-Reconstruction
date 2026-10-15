"""Acceptance report against the handheld ground truth (plan section 7).

After registration (registration.py) the vehicle map is scored in the
handheld frame:

    surface      distance of the map structure points to the handheld
                 surface (point to local plane), per level and by surface
                 orientation; vertical bias of horizontal surfaces; per
                 source tag (L / G / P, plan section 3.5) and, if the map
                 carries sigma, how often |error| <= 2 sigma
    completeness handheld points reproduced by the map (within 5 / 10 cm),
                 among those the vehicle observed at all, by height band
    check points columns measured in the map (registration.py) and surveyed
                 check points: 3D RMSE, CE90 (horizontal), LE90 (vertical)
    vectors      slot corners, detected column centres, marking centre lines
    verdict      the thresholds of the plan (initial values, to be tuned on
                 garages with handheld ground truth)

Outputs (work/acceptance/): report.json, report.md, deviation_level_<i>.png.
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
import shapely
from scipy import spatial

from garage_recon import _core
from garage_recon.acceptance import handheld as handheld_lib
from garage_recon.acceptance import registration as registration_lib
from garage_recon.recon import static_map
from garage_recon.semantics import taxonomy


def error_stats(errors: np.ndarray) -> dict:
    """RMSE / mean / p90 / max of error magnitudes (rows or scalars)."""
    errors = np.asarray(errors, float)
    if errors.ndim == 2:
        errors = np.linalg.norm(errors, axis=1)
    errors = np.abs(errors[np.isfinite(errors)])
    if not len(errors):
        return {"count": 0}
    return {"count": int(len(errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean": float(errors.mean()),
            "p90": float(np.percentile(errors, 90)),
            "max": float(errors.max())}


def checkpoint_stats(errors: np.ndarray) -> dict:
    """3D check point statistics: RMSE (3D / horizontal / vertical), CE90,
    LE90 and the mean error vector (bias)."""
    errors = np.asarray(errors, float).reshape(-1, 3)
    if not len(errors):
        return {"count": 0}
    horizontal = np.linalg.norm(errors[:, :2], axis=1)
    vertical = np.abs(errors[:, 2])
    return {"count": int(len(errors)),
            "rmse_3d": float(np.sqrt(np.mean(np.sum(errors**2, axis=1)))),
            "rmse_horizontal": float(np.sqrt(np.mean(horizontal**2))),
            "rmse_vertical": float(np.sqrt(np.mean(vertical**2))),
            "ce90": float(np.percentile(horizontal, 90)),
            "le90": float(np.percentile(vertical, 90)),
            "max_3d": float(np.linalg.norm(errors, axis=1).max()),
            "bias": errors.mean(axis=0).tolist()}


def _distance_stats(signed: np.ndarray) -> dict:
    absolute = np.abs(signed)
    return {"points": int(len(signed)),
            "median": float(np.median(absolute)),
            "rmse": float(np.sqrt(np.mean(signed**2))),
            "p90": float(np.percentile(absolute, 90)),
            "p95": float(np.percentile(absolute, 95)),
            "within_2cm": float((absolute < 0.02).mean()),
            "within_5cm": float((absolute < 0.05).mean())}


def level_mapping(map_levels: list[dict],
                  truth: handheld_lib.HandheldGroundTruth,
                  gt_from_map: np.ndarray) -> dict[int, int]:
    """Map level id -> handheld level id (nearest floor height)."""
    truth_z = truth.level_z()
    out = {}
    for level in map_levels:
        z = gt_from_map[2, 3] + gt_from_map[2, 2] * level["z"]
        if truth_z:
            out[int(level["id"])] = min(truth_z,
                                        key=lambda k: abs(truth_z[k] - z))
        else:
            out[int(level["id"])] = int(level["id"])
    return out


def surface_report(points: np.ndarray, levels: np.ndarray,
                   sigma: np.ndarray | None, source: np.ndarray,
                   truth: handheld_lib.HandheldGroundTruth, cfg
                   ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Map structure points (handheld frame) against the handheld surface.

    Returns:
        (report, signed distances (NaN where unmatched), surface normals).
    """
    signed, _, normals = _core.surface_distances(
        points, truth.xyz, float(cfg.surface_radius),
        float(cfg.surface_max_distance))
    matched = np.isfinite(signed)
    report = {"points": int(len(points)),
              "matched_fraction": float(matched.mean()) if len(points) else 0}
    if not matched.any():
        return report, signed, normals
    report["all"] = _distance_stats(signed[matched])
    horizontal = matched & (normals[:, 2] > 0.9)
    vertical = matched & (np.abs(normals[:, 2]) < 0.3)
    if horizontal.any():
        report["horizontal"] = _distance_stats(signed[horizontal])
        report["horizontal"]["bias"] = float(np.median(signed[horizontal]))
    if vertical.any():
        report["vertical"] = _distance_stats(signed[vertical])
    report["levels"] = {}
    for level in np.unique(levels[matched]):
        mine = matched & (levels == level)
        stats = _distance_stats(signed[mine])
        flat = mine & horizontal
        if flat.any():
            stats["vertical_bias"] = float(np.median(signed[flat]))
        report["levels"][str(int(level))] = stats
    report["sources"] = {
        str(tag): _distance_stats(signed[matched & (source == tag)])
        for tag in np.unique(source[matched])}
    if sigma is not None:
        ok = matched & np.isfinite(sigma) & (sigma > 0)
        if ok.any():
            report["sigma"] = {
                "median": float(np.median(sigma[ok])),
                "within_1_sigma": float(
                    (np.abs(signed[ok]) <= sigma[ok]).mean()),
                "within_2_sigma": float(
                    (np.abs(signed[ok]) <= 2 * sigma[ok]).mean())}
    return report, signed, normals


def completeness_report(map_points: np.ndarray,
                        truth: handheld_lib.HandheldGroundTruth,
                        sample: int = 400000, seed: int = 0) -> dict:
    """Handheld points reproduced by the map, by height band.

    Only handheld points the vehicle observed at all (a map point within
    0.5 m) count. The bands separate what the parking map needs (the
    floor) from what a vehicle LiDAR sees only at grazing angles (the
    ceiling) and from objects that change between sessions (middle band:
    parked cars, but also walls and columns).
    """
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(truth.xyz), min(sample, len(truth.xyz)),
                      replace=False)
    points = truth.xyz[pick]
    tree = spatial.cKDTree(map_points)
    distance, _ = tree.query(points, distance_upper_bound=0.5)
    observed = np.isfinite(distance)
    floors = np.sort(np.array(list(truth.level_z().values())))
    if len(floors):
        # Height above the floor of the level the point belongs to.
        below = np.searchsorted(floors, points[:, 2] + 0.05) - 1
        height = points[:, 2] - floors[np.clip(below, 0, len(floors) - 1)]
    else:
        height = np.zeros(len(points))
    bands = {"floor": height < 0.1, "middle": (height >= 0.1) & (height < 2.3),
             "top": height >= 2.3}
    out = {"sampled": int(len(pick)),
           "observed_fraction": float(observed.mean())}
    for name, band in bands.items():
        seen = band & observed
        out[name] = {
            "points": int(band.sum()),
            "observed_fraction": float(seen.sum() / max(band.sum(), 1)),
            "within_5cm": float((distance[seen] < 0.05).mean())
            if seen.any() else 0.0,
            "within_10cm": float((distance[seen] < 0.10).mean())
            if seen.any() else 0.0}
    return out


def column_checkpoints(measurements: list[registration_lib.ColumnMeasurement],
                       control_ids: list[str]) -> dict:
    """Column check points (columns not used as control points)."""
    rows, errors = [], []
    for m in measurements:
        role = "control" if m.column_id in control_ids else "check"
        row = {"id": m.column_id, "level": m.level, "role": role,
               "faces": m.faces}
        if m.measured is not None:
            error = m.measured - m.truth
            row.update({"dx": float(error[0]), "dy": float(error[1]),
                        "dz": float(error[2])})
            if role == "check":
                errors.append(error)
        rows.append(row)
    stats = checkpoint_stats(np.array(errors))
    stats["measured"] = sum(m.measured is not None for m in measurements)
    stats["columns"] = len(measurements)
    by_level = {}
    for level in sorted({m.level for m in measurements}):
        level_errors = [m.measured - m.truth for m in measurements
                        if m.level == level and m.measured is not None
                        and m.column_id not in control_ids]
        by_level[str(level)] = checkpoint_stats(np.array(level_errors))
    return {"stats": stats, "levels": by_level, "points": rows}


def surveyed_checkpoints(truth: handheld_lib.HandheldGroundTruth,
                         gt_from_map: np.ndarray) -> dict:
    """Surveyed check points identified in the map (points.csv)."""
    checks = [p for p in truth.points
              if p.role == "check" and p.map_xyz is not None]
    if not checks:
        return {"count": 0}
    measured = registration_lib.transform(np.array([p.map_xyz for p in checks]),
                                          gt_from_map)
    errors = measured - np.array([p.xyz for p in checks])
    out = checkpoint_stats(errors)
    out["points"] = [{"id": p.point_id, "dx": float(e[0]), "dy": float(e[1]),
                      "dz": float(e[2])} for p, e in zip(checks, errors)]
    return out


def _to_truth_2d(points_2d: np.ndarray, z: float,
                 gt_from_map: np.ndarray) -> np.ndarray:
    points = np.column_stack([points_2d, np.full(len(points_2d), z)])
    return registration_lib.transform(points, gt_from_map)


def slot_report(slots: list[dict], truth: handheld_lib.HandheldGroundTruth,
                gt_from_map: np.ndarray, levels: dict[int, int],
                iou_threshold: float = 0.5) -> dict:
    """Slot corners of the map against the digitised slots."""
    predicted = []
    for slot in slots:
        corners = _to_truth_2d(np.asarray(slot["corners"], float),
                               float(slot.get("z", 0.0)), gt_from_map)
        predicted.append((levels.get(int(slot["level"]), -1),
                          shapely.Polygon(corners[:, :2]), corners))
    errors, vertical, used, matched = [], [], set(), 0
    for slot in truth.features["slots"]:
        corners = np.asarray(slot["corners"], float)
        polygon = shapely.Polygon(corners[:, :2])
        best, best_index = 0.0, -1
        for i, (level, candidate, _) in enumerate(predicted):
            if i in used or level != int(slot["level"]):
                continue
            iou = (polygon.intersection(candidate).area
                   / max(polygon.union(candidate).area, 1e-9))
            if iou > best:
                best, best_index = iou, i
        if best < iou_threshold:
            continue
        used.add(best_index)
        matched += 1
        mine = predicted[best_index][2]
        # Corner order may differ: pair each true corner with the nearest.
        for corner in corners:
            k = int(np.argmin(np.linalg.norm(mine[:, :2] - corner[:2], axis=1)))
            errors.append(mine[k, :2] - corner[:2])
            vertical.append(mine[k, 2] - corner[2])
    out = {"truth": len(truth.features["slots"]), "predicted": len(slots),
           "matched": matched}
    out["corners"] = error_stats(np.array(errors).reshape(-1, 2))
    if errors:
        horizontal = np.linalg.norm(errors, axis=1)
        out["corners"]["ce90"] = float(np.percentile(horizontal, 90))
        out["corners_vertical"] = error_stats(np.array(vertical))
    return out


def column_vector_report(columns: list[dict],
                         truth: handheld_lib.HandheldGroundTruth,
                         gt_from_map: np.ndarray, levels: dict[int, int],
                         gate: float = 1.0) -> dict:
    """Column centres found by the semantic pipeline."""
    errors, used = [], set()
    centers = [(levels.get(int(c["level"]), -1),
                _to_truth_2d(np.asarray([c["center"]], float),
                             float(c.get("z0", 0.0)), gt_from_map)[0, :2])
               for c in columns]
    for column in truth.features["columns"]:
        target = np.asarray(column["center"][:2], float)
        candidates = [(np.linalg.norm(center - target), i)
                      for i, (level, center) in enumerate(centers)
                      if level == int(column["level"]) and i not in used]
        if not candidates:
            continue
        distance, i = min(candidates)
        if distance <= gate:
            used.add(i)
            errors.append(centers[i][1] - target)
    out = {"truth": len(truth.features["columns"]), "predicted": len(columns),
           "matched": len(errors)}
    out["centers"] = error_stats(np.array(errors).reshape(-1, 2))
    return out


def line_report(segments: list[dict], truth: handheld_lib.HandheldGroundTruth,
                gt_from_map: np.ndarray, levels: dict[int, int],
                tolerance: float = 0.15, step: float = 0.05) -> dict:
    """Marking centre lines: recall within `tolerance` and lateral offsets."""
    by_level: dict[int, list[np.ndarray]] = {}
    for segment in segments:
        ends = _to_truth_2d(np.array([segment["p0"], segment["p1"]], float),
                            0.0, gt_from_map)[:, :2]
        by_level.setdefault(levels.get(int(segment["level"]), -1),
                            []).append(ends)
    distances = []
    for line in truth.features["lines"]:
        p0 = np.asarray(line["p0"][:2], float)
        p1 = np.asarray(line["p1"][:2], float)
        n = max(2, int(np.linalg.norm(p1 - p0) / step) + 1)
        samples = np.linspace(p0, p1, n)
        best = np.full(n, np.inf)
        for ends in by_level.get(int(line["level"]), []):
            d = ends[1] - ends[0]
            length2 = max(float(d @ d), 1e-12)
            t = np.clip((samples - ends[0]) @ d / length2, 0.0, 1.0)
            foot = ends[0] + t[:, None] * d
            best = np.minimum(best, np.linalg.norm(samples - foot, axis=1))
        distances.append(best)
    if not distances:
        return {"truth": 0}
    distances = np.concatenate(distances)
    hit = distances < tolerance
    return {"truth": len(truth.features["lines"]),
            "predicted": len(segments),
            "recall": float(hit.mean()),
            "offset": error_stats(distances[hit])}


def verdict(report: dict, thresholds) -> list[dict]:
    """Pass / fail of the plan's acceptance criteria."""
    rows = []

    def add(name: str, value, limit: float, smaller_is_better: bool = True):
        if value is None:
            rows.append({"criterion": name, "value": None,
                         "threshold": limit, "pass": None})
            return
        ok = value <= limit if smaller_is_better else value >= limit
        rows.append({"criterion": name, "value": float(value),
                     "threshold": float(limit), "pass": bool(ok)})

    columns = report.get("checkpoints", {}).get("columns", {}).get("stats", {})
    add("checkpoint_rmse_3d", columns.get("rmse_3d"),
        thresholds.checkpoint_rmse)
    surface = report.get("surface", {}).get("all", {})
    add("surface_p95", surface.get("p95"), thresholds.surface_p95)
    corners = report.get("vectors", {}).get("slots", {}).get("corners", {})
    add("slot_corner_rms", corners.get("rmse"), thresholds.slot_corner_rms)
    floor = report.get("completeness", {}).get("floor", {})
    add("floor_completeness", floor.get("within_10cm"),
        thresholds.completeness, smaller_is_better=False)
    return rows


def write_heatmaps(points: np.ndarray, signed: np.ndarray, levels: np.ndarray,
                   out_dir: pathlib.Path, cell: float,
                   full_scale: float = 0.05) -> list[str]:
    """Per level: mean |error| per cell, 0 (blue) .. full_scale (red)."""
    files = []
    ok = np.isfinite(signed)
    for level in np.unique(levels[ok]):
        mine = ok & (levels == level)
        xy = points[mine, :2]
        lo = xy.min(axis=0)
        cells = np.floor((xy - lo) / cell).astype(int)
        shape = cells.max(axis=0) + 1
        flat = cells[:, 1] * shape[0] + cells[:, 0]
        total = np.bincount(flat, np.abs(signed[mine]),
                            minlength=shape[0] * shape[1])
        count = np.bincount(flat, minlength=shape[0] * shape[1])
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        image = mean.reshape(shape[1], shape[0])[::-1]
        scaled = np.clip(np.nan_to_num(image) / full_scale, 0, 1)
        colour = cv2.applyColorMap((scaled * 255).astype(np.uint8),
                                   cv2.COLORMAP_JET)
        colour[~np.isfinite(image)] = 255
        colour = cv2.resize(colour, None, fx=2, fy=2,
                            interpolation=cv2.INTER_NEAREST)
        name = f"deviation_level_{int(level)}.png"
        cv2.imwrite(str(out_dir / name), colour)
        files.append(name)
    return files


def _cm(value) -> str:
    return "–" if value is None else f"{100 * value:.1f} cm"


def _vector_cm(vector) -> str:
    return ", ".join(f"{100 * v:+.1f}" for v in vector) + " cm"


def write_markdown(report: dict, path: pathlib.Path) -> None:
    """Human-readable summary of report.json."""
    reg = report["registration"]
    lines = ["# 测量级验收报告", "",
             f"- 手持真值：`{report['handheld']}`",
             f"- 车载重建：`{report['work']}`"]
    if report.get("features_draft"):
        lines.append("- 真值要素（楼层、柱）由 `import-scan` 自动起草，尚未经人工"
                     "复核；正式签字前须复核 features.json")
    lines += [f"- 配准：{reg['method']}"
              + (f"（控制点 {len(reg['control_ids'])} 个，残差 RMS "
                 f"{_cm(reg.get('control_rms'))}）" if reg["control_ids"] else
                 f"（ICP RMSE {_cm(reg['icp']['rmse'])}）"),
              "", "## 结论", "", "| 指标 | 数值 | 门槛 | 结果 |",
              "|---|---|---|---|"]
    names = {"checkpoint_rmse_3d": "检查点 RMSE（3D）",
             "surface_p95": "表面误差 p95",
             "slot_corner_rms": "车位角点 RMS",
             "floor_completeness": "地面完整度（观测区域内 10 cm）"}
    for row in report["verdict"]:
        if row["criterion"] == "floor_completeness":
            value = ("–" if row["value"] is None
                     else f"{100 * row['value']:.1f}%")
            limit = f"≥ {100 * row['threshold']:.0f}%"
        else:
            value, limit = _cm(row["value"]), f"≤ {_cm(row['threshold'])}"
        result = {True: "通过", False: "未通过", None: "无数据"}[row["pass"]]
        lines.append(f"| {names.get(row['criterion'], row['criterion'])} | "
                     f"{value} | {limit} | {result} |")
    columns = report["checkpoints"]["columns"]["stats"]
    if columns.get("count"):
        lines += ["", "## 检查点（柱，按地图几何量测）", "",
                  f"- 检查点 {columns['count']} 个（共 {columns['columns']} 根柱，"
                  f"可量测 {columns['measured']} 根）",
                  f"- RMSE：3D {_cm(columns['rmse_3d'])}，平面 "
                  f"{_cm(columns['rmse_horizontal'])}，高程 "
                  f"{_cm(columns['rmse_vertical'])}",
                  f"- CE90 {_cm(columns['ce90'])}，LE90 {_cm(columns['le90'])}，"
                  f"最大 {_cm(columns['max_3d'])}",
                  "- 平均误差向量（系统偏差）：" + _vector_cm(columns["bias"])]
        for level, stats in report["checkpoints"]["columns"]["levels"].items():
            if stats.get("count"):
                lines.append(f"- 第 {level} 层：{stats['count']} 个，RMSE 3D "
                             f"{_cm(stats['rmse_3d'])}，偏差 "
                             + _vector_cm(stats["bias"]))
    surface = report["surface"]
    if "all" in surface:
        lines += ["", "## 表面精度（地图结构点到手持表面）", "",
                  "| 范围 | 点数 | 中位 | p90 | p95 | 2 cm 内 | 高程偏差 |",
                  "|---|---|---|---|---|---|---|"]
        rows = [("全部", surface["all"])] + [
            (f"第 {k} 层", v) for k, v in surface["levels"].items()]
        for name, stats in rows:
            lines.append(
                f"| {name} | {stats['points']} | {_cm(stats['median'])} | "
                f"{_cm(stats['p90'])} | {_cm(stats['p95'])} | "
                f"{100 * stats['within_2cm']:.1f}% | "
                f"{_cm(stats.get('vertical_bias'))} |")
        if "sigma" in surface:
            sigma = surface["sigma"]
            lines.append(f"\n- 点的 σ 中位 {_cm(sigma['median'])}；误差在 1σ / 2σ 内的"
                         f"比例 {100 * sigma['within_1_sigma']:.0f}% / "
                         f"{100 * sigma['within_2_sigma']:.0f}%")
        lines.append("- 来源：" + "，".join(
            f"{tag} 类 {v['points']} 点（p95 {_cm(v['p95'])}）"
            for tag, v in surface["sources"].items()))
    completeness = report["completeness"]
    lines += ["", "## 完整度", "",
              f"- 手持点中车辆观测到的比例（0.5 m 内有地图点）："
              f"{100 * completeness['observed_fraction']:.1f}%",
              "", "| 高度带 | 手持点 | 车辆观测到 | 5 cm 内复现 | 10 cm 内复现 |",
              "|---|---|---|---|---|"]
    for band, name in (("floor", "地面（< 0.1 m）"),
                       ("middle", "中部（墙、柱、车辆）"),
                       ("top", "顶部（顶板、机电，≥ 2.3 m）")):
        stats = completeness[band]
        lines.append(f"| {name} | {stats['points']} | "
                     f"{100 * stats['observed_fraction']:.1f}% | "
                     f"{100 * stats['within_5cm']:.1f}% | "
                     f"{100 * stats['within_10cm']:.1f}% |")
    vectors = report["vectors"]
    lines += ["", "## 矢量", ""]
    slots = vectors.get("slots", {})
    if slots.get("truth"):
        corners = slots.get("corners", {})
        lines.append(f"- 车位：真值 {slots['truth']}，匹配 {slots['matched']}；角点 "
                     f"RMS {_cm(corners.get('rmse'))}，CE90 "
                     f"{_cm(corners.get('ce90'))}，最大 {_cm(corners.get('max'))}")
    columns = vectors.get("columns", {})
    if columns.get("truth"):
        centers = columns.get("centers", {})
        lines.append(f"- 柱（语义检测）：真值 {columns['truth']}，匹配 "
                     f"{columns['matched']}；中心 RMS {_cm(centers.get('rmse'))}")
    marks = vectors.get("lines", {})
    if marks.get("truth"):
        lines.append(f"- 标线：召回 {100 * marks['recall']:.0f}%（15 cm 内），"
                     f"中心线偏移 RMS {_cm(marks['offset'].get('rmse'))}")
    if report.get("heatmaps"):
        lines += ["", "## 偏差图", ""]
        lines += [f"![{name}]({name})" for name in report["heatmaps"]]
    path.write_text("\n".join(lines) + "\n")


def run_acceptance(work_dir: str | pathlib.Path,
                   handheld_dir: str | pathlib.Path, cfg,
                   log=print) -> dict:
    """Registers a work directory to a handheld result and writes the report.

    Args:
        work_dir: Work directory with map/ (and semantics/ if available).
        handheld_dir: Handheld result directory (handheld.py).
        cfg: Full configuration (the `acceptance` section is used).
        log: Progress output.

    Returns:
        The report dictionary (also written to work/acceptance/).
    """
    work = pathlib.Path(work_dir)
    acc = cfg.acceptance
    truth = handheld_lib.load(handheld_dir)
    log(f"[accept] handheld cloud {truth.source.name}: {len(truth.xyz)} "
        f"points, {len(truth.features['columns'])} columns, "
        f"{len(truth.features['slots'])} slots")
    point_map = static_map.PointMap.load(work / "map" / "point_map.npz")
    static = ~point_map.dynamic if point_map.dynamic is not None else np.ones(
        len(point_map), bool)
    labels_path = work / "semantics" / "labels.npz"
    if labels_path.exists():
        with np.load(labels_path) as z:
            labels = z["labels"]
        structure = static & np.isin(labels, list(taxonomy.STRUCTURE_IDS))
    else:
        structure = static
    reg = registration_lib.register(point_map.xyz[structure], truth, acc, log)
    measurements = registration_lib.measure_columns(
        point_map.xyz[static], truth.features["columns"], reg.gt_from_map)
    map_levels = json.loads(
        (work / "map" / "levels.json").read_text())["levels"]
    levels = level_mapping(map_levels, truth, reg.gt_from_map)

    points = registration_lib.transform(point_map.xyz[structure],
                                        reg.gt_from_map)
    point_level = np.array([levels.get(int(k), -1)
                            for k in point_map.level[structure]])
    sigma = getattr(point_map, "sigma", None)
    sigma = None if sigma is None else np.asarray(sigma)[structure]
    source = np.full(len(points), "L")
    surface, signed, _ = surface_report(points, point_level, sigma, source,
                                        truth, acc)
    log(f"[accept] surface: {surface.get('all', {}).get('points', 0)} points, "
        f"p95 {_cm(surface.get('all', {}).get('p95'))}")

    semantics = work / "semantics"
    vectors = {}
    if (semantics / "slots.json").exists():
        vectors["slots"] = slot_report(
            json.loads((semantics / "slots.json").read_text()), truth,
            reg.gt_from_map, levels)
    if (semantics / "structure.json").exists():
        vectors["columns"] = column_vector_report(
            json.loads((semantics / "structure.json").read_text())["columns"],
            truth, reg.gt_from_map, levels)
    if (semantics / "markings.json").exists():
        vectors["lines"] = line_report(
            json.loads((semantics / "markings.json").read_text()), truth,
            reg.gt_from_map, levels)

    out_dir = work / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    control_rms = (float(np.sqrt(np.mean([r["dx"]**2 + r["dy"]**2 + r["dz"]**2
                                          for r in reg.control_residuals])))
                   if reg.control_residuals else None)
    report = {
        "handheld": str(pathlib.Path(handheld_dir)),
        "work": str(work),
        "features_draft": bool(truth.features.get("draft", False)),
        "registration": {"method": reg.method,
                         "gt_from_map": reg.gt_from_map.tolist(),
                         "coarse": reg.coarse, "icp": reg.icp,
                         "control_ids": reg.control_ids,
                         "control_residuals": reg.control_residuals,
                         "control_rms": control_rms},
        "level_mapping": {str(k): v for k, v in levels.items()},
        "surface": surface,
        "completeness": completeness_report(
            registration_lib.transform(point_map.xyz[static],
                                       reg.gt_from_map), truth),
        "checkpoints": {
            "columns": column_checkpoints(measurements, reg.control_ids),
            "surveyed": surveyed_checkpoints(truth, reg.gt_from_map)},
        "vectors": vectors,
        "heatmaps": write_heatmaps(points, signed, point_level, out_dir,
                                   float(acc.heatmap_cell)),
    }
    report["verdict"] = verdict(report, acc.thresholds)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    write_markdown(report, out_dir / "report.md")
    for row in report["verdict"]:
        log(f"[accept] {row['criterion']}: {row['value']} (threshold "
            f"{row['threshold']}) -> {row['pass']}")
    return report
