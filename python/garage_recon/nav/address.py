"""Slot addresses ("B2-C-105") and address queries.

A slot's address is its level name, zone and number. Level names come from
the configuration (`nav.level_names`, level 0 = top); zones from the OCR of
the slot's own text, configured polygons, or else the nearest zone sign the
OCR placed on the same level (`nav.zone_sign_radius`); numbers from the OCR
stage. Without a number the address is the generated slot id with the
level name and zone in front ("B1-5A-R03-001"). Queries accept the usual
spoken and written forms: "B2 C区 105号", "b2-c-105", "负二层 C 区 105",
"L0 A 12"; a query with a zone and no number ("5A区", "B1 中央区") finds
every slot of that zone (real garages often sign zones but not slots).
"""

from __future__ import annotations

import re

import numpy as np
import shapely

from garage_recon.semantics import numbers as numbers_lib

_CHINESE_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                   "七": 7, "八": 8, "九": 9, "十": 10}


def level_names(num_levels: int, cfg) -> list[str]:
    """Display names of the levels (level 0 = top)."""
    names = list(cfg.nav.get("level_names") or [])
    return [str(names[i]) if i < len(names) else f"L{i}"
            for i in range(num_levels)]


def assign_zones(goals: list[dict], cfg, landmarks=()) -> None:
    """Zones where the slot's own OCR gave none: the configured polygons,
    else the nearest zone sign (text landmark of kind "zone") on the same
    level within `nav.zone_sign_radius`. `zone_source` says which."""
    zones = [(z["name"], int(z["level"]), shapely.Polygon(z["polygon"]))
             for z in cfg.nav.get("zones") or []]
    signs = [l for l in landmarks if l.get("kind") == "zone"]
    radius = float(cfg.nav.get("zone_sign_radius") or 0.0)
    for goal in goals:
        if goal.get("zone"):
            goal.setdefault("zone_source", "ocr")
            continue
        x, y, _ = goal["park"]["xyz"]
        for name, level, polygon in zones:
            if level == goal["level"] and polygon.contains(shapely.Point(x, y)):
                goal["zone"], goal["zone_source"] = name, "polygon"
                break
        if goal.get("zone") or radius <= 0:
            continue
        best, best_d = None, radius
        for sign in signs:
            if sign.get("level") != goal["level"]:
                continue
            d = float(np.hypot(sign["xyz"][0] - x, sign["xyz"][1] - y))
            if d <= best_d:
                best, best_d = sign["text"], d
        if best is not None:
            goal["zone"], goal["zone_source"] = best, "sign"


def _zone_key(zone) -> str:
    """Zone names compared without case, spaces or a trailing 区."""
    return re.sub(r"区$", "", re.sub(r"\s+", "", str(zone or "").upper()))


def address(goal: dict, names: list[str]) -> str:
    """"<level>-<zone>-<number>", or without a number the slot id with the
    level name and zone in front ("B1-R05-004", "B1-5A-R05-004")."""
    level = names[goal["level"]] if goal["level"] < len(names) else str(
        goal["level"])
    zone = [goal["zone"]] if goal.get("zone") else []
    if goal.get("number"):
        return "-".join([level] + zone + [str(goal["number"])])
    return re.sub(r"^L\d+", "-".join([level] + zone), goal["slot_id"])


def parse_query(text: str, names: list[str]) -> dict:
    """Level index, zone and number of a query (None where absent)."""
    q = re.sub(r"^(导航到|导航去|前往|去往|带我去|去|到)", " ", text.strip().upper())
    q = re.sub(r"负\s*([一二三四五六七八九十]|\d+)\s*[层楼]",
               lambda m: f" B{_CHINESE_DIGITS.get(m.group(1), m.group(1))} ",
               q)
    q = re.sub(r"(停车位|车位|号|区|层|楼|[-_/,，、#])", " ", q)
    tokens = q.split()
    upper = [n.upper() for n in names]
    out = {"level": None, "zone": None, "number": None}
    for token in tokens:
        if token in upper and out["level"] is None:
            out["level"] = upper.index(token)
        elif re.fullmatch(r"[BLF]\d", token) and out["level"] is None:
            out["level"] = -1  # A level name this garage does not have.
        elif token.isdigit() and out["number"] is None:
            out["number"] = token
        elif (re.fullmatch(r"[A-Z]+|\d{1,2}[A-Z]{1,2}|[\u4e00-\u9fff]+", token)
              and out["zone"] is None):  # "C", "5A", "中央".
            out["zone"] = token
        else:
            # "C105" or "A-012" written together.
            m = re.fullmatch(r"([A-Z]+)(\d+)", token)
            if m and out["number"] is None:
                out["zone"] = out["zone"] or m.group(1)
                out["number"] = m.group(2)
    return out


def find(goals: list[dict], text: str, names: list[str]) -> list[dict]:
    """Goals matching a query: its number (and level, zone where given),
    or without a number every goal of its zone. The slot id and the address
    also match directly. Slot attributes in the query ("充电车位",
    "B1无障碍车位") select the slots that have them."""
    key = text.strip().upper()
    direct = [g for g in goals if key in (g["slot_id"].upper(),
                                          str(g.get("address", "")).upper())]
    if direct:
        return direct
    wanted = set(numbers_lib.text_attributes(text))
    if wanted:
        for pattern in numbers_lib.SLOT_ATTRIBUTES.values():
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    query = parse_query(text, names)
    if query["number"] is None and query["zone"] is None and not wanted:
        return []
    out = []
    for goal in goals:
        if not wanted <= set(goal.get("attributes") or []):
            continue
        number = goal.get("number")
        if query["number"] is not None and (
                number is None or int(re.sub(r"\D", "", str(number)) or -1)
                != int(query["number"])):
            continue
        if query["level"] is not None and goal["level"] != query["level"]:
            continue
        if query["zone"] is not None and _zone_key(goal.get("zone")) != (
                _zone_key(query["zone"])):
            continue
        out.append(goal)
    return out
