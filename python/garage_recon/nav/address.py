"""Slot addresses ("B2-C-105") and address queries.

A slot's address is its level name, zone and number. Level names come from
the configuration (`nav.level_names`, level 0 = top); zones from configured
polygons or the OCR stage; numbers from the OCR stage. Missing parts fall
back to the generated slot id. Queries accept the usual spoken and written
forms: "B2 C区 105号", "b2-c-105", "负二层 C 区 105", "L0 A 12".
"""

from __future__ import annotations

import re

import shapely

_CHINESE_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                   "七": 7, "八": 8, "九": 9, "十": 10}


def level_names(num_levels: int, cfg) -> list[str]:
    """Display names of the levels (level 0 = top)."""
    names = list(cfg.nav.get("level_names") or [])
    return [str(names[i]) if i < len(names) else f"L{i}"
            for i in range(num_levels)]


def assign_zones(goals: list[dict], cfg) -> None:
    """Zones from the configured polygons, where OCR gave none."""
    zones = [(z["name"], int(z["level"]), shapely.Polygon(z["polygon"]))
             for z in cfg.nav.get("zones") or []]
    for goal in goals:
        if goal.get("zone") or not zones:
            continue
        x, y, _ = goal["park"]["xyz"]
        for name, level, polygon in zones:
            if level == goal["level"] and polygon.contains(shapely.Point(x, y)):
                goal["zone"] = name
                break


def address(goal: dict, names: list[str]) -> str:
    """"<level>-<zone>-<number>", or without a number the slot id with the
    level name in front ("B1-R05-004")."""
    level = names[goal["level"]] if goal["level"] < len(names) else str(
        goal["level"])
    if goal.get("number"):
        parts = [level] + ([goal["zone"]] if goal.get("zone") else []) + [
            str(goal["number"])]
        return "-".join(parts)
    return re.sub(r"^L\d+", level, goal["slot_id"])


def parse_query(text: str, names: list[str]) -> dict:
    """Level index, zone and number of a query (None where absent)."""
    q = text.strip().upper()
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
        elif re.fullmatch(r"[A-Z]+", token) and out["zone"] is None:
            out["zone"] = token
        else:
            # "C105" or "A-012" written together.
            m = re.fullmatch(r"([A-Z]+)(\d+)", token)
            if m and out["number"] is None:
                out["zone"] = out["zone"] or m.group(1)
                out["number"] = m.group(2)
    return out


def find(goals: list[dict], text: str, names: list[str]) -> list[dict]:
    """Goals matching a query; the slot id also matches directly."""
    direct = [g for g in goals if g["slot_id"].upper() == text.strip().upper()]
    if direct:
        return direct
    query = parse_query(text, names)
    if query["number"] is None:
        return []
    out = []
    for goal in goals:
        number = goal.get("number")
        if number is None or int(re.sub(r"\D", "", str(number)) or -1) != int(
                query["number"]):
            continue
        if query["level"] is not None and goal["level"] != query["level"]:
            continue
        if query["zone"] is not None and (goal.get("zone") or "").upper() != (
                query["zone"]):
            continue
        out.append(goal)
    return out
