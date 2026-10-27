"""Files of the navigation map.

    level_<k>.pgm / .yaml       ROS map_server / nav2 occupancy map (trinary):
                                permanent obstacles occupied, drivable free,
                                the rest (parked cars, unseen) unknown
    level_<k>_layers.npz        all layers (see layers.LevelLayers)
    level_<k>_preview.png       layers, slots and route graph, north up
    graph.json / graph.geojson  route graph (nodes, edges with kind / width)
    goals.json / goals.geojson  slot goals (park and approach poses,
                                address, route length)
    pois.json / pois.geojson    points of interest: the entrance (route
                                origin), exit and gate signs
"""

from __future__ import annotations

import json
import pathlib

import cv2
import numpy as np
import yaml


def write_ros_map(directory: pathlib.Path, layers, name: str) -> None:
    """Occupancy map in the map_server format (PGM + YAML)."""
    g = layers.grid
    image = np.full((g.height, g.width), 205, np.uint8)  # Unknown.
    image[layers.drivable] = 254
    image[layers.structure] = 0
    image = np.flipud(image)  # PGM row 0 is the top (largest y).
    path = directory / f"{name}.pgm"
    with open(path, "wb") as f:
        f.write(f"P5\n{g.width} {g.height}\n255\n".encode())
        f.write(image.tobytes())
    (directory / f"{name}.yaml").write_text(yaml.safe_dump({
        "image": path.name, "mode": "trinary", "resolution": float(g.res),
        "origin": [float(g.x0), float(g.y0), 0.0], "negate": 0,
        "occupied_thresh": 0.65, "free_thresh": 0.25}, sort_keys=False))


def write_preview(path: pathlib.Path, layers, graph, goals: list[dict]
                  ) -> None:
    """Colour preview: drivable white, structure black, parked blue,
    overhead red, low objects orange, slots green, aisles magenta, lanes
    violet, one-way edges with a red arrow."""
    g = layers.grid
    rgb = np.full((g.height, g.width, 3), 128, np.uint8)
    rgb[layers.drivable] = (255, 255, 255)
    rgb[(layers.slot > 0) & layers.known] = (200, 235, 200)
    rgb[layers.low] = (255, 170, 60)
    rgb[layers.parked] = (90, 140, 230)
    rgb[layers.structure] = (0, 0, 0)
    rgb[layers.overhead] = (200, 30, 30)

    def pixels(xy):
        return np.round((np.asarray(xy)[:, :2] - [g.x0, g.y0]) / g.res
                        - 0.5).astype(np.int32)

    for edge in graph.edges:
        if edge.level != g.level:
            continue
        color = {"aisle": (220, 0, 220), "lane": (130, 0, 255),
                 "connector": (0, 160, 0), "ramp": (255, 120, 0),
                 "bridge": (0, 200, 255)}[edge.kind]
        cv2.polylines(rgb, [pixels(edge.points)], False, color,
                      1 if edge.kind == "connector" else 2)
        if edge.one_way:  # Direction of travel at the middle.
            mid = len(edge.points) // 2
            a, b = pixels(edge.points[[max(mid - 1, 0), min(mid + 1,
                                       len(edge.points) - 1)]])
            if np.any(a != b):
                d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
                tip = (a + b) // 2 + np.round(8 * d).astype(np.int32)
                cv2.arrowedLine(rgb, tuple(int(v) for v in (a + b) // 2),
                                tuple(int(v) for v in tip), (255, 0, 0), 2,
                                tipLength=0.6)
    for goal in goals:
        if goal["level"] != g.level:
            continue
        c = pixels([goal["park"]["xyz"]])[0]
        yaw = goal["park"]["yaw_head_in"]
        tip = c + np.round(np.array([np.cos(yaw), np.sin(yaw)]) / g.res
                           ).astype(np.int32)
        cv2.arrowedLine(rgb, tuple(int(v) for v in c),
                        tuple(int(v) for v in tip), (0, 120, 0), 1,
                        tipLength=0.4)
    cv2.imwrite(str(path), cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR))


def _feature(geometry: dict, properties: dict) -> dict:
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def write_graph(directory: pathlib.Path, graph) -> None:
    """graph.json and graph.geojson (edges as lines, nodes as points)."""
    (directory / "graph.json").write_text(json.dumps(graph.to_json()))
    features = [
        _feature({"type": "LineString",
                  "coordinates": np.round(e.points, 3).tolist()},
                 {"edge": k, "a": e.a, "b": e.b, "kind": e.kind,
                  "level": e.level, "width": round(e.width, 2),
                  "one_way": e.one_way, "length": round(e.length, 2),
                  "features": list(e.features)})
        for k, e in enumerate(graph.edges)]
    features += [
        _feature({"type": "Point", "coordinates": np.round(p, 3).tolist()},
                 {"node": k, "level": lv})
        for k, (p, lv) in enumerate(zip(graph.nodes, graph.node_level))]
    (directory / "graph.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}))


def write_goals(directory: pathlib.Path, goals: list[dict],
                names: list[str]) -> None:
    """goals.json (with the level names) and goals.geojson (park points)."""
    (directory / "goals.json").write_text(json.dumps(
        {"level_names": names, "goals": goals}, ensure_ascii=False, indent=1))
    features = []
    for goal in goals:
        properties = {k: v for k, v in goal.items()
                      if k not in ("park", "approach", "entrance", "corners")}
        properties["yaw_head_in"] = goal["park"]["yaw_head_in"]
        features.append(_feature(
            {"type": "Point", "coordinates": goal["park"]["xyz"]},
            properties))
    (directory / "goals.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        ensure_ascii=False))


def write_pois(directory: pathlib.Path, pois: list[dict]) -> None:
    """pois.json and pois.geojson (sign positions)."""
    (directory / "pois.json").write_text(json.dumps(
        pois, ensure_ascii=False, indent=1))
    features = [_feature({"type": "Point", "coordinates": p["xyz"]},
                         {k: v for k, v in p.items() if k != "xyz"})
                for p in pois]
    (directory / "pois.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": features},
        ensure_ascii=False))
