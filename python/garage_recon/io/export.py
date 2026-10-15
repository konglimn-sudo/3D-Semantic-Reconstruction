"""Deliverable writers: labelled point clouds (LAS 1.4 / PLY), GeoJSON layers
and a Lanelet2-style OSM map with parking spaces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..semantics.taxonomy import las_code


def write_ply(path: Path, xyz: np.ndarray, fields: dict[str, np.ndarray] | None = None) -> None:
    fields = fields or {}
    types = {np.dtype("float32"): "float", np.dtype("float64"): "double", np.dtype("uint8"): "uchar",
             np.dtype("int32"): "int", np.dtype("int16"): "short", np.dtype("uint16"): "ushort"}
    cols = [("x", xyz[:, 0].astype(np.float64)), ("y", xyz[:, 1].astype(np.float64)), ("z", xyz[:, 2].astype(np.float64))]
    for k, v in fields.items():
        v = np.asarray(v)
        if v.dtype not in types:
            v = v.astype(np.float32)
        cols.append((k, v))
    dtype = np.dtype([(k, v.dtype) for k, v in cols])
    rec = np.empty(len(xyz), dtype)
    for k, v in cols:
        rec[k] = v
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property {types[v.dtype]} {k}" for k, v in cols] + ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(rec.astype(dtype.newbyteorder("<")).tobytes())


def write_las(path: Path, xyz: np.ndarray, labels: np.ndarray, intensity: np.ndarray | None = None,
              extra: dict[str, np.ndarray] | None = None) -> bool:
    """LAS 1.4, point format 6; garage classes use codes 64 + class id.
    Returns False (and writes nothing) when laspy is not installed."""
    try:
        import laspy
    except ImportError:
        return False
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    for name, arr in (extra or {}).items():
        header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=np.asarray(arr).dtype))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.classification = np.clip(las_code(0) + labels.astype(np.int32), 0, 255).astype(np.uint8)
    if intensity is not None:
        las.intensity = (np.clip(intensity, 0, 1) * 65535).astype(np.uint16)
    for name, arr in (extra or {}).items():
        las[name] = arr
    las.write(str(path))
    return True


def feature(geom_type: str, coords, props: dict) -> dict:
    return {"type": "Feature", "geometry": {"type": geom_type, "coordinates": coords}, "properties": props}


def write_geojson(path: Path, features: list[dict], crs_note: str = "local metric frame (x east, y north)") -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "crs_note": crs_note, "features": features}))


def _latlon(xy: np.ndarray, origin: dict) -> tuple[float, float]:
    """Equirectangular local -> WGS84 (adequate over a garage; use a proper
    projection such as UTM/MGRS for production maps)."""
    R = 6378137.0
    lat0, lon0 = np.radians(origin["lat"]), np.radians(origin["lon"])
    lat = lat0 + xy[1] / R
    lon = lon0 + xy[0] / (R * np.cos(lat0))
    return float(np.degrees(lat)), float(np.degrees(lon))


def write_lanelet2_osm(path: Path, slots, levels_z: dict[int, float], lots: dict[int, np.ndarray], origin: dict) -> None:
    """Parking spaces as `parking_space` linestrings (centre line from the
    entrance, with a `width` tag) and one `parking_lot` area per level, as in
    Autoware's Lanelet2 extension. Elevation and level are kept as tags."""
    nodes, ways = [], []
    nid, wid = [0], [100000]

    def node(x, y, z):
        nid[0] += 1
        lat, lon = _latlon(np.array([x, y]), origin)
        nodes.append(f'  <node id="{nid[0]}" lat="{lat:.9f}" lon="{lon:.9f}">'
                     f'<tag k="local_x" v="{x:.3f}"/><tag k="local_y" v="{y:.3f}"/><tag k="ele" v="{z:.3f}"/></node>')
        return nid[0]

    for s in slots:
        a = s.entrance_mid
        b = s.back_mid
        n0, n1 = node(a[0], a[1], s.z), node(b[0], b[1], s.z)
        wid[0] += 1
        ways.append(f'  <way id="{wid[0]}"><nd ref="{n0}"/><nd ref="{n1}"/>'
                    f'<tag k="type" v="parking_space"/><tag k="width" v="{s.width:.2f}"/>'
                    f'<tag k="level" v="{s.level}"/><tag k="slot_id" v="{s.slot_id}"/>'
                    + (f'<tag k="ref" v="{s.number}"/>' if getattr(s, "number", None) else "")
                    + (f'<tag k="zone" v="{s.zone}"/>' if getattr(s, "zone", None) else "")
                    + f'<tag k="inferred" v="{str(s.inferred).lower()}"/></way>')
    for lv, poly in lots.items():
        ids = [node(x, y, levels_z.get(lv, 0.0)) for x, y in poly]
        wid[0] += 1
        refs = "".join(f'<nd ref="{i}"/>' for i in ids + ids[:1])
        ways.append(f'  <way id="{wid[0]}">{refs}<tag k="type" v="parking_lot"/><tag k="area" v="yes"/>'
                    f'<tag k="level" v="{lv}"/></way>')
    xml = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="garage_recon">']
    path.write_text("\n".join(xml + nodes + ways + ["</osm>"]) + "\n")
