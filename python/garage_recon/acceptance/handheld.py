"""The handheld ground truth: reading the scanner export and its annotations.

A handheld result is a directory with
    cloud.(ply|las|laz|pcd|xyz|txt|csv|npz)  the scan, in its own frame
                                             (gravity aligned, z up)
    features.json  (optional)  levels, columns, slots and marking lines
                               digitised on the scan, in the same frame
    points.csv     (optional)  surveyed points (total station targets, ...):
                               id, role (control | check), x, y, z and,
                               if identified in the vehicle map,
                               map_x, map_y, map_z

features.json:
    {"levels":  [{"id", "z"}],
     "columns": [{"id", "level", "center": [x, y, z], "size": [sx, sy],
                  "yaw_deg", "height"}],
     "slots":   [{"id", "level", "corners": [[x, y, z] x 4]}],
     "lines":   [{"id", "level", "cls", "p0": [x, y, z], "p1": [x, y, z],
                  "width"}]}
"""

from __future__ import annotations

import csv
import dataclasses
import json
import pathlib

import numpy as np

CLOUD_SUFFIXES = (".ply", ".las", ".laz", ".pcd", ".xyz", ".txt", ".csv",
                  ".npz")

_PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
              "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
              "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
              "float": "f4", "float32": "f4", "double": "f8",
              "float64": "f8"}
_PCD_TYPES = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
              ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}


@dataclasses.dataclass
class SurveyPoint:
    """A surveyed point: control points define the frame, checks test it."""

    point_id: str
    role: str  # "control" or "check"
    xyz: np.ndarray  # Handheld frame.
    map_xyz: np.ndarray | None = None  # Vehicle-map frame, if known.


@dataclasses.dataclass
class HandheldGroundTruth:
    """Cloud, features and surveyed points of one handheld result."""

    xyz: np.ndarray
    intensity: np.ndarray | None
    features: dict
    points: list[SurveyPoint]
    source: pathlib.Path

    def level_ids(self) -> list[int]:
        return [int(lv["id"]) for lv in self.features.get("levels", [])]

    def level_z(self) -> dict[int, float]:
        return {int(lv["id"]): float(lv["z"])
                for lv in self.features.get("levels", [])}


def _read_ply(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no end_header")
            text = line.decode("ascii", "replace").strip()
            header.append(text)
            if text == "end_header":
                break
        body = f.read()
    fmt = next((h.split()[1] for h in header if h.startswith("format")), None)
    elements, current = [], None
    for h in header:
        parts = h.split()
        if parts and parts[0] == "element":
            current = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(current)
        elif parts and parts[0] == "property" and current is not None:
            if parts[1] == "list":
                raise ValueError(f"{path}: list properties are not supported")
            current["props"].append((parts[2], _PLY_TYPES[parts[1]]))
    if not elements or elements[0]["name"] != "vertex":
        raise ValueError(f"{path}: the first element must be 'vertex'")
    vertex = elements[0]
    if fmt == "ascii":
        rows = body.decode("ascii").split("\n")[:vertex["count"]]
        data = np.loadtxt(rows, ndmin=2)
        names = [p[0] for p in vertex["props"]]
        column = {n: data[:, k] for k, n in enumerate(names)}
    else:
        order = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(n, order + t) for n, t in vertex["props"]])
        column = np.frombuffer(body, dtype=dtype, count=vertex["count"])
    xyz = np.column_stack([column["x"], column["y"], column["z"]]).astype(
        np.float64)
    names = (column.dtype.names if hasattr(column, "dtype") else
             tuple(column))
    intensity = None
    for key in ("intensity", "scalar_intensity", "reflectance"):
        if key in names:
            intensity = np.asarray(column[key], np.float64)
            break
    return xyz, intensity


def _read_pcd(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray | None]:
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: no DATA line")
            parts = line.decode("ascii", "replace").split()
            if not parts or parts[0].startswith("#"):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break
        body = f.read()
    # Padding fields ("_", e.g. COUNT 4 of U1) may repeat; names must not.
    fields = [n if n not in header["FIELDS"][:k] else f"{n}{k}"
              for k, n in enumerate(header["FIELDS"])]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
    count = int(header.get("POINTS", header.get("WIDTH"))[0])
    if header["DATA"][0] == "ascii":
        data = np.loadtxt(body.decode("ascii").splitlines(), ndmin=2)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        column = {n: data[:, s] for n, s in zip(fields, starts)}
    elif header["DATA"][0] == "binary":
        sizes = [int(s) for s in header["SIZE"]]
        dtype = np.dtype([
            (n, "<" + _PCD_TYPES[(t, s)]) + ((c,) if c > 1 else ())
            for n, t, s, c in zip(fields, header["TYPE"], sizes, counts)])
        column = np.frombuffer(body, dtype=dtype, count=count)
    else:
        raise ValueError(f"{path}: PCD DATA {header['DATA'][0]} not supported")
    xyz = np.column_stack([column["x"], column["y"], column["z"]]).astype(
        np.float64)
    intensity = None
    lower = {n.lower(): n for n in fields}
    if "intensity" in lower:
        intensity = np.asarray(column[lower["intensity"]], np.float64)
    elif "rgb" in fields or "rgba" in fields:
        # PCL packs 8-bit colour into the bits of one float: luminance.
        packed = np.ascontiguousarray(
            column["rgb" if "rgb" in fields else "rgba"]).view(np.uint32)
        r = (packed >> 16) & 255
        g = (packed >> 8) & 255
        b = packed & 255
        intensity = 0.299 * r + 0.587 * g + 0.114 * b
    return xyz, intensity


def read_cloud(path: str | pathlib.Path
               ) -> tuple[np.ndarray, np.ndarray | None]:
    """Reads a point cloud export.

    Args:
        path: PLY (ascii / binary), LAS / LAZ (needs laspy), PCD (ascii /
            binary), XYZ / TXT / CSV (x y z [intensity] per line) or NPZ
            (`xyz` or `points`, optional `intensity`).

    Returns:
        (xyz (N, 3) float64, intensity (N,) or None).

    Raises:
        ValueError: Unknown or unsupported format.
    """
    path = pathlib.Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return _read_ply(path)
    if suffix == ".pcd":
        return _read_pcd(path)
    if suffix in (".las", ".laz"):
        import laspy  # pylint: disable=import-outside-toplevel
        las = laspy.read(path)
        xyz = np.column_stack([las.x, las.y, las.z]).astype(np.float64)
        return xyz, np.asarray(las.intensity, np.float64)
    if suffix == ".npz":
        with np.load(path) as z:
            key = "xyz" if "xyz" in z.files else "points"
            return (z[key].astype(np.float64),
                    z["intensity"].astype(np.float64)
                    if "intensity" in z.files else None)
    if suffix in (".xyz", ".txt", ".csv"):
        with open(path) as f:
            first = f.readline()
        delimiter = "," if "," in first else None
        skip = 0 if _is_number(first.replace(",", " ").split()[0]) else 1
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skip, ndmin=2)
        return (data[:, :3].astype(np.float64),
                data[:, 3].astype(np.float64) if data.shape[1] > 3 else None)
    raise ValueError(f"unsupported cloud format: {path}")


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def read_points(path: pathlib.Path) -> list[SurveyPoint]:
    """Surveyed points (points.csv)."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            map_xyz = None
            if row.get("map_x") not in (None, ""):
                map_xyz = np.array([float(row["map_x"]), float(row["map_y"]),
                                    float(row["map_z"])])
            out.append(SurveyPoint(
                row["id"], row.get("role", "check").strip().lower(),
                np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                map_xyz))
    return out


def load(directory: str | pathlib.Path) -> HandheldGroundTruth:
    """Loads a handheld result directory (see the module docstring).

    Raises:
        FileNotFoundError: No cloud file in the directory.
    """
    directory = pathlib.Path(directory)
    clouds = sorted(p for p in directory.iterdir()
                    if p.stem == "cloud" and p.suffix.lower() in CLOUD_SUFFIXES)
    if not clouds:
        raise FileNotFoundError(f"no cloud.* ({', '.join(CLOUD_SUFFIXES)}) in "
                                f"{directory}")
    xyz, intensity = read_cloud(clouds[0])
    features_path = directory / "features.json"
    features = (json.loads(features_path.read_text())
                if features_path.exists() else {})
    for key in ("levels", "columns", "slots", "lines"):
        features.setdefault(key, [])
    points_path = directory / "points.csv"
    points = read_points(points_path) if points_path.exists() else []
    return HandheldGroundTruth(xyz, intensity, features, points, clouds[0])
