"""Sub-pixel vector fitting of painted lines on the floor orthophoto.

Plan step 4 (vector fitting on the orthophoto). The skeleton of a
thresholded marking mask finds the lines, but its ends are eroded by about
half a line width and its centre is quantised to the raster cells. Each
segment is therefore measured again on a brightness image of the floor (the
camera orthophoto where it was observed, else the LiDAR intensity):

* centre line: at stations along the segment, the brightness profile across
  it gives the stripe edges (half way between paint and floor); a robust
  line fit through the stripe centres;
* ends: along the centre line the end is where the brightness falls half
  way from the paint to the floor beyond it. An end is reliable only if
  floor is observed beyond it; otherwise a car, the wall base or the edge
  of the scan may hide part of the line, and the slot assembly takes the
  end from the other separators of the row.

The image is sampled bilinearly in world coordinates (1 cm steps), so the
results are not tied to the raster cells.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from scipy import ndimage

from garage_recon.recon import bev


@dataclasses.dataclass
class LineFit:
    """A re-measured line: centre line end points, width, end reliability."""

    p0: np.ndarray
    p1: np.ndarray
    width: float
    end_ok: tuple[bool, bool]
    stations: int
    rms: float  # Of the stripe centres about the fitted line (m).


@dataclasses.dataclass
class FitConfig:
    """Sampling and acceptance of the line measurement."""

    station_step: float = 0.05  # m between profiles across the line.
    sample_step: float = 0.01   # m along a profile.
    # Profiles across the line stay within 0.25 m: parked cars leave about
    # 0.3 m of floor beside a separator.
    half_span: float = 0.25
    floor_band: tuple[float, float] = (0.12, 0.25)  # Floor part of a profile.
    end_search: float = 0.6     # m beyond the skeleton end.
    end_floor: tuple[float, float] = (0.12, 0.6)  # Floor beyond an end ...
    end_floor_width: float = 0.1  # ... and this far to either side (m).
    min_contrast: float = 0.08  # Paint above floor (normalised brightness).
    min_valid: float = 0.8      # Observed fraction of a floor sample.
    max_end_shift: float = 0.3  # m from the skeleton end.
    # Images in order of preference; a segment is measured on the first one
    # that works. With 256 px fisheyes the LiDAR intensity is sharper than
    # the camera orthophoto; put "camera" first for high-resolution cameras
    # or a 3DGS orthophoto.
    sources: tuple[str, ...] = ("lidar", "camera")


class FloorImage:
    """Brightness of one level's floor, sampled in world coordinates."""

    def __init__(self, name: str, image: np.ndarray, valid: np.ndarray,
                 grid: bev.BevGrid):
        self.name = name
        self.image = image.astype(np.float64)
        self.valid = valid.astype(np.float64)
        self.grid = grid

    def _coordinates(self, xy: np.ndarray) -> np.ndarray:
        pixels = self.grid.world_to_px(np.atleast_2d(xy)) - 0.5
        return np.stack([pixels[:, 1], pixels[:, 0]])  # (row, col)

    def brightness(self, xy: np.ndarray) -> np.ndarray:
        return ndimage.map_coordinates(self.image, self._coordinates(xy),
                                       order=1, mode="nearest")

    def observed(self, xy: np.ndarray) -> np.ndarray:
        return ndimage.map_coordinates(self.valid, self._coordinates(xy),
                                       order=1, mode="constant", cval=0.0)


def floor_images(rasters: dict, grid: bev.BevGrid) -> list[FloorImage]:
    """Brightness images: "camera" (orthophoto) and "lidar" (intensity).

    Cells under objects (height raster above 0.3 m) are not observed floor.
    """
    covered = (np.nan_to_num(rasters["height_max"]) > 0.3
               if "height_max" in rasters else None)
    out = []
    rgb = rasters.get("rgb")
    if rgb is not None and "texture_nobs" in rasters:
        valid = rasters["texture_nobs"] > 0
        if valid.any():
            luminance = cv2.cvtColor(rgb.astype(np.uint8),
                                     cv2.COLOR_RGB2GRAY) / 255.0
            out.append(("camera", luminance, valid))
    intensity = rasters.get("intensity")
    if intensity is not None and np.isfinite(intensity).any():
        observed = np.isfinite(intensity)
        weight = ndimage.gaussian_filter(observed.astype(float), 1.0)
        filled = ndimage.gaussian_filter(np.where(observed, intensity, 0.0),
                                         1.0)
        valid = weight > 0.15
        out.append(("lidar",
                    np.where(valid, filled / np.maximum(weight, 1e-6), 0.0),
                    valid))
    images = []
    for name, image, valid in out:
        if covered is not None:
            valid = valid & ~covered
        values = image[valid]
        lo, hi = np.percentile(values, 2), np.percentile(values, 99.5)
        normalised = np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        images.append(FloorImage(name, normalised, valid, grid))
    return images


def _stripe_center(profile: np.ndarray, seen: np.ndarray,
                   offsets: np.ndarray,
                   config: FitConfig) -> tuple[float, float] | None:
    """(centre, width) of the bright stripe in a profile across a line.

    The stripe must be observed; of the floor band on each side at least
    half.
    """
    inner = np.abs(offsets) < config.floor_band[0]
    if not seen[inner].all():
        return None
    floor_zone = ((np.abs(offsets) >= config.floor_band[0])
                  & (np.abs(offsets) <= config.floor_band[1]))
    for side in (offsets < 0, offsets > 0):
        if seen[floor_zone & side].mean() < 0.5:
            return None
    floor = np.median(profile[floor_zone & seen])
    peak_index = np.flatnonzero(inner)[np.argmax(profile[inner])]
    paint = profile[peak_index]
    if paint - floor < config.min_contrast:
        return None
    half = 0.5 * (paint + floor)
    left = peak_index
    while left > 0 and profile[left - 1] > half:
        left -= 1
    right = peak_index
    while right + 1 < len(profile) and profile[right + 1] > half:
        right += 1
    if left == 0 or right == len(profile) - 1:
        return None

    def crossing(outside: int, inside: int) -> float:
        a, b = profile[outside], profile[inside]
        t = (half - a) / (b - a) if b != a else 0.5
        return offsets[outside] + t * (offsets[inside] - offsets[outside])

    lo = crossing(left - 1, left)
    hi = crossing(right + 1, right)
    return 0.5 * (lo + hi), hi - lo


def _end(image: FloorImage, origin: np.ndarray, direction: np.ndarray,
         start: float, sign: float, paint: float,
         config: FitConfig) -> tuple[float, bool]:
    """Position of one end along the centre line (`sign` +1: beyond p1).

    Returns:
        (position along the line from `origin`, reliable).
    """
    s = start + sign * np.arange(-0.2, config.end_search + 1e-9,
                                 config.sample_step)
    xy = origin + s[:, None] * direction
    profile = image.brightness(xy)
    beyond = start + sign * np.arange(config.end_floor[0],
                                      config.end_floor[1] + 1e-9,
                                      config.sample_step)
    beyond_xy = origin + beyond[:, None] * direction
    # The floor around the continuation must be observed: a column or a car
    # next to the line hides its end even if the axis itself looks free.
    normal = np.array([-direction[1], direction[0]])
    patch = np.concatenate([beyond_xy + w * normal for w in
                            (-config.end_floor_width, 0.0,
                             config.end_floor_width)])
    if (image.observed(patch) >= config.min_valid).mean() < config.min_valid:
        return start, False
    seen = image.observed(beyond_xy) >= config.min_valid
    # Lower quartile: robust to paint that reaches into the band.
    floor = np.percentile(image.brightness(beyond_xy)[seen], 25)
    if paint - floor < config.min_contrast:
        return start, False
    half = 0.5 * (paint + floor)
    below = np.flatnonzero(profile < half)
    if not len(below) or below[0] == 0:
        return start, False
    k = below[0]
    a, b = profile[k - 1], profile[k]
    t = (a - half) / (a - b) if a != b else 0.5
    position = s[k - 1] + t * (s[k] - s[k - 1])
    if abs(position - start) > config.max_end_shift:
        return start, False
    return float(position), True


def fit_line(p0: np.ndarray, p1: np.ndarray, image: FloorImage,
             config: FitConfig) -> LineFit | None:
    """Re-measures one painted line; None if too few stations are usable."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.linalg.norm(p1 - p0))
    if length < 0.3:
        return None
    direction = (p1 - p0) / length
    normal = np.array([-direction[1], direction[0]])
    offsets = np.arange(-config.half_span, config.half_span + 1e-9,
                        config.sample_step)
    stations = np.arange(0.1, length - 0.1 + 1e-9, config.station_step)
    centers, widths, used = [], [], []
    for s in stations:
        xy = p0 + s * direction + offsets[:, None] * normal
        stripe = _stripe_center(image.brightness(xy),
                                image.observed(xy) >= config.min_valid,
                                offsets, config)
        if stripe is None:
            continue
        centers.append(stripe[0])
        widths.append(stripe[1])
        used.append(s)
    if len(used) < 4:
        return None
    used, centers = np.asarray(used), np.asarray(centers)
    keep = np.ones(len(used), bool)
    for _ in range(3):  # Robust fit: centre offset linear along the line.
        a, b = np.polyfit(used[keep], centers[keep], 1)
        residual = centers - (a * used + b)
        mad = 1.4826 * np.median(np.abs(residual[keep])) + 1e-3
        keep = np.abs(residual) <= 3.0 * mad
        if keep.sum() < 4:
            return None
    slope, intercept = a, b
    origin = p0 + intercept * normal
    axis = direction + slope * normal
    scale = np.linalg.norm(axis)
    axis /= scale
    # Stations are measured along `direction`; on the fitted axis a point at
    # s lies at s * scale.
    start0, end0 = 0.0, length * scale
    paint = float(np.median(image.brightness(
        origin + np.linspace(0.2, max(end0 - 0.2, 0.2), 20)[:, None] * axis)))
    s0, ok0 = _end(image, origin, axis, start0, -1.0, paint, config)
    s1, ok1 = _end(image, origin, axis, end0, 1.0, paint, config)
    rms = float(np.sqrt(np.mean(residual[keep]**2)))
    return LineFit(origin + s0 * axis, origin + s1 * axis,
                   float(np.median(np.asarray(widths)[keep])), (ok0, ok1),
                   int(keep.sum()), rms)


def refine_segments(segments: list, rasters: dict, grid: bev.BevGrid,
                    config: FitConfig | None = None) -> int:
    """Re-measures marking segments in place (p0, p1, width, end_ok).

    Every segment is measured on the first image of `config.sources` that
    works. Segments that cannot be measured keep the skeleton geometry with
    both ends unreliable.

    Returns:
        The number of refined segments.
    """
    config = config or FitConfig()
    by_name = {image.name: image for image in floor_images(rasters, grid)}
    images = [by_name[name] for name in config.sources if name in by_name]
    refined = 0
    for segment in segments:
        best = None
        for image in images:
            best = fit_line(np.asarray(segment.p0), np.asarray(segment.p1),
                            image, config)
            if best is not None:
                break
        if best is None:
            continue
        segment.p0 = best.p0.tolist()
        segment.p1 = best.p1.tolist()
        segment.width = best.width
        segment.end_ok = [bool(best.end_ok[0]), bool(best.end_ok[1])]
        refined += 1
    return refined
