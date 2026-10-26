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
results are not tied to the raster cells. The measurement is C++
(gr/semantics/vector_fit.h); this module prepares the floor images.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
from scipy import ndimage

from garage_recon import _core
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


def line_fit_config(config: FitConfig) -> _core.LineFitConfig:
    out = _core.LineFitConfig()
    for key in ("station_step", "sample_step", "half_span", "end_search",
                "end_floor_width", "min_contrast", "min_valid",
                "max_end_shift"):
        setattr(out, key, getattr(config, key))
    out.floor_band_inner, out.floor_band_outer = config.floor_band
    out.end_floor_near, out.end_floor_far = config.end_floor
    return out


def _image_tuple(image: FloorImage) -> tuple:
    g = image.grid
    return (image.name, np.ascontiguousarray(image.image, np.float64),
            np.ascontiguousarray(image.valid, np.float64), g.x0, g.y0, g.res)


def fit_line(p0: np.ndarray, p1: np.ndarray, image: FloorImage,
             config: FitConfig) -> LineFit | None:
    """Re-measures one painted line (C++ FitLine); None if too few stations
    are usable."""
    fit = _core.fit_line(np.asarray(p0, float), np.asarray(p1, float),
                         _image_tuple(image), line_fit_config(config))
    if fit is None:
        return None
    return LineFit(np.asarray(fit["p0"]), np.asarray(fit["p1"]), fit["width"],
                   tuple(fit["end_ok"]), fit["stations"], fit["rms"])


def refine_segments(segments: list, rasters: dict, grid: bev.BevGrid,
                    config: FitConfig | None = None) -> int:
    """Re-measures marking segments in place (p0, p1, width, end_ok).

    Every segment is measured on the first image of `config.sources` that
    works (C++ RefineSegments). Segments that cannot be measured keep the
    skeleton geometry with both ends unreliable.

    Returns:
        The number of refined segments.
    """
    config = config or FitConfig()
    by_name = {image.name: image for image in floor_images(rasters, grid)}
    images = [_image_tuple(by_name[name]) for name in config.sources
              if name in by_name]
    if not segments or not images:
        return 0
    data = [{"level": s.level, "p0": s.p0, "p1": s.p1, "width": s.width,
             "cls": s.cls, "confidence": s.confidence,
             "end_ok": list(s.end_ok)} for s in segments]
    out, refined = _core.refine_markings(data, images, line_fit_config(config))
    for segment, d in zip(segments, out):
        segment.p0 = np.asarray(d["p0"]).tolist()
        segment.p1 = np.asarray(d["p1"]).tolist()
        segment.width = float(d["width"])
        segment.end_ok = [bool(v) for v in d["end_ok"]]
    return int(refined)
