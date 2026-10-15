"""Vehicle-based 3D reconstruction and automatic semantic labelling of
multi-level underground parking garages.

Sub-packages
------------
recon      Section 4: trajectory (odometry, loops, pose graph, scan-to-map),
           static map, levels/ramps, BEV rasters and floor texture.
semantics  Section 5: structure, markings, parking slots, 4D objects,
           occupancy ground truth and per-frame projected labels.
data       Trip dataset layout, rig calibration and a synthetic garage
           generator used for tests and demos.
"""

__version__ = "0.1.0"

try:
    from . import _core  # noqa: F401
except ImportError as exc:  # pragma: no cover - build hint
    raise ImportError(
        "garage_recon._core is not built. From the repository root run "
        "`cmake -S . -B build && cmake --build build -j` (development layout) "
        "or `pip install .`."
    ) from exc
