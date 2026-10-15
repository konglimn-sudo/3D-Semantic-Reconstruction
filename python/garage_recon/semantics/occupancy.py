"""Per-frame semantic occupancy ground truth in the ego frame (section 5,
module "occupancy"): labelled static map + this frame's object boxes, with
LiDAR and camera visibility masks. Parking needs fine voxels (5-10 cm) so that
wheel stoppers, column corners and parking locks survive.

The generator is C++ (`_core.OccupancyGenerator`): it indexes the static map
once per drive and builds each frame's grid from the vehicle pose, the
frame's boxes and its LiDAR sweep.
"""

from __future__ import annotations

import numpy as np

from .. import _core
from ..core_config import occupancy_config
from ..data.schema import Rig


def occupancy_generator(static_xyz: np.ndarray, static_labels: np.ndarray, rig: Rig, cfg) -> _core.OccupancyGenerator:
    cams = [(rig.cameras[c].make_model(), rig.cameras[c].T_vehicle_cam) for c in rig.fisheye_cameras]
    return _core.OccupancyGenerator(static_xyz, static_labels.astype(np.int32), occupancy_config(cfg),
                                    rig.primary_lidar.T_vehicle_lidar, cams)
