"""Which pose set every stage output was built with.

work/provenance.json maps a stage name to the ids of the pose set it used.
Stages that only use the LiDAR / IMU trajectory compare `trajectory_id`;
stages that use camera poses compare the full `pose_set_id`. QA rejects
outputs whose ids differ from the current pose set (see `stale_stages`).
"""

from __future__ import annotations

import json
import pathlib
import time

_FILE = "provenance.json"


def _read(work_dir: pathlib.Path) -> dict:
    path = work_dir / _FILE
    return json.loads(path.read_text()) if path.exists() else {}


def record(work_dir: str | pathlib.Path, stage: str, pose_set,
           uses_cameras: bool) -> None:
    """Records that `stage` was just built with `pose_set`.

    Args:
        work_dir: The work directory.
        stage: Stage name, e.g. "recon.map" or "semantics.frames".
        pose_set: The PoseSet used.
        uses_cameras: Whether the stage depends on camera poses.
    """
    work = pathlib.Path(work_dir)
    entries = _read(work)
    entries[stage] = {
        "pose_set_id": pose_set.pose_set_id,
        "trajectory_id": pose_set.trajectory_id,
        "uses_cameras": bool(uses_cameras),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (work / _FILE).write_text(json.dumps(entries, indent=1, sort_keys=True))


def stale_stages(work_dir: str | pathlib.Path, pose_set) -> list[str]:
    """Stages whose outputs were built with a different pose set."""
    stale = []
    for stage, entry in sorted(_read(pathlib.Path(work_dir)).items()):
        key = "pose_set_id" if entry.get("uses_cameras") else "trajectory_id"
        if entry.get(key) != getattr(pose_set, key):
            stale.append(stage)
    return stale


def entries(work_dir: str | pathlib.Path) -> dict:
    return _read(pathlib.Path(work_dir))
