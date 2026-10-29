"""Section 4 orchestration: trips -> pose set -> static map -> levels -> BEV.

Odometry, loop closure, the pose graph (IMU preintegration, wheel and bias
factors), the continuous-time trajectory, camera calibration, map
accumulation and dynamic points run in the C++ core; the stages here load
the inputs, call it and write the work directory.

Stages:
    odometry     per-trip LiDAR odometry and keyframe sweeps (kf/)
    graph        loops + pose graph (IMU bias and velocity per keyframe) +
                 LiDAR bundle adjustment (plane factors)
    trajectory   the pose set (continuous trajectories + rig calibration);
                 keyframe sweeps deskewed again with it (kf_final/)
    calibration  camera time offsets and extrinsics (needs a matcher)
    map          static point map (observations fused, sigma per point),
                 levels, dynamic points
    bev          per-level rasters
    texture      floor orthophoto from the fisheyes
    depth_prior  depth model (DA3) fused with the LiDAR map in its holes
    gs           3D Gaussians on the LiDAR skeleton (gsplat 3DGUT)
    gs_verify    Gaussians against held-out LiDAR rays, the map, a reference
    ortho        orthophoto per level from the Gaussians

Work directory layout:
    trips/<trip>/odometry.{npz,tum}, kf/, kf_final/, poses.{npz,tum}
    graph/kf_poses.npz, loops.json, summary.json
    pose_set/pose_set.json, <trip>.npz;  calibration.json
    map/point_map.npz (tiles/*.npz + tiles.json with map.tile_size),
        levels.json, floor_grids.npz
    bev/level_<i>.npz (+ PNG previews)
    depth_prior/<trip>/<camera>/<stamp>.npz, frames.csv, points.npz
    gs/model.npz, model.ply, summary.json, previews/, verify.json
    ortho/level_<i>.png, .json
    provenance.json
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from garage_recon import config as config_lib
from garage_recon import core_config
from garage_recon import provenance
from garage_recon.data import dataset
from garage_recon.models import registry
from garage_recon.recon import backend
from garage_recon.recon import bev
from garage_recon.recon import calibration
from garage_recon.recon import depth_prior
from garage_recon.recon import floor_texture
from garage_recon.recon import keyframes as keyframes_lib
from garage_recon.recon import levels as levels_lib
from garage_recon.recon import odometry
from garage_recon.recon import pose_set as pose_set_lib
from garage_recon.recon import static_map
from garage_recon.recon import trajectory

STAGES = ("odometry", "graph", "trajectory", "calibration", "map", "bev",
          "texture", "depth_prior", "gs", "gs_verify", "ortho")


class ReconPipeline:
    """Runs and resumes the section 4 stages on a work directory."""

    def __init__(self, trip_dirs, work_dir, cfg, log=print,
                 models: registry.ModelSuite | None = None):
        self.datasets = [dataset.TripDataset(d) for d in trip_dirs]
        self.work = pathlib.Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.log = log
        self.models = models or registry.ModelSuite()
        self._pose_set = None
        config_lib.save_config(cfg, self.work / "config.used.yaml")

    # ------------------------------------------------------------- helpers
    def trip_dir(self, data: dataset.TripDataset) -> pathlib.Path:
        return self.work / "trips" / data.trip_id

    def odometries(self) -> list[odometry.TripOdometry]:
        return [odometry.TripOdometry.load(d.trip_id, self.trip_dir(d))
                for d in self.datasets]

    def keyframes(self, final: bool = True) -> keyframes_lib.KeyframeSet:
        """Keyframes; `final` selects the sweeps of the final trajectory."""
        odoms = self.odometries()
        final_dir = self.trip_dir(self.datasets[0]) / keyframes_lib.FINAL_SWEEPS
        sweeps = (keyframes_lib.FINAL_SWEEPS if final and final_dir.is_dir()
                  else keyframes_lib.ODOMETRY_SWEEPS)
        return keyframes_lib.KeyframeSet.from_odometry(odoms, sweeps)

    def pose_set(self) -> pose_set_lib.PoseSet:
        """The current pose set (cached until a stage replaces it)."""
        if self._pose_set is None:
            self._pose_set = pose_set_lib.PoseSet.load(self.work,
                                                       self.datasets)
        return self._pose_set

    def _set_pose_set(self, pose_set: pose_set_lib.PoseSet) -> None:
        pose_set.save(self.work)
        self._pose_set = pose_set

    def kf_poses(self, keyframes: keyframes_lib.KeyframeSet | None = None
                 ) -> np.ndarray:
        """(K, 4, 4) world <- lidar of the keyframes, from the pose set."""
        keyframes = keyframes or self.keyframes()
        return calibration.keyframe_poses(keyframes, self.pose_set())

    def vehicle_poses(self, data: dataset.TripDataset
                      ) -> tuple[np.ndarray, np.ndarray]:
        """(stamps, world <- vehicle) at the LiDAR frames, from the pose set."""
        trip = self.pose_set().trips[self.pose_set().index(data.trip_id)]
        return trip.knot_stamps, trip.knot_poses

    def point_map(self, bounds=None, fields=None, level=None,
                  with_index: bool = False):
        """The map (built in one piece or in tiles), or only the tiles that
        touch `bounds` (x0, y0, x1, y1), the `fields` asked for, the points
        of `level` (static_map.load_point_map)."""
        return static_map.load_point_map(self.work / "map", bounds, fields,
                                         level, with_index)

    def levels(self) -> levels_lib.LevelModel:
        return levels_lib.LevelModel.load(self.work / "map")

    def record(self, stage: str, uses_cameras: bool) -> None:
        provenance.record(self.work, f"recon.{stage}", self.pose_set(),
                          uses_cameras)

    def run(self, stages=STAGES) -> None:
        for stage in stages:
            start = time.time()
            getattr(self, f"stage_{stage}")()
            self.log(f"== stage {stage} done in {time.time() - start:.1f}s")

    # -------------------------------------------------------------- stages
    def stage_odometry(self) -> None:
        """Per-trip LiDAR odometry and keyframe sweeps."""
        for data in self.datasets:
            odometry.run_trip_odometry(data, self.cfg, self.trip_dir(data),
                                       self.log)

    def stage_graph(self) -> None:
        """Loops and the pose graph; IMU bias / velocity per keyframe."""
        odoms = self.odometries()
        keyframes = keyframes_lib.KeyframeSet.from_odometry(odoms)
        pairs = None
        appearance_cfg = self.cfg.loops.get("appearance") or {}
        if appearance_cfg.get("enabled") and self.models.features is not None:
            from garage_recon.recon import appearance  # Needs torch.
            pairs = appearance.appearance_pairs(
                keyframes, self.datasets, self.models.features, self.cfg,
                self.log)
        text_cfg = self.cfg.loops.get("text") or {}
        if text_cfg.get("enabled") and self.models.ocr is not None:
            from garage_recon.recon import text_loops
            found = text_loops.text_pairs(keyframes, self.datasets,
                                          self.models.ocr, self.cfg, self.log)
            pairs = (found if pairs is None else np.unique(
                np.vstack([pairs, found]), axis=0))
        result = backend.run_graph(keyframes, self.datasets, odoms, self.cfg,
                                   self.log, pairs)
        poses = result["poses"]
        out = self.work / "graph"
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "kf_poses.npz", poses=poses, odom=keyframes.odom,
                 trip=keyframes.trip, frame=keyframes.frame,
                 stamp=keyframes.stamp, velocity=result["velocities"],
                 bias=result["biases"],
                 inertial_valid=result["inertial_valid"])
        (out / "loops.json").write_text(
            json.dumps([backend.loop_json(lp) for lp in result["loops"]]))
        summary = dict(result["summary"])
        summary.update({"keyframes": len(keyframes),
                        "loops": len(result["loops"]),
                        "anchors": len(result["anchors"]),
                        "gravity_factors": len(result["gravity"]),
                        "loops_tested": result["loops_tested"],
                        "loops_verified": result["loops_verified"],
                        "ba_rounds": result["ba_rounds"],
                        "trip_alignments": [
                            {**a, "trip": self.datasets[a["trip"]].trip_id,
                             "T": np.asarray(a["T"]).tolist()}
                            for a in result["trip_alignments"]]})
        valid = np.asarray(result["inertial_valid"], bool)
        for t, data in enumerate(self.datasets):
            mine = valid & (keyframes.trip == t) if valid.any() else valid
            if not mine.any():
                continue
            mean = np.asarray(result["biases"])[mine].mean(axis=0)
            summary[f"imu_bias_{data.trip_id}"] = {
                "gyro_deg_s": np.degrees(mean[:3]).round(4).tolist(),
                "accel_m_s2": mean[3:].round(4).tolist()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        for t, (data, odom) in enumerate(zip(self.datasets, odoms)):
            lidar = backend.frame_poses(keyframes, poses, t, odom.stamps,
                                        odom.lidar_poses)
            trajectory.save_frame_poses(self.work, data.trip_id, odom.stamps,
                                        lidar,
                                        data.rig.primary_lidar.T_vehicle_lidar)

    def stage_trajectory(self) -> None:
        """Builds the pose set and re-deskews the keyframe sweeps."""
        with np.load(self.work / "graph" / "kf_poses.npz") as z:
            table = {k: z[k] for k in z.files}
        frame_poses = []
        for data in self.datasets:
            with np.load(self.trip_dir(data) / "poses.npz") as z:
                frame_poses.append((z["stamps"], z["lidar"]))
        pose_set = trajectory.build_pose_set(self.datasets, frame_poses, table)
        self._set_pose_set(pose_set)
        stats = trajectory.redeskew_keyframes(pose_set, self.odometries(),
                                              self.cfg, self.log)
        self.log(f"[trajectory] pose set {pose_set.pose_set_id}: "
                 f"{stats['keyframes']} sweeps re-deskewed, median point "
                 f"shift {stats['median_shift_cm']} cm")
        self.record("trajectory", uses_cameras=False)

    def stage_calibration(self) -> None:
        """Camera time offsets and extrinsics; a new pose set."""
        if not self.cfg.calibration.enabled or self.models.matcher is None:
            self.log("[calibration] skipped (disabled or no image matcher); "
                     "cameras keep the rig.yaml calibration")
            return
        pose_set, report = calibration.calibrate_cameras(
            self.pose_set(), self.keyframes(), self.models.matcher, self.cfg,
            self.log)
        self._set_pose_set(pose_set)
        (self.work / "calibration.json").write_text(json.dumps(
            {"pose_set_id": pose_set.pose_set_id, "cameras": report},
            indent=1))
        self.record("calibration", uses_cameras=True)

    def stage_map(self) -> None:
        """Static point map, levels and dynamic points (in tiles with
        map.tile_size)."""
        cfg = self.cfg
        keyframes = self.keyframes()
        poses = self.kf_poses(keyframes)
        # Vehicle origins on the floor: the levels and the ego swept volume.
        paths = [self.vehicle_poses(d)[1][:, :3, 3] for d in self.datasets]
        out = self.work / "map"
        levels_out, summary = static_map.build_map(keyframes, poses, paths,
                                                   cfg, out, self.log)
        levels = levels_lib.from_core(levels_out)
        n = summary["points"]
        if cfg.map.dynamic.enabled:
            dynamic = summary["moved"] + summary["changed"]
            self.log(f"[map] dynamic points: {dynamic} / {n} "
                     f"({100.0 * dynamic / max(n, 1):.1f}%), moved "
                     f"{summary['moved']}, changed between sessions "
                     f"{summary['changed']}")
        levels.save(out)
        tiles = (f" in {summary['tiles']} tiles"
                 if float(cfg.map.get("tile_size") or 0.0) > 0.0 else "")
        self.log(f"[map] {n} points at {cfg.map.voxel} m{tiles}")
        self.record("map", uses_cameras=False)

    def stage_bev(self) -> None:
        """Per-level rasters (intensity, heights)."""
        levels = self.levels()
        for lv in range(levels.n):
            # One level at a time: its rasters only use its own points.
            point_map = self.point_map(
                fields=("intensity", "level", "dynamic", "height", "normals"),
                level=lv)
            grid = bev.level_grid(point_map, lv, float(self.cfg.bev.res),
                                  float(self.cfg.bev.margin))
            rasters = bev.level_rasters(point_map, levels, grid, self.cfg)
            path = self.work / "bev" / f"level_{lv}.npz"
            bev.save_bev(path, grid, rasters)
            bev.to_png(path.with_name(f"level_{lv}_intensity.png"),
                       rasters["intensity"])
            bev.to_png(path.with_name(f"level_{lv}_height.png"),
                       rasters["height_max"], 0.0, 2.5)
            self.log(f"[bev] level {lv}: {grid.width}x{grid.height} cells")
        self.record("bev", uses_cameras=False)

    def stage_depth_prior(self) -> None:
        """DA3 depth fused with the LiDAR map, in the LiDAR holes (8.2)."""
        if (not self.cfg.depth_prior.enabled or self.models.depth is None
                or not any(d.rig.cameras for d in self.datasets)):
            self.log("[depth_prior] skipped (disabled, no depth model or no "
                     "cameras)")
            return
        depth_prior.run_depth_prior(self.pose_set(), self.point_map(),
                                    self.models.depth, self.cfg,
                                    self.work / "depth_prior", self.log)
        self.record("depth_prior", uses_cameras=True)

    def stage_gs(self) -> None:
        """3D Gaussians on the LiDAR skeleton (8.3)."""
        if not self.cfg.gs.enabled or not any(d.rig.cameras
                                              for d in self.datasets):
            self.log("[gs] skipped (disabled or no cameras)")
            return
        from garage_recon.recon import gs  # Needs torch and gsplat.
        keyframes = self.keyframes()
        gs.run_gs(self.pose_set(), self.point_map(), self.levels(), keyframes,
                  self.kf_poses(keyframes), self.cfg, self.work, self.log)
        self.record("gs", uses_cameras=True)

    def stage_gs_verify(self) -> None:
        """The Gaussians against held-out LiDAR rays, the map and a reference."""
        if not self.cfg.gs.enabled or not (self.work / "gs" / "model.npz").exists():
            self.log("[gs_verify] skipped (no Gaussians)")
            return
        from garage_recon.recon import gs_verify
        frames = {o.trip_id: o.kf_frames for o in self.odometries()}
        reference = None
        handheld = self.cfg.gs_verify.get("reference")
        report = self.work / "acceptance" / "report.json"
        if handheld and report.exists():
            # The registered handheld cloud, in the map frame.
            gt_from_map = np.array(json.loads(report.read_text())[
                "registration"]["gt_from_map"])
            with np.load(pathlib.Path(handheld) / "cloud.npz") as z:
                xyz = z["xyz"].astype(np.float64)
            map_from_gt = np.linalg.inv(gt_from_map)
            reference = xyz @ map_from_gt[:3, :3].T + map_from_gt[:3, 3]
        gs_verify.run_gs_verify(self.pose_set(), self.point_map(), frames,
                                self.cfg, self.work, reference, self.log)
        self.record("gs_verify", uses_cameras=True)

    def stage_ortho(self) -> None:
        """Orthophoto of every level from the Gaussians."""
        if not self.cfg.gs.enabled or not (self.work / "gs" / "model.npz").exists():
            self.log("[ortho] skipped (no Gaussians)")
            return
        from garage_recon.recon import gs_verify
        gs_verify.run_ortho(self.levels(), self.cfg, self.work, self.log)
        self.record("ortho", uses_cameras=True)

    def stage_texture(self) -> None:
        """Floor orthophoto from the camera images (fisheye and pinhole)."""
        if not self.cfg.floor_texture.enabled or not any(
                d.rig.cameras for d in self.datasets):
            self.log("[texture] skipped (disabled or no cameras)")
            return
        point_map, levels = self.point_map(), self.levels()
        for lv in range(levels.n):
            path = self.work / "bev" / f"level_{lv}.npz"
            grid, rasters = bev.load_bev(path)
            rgb, score, observations = floor_texture.fuse_floor_texture(
                self.pose_set(), grid, rasters["floor_z"],
                float(levels.z[lv]), point_map, self.cfg, self.log)
            rasters.update({"rgb": rgb, "texture_score": score,
                            "texture_nobs": observations})
            bev.save_bev(path, grid, rasters)
            bev.to_png(path.with_name(f"level_{lv}_rgb.png"), rgb)
        self.record("texture", uses_cameras=True)
