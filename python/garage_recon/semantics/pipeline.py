"""Section 5 orchestration on top of a finished reconstruction work directory.

Every pose comes from the recon pipeline's pose set (keyframe poses, vehicle
paths, camera exposures with rolling shutter and time offset).

Geometric route (rules): structure, markings, slots, objects, occupancy,
frames, rules, export. Language route (2D models decide "what", the 3D
identity field keeps the views consistent): features, identities, language;
it needs a segmenter and a language encoder (models section).

Outputs (work/semantics/):
    labels.npz                 per-point class ids aligned with the point map
    structure.json             grid yaw, columns (axis labels), wall segments
    markings.json / .geojson   vectorised floor markings
    slots.json / .geojson      parking slots (corners, entrance, occupancy, ids)
    objects/static.json        parked objects per trip
    objects/moving.json        moving-object boxes per keyframe; tracks.json
    occupancy/<trip>/<stamp>.npz   semantics + LiDAR / camera visibility masks
    frames/<trip>/<camera>/<stamp>.json (+ _label.png)   per-frame labels
    features/<trip>/<camera>/<stamp>.npz   masks + mask embeddings
    identities.npz             identity of every carrier point, mask table
    language/labels.npz        per-point classes of the language route
    language/summary.json      identities, classes, cross-view consistency
    issues.json                review queue from consistency rules
    numbers.json               slot numbers and zones by OCR (merged by slots())
    nav/                       navigation map: ROS maps, layers, route graph,
                               slot goals with addresses, qa.json
    export/                    LAS/PLY, GeoJSON, Lanelet2 OSM
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np

from garage_recon import _core
from garage_recon import core_config
from garage_recon import provenance
from garage_recon.io import export
from garage_recon.nav import navmap
from garage_recon.recon import bev
from garage_recon.recon import pipeline as recon_pipeline
from garage_recon.semantics import frame_labels
from garage_recon.semantics import language_field
from garage_recon.semantics import markings as markings_lib
from garage_recon.semantics import numbers as numbers_lib
from garage_recon.semantics import objects2d
from garage_recon.semantics import objects4d
from garage_recon.semantics import occupancy
from garage_recon.semantics import rules
from garage_recon.semantics import slots as slots_lib
from garage_recon.semantics import structure as structure_lib
from garage_recon.semantics import taxonomy

STAGES = ("structure", "markings", "slots", "objects", "objects2d",
          "occupancy", "frames",
          "features", "identities", "language", "numbers", "rules", "nav",
          "export")


def box_timeline(static_by_trip: dict[int, list[objects4d.Box3D]],
                 moving: dict[int, list[objects4d.Box3D]],
                 kf_stamp: np.ndarray,
                 kf_trip: np.ndarray) -> _core.BoxTimeline:
    """Object boxes at any time (C++): the trip's parked objects plus moving
    tracks interpolated between keyframes."""
    timeline = _core.BoxTimeline()
    for t, boxes in static_by_trip.items():
        for box in boxes:
            timeline.add_static(int(t), box.to_core())
    for k, boxes in moving.items():
        for box in boxes:
            timeline.add_moving(int(kf_trip[k]), float(kf_stamp[k]),
                                box.to_core())
    return timeline


class SemanticsPipeline:
    """Runs and resumes the section 5 stages on a work directory."""

    def __init__(self, recon: recon_pipeline.ReconPipeline, cfg=None,
                 log=print):
        self.recon = recon
        self.cfg = cfg or recon.cfg
        self.log = log
        self.out = recon.work / "semantics"
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ loaders
    def labels(self) -> np.ndarray:
        with np.load(self.out / "labels.npz") as z:
            return z["labels"]

    def point_map(self):
        """Point map with the dynamic mask refined by object-level reasoning."""
        point_map = self.recon.point_map()
        with np.load(self.out / "labels.npz") as z:
            if "dynamic" in z:
                point_map.dynamic = z["dynamic"]
        return point_map

    def structure(self):
        data = json.loads((self.out / "structure.json").read_text())
        return (data["yaw"],
                [structure_lib.Column(**c) for c in data["columns"]],
                [structure_lib.Wall(**w) for w in data["walls"]])

    def markings(self) -> list[markings_lib.MarkingSegment]:
        return [markings_lib.MarkingSegment(
            **{k: v for k, v in m.items() if k != "length"})
                for m in json.loads((self.out / "markings.json").read_text())]

    def slots(self) -> list[slots_lib.Slot]:
        """Slots, with the numbers and zones of the numbers stage."""
        slots = [slots_lib.Slot(**s)
                 for s in json.loads((self.out / "slots.json").read_text())]
        path = self.out / "numbers.json"
        if path.exists():
            read = json.loads(path.read_text())["slots"]
            for slot in slots:
                for key, value in read.get(slot.slot_id, {}).items():
                    setattr(slot, key, value)
        return slots

    def boxes(self):
        """(parked boxes per trip index, moving boxes per keyframe)."""
        directory = self.out / "objects"
        trips = [d.trip_id for d in self.recon.datasets]
        static = json.loads((directory / "static.json").read_text())
        moving = json.loads((directory / "moving.json").read_text())
        parked = {trips.index(t): [objects4d.Box3D(**b) for b in v]
                  for t, v in static.items()}
        return parked, {int(k): [objects4d.Box3D(**b) for b in v]
                        for k, v in moving.items()}

    def paths_by_level(self) -> dict[int, np.ndarray]:
        """Driven xy path points of all trips, per level."""
        levels = self.recon.levels()
        out: dict[int, list] = {}
        for data in self.recon.datasets:
            path = self.recon.vehicle_poses(data)[1][:, :3, 3]
            lv, _, _ = levels.assign(path + np.array([0.0, 0.0, 0.1]))
            for level in np.unique(lv[lv >= 0]):
                out.setdefault(int(level), []).append(path[lv == level, :2])
        return {k: np.vstack(v) for k, v in out.items()}

    def record(self, stage: str, uses_cameras: bool) -> None:
        provenance.record(self.recon.work, f"semantics.{stage}",
                          self.recon.pose_set(), uses_cameras)

    def run(self, stages=STAGES) -> None:
        for stage in stages:
            start = time.time()
            getattr(self, f"stage_{stage}")()
            self.log(f"== stage {stage} done in {time.time() - start:.1f}s")

    # ------------------------------------------------------------- stages
    def stage_structure(self) -> None:
        """Floors, ceilings, walls, columns (grid and axis labels)."""
        point_map, levels = self.recon.point_map(), self.recon.levels()
        paths = [self.recon.vehicle_poses(d)[1][:, :2, 3]
                 for d in self.recon.datasets]
        result = structure_lib.extract_structure(point_map, levels, paths,
                                                 self.cfg, self.log)
        # `structure` stays untouched so that later stages can be re-run.
        np.savez_compressed(self.out / "labels.npz", labels=result.labels,
                            structure=result.labels, dynamic=point_map.dynamic)
        (self.out / "structure.json").write_text(
            json.dumps(result.to_json(), indent=1))
        self.record("structure", uses_cameras=False)

    def stage_markings(self) -> None:
        """Vectorised floor markings from the BEV rasters."""
        segments: list[markings_lib.MarkingSegment] = []
        yaw, _, _ = self.structure()
        for path in sorted((self.recon.work / "bev").glob("level_*.npz")):
            grid, rasters = bev.load_bev(path)
            segments += markings_lib.extract_markings(rasters, grid, self.cfg,
                                                      yaw=yaw, log=self.log)
        markings_lib.classify_markings(segments, self.paths_by_level(),
                                       self.cfg)
        (self.out / "markings.json").write_text(
            json.dumps([s.to_json() for s in segments]))
        export.write_geojson(self.out / "markings.geojson", [
            export.feature("LineString", [s.p0, s.p1],
                           {"level": s.level, "cls": s.cls, "width": s.width})
            for s in segments])
        self.record("markings", uses_cameras=False)

    def stage_slots(self) -> None:
        """Parking slots from the markings, columns and paths."""
        yaw, columns, _ = self.structure()
        segments = self.markings()
        paths = self.paths_by_level()
        slots: list[slots_lib.Slot] = []
        for path in sorted((self.recon.work / "bev").glob("level_*.npz")):
            grid, rasters = bev.load_bev(path)
            level = grid.level
            slots += slots_lib.assemble_slots(
                [s for s in segments if s.level == level], level, yaw,
                paths.get(level, np.zeros((0, 2))),
                [c for c in columns if c.level == level], grid,
                rasters["height_max"], rasters["floor_z"], self.cfg, self.log,
                floor_count=rasters.get("floor_count"),
                intensity=rasters.get("intensity"))
        (self.out / "slots.json").write_text(
            json.dumps([s.to_json() for s in slots]))
        export.write_geojson(self.out / "slots.geojson", [
            export.feature("Polygon", [s.corners + s.corners[:1]], {
                "slot_id": s.slot_id, "level": s.level,
                "occupied": s.occupied, "inferred": s.inferred,
                "width": s.width, "depth": s.depth,
                "entrance": [s.corners[0], s.corners[1]]})
            for s in slots])
        self.record("slots", uses_cameras=False)

    def stage_objects(self) -> None:
        """Parked and moving objects; their labels on the map."""
        recon = self.recon
        keyframes = recon.keyframes()
        poses = recon.kf_poses(keyframes)
        point_map, levels = recon.point_map(), recon.levels()
        with np.load(self.out / "labels.npz") as z:
            labels = z["structure"].copy()
        yaw, _, _ = self.structure()
        static_points = ~point_map.dynamic
        index = objects4d.VoxelLabelIndex(point_map.xyz[static_points],
                                          labels[static_points], 0.1)
        slots = self.slots() if (self.out / "slots.json").exists() else []
        tracks, static, moving, moving_points = objects4d.run_objects(
            keyframes, poses, levels, index, point_map, labels, yaw,
            self.paths_by_level(), self.cfg, slots, self.log)
        dynamic = point_map.dynamic | moving_points
        labels[moving_points] = taxonomy.cid("vehicle_moving")
        directory = self.out / "objects"
        directory.mkdir(exist_ok=True)
        (directory / "tracks.json").write_text(json.dumps(tracks))
        (directory / "static.json").write_text(json.dumps(
            {keyframes.trip_ids[t]: [b.to_json() for b in v]
             for t, v in static.items()}))
        (directory / "moving.json").write_text(json.dumps(
            {str(k): [b.to_json() for b in v] for k, v in moving.items()}))
        # Object labels on the map: parked objects take static points or
        # their session's points, moving objects take dynamic points.
        seen = set()
        for t, boxes in static.items():
            for box in boxes:
                inside = (box.contains(point_map.xyz, margin=0.1)
                          & (point_map.height > 0.05))
                inside &= (np.isin(labels, [0, taxonomy.cid("mep")])
                           & (~point_map.dynamic | (point_map.trip == t)))
                labels[inside] = taxonomy.cid(box.cls)
        for boxes in moving.values():
            for box in boxes:
                if box.track_id in seen:
                    continue
                seen.add(box.track_id)
                inside = (box.contains(point_map.xyz, margin=0.2)
                          & point_map.dynamic & (labels == 0))
                labels[inside] = taxonomy.cid(box.cls)
        with np.load(self.out / "labels.npz") as z:
            structure = z["structure"]
        np.savez_compressed(self.out / "labels.npz", labels=labels,
                            structure=structure, dynamic=dynamic)
        point_map.dynamic = dynamic
        for path in sorted((recon.work / "bev").glob("level_*.npz")):
            grid, rasters = bev.load_bev(path)
            rasters["clearance"] = clearance_raster(point_map, labels, grid)
            bev.save_bev(path, grid, rasters)
            bev.to_png(path.with_name(f"level_{grid.level}_clearance.png"),
                       rasters["clearance"], 1.5, 3.5)
        self.record("objects", uses_cameras=False)

    def stage_objects2d(self) -> None:
        """Objects of the 2D detector lifted onto the map (README 5.2)."""
        models = self.recon.models
        if not self.cfg.objects2d.enabled or models.detector is None:
            self.log("[objects2d] skipped (disabled or no detector: "
                     "models.owlv2)")
            return
        result = objects2d.run(self.recon.pose_set(), self.point_map(), models,
                               self.cfg, self.out / "objects2d",
                               self.recon.levels(), self.log)
        with np.load(self.out / "labels.npz") as z:
            stored = {k: z[k] for k in z.files}
        stored["labels"], count = objects2d.merge(
            stored["labels"], result["static"], result["labels"],
            result["confidence"], list(self.cfg.objects2d.override),
            float(self.cfg.objects2d.min_confidence))
        np.savez_compressed(self.out / "labels.npz", **stored)
        self.log(f"[objects2d] {count} map points relabelled")
        self.record("objects2d", uses_cameras=True)

    def stage_occupancy(self) -> None:
        """Per-keyframe semantic occupancy with visibility masks."""
        if not self.cfg.occupancy.enabled:
            return
        recon = self.recon
        keyframes = recon.keyframes()
        pose_set = recon.pose_set()
        point_map = self.point_map()
        labels = self.labels()
        static = ~point_map.dynamic
        timeline = box_timeline(*self.boxes(), keyframes.stamp, keyframes.trip)
        count = 0
        for t, data in enumerate(recon.datasets):
            rig = pose_set.rig(t)
            generator = occupancy.occupancy_generator(
                point_map.xyz[static], labels[static], rig, self.cfg)
            directory = self.out / "occupancy" / data.trip_id
            directory.mkdir(parents=True, exist_ok=True)
            stride = int(self.cfg.occupancy.frame_stride)
            for k in keyframes.indices_of_trip(t)[::stride]:
                stamp = float(keyframes.stamp[k])
                world_from_vehicle = pose_set.world_from_vehicle(t, stamp)[0]
                points, _ = keyframes.load(int(k))
                semantics, mask_lidar, mask_camera = generator.generate(
                    world_from_vehicle, timeline.at(t, stamp), points)
                np.savez_compressed(
                    directory / f"{int(round(stamp * 1e9))}.npz",
                    semantics=semantics, mask_lidar=mask_lidar,
                    mask_camera=mask_camera,
                    T_world_vehicle=world_from_vehicle,
                    voxel=self.cfg.occupancy.voxel,
                    range=np.asarray(self.cfg.occupancy.range))
                count += 1
        self.log(f"[occupancy] {count} frames")
        self.record("occupancy", uses_cameras=True)

    def stage_frames(self) -> None:
        """Per-image labels by projection with the exposure poses."""
        if not self.cfg.frames.enabled:
            return
        recon = self.recon
        keyframes = recon.keyframes()
        pose_set = recon.pose_set()
        point_map = self.point_map()
        labels = self.labels()
        static = ~point_map.dynamic
        slots, segments = self.slots(), self.markings()
        labeler = frame_labels.frame_labeler(point_map.xyz[static],
                                             labels[static], slots, segments,
                                             self.cfg)
        timeline = box_timeline(*self.boxes(), keyframes.stamp, keyframes.trip)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        scores = []
        count = 0
        for t, data in enumerate(recon.datasets):
            rig = pose_set.rig(t)
            for name in rig.cameras:  # Fisheye and pinhole alike.
                calib = rig.cameras[name]
                camera = calib.make_model()
                directory = self.out / "frames" / data.trip_id / name
                directory.mkdir(parents=True, exist_ok=True)
                stride = int(self.cfg.frames.frame_stride)
                for ns in data.camera_stamps_ns(name)[::stride]:
                    exposure = pose_set.camera_exposure(t, name, int(ns))
                    center = pose_set.camera_center_pose(t, name, int(ns))
                    mid_time = 0.5 * sum(pose_set.row_times(t, name, int(ns)))
                    image = data.load_image(name, ns)
                    record, label_image = frame_labels.label_camera_frame(
                        labeler, calib, camera, exposure, image, slots,
                        segments, timeline.at(t, mid_time))
                    record.update({
                        "stamp_ns": int(ns), "trip": data.trip_id,
                        "T_world_cam": np.asarray(exposure).tolist(),
                        "T_world_cam_center": center.tolist(),
                        "pose_set_id": pose_set.pose_set_id})
                    (directory / f"{ns}.json").write_text(json.dumps(record))
                    if self.cfg.frames.write_label_images:
                        cv2.imwrite(str(directory / f"{ns}_label.png"),
                                    label_image)
                    if record["consistency_score"] is not None:
                        scores.append(record["consistency_score"])
                    count += 1
        summary = {
            "frames": count,
            "consistency_median": float(np.median(scores)) if scores else None,
            "consistency_p10": (float(np.percentile(scores, 10)) if scores
                                else None),
        }
        (self.out / "frames" / "summary.json").write_text(
            json.dumps(summary, indent=1))
        self.log(f"[frames] {count} camera frames labelled, consistency "
                 f"median {summary['consistency_median']}")
        self.record("frames", uses_cameras=True)

    # ------------------------------------------------------ language route
    def _language_ready(self, stage: str) -> bool:
        models = self.recon.models
        if models.segmenter is None or models.encoder is None:
            self.log(f"[{stage}] skipped (no segmenter / language encoder)")
            return False
        return True

    def _frame_keys(self) -> list[language_field.FrameKey]:
        return language_field.frame_keys(
            self.recon.pose_set(), int(self.cfg.features.frame_stride))

    def stage_features(self) -> None:
        """Masks and mask embeddings of every image (2D models)."""
        if not self.cfg.features.enabled or not self._language_ready(
                "features"):
            return
        keys = self._frame_keys()
        masks = language_field.extract_features(
            self.recon.pose_set(), self.recon.models, keys,
            self.out / "features")
        self.log(f"[features] {len(keys)} images, {masks} masks with "
                 f"embeddings ({self.recon.models.name} models)")
        self.record("features", uses_cameras=True)

    def _carriers(self) -> tuple[np.ndarray, np.ndarray]:
        """(static map point indices, their carrier subset indices)."""
        point_map = self.point_map()
        static = np.nonzero(~point_map.dynamic)[0]
        carriers = language_field.carrier_points(
            point_map.xyz[static], float(self.cfg.identities.carrier_voxel))
        return static, static[carriers]

    def stage_identities(self) -> None:
        """Cross-view identities of the masks via the 3D map."""
        if not self.cfg.features.enabled or not self._language_ready(
                "identities"):
            return
        keys = self._frame_keys()
        _, carriers = self._carriers()
        point_map = self.point_map()
        normals = (None if point_map.normals is None
                   else point_map.normals[carriers].astype(np.float64))
        result = language_field.associate(
            self.recon.pose_set(), point_map.xyz[carriers], normals, keys,
            self.out / "features", core_config.identity_config(self.cfg))
        masks = result["masks"]
        np.savez_compressed(
            self.out / "identities.npz", carriers=carriers,
            identity=result["point_identity"],
            confidence=result["point_confidence"],
            num_identities=result["num_identities"],
            **{f"mask_{k}": v for k, v in masks.items()},
            view_trip=np.asarray([k.trip_id for k in keys]),
            view_camera=np.asarray([k.camera for k in keys]),
            view_stamp_ns=np.asarray([k.stamp_ns for k in keys], np.int64))
        assigned = result["point_identity"] >= 0
        self.log(f"[identities] {result['num_identities']} identities over "
                 f"{len(keys)} images; {100.0 * assigned.mean():.0f}% of "
                 f"{len(carriers)} carrier points identified")
        self.record("identities", uses_cameras=True)

    def stage_language(self) -> None:
        """Identity classes from the text prompts; consistency."""
        if not self.cfg.features.enabled or not self._language_ready(
                "language"):
            return
        section = self.cfg.language
        keys = self._frame_keys()
        with np.load(self.out / "identities.npz") as z:
            carriers = z["carriers"]
            carrier_identity = z["identity"]
            association = {"masks": {
                k[len("mask_"):]: z[k] for k in z.files
                if k.startswith("mask_")}}
        embeddings = language_field.identity_embeddings(
            association, keys, self.out / "features", section)
        class_ids, class_embeddings = language_field.class_text_embeddings(
            self.recon.models.encoder)
        identity_class, identity_score = language_field.classify(
            embeddings["embeddings"], class_ids, class_embeddings,
            float(section.min_score))
        lookup = np.full(max(int(carrier_identity.max(initial=-1)) + 1,
                             int(embeddings["ids"].max(initial=-1)) + 1, 1),
                         taxonomy.UNLABELED, np.int64)
        lookup[embeddings["ids"]] = identity_class
        carrier_labels = np.where(carrier_identity >= 0,
                                  lookup[np.maximum(carrier_identity, 0)],
                                  taxonomy.UNLABELED)
        point_map = self.point_map()
        xyz = point_map.xyz[carriers]
        # Every point lies within a voxel diagonal of its voxel's carrier.
        reach = 2.0 * float(self.cfg.identities.carrier_voxel)
        labels = language_field.spread_to_points(
            point_map.xyz, xyz, carrier_labels.astype(np.uint8), reach,
            np.uint8(taxonomy.UNLABELED))
        identity = language_field.spread_to_points(
            point_map.xyz, xyz, carrier_identity.astype(np.int64), reach,
            np.int64(-1))
        labels[point_map.dynamic] = taxonomy.UNLABELED
        identity[point_map.dynamic] = -1
        directory = self.out / "language"
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / "labels.npz", labels=labels,
                            identity=identity)
        label_dir = (directory / "frames" if section.write_label_images
                     else None)
        # Frames are rendered from the full-resolution static map.
        static = ~point_map.dynamic
        consistency = language_field.frame_consistency(
            self.recon.pose_set(), point_map.xyz[static], labels[static], keys,
            self.out / "features", class_ids, class_embeddings, section,
            float(self.cfg.map.voxel), float(self.cfg.identities.max_range),
            label_dir)
        rates = np.asarray(consistency["rates"])
        counts = {taxonomy.BY_ID[int(c)].name: int(n) for c, n in
                  zip(*np.unique(labels[labels > 0], return_counts=True))}
        summary = {
            "identities": int(len(embeddings["ids"])),
            "classified": int((identity_class > 0).sum()),
            "point_classes": counts,
            "frames": len(rates),
            "cross_view_consistency_mean": (float(rates.mean()) if len(rates)
                                            else None),
            "cross_view_consistency_p10": (float(np.percentile(rates, 10))
                                           if len(rates) else None),
            "models": self.recon.models.name,
            "pose_set_id": self.recon.pose_set().pose_set_id,
        }
        language_field.write_summary(directory / "summary.json", summary)
        self.log(f"[language] {summary['classified']}/{summary['identities']} "
                 f"identities classified; cross-view consistency "
                 f"{summary['cross_view_consistency_mean']}")
        self.record("language", uses_cameras=True)

    def stage_numbers(self) -> None:
        """Slot numbers and zones by OCR, repaired row by row."""
        if not self.cfg.numbers.enabled or self.recon.models.ocr is None:
            self.log("[numbers] skipped (disabled or no OCR model)")
            return
        slots = [slots_lib.Slot(**s) for s in json.loads(
            (self.out / "slots.json").read_text())]
        summary = numbers_lib.read_numbers(self.recon, slots, self.cfg,
                                           self.log)
        (self.out / "numbers.json").write_text(json.dumps({
            "summary": summary,
            "slots": {s.slot_id: {"number": s.number, "zone": s.zone,
                                  "text_votes": s.text_votes}
                      for s in slots}}, indent=1))
        self.record("numbers", uses_cameras=True)

    def stage_nav(self) -> None:
        """Semantic navigation map: layers, route graph, slot goals, QA."""
        if not self.cfg.nav.enabled:
            return
        navmap.build(self, self.cfg, self.log)
        self.record("nav", uses_cameras=False)

    def stage_rules(self) -> None:
        """Consistency rules into the review queue."""
        _, columns, _ = self.structure()
        issues = (rules.check_slots(self.slots(), columns,
                                    self.paths_by_level(), self.cfg)
                  + rules.check_columns(columns))
        (self.out / "issues.json").write_text(json.dumps(issues, indent=1))
        by_severity = {}
        for issue in issues:
            by_severity[issue["severity"]] = by_severity.get(
                issue["severity"], 0) + 1
        self.log(f"[rules] {len(issues)} issues {by_severity}")
        self.record("rules", uses_cameras=False)

    def stage_export(self) -> None:
        """LAS / PLY, GeoJSON and Lanelet2 deliverables."""
        recon = self.recon
        point_map, levels = self.point_map(), recon.levels()
        labels = self.labels()
        directory = self.out / "export"
        directory.mkdir(exist_ok=True)
        extra = {"level": point_map.level.astype(np.int16),
                 "dynamic": point_map.dynamic.astype(np.uint8)}
        if not export.write_las(directory / "labeled_map.las", point_map.xyz,
                                labels, point_map.intensity, extra):
            self.log("[export] laspy not installed, writing PLY only")
        rgb = taxonomy.palette()[labels]
        export.write_ply(directory / "labeled_map.ply", point_map.xyz,
                         {"red": rgb[:, 0], "green": rgb[:, 1],
                          "blue": rgb[:, 2], "label": labels,
                          "intensity": point_map.intensity})
        _, columns, walls = self.structure()
        export.write_geojson(directory / "columns.geojson", [
            export.feature("Point", c.center, {
                "id": c.id, "level": c.level, "axis": c.axis_label,
                "size": c.size, "completed": c.completed})
            for c in columns])
        export.write_geojson(directory / "walls.geojson", [
            export.feature("LineString", [w.p0, w.p1],
                           {"level": w.level, "thickness": w.thickness})
            for w in walls])
        slots = self.slots()
        lots = {}
        for level in range(levels.n):
            level_slots = [s for s in slots if s.level == level]
            if level_slots:
                points = np.vstack([np.asarray(s.corners)
                                    for s in level_slots])
                lo, hi = points.min(axis=0) - 1.0, points.max(axis=0) + 1.0
                lots[level] = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                                        [hi[0], hi[1]], [lo[0], hi[1]]])
        export.write_lanelet2_osm(
            directory / "lanelet2_map.osm", slots,
            {i: float(z) for i, z in enumerate(levels.z)}, lots,
            dict(self.cfg.export.map_origin))
        self.log(f"[export] wrote {directory}")
        self.record("export", uses_cameras=False)


def clearance_raster(point_map, labels: np.ndarray, grid) -> np.ndarray:
    """Vertical free space above the floor per cell.

    The lowest overhead static structure (ceiling, ducts / pipes, beams,
    signs, lights); 0 on columns and walls; NaN where nothing overhead was
    observed.
    """
    selected = (point_map.level == grid.level) & ~point_map.dynamic
    height = point_map.height
    overhead = (selected
                & np.isin(labels, [taxonomy.cid("ceiling"), taxonomy.cid("mep"),
                                   taxonomy.cid("beam"), taxonomy.cid("sign"),
                                   taxonomy.cid("light")])
                & (height > 0.5))
    args = (grid.x0, grid.y0, grid.res, grid.width, grid.height)
    clear = _core.rasterize(point_map.xyz[overhead, :2],
                            height[overhead].astype(float), *args, "min",
                            np.nan)
    block = (selected
             & np.isin(labels, [taxonomy.cid("column"), taxonomy.cid("wall")])
             & (height > 0.2) & (height < 2.0))
    blocked = _core.rasterize(point_map.xyz[block, :2], np.ones(block.sum()),
                              *args, "count", 0.0) > 0
    clear[blocked] = 0.0
    return clear
