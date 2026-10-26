"""Command line entry point.

    garage-recon synth     --out DATA [--preset small] [--trips 2] [--seed 0]
                           [--ideal] [--no-handheld]
    garage-recon handheld  --data DATA [--seed 0]
    garage-recon convert   (--bag BAG [BAG ...] | --mandeye DIR)
                           --profile PROFILE.yaml --out TRIP [--trip-id ID]
                           [--start S] [--duration S] [--jobs N]
    garage-recon import-scan --cloud EXPORT --out HANDHELD
                           [--trajectory TUM] [--margin 15] [--voxel 0.02]
                           [--no-draft]
    garage-recon recon     --trips DATA/trip_00 ... --work WORK [--gt DATA/gt]
                           [--config cfg.yaml] [--stages ...]
    garage-recon semantics --trips ... --work WORK [--gt DATA/gt] [...]
    garage-recon qa        --trips ... --work WORK
    garage-recon eval      --work WORK --gt DATA/gt
    garage-recon accept    --work WORK --handheld DATA/handheld
                           [--method control|auto] [--config cfg.yaml]
    garage-recon viewer    --work WORK [--work WORK2 ...] --out DIR
                           [--name NAME ...] [--max-points 3000000]
                           [--serve PORT]
    garage-recon demo      --out DIR [--preset tiny|small|demo]

`handheld` adds the synthetic handheld-scanner ground truth (cloud and
digitised features in the scanner's own frame) to a synthetic dataset;
`convert` turns a ROS bag or a Mandeye recording into a trip with a
conversion profile (see convert/profiles/); `import-scan` turns a scanner export into a handheld
ground-truth directory with drafted levels and columns; `accept` registers
a reconstruction to a handheld result and writes the survey acceptance
report (work/acceptance/report.md); `viewer` exports reconstructions (map,
trajectories, semantic vectors, plus PLY / TUM files) to an interactive web
viewer and can serve it.

`--gt` points the oracle models (models.backend: auto | oracle) at the
ground truth of a synthetic dataset; without models the camera calibration
and the language route are skipped.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from garage_recon import config as config_lib
from garage_recon.acceptance import report as acceptance_report
from garage_recon.acceptance import scan_import
from garage_recon.data import synthetic
from garage_recon.data.synthetic import handheld
from garage_recon.eval import synthetic_eval
from garage_recon.io import viewer
from garage_recon.models import registry
from garage_recon.qa import selfcheck
from garage_recon.recon import pipeline as recon_pipeline
from garage_recon.semantics import pipeline as semantics_pipeline


def _log(message: str) -> None:
    print(message, flush=True)


def _pipelines(trips, work, cfg, gt_dir=None):
    """Recon and semantics pipelines with the models for `gt_dir`."""
    models = registry.build(cfg.models, gt_dir)
    recon = recon_pipeline.ReconPipeline(trips, work, cfg, _log, models)
    return recon, semantics_pipeline.SemanticsPipeline(recon, cfg, _log)


def _serve(directory: pathlib.Path, port: int) -> None:
    """Serves `directory` over HTTP until interrupted."""
    import functools  # pylint: disable=import-outside-toplevel
    import http.server  # pylint: disable=import-outside-toplevel
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(directory))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                         handler) as server:
        _log(f"viewer at http://localhost:{port}/ (Ctrl+C to stop)")
        server.serve_forever()


def summarize(report: dict) -> dict:
    """Compact headline numbers of an evaluation report."""
    out = {}
    trajectories = report.get("trajectory", {})
    if trajectories:
        out["ate_rmse_m"] = {k: round(v["ate_rmse"], 3)
                             for k, v in trajectories.items()}
    if "map_accuracy" in report:
        out["map_p90_m"] = round(report["map_accuracy"]["p90"], 3)
    for key in ("columns", "slots"):
        if key in report:
            out[key] = {m: (round(report[key][m], 3)
                            if isinstance(report[key][m], float)
                            else report[key][m])
                        for m in ("precision", "recall") if m in report[key]}
    slots = report.get("slots", {})
    if slots.get("corner_err_mean") is not None:
        out["slot_corner_err_mean_m"] = round(slots["corner_err_mean"], 3)
    if "static_vehicles" in report:
        out["parked_vehicles_f1"] = {
            k: round(v["f1"], 3) for k, v in report["static_vehicles"].items()}
    if "moving_objects" in report:
        out["moving_tracked"] = {k: f"{v['tracked']}/{v['gt']}"
                                 for k, v in report["moving_objects"].items()}
    poses = report.get("poses", {})
    if "camera_reprojection" in poses:
        out["camera_reprojection_px"] = {
            k: round(v["mean_px"], 2)
            for k, v in poses["camera_reprojection"].items()}
    if "imu_bias" in poses:
        out["imu_bias_error"] = poses["imu_bias"]
    semantics = report.get("language", {})
    if semantics:
        out["language"] = {k: semantics[k] for k in
                           ("point_miou", "frame_miou_3d", "frame_miou_2d",
                            "cross_view_consistency") if k in semantics}
    return out


def main(argv=None) -> int:
    """Parses the command line and runs one command; returns the exit code."""
    parser = argparse.ArgumentParser(
        prog="garage-recon", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="cmd", required=True)

    synth = commands.add_parser("synth",
                                help="generate a synthetic garage dataset")
    synth.add_argument("--out", required=True)
    synth.add_argument("--preset", default="small",
                       choices=["tiny", "small", "demo"])
    synth.add_argument("--trips", type=int, default=None)
    synth.add_argument("--seed", type=int, default=0)
    synth.add_argument("--stereo", action="store_true")
    synth.add_argument("--ideal", action="store_true",
                       help="no sensor errors (bias, time offset, extrinsic)")
    synth.add_argument("--no-handheld", action="store_true",
                       help="skip the handheld ground truth")
    synth.add_argument("--no-arrows", action="store_true",
                       help="no floor arrows on the aisles")

    scan = commands.add_parser(
        "handheld", help="add the synthetic handheld ground truth to a dataset")
    scan.add_argument("--data", required=True)
    scan.add_argument("--seed", type=int, default=0)

    convert = commands.add_parser(
        "convert", help="ROS bag -> trip directory (conversion profile)")
    source = convert.add_mutually_exclusive_group(required=True)
    source.add_argument("--bag", nargs="+", help="ROS 1 / ROS 2 bag(s)")
    source.add_argument("--mandeye", default=None,
                        help="Mandeye recording directory (lidarNNNN.laz, "
                             "imuNNNN.csv)")
    convert.add_argument("--profile", required=True,
                         help="profile YAML (e.g. convert/profiles/*.yaml)")
    convert.add_argument("--out", required=True)
    convert.add_argument("--trip-id", default=None)
    convert.add_argument("--start", type=float, default=0.0,
                         help="seconds to skip at the start")
    convert.add_argument("--duration", type=float, default=None)
    convert.add_argument("--jobs", type=int, default=1,
                         help="ROS bags: processes reading time windows in "
                              "parallel (compressed bags)")

    scan_in = commands.add_parser(
        "import-scan", help="scanner export -> handheld ground truth")
    scan_in.add_argument("--cloud", required=True)
    scan_in.add_argument("--out", required=True)
    scan_in.add_argument("--trajectory", default=None,
                         help="TUM path in the scan frame: crop and floors")
    scan_in.add_argument("--margin", type=float, default=15.0)
    scan_in.add_argument("--voxel", type=float, default=0.02)
    scan_in.add_argument("--no-draft", action="store_true",
                         help="do not draft features.json")

    for name, help_text in (
            ("recon", "section 4: pose set, map, levels, BEV"),
            ("semantics", "section 5: rules and language route"),
            ("qa", "ground-truth-free quality report")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--trips", nargs="+", required=True)
        sub.add_argument("--work", required=True)
        sub.add_argument("--config", default=None)
        sub.add_argument("--gt", default=None,
                         help="synthetic ground truth for the oracle models")
        if name != "qa":
            sub.add_argument("--stages", nargs="*", default=None)

    evaluate = commands.add_parser(
        "eval", help="evaluate against synthetic ground truth")
    evaluate.add_argument("--work", required=True)
    evaluate.add_argument("--gt", required=True)

    accept = commands.add_parser(
        "accept", help="survey acceptance against a handheld result")
    accept.add_argument("--work", required=True)
    accept.add_argument("--handheld", required=True)
    accept.add_argument("--method", choices=["control", "auto"], default=None)
    accept.add_argument("--config", default=None)

    view = commands.add_parser(
        "viewer", help="export reconstructions to an interactive web viewer")
    view.add_argument("--work", nargs="+", required=True)
    view.add_argument("--name", nargs="*", default=None,
                      help="one name per work directory (default: its name)")
    view.add_argument("--out", required=True)
    view.add_argument("--max-points", type=int, default=3_000_000)
    view.add_argument("--serve", type=int, default=None, metavar="PORT",
                      help="serve the viewer at http://localhost:PORT")

    locate = commands.add_parser(
        "localize", help="localize a trip in the navigation map of a "
        "reconstruction")
    locate.add_argument("--work", required=True,
                        help="reconstruction with the nav stage")
    locate.add_argument("--trip", required=True)
    locate.add_argument("--out", required=True)
    locate.add_argument("--config", default=None)
    locate.add_argument("--prior", choices=["none", "entrance", "reference"],
                        default="none",
                        help="none: global initialization; entrance: the "
                        "trip's entrance pose; reference: the reference "
                        "trajectory at the start (diagnostic)")
    locate.add_argument("--reference", default=None,
                        help="TUM trajectory (world <- vehicle) or a "
                        "directory with poses.tum, for the errors")
    locate.add_argument("--stride", type=int, default=1)
    locate.add_argument("--start", type=float, default=0.0,
                        help="seconds of the trip skipped")

    demo = commands.add_parser(
        "demo", help="synthesise, reconstruct, label and evaluate")
    demo.add_argument("--out", required=True)
    demo.add_argument("--preset", default="tiny",
                      choices=["tiny", "small", "demo"])
    demo.add_argument("--config", default=None)

    args = parser.parse_args(argv)
    start = time.time()
    if args.cmd == "synth":
        synthetic.generate(args.out, args.preset, args.trips, args.seed,
                           stereo=args.stereo, ideal=args.ideal,
                           arrows=not args.no_arrows, log=_log)
        if not args.no_handheld:
            handheld.simulate_handheld(args.out, args.seed, log=_log)
    elif args.cmd == "handheld":
        handheld.simulate_handheld(args.data, args.seed, log=_log)
    elif args.cmd == "convert":
        # Imported here: rosbags and laspy are optional dependencies.
        # pylint: disable-next=import-outside-toplevel
        from garage_recon.convert import profile as profile_lib
        profile = profile_lib.load(args.profile)
        if args.mandeye:
            # pylint: disable-next=import-outside-toplevel
            from garage_recon.convert import mandeye
            mandeye.convert(args.mandeye, profile, args.out, args.trip_id,
                            args.start, args.duration, log=_log)
        else:
            # pylint: disable-next=import-outside-toplevel
            from garage_recon.convert import rosbag
            rosbag.convert(args.bag, profile, args.out, args.trip_id,
                           args.start, args.duration, log=_log,
                           jobs=args.jobs)
    elif args.cmd == "import-scan":
        scan_import.import_scan(args.cloud, args.out, args.voxel,
                                args.trajectory, args.margin,
                                draft=not args.no_draft, log=_log)
    elif args.cmd in ("recon", "semantics"):
        cfg = config_lib.load_config(args.config)
        recon, semantics = _pipelines(args.trips, args.work, cfg, args.gt)
        target = recon if args.cmd == "recon" else semantics
        if args.stages:
            target.run(args.stages)
        else:
            target.run()
    elif args.cmd == "qa":
        cfg = config_lib.load_config(args.config)
        recon, semantics = _pipelines(args.trips, args.work, cfg, args.gt)
        selfcheck.run_selfcheck(recon, semantics, _log)
    elif args.cmd == "eval":
        synthetic_eval.evaluate(args.work, args.gt, _log)
    elif args.cmd == "accept":
        cfg = config_lib.load_config(args.config)
        if args.method:
            cfg.acceptance.method = args.method
        acceptance_report.run_acceptance(args.work, args.handheld, cfg, _log)
    elif args.cmd == "viewer":
        names = args.name or [pathlib.Path(w).resolve().name
                              for w in args.work]
        out = viewer.export_viewer(args.work, names, args.out,
                                   args.max_points, log=_log)
        if args.serve:
            _serve(out, args.serve)
    elif args.cmd == "localize":
        # pylint: disable-next=import-outside-toplevel
        from garage_recon.nav import localize
        cfg = config_lib.load_config(args.config)
        localize.run(pathlib.Path(args.work), pathlib.Path(args.trip),
                     pathlib.Path(args.out), cfg, args.prior, args.reference,
                     args.stride, args.start, log=_log)
    elif args.cmd == "demo":
        out = pathlib.Path(args.out)
        trips = synthetic.generate(out / "data", args.preset, log=_log)
        cfg = config_lib.load_config(args.config)
        recon, semantics = _pipelines(trips, out / "work", cfg,
                                      out / "data" / "gt")
        recon.run()
        semantics.run()
        selfcheck.run_selfcheck(recon, semantics, _log)
        report = synthetic_eval.evaluate(out / "work", out / "data" / "gt",
                                         lambda *_: None)
        _log(json.dumps(summarize(report), indent=1, ensure_ascii=False))
    _log(f"done in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
