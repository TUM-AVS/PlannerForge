"""CLI dispatcher for the OSM scenario-generation pipeline.

Invoked from PlannerForge's main process as a subprocess in the cr37 conda env:

    conda run -n cr37 python -m osm_pipeline.run stage1 \
        --south 48.10 --west 11.55 --north 48.13 --east 11.60 \
        --classes roads --density medium --sim-seconds 20 \
        --vehicle-types car --simulate-cr --record-cr-gif

The parent process parses stdout for "Label → /abs/path" lines (the same
convention osm/llm_wrapper.py uses) to extract the generated artefact
paths. Stderr/stdout are streamed live to the Gradio log textbox.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _emit(label: str, value) -> None:
    """Print one `Label → value` line on stdout, flushed so the parent
    process sees it in real time."""
    print(f"{label} → {value}", flush=True)


def cmd_stage1(args) -> int:
    from . import (bev_cr, bev_osm, bev_sumo, cr_convert, cr_simulate,
                   gif_cr, osm_fetch, sumo_convert)

    bbox = (args.south, args.west, args.north, args.east)
    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or ["all"]
    vehicle_types = [v.strip() for v in args.vehicle_types.split(",")
                     if v.strip()] or ["car"]

    osm_path = osm_fetch.fetch(location=None, classes=classes, bbox=bbox)
    _emit("OSM", osm_path)

    try:
        net_path = sumo_convert.convert(osm_path)
        _emit("SUMO net", net_path)
        sumo_png = bev_sumo.render(net_path)
        _emit("SUMO BEV", sumo_png)
    except Exception as e:
        _emit("SUMO BEV skipped", f"{type(e).__name__}: {e}")

    try:
        osm_png = bev_osm.render(osm_path, bbox=bbox)
        _emit("OSM BEV", osm_png)
    except Exception as e:
        _emit("OSM BEV skipped", f"{type(e).__name__}: {e}")

    try:
        osm_roads_png = bev_osm.render_roads(osm_path, bbox=bbox)
        _emit("OSM roads", osm_roads_png)
    except Exception as e:
        _emit("OSM roads skipped", f"{type(e).__name__}: {e}")

    cr_path = None
    try:
        cr_path = cr_convert.convert(osm_path)
        _emit("CR scenario", cr_path)
        cr_png = bev_cr.render(cr_path)
        _emit("CR BEV", cr_png)
    except Exception as e:
        _emit("CR skipped", f"{type(e).__name__}: {e}")
        return 1

    if args.simulate_cr:
        sim_steps = max(1, int(round(args.sim_seconds / 0.1)))
        try:
            sim_result = cr_simulate.simulate(
                cr_path,
                vehicle_types=vehicle_types,
                density=args.density,
                simulation_steps=sim_steps,
            )
            sim_xml = sim_result["sim_xml"]
            _emit("CR sim scenario", sim_xml)
            sim_png = bev_cr.render(sim_xml)
            _emit("CR sim BEV", sim_png)
            # Re-render the SUMO BEV from the CR2Sumo-aligned network so
            # what the user sees matches the simulated traffic.
            cr_sim_net = sim_result.get("sumo_net")
            if cr_sim_net is not None and Path(cr_sim_net).exists():
                try:
                    aligned_png = bev_sumo.render(Path(cr_sim_net))
                    _emit("SUMO BEV (cr-aligned)", aligned_png)
                except Exception as e:
                    _emit("SUMO BEV (cr-aligned) skipped",
                          f"{type(e).__name__}: {e}")
            if args.record_cr_gif:
                try:
                    # No fps override: gif_cr derives the frame duration
                    # from the sampling stride so the animation plays at the
                    # same speed as the database/modified GIFs.
                    cr_gif = gif_cr.record(sim_xml, n_frames=sim_steps)
                    _emit("CR sim GIF", cr_gif)
                except Exception as e:
                    _emit("CR GIF skipped", f"{type(e).__name__}: {e}")
        except Exception as e:
            _emit("CR sim skipped", f"{type(e).__name__}: {e}")

    return 0


def cmd_synthesize(args) -> int:
    from . import cr_planning_problem

    sim_xml = Path(args.sim_xml)
    out_path = Path(args.out_path) if args.out_path else None
    goal_lid = args.goal_lanelet_id if args.goal_lanelet_id is not None else None

    pp_xml = cr_planning_problem.synthesize(
        cr_path=sim_xml,
        ego_obstacle_id=args.ego_id,
        goal_lanelet_id=goal_lid,
        goal_position=args.goal_position,
        goal_offset_steps=args.goal_offset_steps,
        goal_length_m=args.goal_length_m,
        goal_width_m=args.goal_width_m,
        out_path=out_path,
    )
    _emit("PP XML", pp_xml)
    info = getattr(cr_planning_problem, "LAST_SYNTH_INFO", None)
    if info:
        _emit("PP info", info)
    return 0


def cmd_list_egos(args) -> int:
    from . import cr_planning_problem
    ids = cr_planning_problem.list_ego_candidates(
        Path(args.sim_xml), limit=args.limit)
    # JSON line so the parent can json.loads() unambiguously.
    print("EGOS_JSON " + json.dumps(ids), flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="osm_pipeline.run")
    sub = p.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stage1",
                        help="OSM fetch + SUMO/CR convert + simulate + BEVs")
    p1.add_argument("--south", type=float, required=True)
    p1.add_argument("--west", type=float, required=True)
    p1.add_argument("--north", type=float, required=True)
    p1.add_argument("--east", type=float, required=True)
    p1.add_argument("--classes", default="roads",
                    help="comma-sep list of CLASS_GROUPS keys (roads, link, all)")
    p1.add_argument("--density", default="medium",
                    choices=["low", "medium", "high"])
    p1.add_argument("--sim-seconds", type=float, default=20.0)
    p1.add_argument("--vehicle-types", default="car",
                    help="comma-sep SUMO vClass keys (car, truck, ...)")
    p1.add_argument("--simulate-cr", action="store_true", default=True)
    p1.add_argument("--no-simulate-cr", dest="simulate_cr",
                    action="store_false")
    p1.add_argument("--record-cr-gif", action="store_true", default=True)
    p1.add_argument("--no-record-cr-gif", dest="record_cr_gif",
                    action="store_false")
    p1.set_defaults(func=cmd_stage1)

    p2 = sub.add_parser("synthesize",
                        help="Build a CR PlanningProblem on top of a sim XML")
    p2.add_argument("--sim-xml", required=True)
    p2.add_argument("--ego-id", type=int, required=True)
    p2.add_argument("--goal-lanelet-id", type=int, default=None,
                    help="omit to use ego's trajectory at goal_offset_steps")
    p2.add_argument("--goal-position", default="middle",
                    choices=["start", "middle", "end"])
    p2.add_argument("--goal-offset-steps", type=int, default=50)
    p2.add_argument("--goal-length-m", type=float, default=6.0)
    p2.add_argument("--goal-width-m", type=float, default=2.0)
    p2.add_argument("--out-path", default=None)
    p2.set_defaults(func=cmd_synthesize)

    p3 = sub.add_parser("list-egos",
                        help="List dynamic-obstacle IDs in a sim XML")
    p3.add_argument("--sim-xml", required=True)
    p3.add_argument("--limit", type=int, default=30)
    p3.set_defaults(func=cmd_list_egos)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
