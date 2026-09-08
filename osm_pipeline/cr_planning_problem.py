"""
Synthesize a CommonRoad PlanningProblem for a chosen ego vehicle, starting
from a populated CR scenario produced by tools/cr_simulate.py.

The ego is one of the existing DynamicObstacles — we promote it:
 - initial_state comes straight from the obstacle
 - goal_state region is either
     a) a circle on a point along a specified lanelet's center polyline,
        picked with goal_position ∈ {"start", "middle", "end"}, or
     b) the ego's own trajectory point at t0 + goal_offset_steps (fallback)
 - the ego is removed from the scenario's dynamic_obstacles so Frenetix
   replaces it with a planned trajectory while the rest of SUMO's traffic
   remains as real obstacles.
"""
from __future__ import annotations

import math
from pathlib import Path

from config import DATA_CR


# Last call to `synthesize` writes a one-line diagnostic here so the
# Stage 2 wrapper can surface it in the result textbox without changing
# the function's return type.
LAST_SYNTH_INFO: str | None = None


# Frenetix planner kinematic limits. The absolute cap is below the
# library's strict 30 m/s ceiling because the reactive planner's
# default sampling matrix struggles to find feasible trajectories at
# motorway entry speeds — even without a tight leader, curvature +
# dense traffic + a 0.5 s replan cycle exhaust the sampling space.
_PLANNER_V_MAX_M_S = 22.0     # ≈ 80 km/h hard cap
_PLANNER_A_DECEL_M_S2 = 8.0   # max comfortable deceleration
_MIN_GAP_M = 5.0              # safety gap kept to the leading vehicle
_LOOKAHEAD_M = 120.0          # how far we scan for a leading vehicle


def _safe_initial_velocity(scenario, ego,
                           lookahead_m: float = _LOOKAHEAD_M,
                           a_decel_max: float = _PLANNER_A_DECEL_M_S2,
                           d_min_gap: float = _MIN_GAP_M,
                           v_max_planner: float = _PLANNER_V_MAX_M_S
                           ) -> tuple[float, float, float]:
    """Compute a kinematically safe initial speed for the ego.

    Returns ``(v_clamped, v_sumo, d_ahead)`` — the velocity to assign,
    the original SUMO velocity, and the distance to the closest forward
    obstacle on the same / next lanelet (``inf`` when nothing is ahead).
    """
    import numpy as np

    v_sumo = float(getattr(ego.initial_state, "velocity", 0.0) or 0.0)
    t0 = int(ego.initial_state.time_step)
    ego_xy = np.asarray(ego.initial_state.position, dtype=float)
    ego_yaw = float(getattr(ego.initial_state, "orientation", 0.0) or 0.0)
    heading = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])

    # Lanelets the ego is on now + their immediate successors. Anything
    # farther downstream is too far to plan against at the very first
    # planning cycle.
    relevant: set[int] = set()
    for lid in (scenario.lanelet_network
                .find_lanelet_by_position([ego_xy])[0] or []):
        relevant.add(int(lid))
        la = scenario.lanelet_network.find_lanelet_by_id(int(lid))
        if la is not None:
            for s in (la.successor or []):
                relevant.add(int(s))

    d_ahead = float("inf")
    for o in scenario.dynamic_obstacles:
        if o.obstacle_id == ego.obstacle_id:
            continue
        st = o.state_at_time(t0)
        if st is None:
            continue
        o_xy = np.asarray(st.position, dtype=float)
        delta = o_xy - ego_xy
        forward = float(delta @ heading)
        if forward <= d_min_gap or forward > lookahead_m:
            continue
        if relevant:
            o_lanelets = scenario.lanelet_network.find_lanelet_by_position(
                [o_xy])[0] or []
            if not (set(int(x) for x in o_lanelets) & relevant):
                continue
        if forward < d_ahead:
            d_ahead = forward

    if d_ahead == float("inf"):
        v_safe = v_max_planner
    else:
        margin = max(0.0, d_ahead - d_min_gap)
        v_safe = math.sqrt(2.0 * a_decel_max * margin)

    v_clamped = min(v_sumo, v_max_planner, v_safe)
    return v_clamped, v_sumo, d_ahead


def synthesize(cr_path: Path,
               ego_obstacle_id: int,
               goal_lanelet_id: int | None = None,
               goal_position: str = "middle",
               goal_offset_steps: int = 50,
               goal_length_m: float = 20.0,
               goal_width_m: float = 4.0,
               total_sim_steps: int | None = None,
               out_path: Path | None = None) -> Path:
    """Return a new CR .xml with a PlanningProblem for `ego_obstacle_id`.

    The goal region is a **rectangle** (matching CR benchmark convention,
    e.g. AUT_Haag-12_1_T-4 uses a 6.0 × 2.0 m oriented rectangle). The
    rectangle is oriented along either the lanelet's direction at the
    chosen point, or the ego's velocity direction at t0 + goal_offset_steps.
    """
    import numpy as np
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad.common.file_writer import (CommonRoadFileWriter,
                                                OverwriteExistingFile)
    from commonroad.geometry.shape import Rectangle
    from commonroad.planning.goal import GoalRegion
    from commonroad.planning.planning_problem import (PlanningProblem,
                                                       PlanningProblemSet)
    from commonroad.scenario.scenario import Tag
    from commonroad.scenario.state import CustomState
    from commonroad.common.util import Interval

    # Reset the diagnostic at the very top so a previous run's message
    # never leaks into this one (e.g. if `_safe_initial_velocity` raises
    # before reaching its own reset, or if validation below raises).
    global LAST_SYNTH_INFO
    LAST_SYNTH_INFO = None

    cr_path = Path(cr_path)
    if not cr_path.exists():
        raise FileNotFoundError(cr_path)

    if out_path is None:
        out_path = DATA_CR / f"{cr_path.stem}_pp_{ego_obstacle_id}.xml"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenario, _ = CommonRoadFileReader(str(cr_path)).open()

    ego = scenario.obstacle_by_id(ego_obstacle_id)
    if ego is None:
        raise ValueError(
            f"obstacle_id={ego_obstacle_id} not in scenario "
            f"(available: {[o.obstacle_id for o in scenario.dynamic_obstacles][:20]}...)")

    initial_state = ego.initial_state

    # ---- snap ego's initial position onto a lanelet polygon ------------
    # SUMO reports the ego's exact (x, y) on the lane centerline, but
    # crdesigner's lanelet polygons are slightly narrower than the SUMO
    # lane, so the point is sometimes a few centimetres OUTSIDE every
    # polygon. Frenetix then crashes here:
    #   helper_functions.py:205  best_lanelet = initial_lanelets[0]
    #   IndexError: list index out of range
    #
    # An earlier version of this snap moved the ego to the nearest
    # centerline VERTEX, but vertices sit exactly on the polygon
    # boundary. At UTM coordinate magnitudes (10^5–10^7 meters) the
    # double-precision ULP is ~mm, so shapely's point-in-polygon test
    # can still read a boundary-coincident point as "outside". We now
    # snap to the polygon's representative_point() — guaranteed strictly
    # interior even for non-convex shapes — which sidesteps the FP edge
    # case entirely.
    try:
        import numpy as _np
        ini_xy = _np.asarray(initial_state.position, dtype=float)
        hit = scenario.lanelet_network.find_lanelet_by_position([ini_xy])[0]
        if not hit:
            best_lid = None; best_d2 = float("inf")
            for la in scenario.lanelet_network.lanelets:
                verts = la.center_vertices
                if verts is None or len(verts) < 1:
                    continue
                arr = _np.asarray(verts, dtype=float)
                d2 = ((arr - ini_xy) ** 2).sum(axis=1).min()
                if d2 < best_d2:
                    best_d2 = d2; best_lid = la.lanelet_id
            if best_lid is not None:
                la = scenario.lanelet_network.find_lanelet_by_id(best_lid)
                snapped_xy = None
                try:
                    pt = la.polygon.shapely_object.representative_point()
                    snapped_xy = (float(pt.x), float(pt.y))
                except Exception:
                    # Polygon malformed — fall back to nearest centerline
                    # vertex. Frenetix may still IndexError, but the
                    # original behaviour is preserved.
                    arr = _np.asarray(la.center_vertices, dtype=float)
                    idx = int(((arr - ini_xy) ** 2).sum(axis=1).argmin())
                    snapped_xy = (float(arr[idx, 0]), float(arr[idx, 1]))
                initial_state.position = _np.array(
                    [snapped_xy[0], snapped_xy[1]])
    except Exception:
        # Snapping is best-effort — if it fails, fall through and let
        # the existing error surface as before.
        pass

    # ---- safe-stopping cap on the ego's initial velocity ---------------
    # SUMO sometimes spawns the ego at highway speed (e.g. 27 m/s) right
    # behind another vehicle. Frenetix's reactive planner then needs a
    # deceleration that exceeds its kinematic limits, finds no feasible
    # trajectory, and terminates the agent after a handful of steps.
    # Cap the initial speed at v = sqrt(2 · a_decel · (d_ahead − gap))
    # so the planner can always brake to a halt before the leader.
    try:
        v_safe, v_sumo, d_ahead = _safe_initial_velocity(scenario, ego)
        if v_safe + 1e-3 < v_sumo:
            initial_state.velocity = float(v_safe)
            d_str = (f"{d_ahead:.1f} m" if math.isfinite(d_ahead)
                     else "no leader in sight")
            LAST_SYNTH_INFO = (
                f"ego v0 capped: {v_sumo:.2f} → {v_safe:.2f} m/s "
                f"(leader {d_str}, decel ≤ {_PLANNER_A_DECEL_M_S2:.1f} m/s²)"
            )
    except Exception as e:
        LAST_SYNTH_INFO = (f"safe-velocity calc skipped: "
                           f"{type(e).__name__}: {e}")

    # --- goal point + orientation ---------------------------------------
    goal_xy = None
    goal_orientation = 0.0  # radians, measured from +X CCW (CommonRoad conv.)
    if goal_lanelet_id is not None:
        # `cr_simulate` shifts every lanelet ID by `+_LANELET_ID_OFFSET`
        # (1,000,000) in the saved `_sim.xml` so vehicle IDs (which we
        # equate to SUMO trip IDs) cannot collide. Users typing the
        # small CR id (e.g. "5") into the UI would otherwise hit
        # "lanelet_id not in scenario". Try both spaces transparently.
        def _resolve_lanelet(net, lid_in: int):
            OFFSET = 1_000_000
            for lid in (lid_in, lid_in + OFFSET, lid_in - OFFSET):
                la = net.find_lanelet_by_id(int(lid))
                if la is not None:
                    return la
            return None
        lanelet = _resolve_lanelet(
            scenario.lanelet_network, int(goal_lanelet_id))
        if lanelet is None:
            raise ValueError(
                f"lanelet_id={goal_lanelet_id} not in scenario "
                f"(also tried ±1,000,000 offset)")
        vertices = lanelet.center_vertices
        if vertices is None or len(vertices) < 2:
            raise RuntimeError(f"lanelet {goal_lanelet_id} has no center_vertices")
        if goal_position == "start":
            idx = 0
        elif goal_position == "end":
            idx = len(vertices) - 1
        else:  # middle
            idx = len(vertices) // 2
        goal_xy = vertices[idx]
        # Orientation = direction from this vertex to the next (or prev).
        if idx == len(vertices) - 1:
            dv = vertices[idx] - vertices[idx - 1]
        else:
            dv = vertices[idx + 1] - vertices[idx]
        goal_orientation = float(np.arctan2(dv[1], dv[0]))
    else:
        target_t = int(initial_state.time_step) + int(goal_offset_steps)
        st = ego.state_at_time(target_t)
        if st is None:
            pred = getattr(ego, "prediction", None)
            states = getattr(getattr(pred, "trajectory", None), "state_list", None) or []
            if not states:
                raise RuntimeError(
                    f"ego {ego_obstacle_id} has no trajectory — can't pick a goal")
            st = states[-1]
            target_t = int(st.time_step)
        goal_xy = np.asarray(st.position, dtype=float)
        goal_orientation = float(getattr(st, "orientation", 0.0) or 0.0)

        # Snap the goal to the nearest point on a real lanelet's centerline.
        # Frenetix builds its reference path from the lanelet network, so a
        # goal floating between lanelets ("No Point of the Reference Path is
        # in the Goal Area") makes the ego agent unreachable.
        lanelet_ids = scenario.lanelet_network.find_lanelet_by_position(
            [goal_xy])[0]
        if lanelet_ids:
            lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_ids[0])
            verts = lanelet.center_vertices
            if verts is not None and len(verts) >= 2:
                dists = np.linalg.norm(verts - goal_xy, axis=1)
                idx = int(np.argmin(dists))
                goal_xy = verts[idx].astype(float)
                if idx == len(verts) - 1:
                    dv = verts[idx] - verts[idx - 1]
                else:
                    dv = verts[idx + 1] - verts[idx]
                goal_orientation = float(np.arctan2(dv[1], dv[0]))

    # Goal region = oriented Rectangle (CR benchmark convention, e.g.
    # AUT_Haag-12 uses 6.0 × 2.0 m). Orientation aligns the rectangle with
    # the road/ego heading so it looks natural on the BEV.
    t0 = int(initial_state.time_step)

    # Frenetix's `agent.py:77-78` computes:
    #     max_time_steps = config_simulation.max_steps
    #                       × planning_problem.goal.state_list[0].time_step.end
    # With `simulation.yaml: max_steps: 1.0` (our setting), the goal-
    # interval END is the canonical Stage-2 horizon. So we set it to
    # exactly `total_sim_steps` (Stage 1's `sim_seconds × 10`) when the
    # caller provides one — this makes Stage 2 plan for the same number
    # of timesteps SUMO simulated.
    if total_sim_steps is not None and total_sim_steps > 0:
        goal_t_end = int(total_sim_steps)
    else:
        # Fallback when no total provided: cap by the obstacle horizon
        # so we don't plan past the no-data zone.
        obs_horizon = 0
        for o in scenario.dynamic_obstacles:
            if o.obstacle_id == ego_obstacle_id:
                continue
            pred = getattr(o, "prediction", None)
            ft = getattr(pred, "final_time_step", None)
            if ft is not None:
                obs_horizon = max(obs_horizon, int(ft))
        goal_t_end = t0 + max(goal_offset_steps, 1) + 50  # small slack
        if obs_horizon > 0:
            goal_t_end = min(goal_t_end, obs_horizon)
    # Always allow at least one step of search.
    goal_t_end = max(goal_t_end, t0 + 2)

    goal_custom = CustomState(
        time_step=Interval(t0 + 1, goal_t_end),
        position=Rectangle(
            length=float(goal_length_m),
            width=float(goal_width_m),
            center=np.array([float(goal_xy[0]), float(goal_xy[1])]),
            orientation=goal_orientation,
        ),
    )
    goal_region = GoalRegion([goal_custom])

    pp = PlanningProblem(
        planning_problem_id=ego_obstacle_id,
        initial_state=initial_state,
        goal_region=goal_region,
    )
    pps = PlanningProblemSet([pp])

    # Remove the ego from obstacles so Frenetix's planner replaces it.
    try:
        scenario.remove_obstacle(ego)
    except Exception:
        # Older commonroad-io: rebuild the obstacle list.
        scenario._dynamic_obstacles = {
            oid: o for oid, o in scenario._dynamic_obstacles.items()
            if oid != ego_obstacle_id
        }

    writer = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=pps,
        author="osm-carla-chat",
        affiliation="",
        source=(f"populated by commonroad-sumo, planning problem added "
                f"for obstacle {ego_obstacle_id}"),
        tags={Tag.URBAN},
    )
    writer.write_to_file(str(out_path), OverwriteExistingFile.ALWAYS)
    return out_path


def list_ego_candidates(cr_sim_path: Path, limit: int = 50) -> list[int]:
    """Return up to `limit` DynamicObstacle IDs from a populated CR scenario.
    Sorted by longest trajectory first (better planning-problem candidates)."""
    from commonroad.common.file_reader import CommonRoadFileReader

    cr_sim_path = Path(cr_sim_path)
    if not cr_sim_path.exists():
        return []
    scenario, _ = CommonRoadFileReader(str(cr_sim_path)).open()
    items = []
    for obs in scenario.dynamic_obstacles:
        pred = getattr(obs, "prediction", None)
        end_t = int(getattr(pred, "final_time_step", 0) or 0)
        start_t = int(getattr(obs.initial_state, "time_step", 0) or 0)
        items.append((end_t - start_t, obs.obstacle_id))
    items.sort(key=lambda it: it[0], reverse=True)
    return [oid for _, oid in items[:limit]]
