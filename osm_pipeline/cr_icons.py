"""
Extra CommonRoad MPRenderer icons for ObstacleType.MOTORCYCLE and
ObstacleType.PRIORITY_VEHICLE. Upstream's commonroad.visualization.icons
ships car / parked_vehicle / taxi / truck / bus / bicycle only — everything
else falls back to a plain colored rectangle, which is indistinguishable
at our OSM-wide zoom levels.

Both new icons use the **obstacle's actual shape** (MPRenderer passes
`vehicle_length` / `vehicle_width` from `obstacle.obstacle_shape`) so they
stay proportional to the real-world vehicle — matching how the stock
car / truck / bus / bicycle icons work.

Call `install()` once before rendering; idempotent.
"""
from __future__ import annotations

import math

import matplotlib as mpl
import numpy as np

from commonroad.scenario.obstacle import ObstacleType
from commonroad.visualization import icons as _cr_icons
from commonroad.visualization.icons import _transform_to_global, draw_car_icon


def _elliptic_arc(center: tuple[float, float],
                  major: float, minor: float,
                  start_angle: float, end_angle: float,
                  n: int = 48) -> np.ndarray:
    angles = np.linspace(start_angle, end_angle, n)
    return np.array([[center[0] + major * math.cos(a),
                      center[1] + minor * math.sin(a)] for a in angles])


def draw_motorcycle_icon(pos_x, pos_y, orientation,
                         vehicle_length=2.2, vehicle_width=0.9,
                         zorder=5, vehicle_color="#ffffff",
                         edgecolor="black", lw=0.5, opacity=1):
    """Top-down motorcycle silhouette (two tires, tank/fairing, seat,
    handlebar, exhaust). Always rendered at car display dimensions."""
    # Norm-box vertices in [-50, 50]^2; x = length direction, y = width.
    v_rear_tire = _elliptic_arc((-32, 0), 14, 7, 0, 2 * np.pi)
    v_front_tire = _elliptic_arc((32, 0), 14, 7, 0, 2 * np.pi)
    v_tank = _elliptic_arc((4, 0), 22, 18, 0, 2 * np.pi)
    v_seat = _elliptic_arc((-18, 0), 14, 14, 0, 2 * np.pi)
    v_handlebar = np.array([[22, -40], [30, -40], [30, 40], [22, 40]])
    v_frame = np.array([[28, -4], [28, 4], [-32, 4], [-32, -4]])

    parts = [
        (v_rear_tire,  "#1a1a1a"),
        (v_front_tire, "#1a1a1a"),
        (v_frame,      "#333333"),
        (v_tank,       vehicle_color),
        (v_seat,       "#222222"),
        (v_handlebar,  "#333333"),
    ]
    patches = []
    for verts, fc in parts:
        vt = _transform_to_global(
            vertices=verts, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
            vehicle_length=vehicle_length, vehicle_width=vehicle_width,
        )
        patches.append(mpl.patches.Polygon(
            vt, fc=fc, ec=edgecolor, lw=lw, zorder=zorder,
            alpha=opacity, closed=True))
    return patches


def draw_taxi_icon(pos_x, pos_y, orientation,
                   vehicle_length=4.8, vehicle_width=2.0,
                   zorder=5, vehicle_color="#FFCC33",
                   edgecolor="black", lw=0.5, opacity=1):
    """Taxi: standard car silhouette in taxi yellow + a small roof
    sign so it's visually distinguishable from a regular car at a
    glance. The stock CommonRoad registry maps TAXI → draw_car_icon
    which means taxis look identical to cars."""
    patches = draw_car_icon(
        pos_x=pos_x, pos_y=pos_y, orientation=orientation,
        vehicle_length=vehicle_length, vehicle_width=vehicle_width,
        zorder=zorder, vehicle_color=vehicle_color,
        edgecolor=edgecolor, lw=lw, opacity=opacity,
    )
    # Roof sign: short rectangle perpendicular to the length axis,
    # tinted yellow so it pops on the orange Frenetix obstacle palette.
    v_sign_body = np.array([[-9, -16], [9, -16], [9, 16], [-9, 16]])
    v_sign_band = np.array([[-3, -16], [3, -16], [3, 16], [-3, 16]])
    for verts, fc in [(v_sign_body, "#FFE066"), (v_sign_band, "#1a1a1a")]:
        vt = _transform_to_global(
            vertices=verts, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
            vehicle_length=vehicle_length, vehicle_width=vehicle_width,
        )
        patches.append(mpl.patches.Polygon(
            vt, fc=fc, ec=edgecolor, lw=lw, zorder=zorder + 0.1,
            alpha=opacity, closed=True))
    return patches


def draw_priority_vehicle_icon(pos_x, pos_y, orientation,
                               vehicle_length=5.5, vehicle_width=2.1,
                               zorder=5, vehicle_color="#ffffff",
                               edgecolor="black", lw=0.5, opacity=1):
    """Emergency vehicle: stock car silhouette + red/blue rooftop lightbar.
    Always rendered at car display dimensions so it's visible on BEVs."""
    patches = draw_car_icon(
        pos_x=pos_x, pos_y=pos_y, orientation=orientation,
        vehicle_length=vehicle_length, vehicle_width=vehicle_width,
        zorder=zorder, vehicle_color=vehicle_color,
        edgecolor=edgecolor, lw=lw, opacity=opacity,
    )
    # Rooftop lightbar — two halves for red/blue, centered across the roof.
    v_light_red  = np.array([[-7, -28], [7, -28], [7, 0],  [-7, 0]])
    v_light_blue = np.array([[-7, 0],   [7, 0],   [7, 28], [-7, 28]])
    for verts, fc in [(v_light_red, "#D43A2F"), (v_light_blue, "#2F77DE")]:
        vt = _transform_to_global(
            vertices=verts, pos_x=pos_x, pos_y=pos_y, orientation=orientation,
            vehicle_length=vehicle_length, vehicle_width=vehicle_width,
        )
        patches.append(mpl.patches.Polygon(
            vt, fc=fc, ec=edgecolor, lw=lw, zorder=zorder + 0.1,
            alpha=opacity, closed=True))
    return patches


_INSTALLED = False


def _patch_walenet_scene_image() -> None:
    """wale-net's `generate_self_rendered_sc_img(...)` rasterises the
    scenario's lanelet network into an image fed to the prediction
    network. Internally it computes `curve_length[-1]` for each
    boundary line, then `np.arange(0, curve_length[-1], step)`.
    Whenever a lanelet has NaN vertex coords (a known crdesigner OSM→CR
    artefact for ill-formed boundary geometry), the cumulative norm
    propagates NaN, and `np.arange(0, nan, step)` raises:

        ValueError: arange: cannot compute length

    Wrap the function so that any rendering failure substitutes a
    blank `(res, res, 3)` zero image. wale-net loses geometric context
    for that single prediction, but the planner keeps stepping.
    """
    try:
        from prediction.utils import preprocessing as _wp
    except Exception:
        return
    orig = getattr(_wp, "generate_self_rendered_sc_img", None)
    if orig is None or getattr(orig, "_osm_walenet_patched", False):
        return

    def patched(watch_radius, scenario, curr_pos, curr_orient,
                res=256, light_lane_dividers=True):
        try:
            return orig(watch_radius, scenario, curr_pos, curr_orient,
                        res=res, light_lane_dividers=light_lane_dividers)
        except Exception:
            # wale-net wraps this with `FloatTensor(x).unsqueeze(0)
            # .unsqueeze(0)` then feeds into a `Conv2d(in_channels=1)`.
            # The expected leading shape is therefore 2D (H, W); a 3D
            # (H, W, 3) would unsqueeze to 5D and crash Conv2d.
            import numpy as _np
            return _np.zeros((res, res), dtype=_np.uint8)

    patched._osm_walenet_patched = True
    _wp.generate_self_rendered_sc_img = patched
    # `prediction.main` does
    #     from prediction.utils.preprocessing import generate_self_rendered_sc_img
    # so it captured the original at import time — replace it there too.
    try:
        import prediction.main as _wm
        if hasattr(_wm, "generate_self_rendered_sc_img"):
            _wm.generate_self_rendered_sc_img = patched
    except Exception:
        pass


def _patch_walenet_obstacles_in_scenario() -> None:
    """wale-net's `WaleNet._obstacles_in_scenario(time_step, ids)` does:

        self.scenario._dynamic_obstacles[obst].prediction.final_time_step
                                              ^^^^^^^^^^^

    without checking that `prediction` is non-None. Some of our
    obstacles (e.g. SUMO entries that happen to have no future
    states beyond their initial_state, or anything constructed
    without a TrajectoryPrediction) have `prediction is None`,
    so the access raises:

        AttributeError: 'NoneType' object has no attribute 'final_time_step'

    Wrap the method to filter those obstacles out before applying the
    time-step bounds check.
    """
    try:
        from prediction.main import WaleNet
    except Exception:
        return
    orig = getattr(WaleNet, "_obstacles_in_scenario", None)
    if orig is None or getattr(orig, "_osm_walenet_patched", False):
        return

    def patched(self, time_step, obstacle_id_list):
        out = []
        for obst in obstacle_id_list:
            o = self.scenario._dynamic_obstacles.get(obst)
            if o is None:
                continue
            pred = getattr(o, "prediction", None)
            if pred is None:
                continue
            try:
                final_t = pred.final_time_step
                init_t = pred.initial_time_step
            except Exception:
                continue
            if final_t > time_step and (init_t - 1) <= time_step:
                out.append(obst)
        return out

    patched._osm_walenet_patched = True
    WaleNet._obstacles_in_scenario = patched


def _patch_walenet_transform_trajectories() -> None:
    """wale-net's `transform_trajectories(traj_list, now_point, theta)` does
    `tr - now_point` on raw Python lists. When an obstacle's recorded
    history is shorter than wale-net's history window, the helper pads
    with the Python list `[nan, nan]`, so both operands end up as plain
    lists and `list - list` raises:

        TypeError: unsupported operand type(s) for -: 'list' and 'list'

    Coerce both sides to numpy arrays at the function entry — restores
    the intended broadcast subtraction without changing behaviour for
    callers that already pass numpy arrays.
    """
    try:
        from prediction.utils import geometry as _wg
    except Exception:
        return
    orig = _wg.transform_trajectories
    if getattr(orig, "_osm_walenet_patched", False):
        return

    def patched(trajectories_list, now_point, theta):
        import numpy as _np
        np_now = _np.asarray(now_point, dtype=float)
        rot_mat = _np.array(
            [[_np.cos(theta), -_np.sin(theta)],
             [_np.sin(theta),  _np.cos(theta)]])
        out = []
        for tr in trajectories_list:
            tr_arr = _np.asarray(tr, dtype=float)
            out.append((tr_arr - np_now) @ rot_mat)
        return out

    patched._osm_walenet_patched = True
    _wg.transform_trajectories = patched
    # `prediction.main` imports the symbol with `from … import …`,
    # binding the original reference into THAT module's namespace at
    # import time. Patching `prediction.utils.geometry` alone leaves
    # `prediction.main.transform_trajectories` pointing at the unpatched
    # function. Replace it there too. (Same gotcha as the Frenetix
    # lanelet-lookup patch below.)
    try:
        import prediction.main as _wm
        if hasattr(_wm, "transform_trajectories"):
            _wm.transform_trajectories = patched
    except Exception:
        pass


def _patch_walenet_transform_back() -> None:
    """Sibling of `_patch_walenet_transform_trajectories`. wale-net's
    `transform_back(trajectory, translation, rotation)` does:

        translation = -translation

    which raises `TypeError: bad operand type for unary -: 'list'`
    when `translation` arrives as a Python list (the typical case
    when the upstream `_postprocessing` reads it from
    `self.translation_dict[obstacle_id]` — same dict that
    `transform_trajectories` populated with NaN-padded list entries).

    Coerce `translation` to a numpy array on entry; the rest of the
    function already uses numpy ops, so the fix is one line.
    """
    try:
        from prediction.utils import geometry as _wg
    except Exception:
        return
    orig = _wg.transform_back
    if getattr(orig, "_osm_walenet_patched", False):
        return

    import functools

    @functools.wraps(orig)
    def patched(trajectory, translation, rotation):
        import numpy as _np
        translation = _np.asarray(translation, dtype=float)
        return orig(trajectory, translation, rotation)

    patched._osm_walenet_patched = True
    _wg.transform_back = patched
    # Replace the import-time binding in prediction.main too.
    try:
        import prediction.main as _wm
        if hasattr(_wm, "transform_back"):
            _wm.transform_back = patched
    except Exception:
        pass


def _patch_frenetix_lanelet_lookup() -> None:
    """Frenetix's `find_lanelet_by_position_and_orientation` indexes
    `initial_lanelets[0]` without checking that the list is non-empty,
    so an ego whose initial position falls a few centimetres outside
    every lanelet polygon (typical with crdesigner-narrowed lanelets
    + SUMO centerline coords) crashes the simulation init with
    `IndexError: list index out of range`. Wrap the function so that
    when `find_lanelet_by_position` returns nothing, we substitute the
    nearest lanelet by Euclidean distance to its centerline."""
    try:
        from cr_scenario_handler.utils import helper_functions as _hf
    except Exception:
        return

    orig = _hf.find_lanelet_by_position_and_orientation
    if getattr(orig, "_osm_lanelet_lookup_patched", False):
        return

    def patched(lanelet_network, position, orientation):
        try:
            initial = lanelet_network.find_lanelet_by_position(
                [position])[0]
        except Exception:
            initial = []
        if initial:
            return orig(lanelet_network, position, orientation)
        # Fallback: pick the lanelet whose centerline has the closest
        # vertex to `position`. Shadow the network's lookup for the
        # duration of the call so the original function sees a
        # non-empty result.
        import numpy as _np
        try:
            xy = _np.asarray(position, dtype=float)
        except Exception:
            return orig(lanelet_network, position, orientation)
        best_lid = None; best_d2 = float("inf")
        for la in lanelet_network.lanelets:
            verts = la.center_vertices
            if verts is None or len(verts) < 1:
                continue
            arr = _np.asarray(verts, dtype=float)
            d2 = ((arr - xy) ** 2).sum(axis=1).min()
            if d2 < best_d2:
                best_d2 = d2; best_lid = la.lanelet_id
        if best_lid is None:
            return orig(lanelet_network, position, orientation)
        # Temporarily patch find_lanelet_by_position to return our
        # fallback so the original function's logic (which picks the
        # best-aligned lanelet by orientation) still runs.
        original_finder = lanelet_network.find_lanelet_by_position
        def _shim(positions):
            return [[best_lid] for _ in positions]
        lanelet_network.find_lanelet_by_position = _shim  # type: ignore
        try:
            return orig(lanelet_network, position, orientation)
        finally:
            lanelet_network.find_lanelet_by_position = original_finder

    patched._osm_lanelet_lookup_patched = True
    _hf.find_lanelet_by_position_and_orientation = patched
    # Other modules already imported the symbol with `from ... import
    # find_lanelet_by_position_and_orientation`, which binds the
    # original reference into THEIR namespace. Replace it there too.
    for mod_name in (
        "cr_scenario_handler.simulation.simulation",
        "frenetix_motion_planner.utility.reachable_set",
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "find_lanelet_by_position_and_orientation"):
                mod.find_lanelet_by_position_and_orientation = patched
        except Exception:
            pass


def _patch_frenetix_plot_limits() -> None:
    """Frenetix's `visualize_agent_at_timestep` builds plot_limits as a
    fixed `plot_window`-radius square centred on the ego's *initial*
    position only. Goals placed several seconds of driving ahead of
    the spawn (the typical case with goal_offset_steps = 50) end up
    off-screen even though `planning_problem.draw(...)` is called.
    Wrap the function so that when it's responsible for creating its
    own renderer, the plot covers `(ego_start, goal_center, margin)`.
    """
    try:
        from cr_scenario_handler.utils import visualization as _viz
    except Exception:
        return

    orig = _viz.visualize_agent_at_timestep
    if getattr(orig, "_osm_plot_patched", False):
        return

    def patched(scenario, planning_problem, ego, timestep, config,
                log_path, *args, **kwargs):
        if kwargs.get("rnd") is None:
            try:
                import numpy as np
                from commonroad.visualization.mp_renderer import MPRenderer
                pts = []
                try:
                    pts.append(np.asarray(
                        ego.prediction.trajectory.state_list[0].position))
                except Exception:
                    pass
                try:
                    gs = planning_problem.goal.state_list[0]
                    if hasattr(gs.position, "center"):
                        pts.append(np.asarray(gs.position.center))
                    elif hasattr(gs.position, "shapes"):
                        pts.append(np.asarray(gs.position.shapes[0].center))
                except Exception:
                    pass
                if len(pts) >= 2:
                    arr = np.array(pts)
                    margin = max(20.0, float(kwargs.get("plot_window") or 30.0))
                    left, right = float(arr[:, 0].min()), float(arr[:, 0].max())
                    bot,  top   = float(arr[:, 1].min()), float(arr[:, 1].max())
                    cx, cy = (left + right) / 2, (bot + top) / 2
                    side = max(right - left, top - bot) + 2 * margin
                    kwargs["rnd"] = MPRenderer(
                        plot_limits=[cx - side / 2, cx + side / 2,
                                     cy - side / 2, cy + side / 2],
                        figsize=(10, 10),
                    )
                    # Tell the wrapped function not to overwrite our renderer.
                    kwargs["plot_window"] = 0
            except Exception:
                pass
        return orig(scenario, planning_problem, ego, timestep, config,
                    log_path, *args, **kwargs)

    patched._osm_plot_patched = True
    _viz.visualize_agent_at_timestep = patched


def _patch_mp_renderer_for_none_prediction() -> None:
    """CommonRoad's MPRenderer only draws the obstacle icon when the
    obstacle has a `TrajectoryPrediction`. Frenetix's ego at the very
    last simulation step often has `prediction=None` (the planner has
    nothing left to predict), which makes the renderer fall through to
    a plain rectangle. Wrap `draw_dynamic_obstacle` to attach a stub
    one-state trajectory in that situation so the icon code path
    fires; the stub is removed right after the draw call so we don't
    persist any side-effects on the obstacle."""
    from commonroad.visualization.mp_renderer import MPRenderer
    from commonroad.prediction.prediction import TrajectoryPrediction
    from commonroad.scenario.trajectory import Trajectory

    orig = MPRenderer.draw_dynamic_obstacle
    if getattr(orig, "_osm_no_pred_patched", False):
        return

    def patched(self, obj, draw_params=None, *args, **kwargs):
        added_stub = False
        try:
            tb = getattr(draw_params, "time_begin", None)
            if (obj.prediction is None
                    and getattr(obj, "obstacle_shape", None) is not None
                    and obj.initial_state is not None
                    and (tb is None
                         or tb == obj.initial_state.time_step)):
                traj = Trajectory(
                    initial_time_step=obj.initial_state.time_step,
                    state_list=[obj.initial_state])
                obj.prediction = TrajectoryPrediction(traj, obj.obstacle_shape)
                added_stub = True
        except Exception:
            pass
        try:
            return orig(self, obj, draw_params, *args, **kwargs)
        finally:
            if added_stub:
                try: obj.prediction = None
                except Exception: pass

    patched._osm_no_pred_patched = True
    MPRenderer.draw_dynamic_obstacle = patched


def install() -> None:
    """Register the two new icons on commonroad.visualization.icons so any
    MPRenderer call picks them up via `_obstacle_icon_assignment()`.

    The icon-registry patch is one-shot (gated by `_INSTALLED`), but the
    Frenetix patches MUST keep re-attempting on every call: the first
    `install()` typically runs at app startup before the `osm/vendor/…`
    paths are on `sys.path`, so `cr_scenario_handler` isn't importable
    yet and `_patch_frenetix_lanelet_lookup` returns early. The second
    call (from inside `frenetix_run.run()`, after vendor injection) is
    when the patch can actually land — so we must let it through.
    The patch helpers are themselves idempotent via per-function
    `_osm_*_patched` flags on the wrapped target.
    """
    global _INSTALLED
    if not _INSTALLED:
        original = _cr_icons._obstacle_icon_assignment

        def _patched():
            d = original()
            d[ObstacleType.MOTORCYCLE] = draw_motorcycle_icon
            d[ObstacleType.PRIORITY_VEHICLE] = draw_priority_vehicle_icon
            # Override stock TAXI (which is just draw_car_icon) so taxis
            # are visually distinguishable from regular cars.
            d[ObstacleType.TAXI] = draw_taxi_icon
            return d

        _cr_icons._obstacle_icon_assignment = _patched
        _INSTALLED = True

    # Always retry these — idempotent, and only succeeds once the
    # vendored Frenetix modules are importable.
    _patch_mp_renderer_for_none_prediction()
    _patch_frenetix_plot_limits()
    _patch_frenetix_lanelet_lookup()
    _patch_walenet_obstacles_in_scenario()
    _patch_walenet_transform_trajectories()
    _patch_walenet_transform_back()
    _patch_walenet_scene_image()
