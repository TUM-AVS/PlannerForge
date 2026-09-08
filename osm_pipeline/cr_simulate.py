"""
Run a SUMO-backed traffic simulation on a CommonRoad scenario and write the
populated result (with one <dynamicObstacle> per vehicle trajectory) back as
a new .xml.

The heavy lifting is done by the `commonroad-sumo` PyPI package
(`NonInteractiveSumoSimulation.from_scenario(...).run(steps)`), which handles:
  - converting the CR lanelet network → SUMO .net.xml
  - generating random traffic (`SumoTrafficGenerationMode.RANDOM`)
  - stepping the SUMO simulation headlessly
  - collecting each vehicle's trajectory as a CommonRoad DynamicObstacle

Self-contained — no imports from PlannerForge or any other sibling repo.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path

from config import DATA_BEV_SUMO, DATA_CR, DATA_NET


@contextlib.contextmanager
def _silence_stdout():
    """Suppress upstream debug prints (e.g. commonroad-sumo's
    `print(lanelet.distance)` loop in `_get_total_lanelet_length`) that
    would otherwise flood the Gradio stage-1 log."""
    old = sys.stdout
    try:
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def _ensure_good_sumo_home() -> None:
    """commonroad-sumo reads os.environ['SUMO_HOME'] to locate bundled tools
    like randomTrips.py. The user's shell profile may point at a stale
    `/tmp/commonroad-interactive-scenarios/install/sumo` — override with the
    eclipse-sumo package's real data dir (which ships the tools)."""
    try:
        import sumo
        real = os.path.dirname(sumo.__file__)
    except Exception:
        return
    cur = os.environ.get("SUMO_HOME", "")
    tools_ok = cur and os.path.exists(os.path.join(cur, "tools", "randomTrips.py"))
    if not tools_ok and os.path.exists(os.path.join(real, "tools", "randomTrips.py")):
        os.environ["SUMO_HOME"] = real


_ensure_good_sumo_home()


def _segments_intersect_2d(p1, p2, p3, p4) -> bool:
    """True iff line segments (p1→p2) and (p3→p4) cross AT AN INTERIOR
    POINT (shared endpoints don't count). Used by the self-intersection
    detector for lanelet centerlines."""
    import numpy as np
    p1, p2, p3, p4 = (np.asarray(p, dtype=float) for p in (p1, p2, p3, p4))

    def _ccw(a, b, c) -> float:
        return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

    # Exclude shared endpoints — adjacent / touching segments aren't a problem.
    if np.allclose(p1, p3) or np.allclose(p1, p4):
        return False
    if np.allclose(p2, p3) or np.allclose(p2, p4):
        return False

    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def _polyline_self_intersects(verts) -> bool:
    """True iff a 2D polyline crosses itself at an interior point.
    O(n²) but n is small for individual lanelet centerlines (<30)."""
    import numpy as np
    if verts is None or len(verts) < 4:
        return False
    pts = np.asarray(verts, dtype=float)
    n = len(pts) - 1  # number of segments
    for i in range(n):
        a, b = pts[i], pts[i + 1]
        # Skip the adjacent next segment — sharing endpoint is OK.
        for j in range(i + 2, n):
            c, d = pts[j], pts[j + 1]
            if _segments_intersect_2d(a, b, c, d):
                return True
    return False


def _sanitize_lanelets(scenario) -> None:
    """crdesigner's OSM→CR output leaves some lanelet attributes at `None`
    and its intersection metadata can reference lanelet IDs that don't
    resolve. commonroad-sumo chokes on both. Sanitize the scenario in place
    so `NonInteractiveSumoSimulation.from_scenario` can ingest it.

    Also filters lanelets that netconvert refuses: degenerate (<3 vertices,
    mismatched left/right boundary length) and self-intersecting (centerline
    polyline crosses itself). These are the specific cases that produce
    `Invocation of SUMO application netconvert failed with status code 1`
    when CR2SumoMapConverter chains through to netconvert."""
    try:
        from commonroad.scenario.lanelet import LineMarking
    except ImportError:
        LineMarking = None

    # 1. Fill in missing attributes. Defaults:
    #    - lanelet_type       : empty set
    #    - user_one_way       : all VEHICLE classes (+BICYCLE) so SUMO's
    #                            per-class allow list isn't disallow=all
    #    - traffic_signs/lights: empty set
    try:
        from commonroad.scenario.lanelet import RoadUser
        _default_users = {
            RoadUser.VEHICLE, RoadUser.CAR, RoadUser.TRUCK, RoadUser.BUS,
            RoadUser.MOTORCYCLE, RoadUser.BICYCLE, RoadUser.TAXI,
            RoadUser.PRIORITY_VEHICLE,
        }
    except Exception:
        _default_users = None

    valid_ids = set()
    for lanelet in scenario.lanelet_network.lanelets:
        valid_ids.add(lanelet.lanelet_id)
        if getattr(lanelet, "lanelet_type", None) is None:
            lanelet.lanelet_type = set()
        # Always install the permissive vehicle-class set on user_one_way
        # — crdesigner's OSM→CR leaves it either None or too narrow, and
        # commonroad-sumo uses this when emitting the SUMO .net.xml's
        # edge `allow=` attribute. Without all vClasses included, the
        # per-class randomTrips.py pass finds no valid edges and generates
        # zero trips for that class.
        cur = getattr(lanelet, "user_one_way", None)
        if _default_users is not None:
            lanelet.user_one_way = (set(cur) if cur else set()) | _default_users
        if getattr(lanelet, "user_bidirectional", None) is None:
            lanelet.user_bidirectional = set()
        if getattr(lanelet, "traffic_signs", None) is None:
            lanelet.traffic_signs = set()
        if getattr(lanelet, "traffic_lights", None) is None:
            lanelet.traffic_lights = set()
        if (LineMarking is not None
                and getattr(lanelet, "line_marking_left_vertices", None) is None):
            lanelet.line_marking_left_vertices = LineMarking.UNKNOWN
        if (LineMarking is not None
                and getattr(lanelet, "line_marking_right_vertices", None) is None):
            lanelet.line_marking_right_vertices = LineMarking.UNKNOWN

    # 2. Drop intersections and traffic-light references entirely.
    # crdesigner's OSM→CR output leaves them in an inconsistent state
    # (lanelet IDs that don't resolve; traffic lights on multi-successor
    # lanelets that commonroad-sumo's direction resolver can't handle).
    # commonroad-sumo rebuilds intersections from the lanelet topology
    # on its own, and our SUMO net will have traffic lights added at the
    # netconvert step downstream.
    net = scenario.lanelet_network
    try:
        net._intersections = {}
    except Exception:
        pass
    try:
        net._traffic_lights = {}
    except Exception:
        pass
    for lanelet in scenario.lanelet_network.lanelets:
        # Clear per-lanelet TLS references too.
        try:
            lanelet.traffic_lights = set()
        except Exception:
            pass

    # 3. Strip dangling neighbor / successor / predecessor references.
    # crdesigner's OSM→CR output occasionally keeps an `adj_left` or
    # `adj_right` pointing at a lanelet ID that was filtered out of the
    # network (same for successor/predecessor lists). commonroad-sumo's
    # CR2SumoMapConverter does a raw dict lookup on those IDs and raises
    # KeyError. Null them out via the private attributes (the public
    # setters on Lanelet don't exist or require paired arguments).
    for lanelet in scenario.lanelet_network.lanelets:
        if lanelet.adj_left is not None and lanelet.adj_left not in valid_ids:
            lanelet._adj_left = None
            lanelet._adj_left_same_direction = None
        if lanelet.adj_right is not None and lanelet.adj_right not in valid_ids:
            lanelet._adj_right = None
            lanelet._adj_right_same_direction = None
        if lanelet.successor:
            kept = [s for s in lanelet.successor if s in valid_ids]
            if len(kept) != len(lanelet.successor):
                lanelet._successor = kept
        if lanelet.predecessor:
            kept = [p for p in lanelet.predecessor if p in valid_ids]
            if len(kept) != len(lanelet.predecessor):
                lanelet._predecessor = kept

    # 4. Filter out lanelets that netconvert refuses — these are the
    # specific cases observed in Q002 / Q009 / Q011 / etc. that crash
    # CR2SumoMapConverter with "netconvert failed with status code 1".
    # We drop them HERE (before the converter runs) so traffic can still
    # be generated on the remaining clean network. Then re-clean any
    # dangling refs introduced by the removal.
    to_remove = set()
    for lanelet in scenario.lanelet_network.lanelets:
        cv = getattr(lanelet, "center_vertices", None)
        # 4a) Centerline too short to be a valid SUMO edge.
        if cv is None or len(cv) < 3:
            to_remove.add(lanelet.lanelet_id)
            continue
        # 4b) Boundary mismatch — left and right must have the same length.
        lv = getattr(lanelet, "left_vertices", None)
        rv = getattr(lanelet, "right_vertices", None)
        if lv is None or rv is None or len(lv) != len(rv) or len(lv) < 2:
            to_remove.add(lanelet.lanelet_id)
            continue
        # 4c) Self-intersecting centerline. This is THE common cause of
        # netconvert's "Invalid lanelet shape! Self-intersection" warning
        # being followed by a hard exit code 1.
        if _polyline_self_intersects(cv):
            to_remove.add(lanelet.lanelet_id)
            continue

    if to_remove:
        net = scenario.lanelet_network
        # Drop from the internal storage. The public API doesn't expose
        # `remove_lanelet` on all CR versions, so we hit `_lanelets`.
        try:
            for lid in to_remove:
                net._lanelets.pop(lid, None)
        except Exception:
            pass
        valid_ids -= to_remove
        # Re-clean dangling references on the survivors.
        for lanelet in scenario.lanelet_network.lanelets:
            if lanelet.adj_left in to_remove:
                lanelet._adj_left = None
                lanelet._adj_left_same_direction = None
            if lanelet.adj_right in to_remove:
                lanelet._adj_right = None
                lanelet._adj_right_same_direction = None
            if lanelet.successor:
                lanelet._successor = [s for s in lanelet.successor
                                       if s not in to_remove]
            if lanelet.predecessor:
                lanelet._predecessor = [p for p in lanelet.predecessor
                                         if p not in to_remove]


# UI "SUMO traffic density" preset → max number of <vehicle> entries we
# keep in the routes file. commonroad-sumo's RandomTripsTrafficGenerator
# writes ~20 000 trips into the .rou.xml regardless of map size, which
# overwhelms the per-network capacity in any typical 30 s window — so
# scaling alone is invisible. By truncating the routes file BEFORE
# simulation we directly control how many trips even attempt to spawn.
# Network capacity still caps the realised count; on small bboxes
# `medium` and `high` may both saturate, but `low` will always be
# visibly sparser.
_DENSITY_TO_MAX_TRIPS = {
    "low":     20,
    "medium":  50,
    "high":   100,
}


# Constant offset added to every lanelet ID in the saved CR sim XML so
# that vehicle IDs (now equal to SUMO trip ids 0, 1, 2, …) cannot
# collide with lanelet IDs when the scenario is reloaded by
# CommonRoadFileReader.
_LANELET_ID_OFFSET = 1_000_000


def _retrace_sumo_routes_from_cr_trajectories(
        scenario, sumo_net_path: Path, sumo_rou_path: Path,
        cr_id_to_sumo_id: dict[int, int]) -> None:
    """Rewrite the persisted SUMO `.vehicles.rou.xml` so each
    `<route edges="…">` lists only the SUMO edges actually traversed
    by the corresponding CR DynamicObstacle within the simulation
    horizon. Equivalent in spirit to commonroad-sumo's
    `UnsafeResimulationTrafficGenerator` (PlannerForge's pattern), but
    runs after the fact so we can keep using `RandomTripsTrafficGenerator`
    for trip *generation* while still ending up with route entries
    whose first / last edge match each CR obstacle's first / last
    position.

    Effect: SUMO's `<route>` final edge ≡ CR's last trajectory state's
    nearest edge for every vehicle that ran in the sim window.
    """
    if not (sumo_net_path.exists() and sumo_rou_path.exists()):
        return
    try:
        import sumolib
        import xml.etree.ElementTree as ET
    except Exception:
        return

    net = sumolib.net.readNet(str(sumo_net_path))

    def _nearest_edge_id(xy):
        cands = net.getNeighboringEdges(float(xy[0]), float(xy[1]), 12.0)
        if not cands:
            return None
        cands.sort(key=lambda eo: eo[1])
        # Skip junction-internal edges (id starts with ':') — those never
        # appear in <route edges="…"> so they'd cause spurious mismatches.
        for edge, _dist in cands:
            eid = edge.getID()
            if not eid.startswith(":"):
                return eid
        return None

    # Map keyed by SUMO trip id (the routes file's vehicle id space).
    traversed: dict[int, list[str]] = {}
    for o in scenario.dynamic_obstacles:
        sumo_id = cr_id_to_sumo_id.get(int(o.obstacle_id))
        if sumo_id is None:
            continue
        states = [o.initial_state]
        pred = getattr(o, "prediction", None)
        if pred is not None and getattr(pred, "trajectory", None):
            states = states + list(pred.trajectory.state_list)
        edge_seq: list[str] = []
        last = None
        for st in states:
            e = _nearest_edge_id(st.position)
            if e is None or e == last:
                continue
            edge_seq.append(e)
            last = e
        if edge_seq:
            traversed[int(sumo_id)] = edge_seq

    tree = ET.parse(str(sumo_rou_path))
    root = tree.getroot()
    for v in root.findall("vehicle"):
        try:
            vid = int(v.get("id"))
        except Exception:
            continue
        if vid not in traversed:
            # Vehicle never spawned (sim ended before depart) — drop it
            # so the routes file lists only realised trips.
            root.remove(v)
            continue
        rt = v.find("route")
        if rt is None:
            continue
        rt.set("edges", " ".join(traversed[vid]))
    tree.write(str(sumo_rou_path), xml_declaration=True, encoding="UTF-8")


def _rewrite_sim_xml_ids(sim_xml: Path,
                         cr_to_sumo_int: dict[int, int],
                         lanelet_offset: int) -> None:
    """Rewrite IDs in a populated CR sim XML in-place so that:
      * `<dynamicObstacle id="X">` X = SUMO trip id (X' = cr_to_sumo[X])
      * every lanelet id (and predecessor / successor / adjacent ref)
        is shifted by `lanelet_offset`."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(str(sim_xml))
    root = tree.getroot()

    def _bare(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for elem in root.iter():
        t = _bare(elem.tag)
        if t == "lanelet":
            if elem.get("id") is not None:
                elem.set("id", str(int(elem.get("id")) + lanelet_offset))
            if elem.get("ref") is not None:
                elem.set("ref", str(int(elem.get("ref")) + lanelet_offset))
        elif t in ("predecessor", "successor",
                   "adjacentLeft", "adjacentRight"):
            if elem.get("ref") is not None:
                elem.set("ref", str(int(elem.get("ref")) + lanelet_offset))
        elif t == "dynamicObstacle":
            cr_id = int(elem.get("id"))
            elem.set("id", str(cr_to_sumo_int.get(cr_id, cr_id)))

    tree.write(str(sim_xml), xml_declaration=True, encoding="UTF-8")


# UI vehicle-mix → CommonRoad ObstacleType.  11 UI types collapse to 7 CR
# types (ObstacleType has no TRAILER/DELIVERY/COACH/MOPED — closest-match).
_UI_TYPE_TO_CR = {
    "car":        "CAR",
    "truck":      "TRUCK",
    "trailer":    "TRUCK",
    "delivery":   "TRUCK",
    "bus":        "BUS",
    "coach":      "BUS",
    "taxi":       "TAXI",
    "emergency":  "PRIORITY_VEHICLE",
    "bicycle":    "BICYCLE",
    "motorcycle": "MOTORCYCLE",
    "moped":      "MOTORCYCLE",
}


def _build_traffic_generator(vehicle_types: list[str]):
    """Translate the UI's vehicle-mix selection into a commonroad-sumo
    RandomTripsTrafficGenerator with matching `veh_distribution`.

    Also registers driving-model parameters for ObstacleTypes that
    commonroad-sumo ships without defaults (TAXI, PRIORITY_VEHICLE,
    MOTORCYCLE) — otherwise the traffic-generator's shape-provider raises
    ValueError when sampling those types."""
    from commonroad.scenario.obstacle import ObstacleType
    from commonroad.common.util import Interval
    from commonroad_sumo.cr2sumo.traffic_generator.random_trips_traffic_generator import (  # noqa: E501
        RandomTripsTrafficGenerator, RandomTripsTrafficGeneratorConfig)
    from commonroad_sumo.interface.driving_model_parameters_provider import (
        StaticDrivingModelParametersProvider, DrivingModelParameters)

    # Fill in the missing driving-model entries (closest-match physics).
    provider = StaticDrivingModelParametersProvider()
    extras = {
        ObstacleType.TAXI: DrivingModelParameters(
            length=5.0, width=2.0, accel=Interval(2, 2.9),
            decel=Interval(4, 6.5), max_speed=180 / 3.6, min_gap=2.5),
        ObstacleType.PRIORITY_VEHICLE: DrivingModelParameters(
            length=5.5, width=2.1, accel=Interval(3, 4.0),
            decel=Interval(5, 7.0), max_speed=200 / 3.6, min_gap=2.0),
        ObstacleType.MOTORCYCLE: DrivingModelParameters(
            length=2.2, width=0.9, accel=Interval(1.5, 3.0),
            decel=Interval(4, 6.5), max_speed=160 / 3.6, min_gap=2.0),
    }
    for k, v in extras.items():
        provider._driving_model_parameters.setdefault(k, v) \
            if hasattr(provider, "_driving_model_parameters") \
            else provider.DEFAULT_DRIVING_MODEL_PARAMETERS.setdefault(k, v)

    distribution: dict[ObstacleType, float] = {}
    for t in vehicle_types:
        cr_name = _UI_TYPE_TO_CR.get(t)
        if not cr_name:
            continue
        cr_enum = getattr(ObstacleType, cr_name, None)
        if cr_enum is None:
            continue
        # Each UI tick contributes 1.0 share; duplicates (e.g. car + truck
        # + trailer all → CAR/TRUCK) sum naturally.
        distribution[cr_enum] = distribution.get(cr_enum, 0.0) + 1.0
    if not distribution:
        distribution = {ObstacleType.CAR: 1.0}
    cfg = RandomTripsTrafficGeneratorConfig(
        veh_distribution=distribution,
        veh_params_provider=provider,
    )
    return RandomTripsTrafficGenerator(cfg)


# CR ObstacleType → SUMO vClass / guiShape names, matching the values in
# commonroad_sumo.interface.util._VEHICLE_CLASS_CR2SUMO but keeping us
# decoupled from its internal representation.
_SUMO_VCLASS = {
    "CAR":              "passenger",
    "TRUCK":            "truck",
    "BUS":              "bus",
    "TAXI":             "taxi",
    "PRIORITY_VEHICLE": "emergency",
    "BICYCLE":          "bicycle",
    "MOTORCYCLE":       "motorcycle",
}

# Mapping from UI vehicle-mix labels directly to the SUMO vClass we want
# in the injected <vTypeDistribution>. Collapses our 11 UI types into the
# 8 vClasses SUMO recognises without extra vType params.
_UI_TYPE_TO_SUMO_VCLASS = {
    "car":        "passenger",
    "truck":      "truck",
    "trailer":    "trailer",
    "delivery":   "delivery",
    "bus":        "bus",
    "coach":      "coach",
    "taxi":       "taxi",
    "emergency":  "emergency",
    "bicycle":    "bicycle",
    "motorcycle": "motorcycle",
    "moped":      "moped",
}


def _truncate_routes(project, max_trips: int, sim_seconds: float) -> None:
    """Keep only `max_trips` <vehicle> entries in the project's routes
    file AND spread their `depart` attributes uniformly across
    `[0, sim_seconds]`.

    Truncation honours the "SUMO traffic density" preset (the library
    writes ~20k trips by default, far above network capacity).
    Re-spreading the depart times prevents the visible artifact where
    every vehicle leaves in the first ~2 s and the rest of the GIF is
    an empty road — without it, both SUMO and CR animations look
    finished after the first few frames."""
    from commonroad_sumo.sumolib.sumo_project import SumoFileType
    import xml.etree.ElementTree as ET

    try:
        rou_path = project.get_file_path(SumoFileType.VEHICLE_ROUTES)
    except Exception:
        rou_path = None
    if rou_path is None or not rou_path.exists():
        cands = list(Path(project.project_path).glob("*.vehicles.rou.xml"))
        if not cands:
            cands = list(Path(project.project_path).glob("*.rou.xml"))
        if not cands:
            return
        rou_path = cands[0]

    tree = ET.parse(str(rou_path))
    root = tree.getroot()
    vehicles = list(root.findall("vehicle"))
    if not vehicles:
        return
    vehicles.sort(key=lambda v: float(v.get("depart", "0")))
    kept = vehicles[:max_trips] if len(vehicles) > max_trips else vehicles
    for v in vehicles[len(kept):]:
        root.remove(v)

    n = len(kept)
    if n > 0 and sim_seconds > 0:
        spacing = sim_seconds / n
        for i, v in enumerate(kept):
            v.set("depart", f"{i * spacing:.2f}")

    tree.write(str(rou_path), xml_declaration=True, encoding="utf-8")


def _default_random_generator():
    """Same as SumoTrafficGenerationMode.RANDOM but returns the instance
    so we can pass it directly to generate_traffic()."""
    from commonroad_sumo.cr2sumo.traffic_generator.random_trips_traffic_generator import (  # noqa: E501
        RandomTripsTrafficGenerator)
    return RandomTripsTrafficGenerator()


def _inject_vtype_distribution(project, vehicle_types: list[str]) -> None:
    """Post-process commonroad-sumo's generated routes file by inserting a
    `<vTypeDistribution id="DEFAULT_VEHTYPE">` element. All vehicles in the
    file have no `type=` attribute, so SUMO resolves them against this
    distribution at simulation time."""
    from commonroad_sumo.sumolib.sumo_project import SumoFileType

    path = None
    try:
        path = project.get_file_path(SumoFileType.VEHICLE_ROUTES)
    except Exception:
        # Fallback: look for *.vehicles.rou.xml in the project dir.
        import glob, os
        hits = glob.glob(os.path.join(str(project.project_path),
                                      "*.vehicles.rou.xml"))
        if hits:
            from pathlib import Path as _P
            path = _P(hits[0])
    if path is None or not path.exists():
        return

    chosen = [t for t in vehicle_types if t in _UI_TYPE_TO_SUMO_VCLASS]
    if not chosen:
        chosen = ["car"]
    n = len(chosen)

    # Equal-probability vType entries under a single vTypeDistribution.
    entries: list[str] = []
    for i, t in enumerate(chosen):
        vclass = _UI_TYPE_TO_SUMO_VCLASS[t]
        entries.append(
            f'    <vType id="osm_{t}" vClass="{vclass}" guiShape="{vclass}" '
            f'probability="{1.0 / n:.4f}"/>'
        )
    dist_xml = (
        '  <vTypeDistribution id="DEFAULT_VEHTYPE">\n'
        + "\n".join(entries) + "\n"
        + "  </vTypeDistribution>\n"
    )

    text = path.read_text()
    if "DEFAULT_VEHTYPE" in text:
        # already has one — leave as-is
        return
    # Insert right after the <routes ...> opening tag.
    import re
    m = re.search(r"<routes[^>]*>", text)
    if not m:
        return
    insertion_point = m.end()
    new_text = text[:insertion_point] + "\n" + dist_xml + text[insertion_point:]
    path.write_text(new_text)


def _patch_sumo_vclass_mapping() -> None:
    """Upstream commonroad-sumo has two limitations we patch at import time:

    (1) `_VEHICLE_CLASS_SUMO2CR` only maps 7 of the ~25 SUMO vClasses to
        CommonRoad ObstacleType — we add fallbacks for trailer/delivery/
        coach/moped so SUMO→CR extraction doesn't ValueError on them.

    (2) `_obstacle_is_vehicle()` only accepts CAR/TRUCK/BICYCLE/BUS/
        MOTORCYCLE — so TAXI and PRIORITY_VEHICLE obstacles trip
        `SumoInterfaceError` on the first sync step even though the
        underlying vehicle interface can handle them fine. We widen the
        gate to include both.

    Both patches are idempotent."""
    try:
        from commonroad.scenario.obstacle import ObstacleType
        from commonroad_sumo.interface import util as _u
        from commonroad_sumo.interface import sumo_simulation_interface as _ssi
        # `_VEHICLE_CLASS_SUMO2CR` is keyed by SumoVehicleClass
        # (commonroad_sumo.backend.types), NOT by sumolib.net.VehicleType —
        # they're two separate enums even though they share member names.
        from commonroad_sumo.backend.types import SumoVehicleClass as _SVC_extras
    except Exception:
        return

    extras = {
        _SVC_extras.TRAILER:  ObstacleType.TRUCK,
        _SVC_extras.DELIVERY: ObstacleType.TRUCK,
        _SVC_extras.COACH:    ObstacleType.BUS,
        _SVC_extras.MOPED:    ObstacleType.MOTORCYCLE,
    }
    for k, v in extras.items():
        _u._VEHICLE_CLASS_SUMO2CR.setdefault(k, v)

    _orig_is_vehicle = getattr(_ssi, "_obstacle_is_vehicle", None)
    if _orig_is_vehicle is not None and not getattr(
            _orig_is_vehicle, "__osm_patched__", False):
        def _is_vehicle(obstacle_type):
            return (_orig_is_vehicle(obstacle_type)
                    or obstacle_type is ObstacleType.TAXI
                    or obstacle_type is ObstacleType.PRIORITY_VEHICLE)
        _is_vehicle.__osm_patched__ = True
        _ssi._obstacle_is_vehicle = _is_vehicle

    # (3) commonroad-sumo's SumoVehicleClass has a typo — TAXI is mapped to
    # "traxi" (should be "taxi"). Patch `from_sumo_str` to accept the
    # correct SUMO string too. Ref: backend/types.py line 226.
    try:
        from commonroad_sumo.backend.types import SumoVehicleClass as _SVC
    except Exception:
        _SVC = None
    if _SVC is not None and not getattr(
            getattr(_SVC, "from_sumo_str", None), "__osm_patched__", False):
        _orig_from = _SVC.from_sumo_str.__func__
        def _patched_from(cls, vehicle_class_str: str):
            if vehicle_class_str == "taxi":
                return cls.TAXI
            return _orig_from(cls, vehicle_class_str)
        _patched_from.__osm_patched__ = True
        _SVC.from_sumo_str = classmethod(_patched_from)


def simulate(cr_path: Path,
             out_xml: Path | None = None,
             simulation_steps: int = 300,
             random_seed: int = 1234,
             vehicle_types: list[str] | None = None,
             density: str = "medium") -> dict:
    """SUMO-driven traffic simulation on a CommonRoad scenario.

    Args:
        cr_path: input CommonRoad .xml (from tools.cr_convert.convert)
        out_xml: output path (default: data/cr/<stem>_sim.xml)
        simulation_steps: number of SUMO steps (at the scenario's dt)
        random_seed: seed for SumoSimulationConfig
        vehicle_types: names from the UI's "SUMO vehicle mix" group
        density: UI "SUMO traffic density" preset, mapped to
            commonroad-sumo's `max_veh_per_km`.

    Returns a dict with the populated CR .xml and the persisted
    commonroad-sumo SUMO project files (so `sumo_render.launch()` can
    replay the *same* vehicles + IDs in sumo-gui):
        {
            "sim_xml":  Path to populated CommonRoad .xml,
            "sumo_cfg": Path to the persisted <project>.sumo.cfg,
            "sumo_net": Path to the persisted SUMO .net.xml,
            "sumo_rou": Path to the persisted vehicle routes file,
            "sumo_dir": Path to the directory containing all of the above,
        }
    """
    from commonroad.common.file_reader import CommonRoadFileReader
    from commonroad_sumo import (
        NonInteractiveSumoSimulation,
        SumoSimulationConfig,
    )

    cr_path = Path(cr_path)
    if not cr_path.exists():
        raise FileNotFoundError(cr_path)

    if out_xml is None:
        out_xml = DATA_CR / f"{cr_path.stem}_sim.xml"
    out_xml = Path(out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    scenario, _ = CommonRoadFileReader(str(cr_path)).open()
    _sanitize_lanelets(scenario)
    _patch_sumo_vclass_mapping()

    traffic_generator = (
        _build_traffic_generator(vehicle_types or [])
        if vehicle_types else _default_random_generator()
    )
    from commonroad_sumo.cr2sumo.map_converter.map_converter import (
        CR2SumoMapConverter)
    converter = CR2SumoMapConverter(scenario)
    project = converter.create_sumo_files()
    if project is None:
        raise RuntimeError("CR2SumoMapConverter.create_sumo_files failed")
    with _silence_stdout():
        traffic_generator.generate_traffic(scenario, project)
    _inject_vtype_distribution(project, vehicle_types or ["car"])
    # Apply UI "SUMO traffic density" by truncating the routes file
    # AND spread depart times across the sim window so vehicles enter
    # continuously rather than all at t=0.
    max_trips = _DENSITY_TO_MAX_TRIPS.get(str(density).lower())
    sim_seconds_window = float(simulation_steps) * float(scenario.dt)
    if max_trips is not None:
        _truncate_routes(project, max_trips, sim_seconds_window)
    else:
        # Even without a density cap, spread the departures so traffic
        # is visible across the whole window.
        _truncate_routes(project, max_trips=10**9,
                         sim_seconds=sim_seconds_window)

    sim = NonInteractiveSumoSimulation(
        scenario, project,
        simulation_config=SumoSimulationConfig(random_seed=random_seed),
    )
    with _silence_stdout():
        result = sim.run(simulation_steps=simulation_steps)

    result.write_to_file(str(out_xml))
    if not out_xml.exists():
        raise RuntimeError(f"SUMO simulation produced no output at {out_xml}")

    # ID-alignment post-process. commonroad-sumo's `IdMapper` allocates
    # CR obstacle IDs that skip lanelet IDs already in the scenario, so
    # SUMO trip "1" → CR obstacle "3" (or some other non-conflicting
    # int) by default. Rewrite the saved XML so:
    #   * dynamicObstacle id == SUMO trip id (literal equality)
    #   * lanelet ids and their refs are shifted by _LANELET_OFFSET so
    #     they don't collide with the now-small vehicle ids when the
    #     scenario is reloaded.
    # Exposes a `cr2sumo` mapping to the caller for downstream tools.
    cr_to_sumo: dict[int, int] = {}
    for s_id, c_id in getattr(sim, "_id_mapper", None)._sumo2cr.items() \
            if getattr(sim, "_id_mapper", None) is not None else []:
        if isinstance(s_id, str) and s_id.isdigit():
            cr_to_sumo[int(c_id)] = int(s_id)
    _rewrite_sim_xml_ids(out_xml, cr_to_sumo,
                         lanelet_offset=_LANELET_ID_OFFSET)

    # Copy commonroad-sumo's ephemeral SumoProject (net + routes + cfg) out
    # of its TemporaryDirectory so `sumo_render.launch(replay_cfg=...)` can
    # run sumo-gui on the SAME vehicles + IDs that populated the CR xml.
    # Must happen before the SumoProject object is garbage-collected.
    persisted = (DATA_NET / f"{cr_path.stem}_cr_sim").resolve()
    if persisted.exists():
        shutil.rmtree(persisted)
    persisted.mkdir(parents=True, exist_ok=True)
    for f in Path(project.project_path).iterdir():
        if f.is_file():
            shutil.copy2(f, persisted / f.name)

    # commonroad-sumo emits <scenario>.sumo.cfg (note the dot); earlier
    # SUMO builds used .sumocfg — accept both.
    sumo_cfg = (next(persisted.glob("*.sumo.cfg"), None)
                or next(persisted.glob("*.sumocfg"), None))
    sumo_net = next(persisted.glob("*.net.xml"), None)
    sumo_rou = (next(persisted.glob("*.vehicles.rou.xml"), None)
                or next(persisted.glob("*.rou.xml"), None))
    if sumo_cfg is None:
        raise RuntimeError(
            f"commonroad-sumo did not emit a SUMO config in {persisted}")

    # PlannerForge-style consistency: rewrite the persisted SUMO routes
    # file so each <route edges> reflects only the edges actually
    # traversed in the CR simulation horizon. After this, SUMO's
    # first/last route edge == CR obstacle's first/last position-edge
    # for every vehicle (initial AND final lanelet match).
    if sumo_rou is not None and sumo_net is not None:
        try:
            cr_id_to_sumo_id: dict[int, int] = {}
            mapper = getattr(sim, "_id_mapper", None)
            if mapper is not None:
                for s_id, c_id in mapper._sumo2cr.items():
                    if isinstance(s_id, str) and s_id.isdigit():
                        cr_id_to_sumo_id[int(c_id)] = int(s_id)
            _retrace_sumo_routes_from_cr_trajectories(
                sim._scenario, Path(sumo_net), Path(sumo_rou),
                cr_id_to_sumo_id)
        except Exception:
            # Don't fail the whole pipeline if the retrace hits a
            # geometry edge case — the unfiltered routes file is
            # still functional, just not PlannerForge-aligned.
            pass

    # Debug-artifact sidecar (NOT a runtime dependency — nothing in the
    # pipeline reads this file). Persist the lanelet_id ↔ SUMO edge_id
    # mapping that CR2SumoMapConverter built so it's discoverable for
    # ad-hoc debugging and any future tool that needs to translate
    # between the two ID spaces. Keys use the **post-shift** lanelet
    # IDs that the saved _sim.xml uses, so the mapping remains valid
    # when reloading the CR scenario.
    try:
        import json
        mapping = {}
        for k, v in getattr(converter, "lanelet_id2lane_id", {}).items():
            mapping[str(int(k) + _LANELET_ID_OFFSET)] = str(v)
        (persisted / "lanelet_id2lane_id.json").write_text(
            json.dumps(mapping, indent=2))
    except Exception:
        pass

    return {
        "sim_xml":  out_xml,
        "sumo_cfg": sumo_cfg,
        "sumo_net": sumo_net,
        "sumo_rou": sumo_rou,
        "sumo_dir": persisted,
    }
