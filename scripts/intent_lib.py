"""Resolve an LLM-emitted OSM-intent JSON into concrete kwargs for the
existing osm_pipeline_subprocess functions, then drive the full Path B
pipeline (fetch → simulate → ego/goal → synthesize → optional Frenetix).

Two layers:

1. **Resolvers** convert intent-JSON fields into concrete values:
   - `geocode_city(city, country_code) -> bbox`   via Nominatim
   - `resolve_ego(intent.ego, sim_xml)    -> ego_id`
   - `intent_to_stage1_kwargs(intent)     -> dict`
   - `intent_to_synth_kwargs(intent, ego_id) -> dict`

2. `run_intent_pipeline(intent, out_dir, ...)` ties them together,
   returns a flat dict with per-stage flags + timings + the same shape
   as `ablation_lib.run_one_query` so downstream tooling is reusable.
"""
from __future__ import annotations

import math
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from commonroad_interface import osm_pipeline_subprocess as osm_sub  # noqa: E402


# ── Defaults (mirrors the schema docstring in osm_intent_prompt.txt) ───

DEFAULT_BBOX_HALF_EXTENT_DEG = 0.0009   # ~100 m N-S → 200 m × 200 m total area
                                       # Sized to match the median curated CommonRoad
                                       # scenario (197×199 m, 18 lanelets — measured
                                       # across 40 random CollectedScenarios). Earlier
                                       # default (0.005°) produced ~1 km² areas with
                                       # 2000-4000 lanelets that overwhelmed netconvert
                                       # and caused ~40% of queries to fail at simulate.
DEFAULT_SIM_DURATION_S = 20.0
DEFAULT_DENSITY = "low"
DEFAULT_VEHICLE_TYPES = ["car"]
DEFAULT_ROAD_CLASSES = ["roads"]
DEFAULT_GOAL_OFFSET_STEPS = 50
DEFAULT_GOAL_POSITION = "middle"
DEFAULT_GOAL_LENGTH_M = 6.0
DEFAULT_GOAL_WIDTH_M = 2.0

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = "PlannerForge/benchmark"

# ISO 3166-1 alpha-3 → alpha-2. Nominatim's `countrycodes` query
# parameter accepts alpha-2 only; without it, "Munich" matches a small
# town in Spain instead of the Bavarian capital. Limited to countries
# the ablation queries are likely to visit.
ALPHA3_TO_ALPHA2 = {
    "DEU": "de", "AUT": "at", "CHE": "ch", "FRA": "fr", "ESP": "es",
    "ITA": "it", "GBR": "gb", "USA": "us", "CAN": "ca", "MEX": "mx",
    "JPN": "jp", "CHN": "cn", "KOR": "kr", "IND": "in", "AUS": "au",
    "NZL": "nz", "BRA": "br", "ARG": "ar", "NLD": "nl", "BEL": "be",
    "LUX": "lu", "DNK": "dk", "SWE": "se", "NOR": "no", "FIN": "fi",
    "POL": "pl", "CZE": "cz", "PRT": "pt", "IRL": "ie", "GRC": "gr",
}


# ── Resolvers ──────────────────────────────────────────────────────────

def geocode_city(city: str,
                 country_code: str | None = None,
                 half_extent_deg: float = DEFAULT_BBOX_HALF_EXTENT_DEG,
                 ) -> tuple[float, float, float, float] | None:
    """Look up `city` via Nominatim, return a small bbox centred on it.

    Returns (south, west, north, east). Nominatim itself reports a much
    larger bbox (the full city) which would blow past MAX_BBOX_KM2 — we
    instead grab the centroid (lat/lon) and synthesize a fixed-size
    rectangle of half_extent_deg in each direction (~1 km × 1 km).
    """
    params = {"q": city, "format": "json", "limit": 1}
    if country_code:
        alpha2 = ALPHA3_TO_ALPHA2.get(country_code.upper())
        if alpha2:
            params["countrycodes"] = alpha2
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params=params,
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=15,
        )
    except Exception as e:
        print(f"  geocode network error: {type(e).__name__}: {e}",
              file=sys.stderr)
        return None
    if resp.status_code != 200:
        print(f"  geocode HTTP {resp.status_code} for {city!r}",
              file=sys.stderr)
        return None
    data = resp.json()
    if not data:
        return None
    try:
        lat = float(data[0]["lat"])
        lon = float(data[0]["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    south = lat - half_extent_deg
    north = lat + half_extent_deg
    # Compensate for longitude shrinkage by latitude so the patch stays
    # roughly square in km terms.
    cos_lat = max(math.cos(math.radians(lat)), 0.05)
    half_lon = half_extent_deg / cos_lat
    west = lon - half_lon
    east = lon + half_lon
    return (south, west, north, east)


def intent_to_stage1_kwargs(intent: dict) -> tuple[dict, str | None]:
    """Convert intent.location/road_classes/sim into kwargs for
    `osm_pipeline_subprocess.run_stage1`. Returns (kwargs, error_msg).
    error_msg is set when the bbox can't be determined (e.g. geocoding
    failed AND no explicit bbox was given)."""
    loc = (intent or {}).get("location") or {}
    bbox = loc.get("bbox")
    if not bbox:
        city = loc.get("city")
        if not city:
            return {}, "no bbox and no city in intent.location"
        bbox = geocode_city(city, loc.get("country_code"))
        if bbox is None:
            return {}, f"geocode failed for city={city!r}"
    bbox = tuple(float(x) for x in bbox)
    if len(bbox) != 4 or not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        return {}, f"invalid bbox: {bbox}"

    classes = intent.get("road_classes") or DEFAULT_ROAD_CLASSES
    sim = intent.get("sim") or {}
    kwargs = {
        "bbox":          bbox,
        "classes":       classes,
        "density":       sim.get("density") or DEFAULT_DENSITY,
        "sim_seconds":   float(sim.get("duration_s")
                               or DEFAULT_SIM_DURATION_S),
        "vehicle_types": sim.get("vehicle_types") or DEFAULT_VEHICLE_TYPES,
        "simulate_cr":   True,
        "record_cr_gif": False,
    }
    return kwargs, None


# The downstream Frenetix motion planner is implemented only for cars.
# Every ego-resolution strategy filters the candidate obstacle list to
# CR ObstacleType=="car" before picking, so we never end up promoting a
# truck/bus/motorcycle to ego — that would fail at planning time.
EGO_CR_TYPE = "car"


def _ego_type_map(sim_xml: Path) -> dict[int, str]:
    """Return obstacle_id -> obstacle_type_name (e.g. 'car', 'truck')."""
    try:
        from commonroad.common.file_reader import CommonRoadFileReader
    except ImportError as e:
        raise RuntimeError(f"commonroad import failed: {e}")
    scen, _ = CommonRoadFileReader(str(sim_xml)).open()
    out: dict[int, str] = {}
    for obs in scen.dynamic_obstacles:
        try:
            out[int(obs.obstacle_id)] = obs.obstacle_type.value
        except Exception:
            out[int(obs.obstacle_id)] = "unknown"
    return out


def resolve_ego(ego_intent: dict | None,
                sim_xml: Path) -> tuple[int | None, str]:
    """Pick an ego obstacle ID per the intent.ego strategy.

    Returns (ego_id, error). error is empty on success.

    Frenetix is car-only, so this function filters the candidate list
    to CR obstacle type "car" before applying any strategy. The intent
    may name a non-car vehicle for ego.by_type — it gets silently
    downgraded to the first car (the prompt teaches the LLM to emit
    strategy="first" in that case, but we defend in depth here).
    """
    ego_intent = ego_intent or {}
    strategy = (ego_intent.get("strategy") or "first").lower()
    try:
        ids = osm_sub.list_egos(sim_xml, limit=30)
    except Exception as e:
        return None, f"list_egos failed: {type(e).__name__}: {e}"
    if not ids:
        return None, "no dynamic obstacles in sim XML"

    try:
        type_map = _ego_type_map(sim_xml)
    except Exception as e:
        return None, f"type lookup failed: {type(e).__name__}: {e}"
    # Filter to cars only — Frenetix can only plan car trajectories.
    car_ids = [oid for oid in ids
               if type_map.get(oid, "").lower() == EGO_CR_TYPE]
    if not car_ids:
        return None, ("no car in sim XML; types available: "
                      f"{sorted(set(type_map.values()))[:6]}")

    if strategy == "first":
        return int(car_ids[0]), ""

    if strategy == "by_index":
        idx = ego_intent.get("index")
        if not isinstance(idx, int) or idx < 0:
            return None, f"by_index requires non-negative int, got {idx!r}"
        if idx >= len(car_ids):
            return None, (f"by_index={idx} out of range "
                          f"(have {len(car_ids)} cars)")
        return int(car_ids[idx]), ""

    if strategy == "by_type":
        want = (ego_intent.get("vehicle_type") or "").lower()
        if want and want != EGO_CR_TYPE:
            # The user named a non-car vehicle as ego, but the planner
            # can't handle it. Downgrade to the first car. The prompt
            # is supposed to prevent this at extraction time; this is
            # the defensive fallback.
            return int(car_ids[0]), ""
        return int(car_ids[0]), ""

    return None, f"unknown ego strategy {strategy!r}"


_BENCHMARK_TOKEN = re.compile(r"[^A-Za-z0-9]+")


def _slug_for_benchmark(name: str | None) -> str:
    """Sanitise a city / road name into a 2-3 char MAP token per
    CommonRoad 2020a §2.1.2 ("MAP is for rural scenarios a two/three
    letter city code (e.g. Muc) and for highways/major roads the road
    code (e.g. A9 or Lanker)"). Strategy:
      - drop non-alphanumeric chars (spaces, punctuation, accents)
      - if input already short (≤3 chars, e.g. "A8", "I95") keep as-is
      - otherwise take first 3 chars, capitalise initial
    Empty input → 'Gen' (artificial / generated)."""
    if not name:
        return "Gen"
    s = _BENCHMARK_TOKEN.sub("", name)
    if not s:
        return "Gen"
    if len(s) <= 3:
        return s
    # Preserve title-case if the original was a proper noun
    return s[0].upper() + s[1:3].lower()


def _qid_to_int(qid: str) -> int:
    """Pull the numeric suffix off a Q-id ('Q017' → 17). Falls back to
    1 when the id doesn't follow the convention."""
    m = re.search(r"(\d+)$", qid or "")
    return int(m.group(1)) if m else 1


def rewrite_benchmark_id(xml_path: Path, intent: dict, qid: str) -> str | None:
    """Replace crdesigner's default 'ZAM_MUC-1' benchmark_id with one
    derived from the LLM intent + the query id.

    Returns the new id on success (e.g. 'DEU_Munich-7_1_T-1'), or None
    if rewriting failed — the XML is left untouched in that case so the
    pipeline still works downstream.
    """
    try:
        from commonroad.common.file_reader import CommonRoadFileReader
        from commonroad.common.file_writer import (CommonRoadFileWriter,
                                                    OverwriteExistingFile)
        from commonroad.scenario.scenario import ScenarioID
    except ImportError as e:
        print(f"  rewrite_benchmark_id: commonroad import failed: {e}",
              file=sys.stderr)
        return None
    try:
        scenario, pps = CommonRoadFileReader(str(xml_path)).open()
    except Exception as e:
        print(f"  rewrite_benchmark_id: read failed for {xml_path}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None

    loc = (intent or {}).get("location") or {}
    cc = (loc.get("country_code") or "ZAM").upper()
    if not (len(cc) == 3 and cc.isalpha()):
        cc = "ZAM"
    map_name = _slug_for_benchmark(loc.get("city"))
    map_id = _qid_to_int(qid)
    new_bid = f"{cc}_{map_name}-{map_id}_1_T-1"
    try:
        scenario.scenario_id = ScenarioID.from_benchmark_id(new_bid, "2020a")
    except Exception as e:
        print(f"  rewrite_benchmark_id: ScenarioID parse failed for "
              f"{new_bid!r}: {type(e).__name__}: {e}", file=sys.stderr)
        return None
    try:
        writer = CommonRoadFileWriter(
            scenario, pps or None,
            "PlannerForge OSM intent pipeline",
            "TUM",
            "OpenStreetMap + commonroad-sumo",
            tags=set(),
        )
        writer.write_to_file(str(xml_path), OverwriteExistingFile.ALWAYS)
    except Exception as e:
        print(f"  rewrite_benchmark_id: write failed for {xml_path}: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None
    return new_bid


def _synth_traffic_free_planning_problem(
        cr_path: Path,
        out_xml: Path,
        intent: dict | None = None) -> bool:
    """Build a CR XML with a <planningProblem> on top of a crdesigner-
    produced static scenario (no dynamic obstacles).

    Used as a fallback when commonroad-sumo's CR2SumoMapConverter chokes
    on the lanelet network — we still have a valid CR XML from
    cr_convert, so we synthesise an ego trajectory programmatically
    instead of giving up on the whole pipeline.

    Strategy:
      1. Open `cr_path` via CommonRoadFileReader.
      2. Pick the longest lanelet (by aggregate centre-vertex distance)
         as the ego start.
      3. Initial state: position = centre vertex at index 1, velocity = 0,
         orientation = polyline tangent at the start.
      4. Goal region: a rectangle centred on the lanelet's final centre
         vertex (or its longest successor's mid-point), oriented along
         that lanelet's direction. Dimensions come from intent.goal
         (default 6 m × 2 m).
      5. Write the resulting scenario+planning-problem-set to `out_xml`.

    Returns True on success, False if anything failed (no lanelets,
    write error, etc.) — caller decides whether that counts as a
    pipeline failure or a different fallback path.
    """
    try:
        import math
        import numpy as np
        from commonroad.common.file_reader import CommonRoadFileReader
        from commonroad.common.file_writer import (
            CommonRoadFileWriter, OverwriteExistingFile)
        from commonroad.common.util import Interval
        from commonroad.geometry.shape import Rectangle
        from commonroad.planning.goal import GoalRegion
        from commonroad.planning.planning_problem import (
            PlanningProblem, PlanningProblemSet)
        from commonroad.scenario.state import CustomState, InitialState
    except ImportError as e:
        print(f"  no-traffic synth: commonroad import failed: {e}",
              file=sys.stderr)
        return False
    try:
        scenario, _ = CommonRoadFileReader(str(cr_path)).open()
    except Exception as e:
        print(f"  no-traffic synth: read failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False

    net = scenario.lanelet_network
    if not net.lanelets:
        print("  no-traffic synth: scenario has no lanelets", file=sys.stderr)
        return False

    def _polyline_length(verts) -> float:
        if verts is None or len(verts) < 2:
            return 0.0
        arr = np.asarray(verts, dtype=float)
        diff = np.diff(arr, axis=0)
        return float(np.linalg.norm(diff, axis=1).sum())

    # 1. Pick longest lanelet.
    lanelets_ranked = sorted(
        ((la, _polyline_length(la.center_vertices)) for la in net.lanelets),
        key=lambda p: p[1], reverse=True)
    ego_la = None
    for la, L in lanelets_ranked:
        if L >= 8.0 and la.center_vertices is not None and len(la.center_vertices) >= 3:
            ego_la = la
            break
    if ego_la is None:
        print("  no-traffic synth: no lanelet long enough (>=8m)",
              file=sys.stderr)
        return False

    cv = np.asarray(ego_la.center_vertices, dtype=float)
    ini_pos = cv[1]
    # Orientation from the local tangent.
    tan = cv[2] - cv[0]
    ini_ori = math.atan2(float(tan[1]), float(tan[0]))

    # 2. Find a goal point. Walk along the lanelet's longest successor
    # chain (depth-1) for a richer test scenario when the start lanelet
    # is short; otherwise just use the end of the start lanelet.
    goal_la = ego_la
    if ego_la.successor:
        for succ_id in ego_la.successor:
            succ = net.find_lanelet_by_id(succ_id)
            if succ is None or succ.center_vertices is None:
                continue
            if _polyline_length(succ.center_vertices) > _polyline_length(goal_la.center_vertices):
                goal_la = succ
                break
    gv = np.asarray(goal_la.center_vertices, dtype=float)
    goal_pos = gv[-2]
    goal_tan = gv[-1] - gv[-3] if len(gv) >= 3 else gv[-1] - gv[0]
    goal_ori = math.atan2(float(goal_tan[1]), float(goal_tan[0]))

    # 3. Pull goal dimensions from intent (defaults: 6 x 2 m).
    goal = ((intent or {}).get("goal") or {})
    gl = float(goal.get("rectangle_length_m") or DEFAULT_GOAL_LENGTH_M)
    gw = float(goal.get("rectangle_width_m") or DEFAULT_GOAL_WIDTH_M)

    initial_state = InitialState(
        position=np.array([float(ini_pos[0]), float(ini_pos[1])]),
        orientation=ini_ori,
        velocity=0.0,
        yaw_rate=0.0,
        slip_angle=0.0,
        time_step=0,
    )

    goal_rect = Rectangle(length=gl, width=gw,
                           center=np.array([float(goal_pos[0]),
                                            float(goal_pos[1])]),
                           orientation=goal_ori)
    goal_state = CustomState(
        position=goal_rect,
        time_step=Interval(40, 60),
    )
    pp = PlanningProblem(planning_problem_id=1000,
                          initial_state=initial_state,
                          goal_region=GoalRegion([goal_state]))
    pps = PlanningProblemSet(planning_problem_list=[pp])

    try:
        out_xml.parent.mkdir(parents=True, exist_ok=True)
        writer = CommonRoadFileWriter(
            scenario, pps,
            "PlannerForge OSM intent pipeline (no-traffic fallback)",
            "TUM",
            "OpenStreetMap (CR2SumoMapConverter bypassed)",
            tags=set(),
        )
        writer.write_to_file(str(out_xml), OverwriteExistingFile.ALWAYS)
        return True
    except Exception as e:
        print(f"  no-traffic synth: write failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        return False


def intent_to_synth_kwargs(intent: dict, ego_id: int) -> dict:
    """Convert intent.goal into kwargs for run_synthesize."""
    goal = (intent or {}).get("goal") or {}
    strategy = (goal.get("strategy") or "trajectory_offset").lower()
    out = {
        "ego_id":            ego_id,
        "goal_position":     goal.get("position_keyword") or DEFAULT_GOAL_POSITION,
        "goal_offset_steps": int(goal.get("offset_steps")
                                 or DEFAULT_GOAL_OFFSET_STEPS),
        "goal_length_m":     float(goal.get("rectangle_length_m")
                                    or DEFAULT_GOAL_LENGTH_M),
        "goal_width_m":      float(goal.get("rectangle_width_m")
                                    or DEFAULT_GOAL_WIDTH_M),
        "goal_lanelet_id":   None,
    }
    if strategy == "specific_lanelet":
        lid = goal.get("lanelet_id")
        if isinstance(lid, int):
            out["goal_lanelet_id"] = int(lid)
    return out


# ── End-to-end pipeline driver ─────────────────────────────────────────

def _blank_result(qid: str) -> dict:
    return {
        "query_id":        qid,
        "geocode_ok":      False,
        "fetch_ok":        False,
        "netconvert_ok":   False,
        "cr_convert_ok":   False,
        "simulate_ok":     False,
        "ego_resolve_ok":  False,
        "ego_id":          None,
        "n_egos":          0,
        "synth_ok":        False,
        "xml_load_ok":     False,
        "n_lanelets":      0,
        "n_obstacles":     0,
        "has_pp":          False,
        "frenetix_ok":     False,
        "gif_present":     "",
        "t_stage1_s":      0.0,
        "t_synth_s":       0.0,
        "t_frenetix_s":    0.0,
        "t_total_s":       0.0,
        "resolved_bbox":   None,
        "osm_path":        None,
        "cr_sim_path":     None,
        "error_stage":     "",
        "error_msg":       "",
    }


def _drain(gen) -> tuple[list[str], dict]:
    """Drain an osm_sub generator: collect log lines, return ([lines], paths)."""
    lines: list[str] = []
    paths: dict = {}
    for item in gen:
        if isinstance(item, tuple) and item and item[0] == "RESULT":
            paths = item[1]
            continue
        lines.append(str(item))
    return lines, paths


def _write_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def run_intent_pipeline(intent: dict,
                        qid: str,
                        out_dir: Path,
                        *,
                        skip_frenetix: bool = False,
                        planner_cfg: dict | None = None,
                        frenetix_timeout_s: float = 600.0) -> dict:
    """Drive Path B end-to-end from an LLM-emitted intent dict.

    out_dir must be a per-(condition,qid) directory; the function writes
    Original/<qid>.xml + stage1.log/synth.log/frenetix.log there. Returns
    a flat dict with per-stage flags + timings + error_stage/error_msg.
    """
    import subprocess
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "Original").mkdir(parents=True, exist_ok=True)
    result = _blank_result(qid)
    t_start = time.time()

    # 1. Resolve location → bbox + stage1 kwargs.
    stage1_kwargs, err = intent_to_stage1_kwargs(intent)
    if err:
        result["error_stage"] = "geocode"
        result["error_msg"] = err
        result["t_total_s"] = time.time() - t_start
        return result
    result["geocode_ok"] = True
    result["resolved_bbox"] = list(stage1_kwargs["bbox"])

    # 2. Run Stage 1.
    t0 = time.time()
    try:
        lines, paths = _drain(osm_sub.run_stage1(**stage1_kwargs))
    except Exception as e:
        result["error_stage"] = "fetch"
        result["error_msg"] = f"{type(e).__name__}: {e}"
        result["t_stage1_s"] = time.time() - t0
        result["t_total_s"] = time.time() - t_start
        return result
    result["t_stage1_s"] = time.time() - t0
    _write_log(out_dir / "stage1.log", lines)

    result["fetch_ok"]      = bool(paths.get("OSM"))
    result["netconvert_ok"] = bool(paths.get("SUMO net"))
    result["cr_convert_ok"] = bool(paths.get("CR scenario"))
    sim_xml_str = paths.get("CR sim scenario")
    result["simulate_ok"]   = bool(sim_xml_str and Path(sim_xml_str).exists())

    # ── Layer B: bbox-shrink retry ────────────────────────────────────
    # If cr_convert succeeded but cr_simulate (CR2SumoMapConverter)
    # didn't, the LLM extraction is fine — the bbox is just too big /
    # contains too much pathological geometry for netconvert. Retry the
    # whole stage1 once with the bbox area halved (centre kept, half-
    # extent × 0.5 in each direction). Smaller areas usually have
    # cleaner geometry and the converter often succeeds on the retry.
    if (not result["simulate_ok"]) and result["cr_convert_ok"]:
        s, w, n, e = stage1_kwargs["bbox"]
        cy, cx = (s + n) / 2.0, (w + e) / 2.0
        hy, hx = (n - s) / 4.0, (e - w) / 4.0   # half of the half-extent → quarter
        shrunk = (cy - hy, cx - hx, cy + hy, cx + hx)
        retry_kwargs = {**stage1_kwargs, "bbox": shrunk}
        t0 = time.time()
        try:
            lines2, paths2 = _drain(osm_sub.run_stage1(**retry_kwargs))
        except Exception:
            lines2, paths2 = [], {}
        result["t_stage1_s"] = result.get("t_stage1_s", 0) + (time.time() - t0)
        _write_log(out_dir / "stage1_retry_shrunk.log", lines2)
        sim2 = paths2.get("CR sim scenario")
        if sim2 and Path(sim2).exists():
            # Retry succeeded — swap in the new artefacts and continue.
            paths = paths2
            sim_xml_str = sim2
            result["simulate_ok"]   = True
            result["fetch_ok"]      = bool(paths.get("OSM"))
            result["netconvert_ok"] = bool(paths.get("SUMO net"))
            result["cr_convert_ok"] = bool(paths.get("CR scenario"))
            result["bbox_retry"]    = "shrunk"
            result["resolved_bbox"] = list(shrunk)

    # ── Layer C: bbox-shift retry ─────────────────────────────────────
    # If the bbox-shrink retry still didn't produce a SUMO net (Toronto
    # downtown is the canonical case: Nominatim's centroid lands on a
    # pedestrianised block where netconvert refuses the geometry, and
    # the shrunk bbox lands inside the same dead zone), step the bbox by
    # one full width to the N / E / S / W. Keeping the original size — a
    # neighbour cell of the same area is usually enough to escape the
    # bad geometry without losing the requested scale. Stop at the first
    # direction that yields simulate_ok=True.
    if (not result["simulate_ok"]) and result["cr_convert_ok"]:
        orig_s, orig_w, orig_n, orig_e = stage1_kwargs["bbox"]
        dlat = orig_n - orig_s
        dlon = orig_e - orig_w
        for dir_label, (dy, dx) in [
            ("N", ( dlat,    0.0)),
            ("S", (-dlat,    0.0)),
            ("E", ( 0.0,     dlon)),
            ("W", ( 0.0,    -dlon)),
        ]:
            shifted = (orig_s + dy, orig_w + dx,
                       orig_n + dy, orig_e + dx)
            retry_kwargs = {**stage1_kwargs, "bbox": shifted}
            t0 = time.time()
            try:
                lines3, paths3 = _drain(osm_sub.run_stage1(**retry_kwargs))
            except Exception:
                lines3, paths3 = [], {}
            result["t_stage1_s"] = result.get("t_stage1_s", 0) + (time.time() - t0)
            _write_log(out_dir / f"stage1_retry_shift_{dir_label}.log", lines3)
            sim3 = paths3.get("CR sim scenario")
            if sim3 and Path(sim3).exists():
                paths = paths3
                sim_xml_str = sim3
                result["simulate_ok"]   = True
                result["fetch_ok"]      = bool(paths.get("OSM"))
                result["netconvert_ok"] = bool(paths.get("SUMO net"))
                result["cr_convert_ok"] = bool(paths.get("CR scenario"))
                result["bbox_retry"]    = f"shifted_{dir_label}"
                result["resolved_bbox"] = list(shifted)
                break

    # Record source-artifact paths so the evaluator can re-inspect them
    # without re-parsing stage1.log.
    if paths.get("OSM"):
        result["osm_path"] = str(paths["OSM"])
    if sim_xml_str:
        result["cr_sim_path"] = str(sim_xml_str)
    if not result["simulate_ok"]:
        # No-traffic fallback (mirrors osm/'s philosophy: cr_convert
        # and cr_simulate are independent artifacts). If the basic CR
        # XML is on disk but CR2SumoMapConverter choked, build a
        # planning problem on the static lanelet network instead of
        # failing the whole pipeline.
        cr_convert_path = paths.get("CR scenario")
        if (result["cr_convert_ok"] and cr_convert_path
                and Path(cr_convert_path).exists()):
            target_xml = out_dir / "Original" / f"{qid}.xml"
            t0 = time.time()
            ok = _synth_traffic_free_planning_problem(
                Path(cr_convert_path), target_xml, intent)
            result["t_synth_s"] = time.time() - t0
            if ok:
                # Mark the fallback path explicitly so downstream
                # KPIs can distinguish "traffic-rich" from "static"
                # successes if desired.
                result["traffic_fallback"] = True
                result["synth_ok"] = True
                result["n_obstacles"] = 0
                # The traffic-free PP places ego on a long lanelet;
                # ego_id is synthetic (1000) so list_egos didn't run.
                result["n_egos"] = 0
                result["ego_id"] = 1000
                result["ego_resolve_ok"] = True
                _rebid = rewrite_benchmark_id(target_xml, intent, qid)
                if _rebid:
                    result["benchmark_id"] = _rebid
                # XML post-validation (mirrors the inline check used
                # after the normal synth path).
                try:
                    from commonroad.common.file_reader import CommonRoadFileReader
                    _scen, _pps = CommonRoadFileReader(str(target_xml)).open()
                    result["xml_load_ok"] = True
                    result["n_lanelets"] = len(_scen.lanelet_network.lanelets)
                    result["n_obstacles"] = len(_scen.dynamic_obstacles)
                    result["has_pp"] = _pps is not None and bool(
                        getattr(_pps, "planning_problem_dict", {}))
                except Exception as e:
                    result["error_stage"] = "xml_load"
                    result["error_msg"] = f"{type(e).__name__}: {e}"
                result["t_total_s"] = time.time() - t_start
                return result
            # else: fall through to the original failure-attribution
            # block below.
        for flag, stage in (("fetch_ok", "fetch"),
                            ("netconvert_ok", "netconvert"),
                            ("cr_convert_ok", "cr_convert"),
                            ("simulate_ok", "simulate")):
            if not result[flag]:
                result["error_stage"] = stage
                # Pull a "skipped" message if present.
                for k, v in paths.items():
                    if k.endswith("skipped"):
                        result["error_msg"] = f"{k}: {v}"
                        break
                if not result["error_msg"]:
                    result["error_msg"] = f"{stage} produced no output"
                break
        result["t_total_s"] = time.time() - t_start
        return result
    sim_xml = Path(sim_xml_str)

    # 3. Resolve ego via intent strategy.
    try:
        ids = osm_sub.list_egos(sim_xml, limit=30)
        result["n_egos"] = len(ids)
    except Exception as e:
        result["error_stage"] = "no_egos"
        result["error_msg"] = f"list_egos: {type(e).__name__}: {e}"
        result["t_total_s"] = time.time() - t_start
        return result
    ego_id, ego_err = resolve_ego(intent.get("ego"), sim_xml)
    if ego_id is None:
        result["error_stage"] = "ego_resolve"
        result["error_msg"] = ego_err
        result["t_total_s"] = time.time() - t_start
        return result
    result["ego_resolve_ok"] = True
    result["ego_id"] = int(ego_id)

    # 4. Synthesize planning problem.
    target_xml = out_dir / "Original" / f"{qid}.xml"
    syn_kwargs = intent_to_synth_kwargs(intent, ego_id)
    t0 = time.time()
    try:
        lines, paths = _drain(osm_sub.run_synthesize(
            sim_xml=sim_xml, out_path=target_xml, **syn_kwargs))
    except Exception as e:
        result["error_stage"] = "synth"
        result["error_msg"] = f"{type(e).__name__}: {e}"
        result["t_synth_s"] = time.time() - t0
        result["t_total_s"] = time.time() - t_start
        return result
    result["t_synth_s"] = time.time() - t0
    _write_log(out_dir / "synth.log", lines)
    pp_xml_str = paths.get("PP XML")
    if not (pp_xml_str and Path(pp_xml_str).exists()):
        result["error_stage"] = "synth"
        result["error_msg"] = "synth did not produce PP XML"
        result["t_total_s"] = time.time() - t_start
        return result
    if Path(pp_xml_str).resolve() != target_xml.resolve():
        shutil.copy2(pp_xml_str, target_xml)
    result["synth_ok"] = True

    # Stamp the XML with a benchmark_id that reflects the actual query
    # location, not crdesigner's hardcoded "ZAM_MUC-1" default.
    new_bid = rewrite_benchmark_id(target_xml, intent, qid)
    if new_bid:
        result["benchmark_id"] = new_bid

    # 5. XML post-validation.
    try:
        from commonroad.common.file_reader import CommonRoadFileReader
        scen, pps = CommonRoadFileReader(str(target_xml)).open()
        result["xml_load_ok"]  = True
        result["n_lanelets"]   = len(scen.lanelet_network.lanelets)
        result["n_obstacles"]  = len(scen.dynamic_obstacles)
        result["has_pp"]       = pps is not None and bool(
            getattr(pps, "planning_problem_dict", {}))
    except Exception as e:
        result["error_stage"] = "xml_load"
        result["error_msg"] = f"{type(e).__name__}: {e}"
        result["t_total_s"] = time.time() - t_start
        return result

    if skip_frenetix:
        result["t_total_s"] = time.time() - t_start
        return result

    # 6. Frenetix-with-ego.
    cfg = planner_cfg or {
        "venv_path":   str((_ROOT / "Frenetix-Motion-Planner" / "venv" / "bin" / "activate").resolve()),
        "script_path": str((_ROOT / "Frenetix-Motion-Planner" / "main_batch.py").resolve()),
        "log_dir":     str((_ROOT / "Frenetix-Motion-Planner" / "logs").resolve()),
    }
    t0 = time.time()
    cmd = (f"source {cfg['venv_path']} && python {cfg['script_path']} "
           f"--input-file {target_xml}")
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )
    assert proc.stdout is not None
    flines: list[str] = []
    deadline = time.time() + frenetix_timeout_s
    try:
        while True:
            if time.time() > deadline and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
                flines.append("[KILLED: timeout]")
                result["error_stage"] = "timeout"
                result["error_msg"] = f"Frenetix exceeded {frenetix_timeout_s:.0f}s"
                _write_log(out_dir / "frenetix.log", flines)
                result["t_frenetix_s"] = time.time() - t0
                result["t_total_s"] = time.time() - t_start
                return result
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if line:
                flines.append(line.rstrip("\n"))
    finally:
        if proc.stdout:
            proc.stdout.close()
    rc = proc.wait()
    result["t_frenetix_s"] = time.time() - t0
    _write_log(out_dir / "frenetix.log", flines)
    if rc != 0:
        result["error_stage"] = "frenetix"
        result["error_msg"] = f"Frenetix exit code {rc}"
    else:
        result["frenetix_ok"] = True
        gifs = sorted((Path(cfg["log_dir"]) / qid).glob("*.gif")) \
            if (Path(cfg["log_dir"]) / qid).exists() else []
        if gifs:
            result["gif_present"] = str(gifs[0])

    result["t_total_s"] = time.time() - t_start
    return result
