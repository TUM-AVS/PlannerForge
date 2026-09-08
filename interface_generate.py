"""Generate tab: scenario generation from scratch via the vendored OSM
pipeline. After the user picks a bbox + ego + goal, the produced CR XML
(with PlanningProblem baked in) is dropped into Scenarios/<Name>/Original/
and the active tab switches to Main App so the existing modify / motion-
planner flow takes over from step 7.

Ported from osm/interface_map.py. Map iframe + bbox listener JS are
identical; the run handlers call commonroad_interface.osm_pipeline_subprocess
instead of osm's in-process OSMChatAgent.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

from config import CLASS_GROUPS
from commonroad_interface import osm_pipeline_subprocess as osm_sub
from prompts_lib import build_prompt

# scripts/ lives next to this file at the PlannerForge root.
_SCRIPTS_DIR = (Path(__file__).resolve().parent / "scripts")
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from intent_lib import geocode_city, rewrite_benchmark_id  # noqa: E402


STATIC_DIR = (Path(__file__).parent / "static").resolve()
MAP_URL = "gradio_api/file=" + str(STATIC_DIR / "map.html")
GIF_VIEWER_URL = "gradio_api/file=" + str(STATIC_DIR / "gif_viewer.html")

# Just the map iframe. All bbox sync — draw → textbox, and parsed/typed bbox
# → map view — is driven from inside static/map.html, NOT here: gr.HTML can't
# execute injected <script> tags and gradio 5.47 exposes no js_on_load/js hook,
# so any parent-side listener never runs. map.html is a real document (loaded
# via src=), so its own script runs and reaches this textbox through
# window.parent (same origin). See static/map.html for both directions.
IFRAME_HTML = f"""
<iframe src="/{MAP_URL}"
        style="width:100%; height:540px; border:1px solid #ccc; border-radius:6px;">
</iframe>
"""


def _generate_scenario_name(bbox: tuple[float, float, float, float]) -> str:
    """OSM_<bbox6char>_<YYYYMMDD-HHMMSS>. The folder + XML stem must match
    because PlannerForge's select_initial_gif derives paths from this name."""
    h = hashlib.sha1(",".join(f"{x:.6f}" for x in bbox).encode()).hexdigest()[:6]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"OSM_{h}_{stamp}"


def _parse_bbox(text: str) -> tuple[float, float, float, float] | None:
    try:
        parts = [float(x) for x in text.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def _gif_carrier(gif_path: str | None = None) -> str:
    """HTML for the carrier span the gif_viewer.html iframe polls. Holds the
    served URL of the current CR sim GIF (empty = nothing to show)."""
    url = ("/gradio_api/file=" + gif_path) if gif_path else ""
    return f'<span id="cr_gif_url" data-url="{url}"></span>'


# Vehicle classes the OSM intent prompt knows about (sim.vehicle_types domain).
_EGO_VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle", "bicycle",
                      "trailer", "delivery", "coach", "taxi", "emergency", "moped"]


def _preferred_ego_type_from_query(q: str) -> str:
    """Best-effort recovery of the vehicle type the user named as the ego.

    The extraction prompt deliberately downgrades a named non-car ego (e.g.
    "taxi as ego") to strategy="first" because the planner is car-only, so the
    named type is dropped from intent.ego. We recover it from the raw query
    purely to *pre-select* the matching obstacle in the Stage-2 dropdown — the
    user can still change it, and synthesis accepts any ego id. Returns "" if
    no vehicle type is clearly tied to the word "ego"."""
    if not q or "ego" not in q.lower():
        return ""
    import re as _re
    ql = q.lower()
    for t in _EGO_VEHICLE_TYPES:
        # "<type> … ego" or "ego … <type>" within a short window
        if (_re.search(rf"\b{t}\b[\w\s,'-]{{0,24}}\bego\b", ql)
                or _re.search(rf"\bego\b[\w\s,'-]{{0,24}}\b{t}\b", ql)):
            return t
    return ""


# Trailing scenario-type / positional nouns that aren't part of a place name —
# drop them so e.g. "Munich intersection" geocodes as "Munich" and "Cologne
# cathedral vicinity" as "Cologne cathedral".
_SCENARIO_TAIL_WORDS = {
    "intersection", "intersections", "roundabout", "roundabouts", "highway",
    "motorway", "ramp", "ramps", "junction", "junctions", "crossroad",
    "crossroads", "scenario", "scene", "segment", "corridor", "area",
    "vicinity", "approach", "surroundings", "perimeter", "urban", "rural",
    "suburban",
}


def _geocode_query(nl_query: str, city: str) -> str:
    """Best geocoder query for the user's location.

    The extraction prompt keeps only `location.city`, so a named landmark or
    district (e.g. "Schönbrunn Palace", "Brandenburg Gate") is dropped and we'd
    geocode the bare city centroid. Recover the place phrase from the raw query:
    the text before the first comma / scenario verb, with a trailing
    scenario-type noun dropped, and the city appended if it isn't already
    present so the geocoder disambiguates. Returns the city itself when no extra
    place is found."""
    import re as _re
    q = (nl_query or "").strip()
    head = _re.split(r"[,;]", q, maxsplit=1)[0]
    head = _re.split(
        r"\b(ego|generate|create|simulat\w*|build|make|run|execute|where|with"
        r"|featuring|containing|includ\w*)\b",
        head, maxsplit=1, flags=_re.I)[0]
    head = head.strip(" .,;:\t")
    words = head.split()
    while words and words[-1].lower() in _SCENARIO_TAIL_WORDS:
        words = words[:-1]
    head = " ".join(words).strip()
    if not head:
        return city or ""
    if city and city.lower() not in head.lower():
        return f"{head}, {city}"
    return head


def build_generate_tab(
    handler,
    tabs,
    main_app_tab_id: int,
    *,
    history,
    chatbot,
    image_output,
    image_list_state,
    index_state,
    image_caption,
    run_frenetix,
    user_input,
) -> None:
    """Define the Generate tab body. Mutates state by writing to the
    passed-in Main App components when the user finalises a scenario.

    `handler` is the shared ChatHandler so the Generate tab can call
    `handler.engine.load_generated_scenario(name)` and bump `handler.step`
    so the next chat message routes to step-7 (modify/planner) handling.
    """

    gr.Markdown(
        "### Generate a CommonRoad scenario from scratch\n"
        "**Easiest path**: type a free-text description below and click "
        "**Parse with LLM** — the rest of the form auto-fills. You can "
        "still tweak any widget afterwards.\n\n"
        "**Manual path**: skip the description box, draw a bbox on the "
        "map, set options yourself, click **Run Stage 1**, pick ego + "
        "goal, click **Save & open in Main App**."
    )

    # ── NL → intent (optional, drives all the widgets below) ───────────
    nl_query_tb = gr.Textbox(
        label="Describe your scenario (optional)",
        placeholder=("e.g. 'Munich intersection scenario, ego is the "
                     "truck, dense traffic, simulate 30 seconds'"),
        lines=2,
    )
    with gr.Row():
        parse_btn = gr.Button("Parse with LLM",
                              variant="secondary", size="sm")
        parse_log = gr.Textbox(
            label="Parsed intent",
            lines=4, max_lines=12, interactive=False,
            placeholder="LLM output appears here after parsing.",
        )
    # Carries the ego/goal portion of the parsed intent across to the
    # Stage-2 ego-resolution step. Each value in this state is the full
    # intent dict (or None if not parsed yet).
    intent_state = gr.State(value=None)

    gr.HTML(value=IFRAME_HTML)

    bbox_tb = gr.Textbox(
        label="Selected bbox (south,west,north,east)",
        elem_id="bbox_input_generate",
        interactive=True,
        placeholder="draw a rectangle, or Parse a description to fly the map here",
    )
    # NOTE: the map ⇄ this textbox sync (both directions) is handled inside
    # static/map.html by polling/writing this element via window.parent — see
    # the comment on IFRAME_HTML above for why it can't be wired from Python.

    with gr.Row():
        classes = gr.CheckboxGroup(
            choices=list(CLASS_GROUPS.keys()),
            value=["roads"],
            label="Road classes",
        )
        density = gr.Dropdown(
            choices=["low", "medium", "high"],
            value="medium",
            label="SUMO traffic density",
        )
        sim_seconds = gr.Slider(
            minimum=5, maximum=120, value=20, step=5,
            label="SUMO simulation duration (s)",
        )

    with gr.Row():
        vehicle_types = gr.CheckboxGroup(
            choices=["car", "truck", "trailer", "delivery",
                     "bus", "coach", "taxi", "emergency",
                     "bicycle", "motorcycle", "moped"],
            value=["car"],
            label="SUMO vehicle mix",
        )
        record_cr_gif = gr.Checkbox(value=True, label="Record CR sim as GIF")

    gr.Markdown("### Stage 1 — fetch, convert, simulate")
    run_btn = gr.Button("Run Stage 1", variant="primary")
    out_log = gr.Textbox(
        label="Stage 1 backend log (live)",
        lines=18, max_lines=40, interactive=False, autoscroll=True,
        placeholder=("Backend log streams here while the OSM pipeline runs "
                     "in the cr37 conda env. Final summary appears at the "
                     "bottom when it's done."),
    )

    with gr.Row(equal_height=True):
        osm_img = gr.Image(label="OSM map", interactive=False,
                           type="filepath", height=300)
        sumo_img = gr.Image(label="SUMO BEV", interactive=False,
                            type="filepath", height=300)
        cr_img = gr.Image(label="CommonRoad BEV", interactive=False,
                          type="filepath", height=300)
        cr_sim_img = gr.Image(label="CommonRoad sim BEV", interactive=False,
                              type="filepath", height=300)
    # CommonRoad sim GIF — shown in a self-contained zoom/pan viewer iframe
    # (static/gif_viewer.html) so you can zoom in/out and drag without
    # maximising the window. gr.Image was replaced because gr.HTML can't run
    # injected <script>, so in-place zoom buttons couldn't fire; the iframe is a
    # real document whose JS runs, and it polls the carrier span below for the
    # GIF URL (same cross-frame pattern as the map). See static/gif_viewer.html.
    gr.Markdown("### CommonRoad sim GIF (animated) — scroll to zoom, drag to "
                "pan, or use the +/− buttons")
    gr.HTML(value=(
        f'<iframe src="/{GIF_VIEWER_URL}" '
        f'style="width:100%; height:460px; border:1px solid #ccc; '
        f'border-radius:6px;"></iframe>'
    ))
    # Carrier the viewer polls for the current GIF URL (set by _run_stage1).
    cr_sim_gif = gr.HTML(value=_gif_carrier())

    # Carry the produced sim XML path forward into Stage 2.
    sim_xml_state = gr.State(value=None)
    scenario_name_state = gr.State(value=None)

    def _run_stage1(bbox_text, classes_sel, density_val, sim_seconds_val,
                    vehicle_types_val, record_cr_gif_val):
        bbox = _parse_bbox(bbox_text or "")
        # 6 trailing slots: osm_img, sumo_img, cr_img, cr_sim_img,
        # cr_sim_gif (carrier HTML), sim_xml_state.
        empty_tail = (None, None, None, None, _gif_carrier(), None)
        if bbox is None:
            yield (("draw a rectangle on the map first.",) + empty_tail)
            return

        log_buf: list[str] = []
        result_paths: dict[str, str] = {}
        for item in osm_sub.run_stage1(
            bbox=bbox,
            classes=classes_sel or ["roads"],
            density=density_val or "medium",
            sim_seconds=float(sim_seconds_val or 20),
            vehicle_types=vehicle_types_val or ["car"],
            simulate_cr=True,
            record_cr_gif=bool(record_cr_gif_val),
        ):
            if isinstance(item, tuple) and item and item[0] == "RESULT":
                result_paths = item[1]
                continue
            log_buf.append(str(item))
            yield ("\n".join(log_buf), None, None, None, None,
                   _gif_carrier(), None)

        osm_png = result_paths.get("OSM BEV")
        sumo_png = (result_paths.get("SUMO BEV (cr-aligned)")
                    or result_paths.get("SUMO BEV"))
        cr_png = result_paths.get("CR BEV")
        cr_sim_png = result_paths.get("CR sim BEV")
        cr_gif = result_paths.get("CR sim GIF")
        sim_xml_path = result_paths.get("CR sim scenario")

        # Generate the scenario name now so Stage 2's synthesize can save
        # the PP XML directly into Scenarios/<name>/Original/.
        scen_name = _generate_scenario_name(bbox) if sim_xml_path else None

        summary = ("\n".join(log_buf)
                   + "\n\n=== Stage 1 SUMMARY ===\n"
                   + ("\n".join(f"{k} → {v}"
                                for k, v in result_paths.items())
                      or "(no result paths emitted)"))
        if not sim_xml_path:
            summary += "\n\nStage 2 disabled — no CR sim XML produced."
        else:
            summary += f"\n\nScenario name: {scen_name}"

        yield (summary, osm_png, sumo_png, cr_png, cr_sim_png,
               _gif_carrier(cr_gif), sim_xml_path)

    run_evt = run_btn.click(
        fn=_run_stage1,
        inputs=[bbox_tb, classes, density, sim_seconds,
                vehicle_types, record_cr_gif],
        outputs=[out_log, osm_img, sumo_img, cr_img, cr_sim_img,
                 cr_sim_gif, sim_xml_state],
    )

    # ── Parse-with-LLM click handler ───────────────────────────────────
    def _parse_intent(nl_query):
        # outputs: parse_log, intent_state,
        #          bbox_tb, classes, density, sim_seconds, vehicle_types,
        #          goal_lanelet_tb, goal_position_dd, goal_offset_sl,
        #          goal_length_sl, goal_width_sl
        def _bail(msg):
            return (msg, None,
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update())

        if not (nl_query and nl_query.strip()):
            return _bail("type a description first.")

        api_key = (os.getenv("QWEN_API_KEY") or "").strip()
        base_url = (os.getenv("QWEN_BASE_URL") or "").strip()
        model = (os.getenv("QWEN_MODEL") or "qwen3.6-plus").strip()
        if not (api_key and base_url):
            return _bail("QWEN_API_KEY / QWEN_BASE_URL not set in .env — "
                         "see tests/test_qwen.py.")
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as e:
            return _bail(f"langchain_openai not installed: {e}")

        system_prompt = build_prompt("osm_intent", techniques=["cp", "icl"])
        llm = ChatOpenAI(model=model, api_key=api_key, base_url=base_url,
                         temperature=0.0, max_retries=1, timeout=120,
                         extra_body={"enable_thinking": False})
        try:
            resp = llm.invoke([SystemMessage(content=system_prompt),
                               HumanMessage(content=nl_query)])
            raw = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            return _bail(f"LLM call failed: {type(e).__name__}: {e}")

        import re as _re
        cleaned = _re.sub(r"```(?:json)?\s*", "", raw)
        cleaned = _re.sub(r"```\s*$", "", cleaned, flags=_re.MULTILINE)
        try:
            from regex import regex as _rx
            m = _rx.search(r"\{(?:[^{}]|(?R))*\}", cleaned)
            if m is None:
                raise ValueError("no JSON object found")
            intent = json.loads(m.group(0))
        except Exception as e:
            return _bail(f"could not parse JSON: {e}\n\nRaw response:\n{raw}")

        # Required-field check (city and scenario_type per the design).
        loc = intent.get("location") or {}
        bbox = loc.get("bbox")
        city = loc.get("city")
        country_code = loc.get("country_code")
        if not city and not bbox:
            return _bail("intent missing both city and bbox — please name "
                         "a city or include lat/lon corners in your "
                         "description.\n\nParsed intent:\n"
                         + json.dumps(intent, indent=2))

        scen_type = intent.get("scenario_type") or "any"

        # Resolve bbox if only a city was given.
        bbox_str = ""
        geocoded_place = None
        if bbox and isinstance(bbox, list) and len(bbox) == 4:
            bbox_str = ",".join(f"{float(x):.6f}" for x in bbox)
        elif city:
            # Prefer a specific place the user named (landmark/district) over the
            # bare city centroid — the extractor only keeps the city, so recover
            # the place phrase from the raw query and geocode that, falling back
            # to the bare city when the place doesn't resolve.
            geo_q = _geocode_query(nl_query, city)
            resolved = geocode_city(geo_q, country_code)
            if resolved is None and geo_q.strip().lower() != (city or "").strip().lower():
                geo_q = city
                resolved = geocode_city(city, country_code)
            if resolved is None:
                return _bail(f"could not geocode {geo_q!r} / "
                             f"country={country_code!r}.\n\nParsed intent:\n"
                             + json.dumps(intent, indent=2))
            geocoded_place = geo_q
            bbox_str = ",".join(f"{x:.6f}" for x in resolved)

        # Map intent.road_classes onto the CheckboxGroup choices, falling
        # back gracefully if the LLM emitted something off-list.
        rc = intent.get("road_classes") or ["roads"]
        rc = [c for c in rc if c in CLASS_GROUPS] or ["roads"]

        sim = intent.get("sim") or {}
        density_val = sim.get("density") or "medium"
        if density_val not in ("low", "medium", "high"):
            density_val = "medium"
        sim_s = float(sim.get("duration_s") or 20.0)
        vt = sim.get("vehicle_types") or ["car"]

        goal = intent.get("goal") or {}
        glanelet = goal.get("lanelet_id")
        glanelet_str = str(glanelet) if isinstance(glanelet, int) else ""
        gpos = goal.get("position_keyword") or "middle"
        if gpos not in ("start", "middle", "end"):
            gpos = "middle"
        goff = int(goal.get("offset_steps") or 50)
        glen = float(goal.get("rectangle_length_m") or 6.0)
        gwid = float(goal.get("rectangle_width_m") or 2.0)

        # The extractor downgrades a named non-car ego (incl. taxi) to "first"
        # for the car-only planner, dropping the named type. Recover it from the
        # query so Stage 2 can pre-select the matching obstacle.
        pref_ego = _preferred_ego_type_from_query(nl_query)
        intent["_preferred_ego_type"] = pref_ego

        summary = (
            f"Parsed (model: {model})\n"
            f"  location: city={city!r}, country={country_code!r}, "
            f"scenario_type={scen_type!r}\n"
            + (f"  geocoded place: {geocoded_place!r}\n"
               if geocoded_place
               and geocoded_place.strip().lower() != (city or "").strip().lower()
               else "")
            + f"  road_classes={rc}, density={density_val}, "
            f"sim_seconds={sim_s}, vehicle_types={vt}\n"
            f"  ego: {intent.get('ego')}"
            + (f"  (you named '{pref_ego}' → pre-selected in Stage 2)"
               if pref_ego and pref_ego != "car" else "")
            + "\n"
            f"  goal: strategy={goal.get('strategy')}, "
            f"offset_steps={goff}, lanelet={glanelet_str or '∅'}, "
            f"pos={gpos}, rect={glen}×{gwid} m\n"
            f"\nbbox auto-filled: {bbox_str}"
        )

        return (summary, intent,
                gr.update(value=bbox_str),
                gr.update(value=rc),
                gr.update(value=density_val),
                gr.update(value=sim_s),
                gr.update(value=vt),
                gr.update(value=glanelet_str),
                gr.update(value=gpos),
                gr.update(value=goff),
                gr.update(value=glen),
                gr.update(value=gwid))

    # ── Stage 2 — ego + goal ────────────────────────────────────────────
    gr.Markdown("### Stage 2 — pick ego vehicle + goal")
    with gr.Row():
        ego_dd = gr.Dropdown(
            choices=[], value=None, label="Ego vehicle ID",
            interactive=True, allow_custom_value=True,
        )
        refresh_ego_btn = gr.Button("Refresh ego list", size="sm")
    with gr.Row():
        goal_lanelet_tb = gr.Textbox(
            label="Goal lanelet ID (optional)",
            placeholder=("blank → use ego's trajectory at offset; accepts "
                         "both the small CR id (e.g. 5) and the post-sim "
                         "shifted form (e.g. 1000005)"),
        )
        goal_position_dd = gr.Dropdown(
            choices=["start", "middle", "end"], value="middle",
            label="Point on lanelet",
        )
        goal_offset_sl = gr.Slider(
            minimum=10, maximum=200, value=50, step=10,
            label="Goal offset (sim steps) — used when no lanelet given",
        )
    with gr.Row():
        goal_length_sl = gr.Slider(
            minimum=2, maximum=30, value=6, step=0.5,
            label="Goal rectangle length (m)",
        )
        goal_width_sl = gr.Slider(
            minimum=1, maximum=8, value=2, step=0.5,
            label="Goal rectangle width (m)",
        )

    save_btn = gr.Button("Save & open in Main App", variant="primary")
    stage2_log = gr.Textbox(
        label="Stage 2 backend log",
        lines=10, max_lines=20, interactive=False, autoscroll=True,
    )

    def _refresh_ego(sim_xml_val, intent_val):
        """Populate the ego dropdown from list_egos, labelling each ID with its
        vehicle type (e.g. "7 — taxi") so the user can tell them apart. Pre-select
        per the parsed intent: an explicit ego strategy ("by_type"/"by_index"),
        or — since the extractor downgrades a named non-car ego (e.g. taxi) to
        "first" for the car-only planner — the vehicle type the user named in
        their query (intent["_preferred_ego_type"]). Falls back to the first id."""
        if not sim_xml_val or not Path(sim_xml_val).exists():
            return gr.update(choices=[], value=None)
        try:
            ids = osm_sub.list_egos(Path(sim_xml_val), limit=30)
        except Exception as e:
            print(f"list_egos failed: {type(e).__name__}: {e}")
            return gr.update(choices=[], value=None)
        if not ids:
            return gr.update(choices=[], value=None)

        # id -> vehicle type (e.g. "car", "taxi"); empty on any failure.
        type_map: dict[int, str] = {}
        try:
            from intent_lib import _ego_type_map  # type: ignore
            type_map = _ego_type_map(Path(sim_xml_val))
        except Exception as e:
            print(f"_ego_type_map failed: {type(e).__name__}: {e}")

        def _ty(oid) -> str:
            return (type_map.get(int(oid), "") or "").lower()

        selected = ids[0]
        ego = (intent_val or {}).get("ego") or {}
        strat = (ego.get("strategy") or "first").lower()
        # Vehicle type wanted for the ego: explicit by_type, else the name we
        # recovered from the query before the car-only downgrade.
        want = (ego.get("vehicle_type") or "").lower() if strat == "by_type" else ""
        if not want:
            want = ((intent_val or {}).get("_preferred_ego_type") or "").lower()

        if strat == "by_index":
            idx = ego.get("index")
            if isinstance(idx, int) and 0 <= idx < len(ids):
                selected = ids[idx]
        elif want:
            for oid in ids:
                if _ty(oid) == want:
                    selected = oid
                    break

        # "<id> — <type>" shown to the user; the bare "<id>" is the value.
        choices = [(f"{oid} — {type_map.get(int(oid), 'unknown')}", str(oid))
                   for oid in ids]
        return gr.update(choices=choices, value=str(selected))

    refresh_ego_btn.click(
        fn=_refresh_ego, inputs=[sim_xml_state, intent_state],
        outputs=[ego_dd])

    # Auto-populate the Stage-2 ego dropdown (with vehicle types) once Stage 1
    # finishes, so the user doesn't have to click "Refresh ego list".
    run_evt.then(fn=_refresh_ego, inputs=[sim_xml_state, intent_state],
                 outputs=[ego_dd])

    # Wire the Parse-with-LLM button (handler defined above) now that
    # all its downstream widgets are in scope.
    parse_btn.click(
        fn=_parse_intent,
        inputs=[nl_query_tb],
        outputs=[parse_log, intent_state,
                 bbox_tb, classes, density, sim_seconds, vehicle_types,
                 goal_lanelet_tb, goal_position_dd, goal_offset_sl,
                 goal_length_sl, goal_width_sl],
    )

    def _save_and_open(sim_xml_val, bbox_text, ego_val,
                       goal_lanelet_val, goal_pos, goal_off,
                       goal_len, goal_wid,
                       intent_state_val,
                       history_val, image_list_val, index_val):
        # outputs (10): stage2_log, tabs, history, chatbot,
        #               image_output, image_list_state, index_state,
        #               image_caption, run_frenetix, user_input
        def _bail(msg):
            return (msg, gr.update(), history_val, history_val,
                    gr.update(), image_list_val, index_val,
                    gr.update(), gr.update(), gr.update())

        if not sim_xml_val or not Path(sim_xml_val).exists():
            return _bail("Run Stage 1 first — no CR sim XML yet.")
        # Dropdown values are bare ids; be defensive in case a label like
        # "7 — taxi" arrives (e.g. via a typed custom value).
        ego_digits = "".join(ch for ch in str(ego_val or "") if ch.isdigit())
        if not ego_digits:
            return _bail(f"pick an ego first (got {ego_val!r}).")
        ego_id = int(ego_digits)

        lanelet_id = None
        if goal_lanelet_val and str(goal_lanelet_val).strip():
            try:
                lanelet_id = int(str(goal_lanelet_val).strip())
            except ValueError:
                return _bail(f"bad lanelet id: {goal_lanelet_val!r}")

        bbox = _parse_bbox(bbox_text or "")
        if bbox is None:
            return _bail("internal: bbox lost — re-run Stage 1.")
        name = _generate_scenario_name(bbox)
        target_dir = (Path("Scenarios") / name / "Original").resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        target_xml = target_dir / f"{name}.xml"

        log_buf: list[str] = [f"Synthesising planning problem for ego={ego_id}…"]
        result_paths: dict[str, str] = {}
        for item in osm_sub.run_synthesize(
            sim_xml=Path(sim_xml_val),
            ego_id=ego_id,
            goal_lanelet_id=lanelet_id,
            goal_position=goal_pos or "middle",
            goal_offset_steps=int(goal_off),
            goal_length_m=float(goal_len),
            goal_width_m=float(goal_wid),
            out_path=target_xml,
        ):
            if isinstance(item, tuple) and item and item[0] == "RESULT":
                result_paths = item[1]
                continue
            log_buf.append(str(item))

        pp_xml = result_paths.get("PP XML")
        if not pp_xml or not Path(pp_xml).exists():
            log_buf.append("Synthesis did not produce a PP XML — aborting.")
            return _bail("\n".join(log_buf))

        # If synthesize wrote to a different out_path (e.g. fallback), copy
        # it into the canonical location so select_initial_gif finds it.
        pp_xml_path = Path(pp_xml)
        if pp_xml_path.resolve() != target_xml.resolve():
            shutil.copy2(pp_xml_path, target_xml)
            log_buf.append(f"Copied PP XML → {target_xml}")

        # Replace crdesigner's hardcoded "ZAM_MUC-1" benchmark_id with
        # one derived from the parsed intent + this scenario's name, so
        # the generated XML is properly attributable to the user's
        # query location.
        intent_for_bid = intent_state_val or {
            "location": {
                "city": None,
                "country_code": "ZAM",
                "bbox": list(bbox),
            }
        }
        new_bid = rewrite_benchmark_id(target_xml, intent_for_bid, name)
        if new_bid:
            log_buf.append(f"benchmark_id → {new_bid}")

        log_buf.append(f"Rendering scenario GIF (Frenetix without ego)…")
        gifs = handler.engine.load_generated_scenario(name)
        if not gifs:
            log_buf.append("GIF rendering failed — scenario saved but no GIF.")
            return _bail("\n".join(log_buf))

        log_buf.append(f"Loaded {name} — switching to Main App tab.")
        handler.step = 8  # next chat() call will route as step-7 modify/planner

        new_history = list(history_val or [])
        new_history.append({
            "role": "assistant",
            "content": (
                f"Scenario `{name}` generated from OSM bbox and loaded. "
                f"Type a modification request (e.g. 'add a slower car ahead') "
                f"or click **Run with Motion Planner** to plan."
            ),
        })

        return (
            "\n".join(log_buf),
            gr.update(selected=main_app_tab_id),  # switch tabs
            new_history,                           # history state
            new_history,                           # chatbot widget
            gr.update(value=gifs[0], visible=True),  # image_output
            gifs,                                  # image_list_state
            0,                                     # index_state
            gr.update(value=f"**{name}** — generated scenario",
                      visible=True),               # image_caption
            gr.update(visible=True,
                      value="Run with Motion Planner"),  # run_frenetix
            gr.update(visible=True),               # user_input
        )

    save_btn.click(
        fn=_save_and_open,
        inputs=[sim_xml_state, bbox_tb, ego_dd, goal_lanelet_tb,
                goal_position_dd, goal_offset_sl, goal_length_sl,
                goal_width_sl,
                intent_state,
                history, image_list_state, index_state],
        outputs=[stage2_log, tabs, history, chatbot, image_output,
                 image_list_state, index_state, image_caption,
                 run_frenetix, user_input],
    )
