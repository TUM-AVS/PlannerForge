<!-- The guided tour that used to live in README.md; moved here unchanged so the
     README can stay short. -->
# Using PlannerForge

Launching `interface.py` opens a Gradio chat in your browser. You drive the
whole pipeline by typing what you want; the module router figures out which
stage you mean and runs it. A typical session:

**1. Select a scenario** — describe the situation you want to test.
> *"Find an urban intersection scenario in Germany with at least five vehicles."*

PlannerForge retrieves the best match from the 582-scenario corpus and shows
its bird's-eye render. You can also ask for one by name
(*"load DEU_Muc-2_1_T-1"*) or by similarity (*"something like that but busier"*).

**2. Modify it** — reshape the selected scenario. Four edit families:
> *"Make the car ahead of the ego drive aggressively."*  (Behaviour)
> *"Reroute the truck so it turns left across our path."*  (Trajectory)
> *"Add two more vehicles near the ego."*  (Participant)
> *"Move the goal region further down the road."*  (Goal)

Edits are applied through the CommonRoad–SUMO interface and re-simulated so
the result stays physically valid.

**3. Generate from scratch** — synthesize a brand-new scenario.
> *"Generate a scenario in Munich with moderate traffic and a left-turn goal."*

This runs the OSM → SUMO → CommonRoad pipeline (`osm_pipeline/`); you can also
draw a bounding box on the map in the *Generate* tab.

**4. Test a planner** — run a motion planner on the current scenario.
> *"Run the Frenetix planner on this scenario."*
> *"Batch-test all selected scenarios with MP-RBFN."*

**5. Tune the planner** — change cost-function behaviour in natural language.
> *"The ego keeps colliding — make it keep a larger safety distance."*
> *"Reset the planner to the default cost preset."*

PlannerForge emits schema-conformant overrides to the planner's `cost.yaml`,
re-runs, and reports whether the outcome improved.

**6. Analyse** — summarize and compare runs.
> *"Summarize the batch results and tell me which config was safest."*

## Where things land on disk

| Path | Written by |
|---|---|
| `Scenarios/<name>/Original/` | a scenario selected from the corpus: the CommonRoad XML, its SUMO files, the preview PNG and the GIF |
| `Scenarios/<name>/Modified/<name>_<suffix>/` | one directory per modification. The suffix encodes the edit types applied (`T`, `B`, `P`, `G`) plus a random tag |
| `data/osm/` | the OSM → SUMO → CommonRoad generation pipeline's intermediates and its BEV renders |
| `<planner>/logs/<scenario>/` | planner run output: the animation, cost logs, trajectories |
| `<planner>/logs/score_overview.csv` | the batch outcome table the analysis charts read |
| `chroma/` | the ChromaDB index built by `add_scenarios.py` |
| `CollectedScenarios/` | the 582 scenario XMLs, downloaded from Hugging Face by `add_scenarios.py` |

`CollectedScenarios/`, `Scenarios/`, `data/`, `chroma/` and the planner `logs/`
directories are all downloaded or regenerable, and none of them are tracked by git.

## Scenario pictures

Every picture the app produces — the corpus preview PNG, the selected and
modified scenario GIFs, and the renders of a generated scenario — is drawn
through [`cr_draw_style.py`](../cr_draw_style.py), so they share one look:
lanelet IDs at a fixed small size, vehicle IDs, traffic signs and lights,
grey static obstacles, and the same figure geometry and frame pacing. If you
add a renderer, import that module rather than setting draw parameters
locally.
