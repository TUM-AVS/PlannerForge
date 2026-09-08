# Integrating your own motion planner

PlannerForge drives a motion planner as an **external subprocess**. It does not
import your code, so your planner keeps its own environment, its own Python
version and its own dependencies. The whole contract is a launch command, a
handful of output files, and one optional YAML.

Two planners ship as reference integrations:

| | Frenetix | MP-RBFN |
|---|---|---|
| launched via | venv `activate` script | interpreter path |
| cost weights | yes (`cost.yaml`) | no |
| cost logs | yes | no |
| extra env | none | two variables |

Frenetix is the fuller example (cost tuning included); MP-RBFN is the minimal
one. Read `process_engine.py` around `self.planners` alongside this document.

## 1. Register the planner

Add an entry to `self.planners` in `process_engine.py`. The dictionary key is
the display name shown in the UI's planner selector.

```python
"MY-PLANNER": {
    # Either a path ending in "activate" (sourced in bash before `python`),
    # or a path to a Python interpreter (invoked directly). The suffix decides
    # which — see the note below.
    "venv_path": str(Path("My-Planner/venv/bin/activate").resolve()),

    # Runner script. Must accept the two CLI forms in section 2.
    "script_path": str(Path("My-Planner/run_my_planner.py").resolve()),

    # Where results are written. Must be a directory; one subdirectory per
    # scenario is created underneath it (section 3).
    "log_dir": str(Path("My-Planner/logs").resolve()),

    # "y" if you write per-trajectory cost logs (section 4), else "n".
    "cost_support": "n",

    # Path to a tunable cost-weight YAML (section 5), or "" if you have none.
    "weights_file": "",

    # Optional. Extra environment variables for this planner's subprocesses
    # only; overlaid on os.environ.
    "env": {},
}
```

**The `venv_path` dual mode is decided by the string, not by a flag.** If the
value ends with `activate`, PlannerForge runs
`bash -c "source <venv_path> && python <script_path> ..."`. Otherwise it runs
`bash -c "<venv_path> <script_path> ..."` and treats the value as an
interpreter. A conda environment therefore needs the interpreter form —
`/path/to/envs/myenv/bin/python`, not the `activate` script.

To let users pick the planner by typing (*"run my planner on this scenario"*)
rather than only through the selector, add a keyword to `_planner_from_text` in
`process_engine.py`, next to the existing `rbfn` and `frenetix` matches.

## 2. The command line

Your script is called in exactly two ways.

**Single scenario** — one CommonRoad XML, absolute path:

```bash
<python> run_my_planner.py --input-file /abs/path/DEU_Weimar-71_1_T-4.xml
```

**Batch** — a CSV listing scenarios, absolute path, plus `--batch`:

```bash
<python> run_my_planner.py --input-file /abs/path/scenario_batch_list.csv --batch
```

The CSV is **one scenario name per line, no header, no `.xml` suffix**:

```
AUT_Haag-12_1_T-4
BEL_Aarschot-3_1_T-1
BEL_Brussels-29_1_T-1
```

Resolving those names to files is your script's job — the reference planners
look them up in the scenario corpus they were configured with.

Requirements:

- **Exit 0 on success, non-zero on failure.** A non-zero exit is surfaced to
  the user as a failed run.
- **Write progress to stdout/stderr.** Both are streamed line-by-line into the
  UI's console panel while the run is in flight, so print something.
- Accept unknown extra arguments gracefully if you can; PlannerForge currently
  passes only the two flags above.

## 3. What you must write

### The animation

```
<log_dir>/<scenario_name>/<anything>.gif
```

PlannerForge finds it with `glob("*.gif")` in that directory, so the file name
is up to you — the reference planners use `<scenario_name>.gif`. This is the
GIF the user sees after a run.

If no GIF is present after a successful exit, PlannerForge reports that the
planner found no feasible trajectory. Write a GIF whenever you produced any
frames, even for a failed scenario.

To make your animation look like the rest of the app (same lanelet-ID font,
vehicle-ID font, traffic signs and colours), render it through the shared
style module in the repository root:

```python
from cr_draw_style import (FIGSIZE, DPI, FRAME_DURATION_MS, BBOX_INCHES,
                           PAD_INCHES, CustomRenderer, apply_rcparams,
                           build_draw_params, install_icons)

apply_rcparams()
install_icons()
rnd = CustomRenderer(figsize=FIGSIZE)
scenario.draw(rnd, draw_params=build_draw_params(timestep))
rnd.render()
```

This is optional — any GIF is accepted — but it is what keeps generated,
selected and modified scenarios visually identical.

### The score table

```
<log_dir>/score_overview.csv
```

**Semicolon-separated**, one row per scenario (or per agent, if you simulate
several), with this header:

```
scenario;agent;timestep;status;message;result;collision_type;colliding_object_id
```

| Column | Meaning | Required |
|---|---|---|
| `scenario` | scenario name. The `.xml` suffix is optional — nothing downstream depends on it | yes |
| `agent` | agent identifier. Free-form: a numeric ego id or a fixed string both work | yes |
| `timestep` | timestep the run ended on. Used for the run-length distribution plot | yes |
| `status` | outcome. Write `success` for a solved scenario; anything else counts as a failure | see below |
| `message` | failure reason as free text. Drives the failure-breakdown chart | on failure |
| `result` | outcome, or a planner-specific score | see below |
| `collision_type` | e.g. `front_collision`. Leave empty if not applicable | no |
| `colliding_object_id` | obstacle id involved in a collision, empty otherwise | no |

**Outcome rule.** A scenario counts as solved when **either** `result` **or**
`status` reads `success`/`successful` (case-insensitive). Write the verdict in
one of the two; the other may carry a planner-specific value. Frenetix writes
`result=Success|Failed` with a numeric `status` code; MP-RBFN writes
`status=success|fail` and puts a score in `result`. Both are read correctly.

Known `message` values get distinct colours in the charts, so prefer them where
they fit: `collision`, `no valid or feasible trajectory found`,
`time limit reached`, `timeout reached`, `goal reached faster`,
`goal reached out of time`.

Append to this file across a batch — the analysis plots read the whole table.

## 4. Cost logs (only if `cost_support: "y"`)

```
<log_dir>/<scenario_name>/cost/cost_<timestep>.json
```

One file per planning step, each mapping trajectory ids to their cost
breakdown:

```json
{
  "trajectory_id:141": {
    "cost": 7.74,
    "costMap": {
      "lateral_jerk": [0.15, 0.03],
      "prediction": [4.02, 2.01],
      "distance_to_reference_path": [0.0, 0.0]
    }
  }
}
```

Each `costMap` entry is `[raw_value, weighted_value]`. PlannerForge reads the
minimum `cost` per step and sums it across the scenario, and reports that
number to the LLM when it analyses a run. Setting `cost_support: "n"` skips
this entirely and reports a cost of `-1.0`.

## 5. Cost tuning (only if `weights_file` is set)

Point `weights_file` at a YAML file of tunable weights. When the user asks for
a behavioural change in natural language (*"keep a larger safety distance"*),
the LLM **rewrites the whole file**, preserving key names, ordering, comments
and indentation, and your planner simply reads it on the next run.

The Frenetix file has two top-level blocks — 13 weights under `cost_weights`
and 3 under `external_cost_weights`:

```yaml
cost_weights:
    acceleration: 0.3
    jerk: 0.3
    ...
external_cost_weights:
    occ_pm: 0.0
```

Your schema does not have to match; the prompt is generated from the file's own
contents. Two things to know:

- The LLM is instructed to refuse unknown parameter names and list the valid
  ones, so keep the names self-describing.
- The batch charts label the active configuration by comparing the file against
  the presets in `utils/data_utils.py::PRESETS` using **exact float equality**.
  A file that matches no preset is labelled `Custom`. To get your own named
  presets in the charts, add them to that dictionary.

## 6. Minimal working stub

```python
#!/usr/bin/env python3
"""Minimal PlannerForge-compatible planner runner."""
import argparse, csv
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
CORPUS = Path(__file__).resolve().parent.parent / "CollectedScenarios"
HEADER = ["scenario", "agent", "timestep", "status", "message",
          "result", "collision_type", "colliding_object_id"]


def run_one(xml_path: Path) -> dict:
    """Plan on one scenario. Replace with your planner."""
    out_dir = LOG_DIR / xml_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    # ... plan, simulate, and write out_dir/<stem>.gif ...
    return {"scenario": xml_path.stem, "agent": "my-planner", "timestep": 42,
            "status": "success", "message": "", "result": "Success",
            "collision_type": "", "colliding_object_id": ""}


def write_scores(rows: list[dict]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "score_overview.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, delimiter=";")
        if new:
            w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True)
    ap.add_argument("--batch", action="store_true")
    args = ap.parse_args()

    if args.batch:
        names = [ln.strip() for ln in Path(args.input_file).read_text().splitlines()
                 if ln.strip()]
        paths = [CORPUS / f"{n.removesuffix('.xml')}.xml" for n in names]
    else:
        paths = [Path(args.input_file)]

    rows = []
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {p.stem}", flush=True)   # streamed to the UI
        rows.append(run_one(p))
    write_scores(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 7. Checklist

- [ ] entry added to `self.planners`, `venv_path` in the right one of the two forms
- [ ] `--input-file <xml>` runs one scenario
- [ ] `--input-file <csv> --batch` runs a list
- [ ] exit code 0 on success, non-zero on failure
- [ ] progress printed to stdout
- [ ] `<log_dir>/<scenario>/*.gif` written
- [ ] `<log_dir>/score_overview.csv` written with the 8 columns and `success` in
      `status` or `result`
- [ ] cost JSONs written, or `cost_support: "n"`
- [ ] `weights_file` set, or `""`
- [ ] keyword added to `_planner_from_text` (optional)
