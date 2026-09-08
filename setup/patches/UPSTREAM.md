# Vendored planner upstreams and local patches

This release integrates two LGPL-3.0 motion planners as git submodules plus
local patches, instead of vendoring full copies.

The patches reproduce, byte for byte, the git-tracked WORKTREE content of the
planner directories in the source repo `PlannerForge` (commit `721394a`, branch
`main`, snapshot taken 2026-07-17). Untracked files (logs, venvs, outputs)
are not part of the patches.

## Frenetix-Motion-Planner

- Upstream: https://github.com/TUM-AVS/Frenetix-Motion-Planner
- Base commit: `fc6e7708ba384f991c2a0a07b7300ab52276c70f` (2024-08-29, upstream HEAD at snapshot time)
- Patch: `frenetix.patch` (144 KB; 25 files, +3392 / -239 lines)
- Apply: `git -C Frenetix-Motion-Planner apply -p1 --whitespace=nowarn <this-dir>/frenetix.patch`

Match quality: base commit chosen by scoring every upstream commit against the
vendored snapshot via blob hashes. At the base, excluding the known locally
modified files, only 5 files differ in content (`.gitignore`,
`configurations/simulation/prediction.yaml`,
`configurations/simulation/simulation.yaml`,
`cr_scenario_handler/utils/configuration_builder.py`,
`frenetix_motion_planner/cost_functions/partial_cost_functions.py` - all
additional local edits), 1 upstream file was deleted locally (`main.py`) and 7
files are local additions. The next-best commit differs in 30 paths, so the
base is unambiguous.

Files touched by `frenetix.patch`:

```
M  .gitignore
A  batch_presets/batch_100_scenarios.csv
A  batch_presets/batch_200_scenarios.csv
A  batch_presets/batch_300_scenarios.csv
A  batch_presets/batch_400_scenarios.csv
A  batch_presets/batch_500_scenarios.csv
M  configurations/frenetix_motion_planner/cost.yaml
A  configurations/frenetix_motion_planner/cost_old.yaml
M  configurations/simulation/prediction.yaml
M  configurations/simulation/simulation.yaml
M  configurations/simulation/visualization.yaml
M  cr_scenario_handler/simulation/agent.py
M  cr_scenario_handler/simulation/agent_batch.py
M  cr_scenario_handler/simulation/simulation.py
M  cr_scenario_handler/utils/configuration_builder.py
M  cr_scenario_handler/utils/general.py
M  cr_scenario_handler/utils/goalcheck.py
M  cr_scenario_handler/utils/visualization.py
M  frenetix_motion_planner/cost_functions/partial_cost_functions.py
D  main.py
A  main_batch.py
M  main_multiagent.py
A  main_without_ego.py
A  requirements.txt
A  scenario_batch_list.csv
```

Note: `main_batch.py`, `main_without_ego.py` and `requirements.txt` do not
exist upstream at the base commit; they are shipped as new files by the patch.

## RBFN-Motion-Primitives (MP-RBFN)

- Upstream: https://github.com/TUM-AVS/RBFN-Motion-Primitives
  (the vendored README still cites the old name `TUM-AVS/MP-RBFN`; the
  repository was renamed upstream, the old URL now returns 404)
- Base commit: `223c11ddcb3436a28666d86e7b7f4a23833d35d6` (2025-05-05, commit "ITSC25")
- Patch: `rbfn.patch` (27 MB; 39 files, +547101 / -86402 lines)
- Apply: `git -C RBFN-Motion-Primitives apply -p1 --whitespace=nowarn <this-dir>/rbfn.patch`

Match quality: 47 of 73 tracked files match the base commit by exact blob
hash. The remaining differences are the deliberate local modification layer,
not base ambiguity: 8 code/config/doc files edited locally, 13 upstream files
deleted locally (doc images and the ZAM example scenarios) and 18 files added
locally (batch simulation scripts, 16 CommonRoad scenario XMLs, a batch
README). No other upstream commit matches any of the differing files better.
Commits `c8e86c2` and `7152a88` (2025-07-18) tie on this score but only add a
README citation block absent from the vendored copy, so the older `223c11d`
was chosen. The most recent upstream commit (`75450b7`, 2026-03-11) is a repo
restructure and matches far worse.

Patch size note: the 27 MB is dominated by the 16 added CommonRoad scenario
XMLs (about 12 MB raw). The code changes themselves are small.

Files touched by `rbfn.patch`:

```
M  README.md
A  README_BATCH_SIMULATION.md
D  doc/Figure_1.png
D  doc/MP_RBFN.png
D  doc/ZAM_Over-1_2.gif
D  doc/ZAM_Tjunction-1_27_T-1.gif
D  doc/framework_no_accel.png
A  example_scenarios/ARG_Carcarana-1_8_T-1.xml
A  example_scenarios/BEL_Antwerp-10_7_T-1.xml
A  example_scenarios/BEL_Antwerp-13_5_T-1.xml
A  example_scenarios/BEL_Antwerp-1_14_T-1.xml
A  example_scenarios/BEL_Brussels-51_2_T-1.xml
A  example_scenarios/BEL_Brussels-82_4_T-1.xml
A  example_scenarios/BEL_Zwevegem-1_6_T-1.xml
A  example_scenarios/DEU_Arnstadt-46_1_T-1.xml
A  example_scenarios/DEU_Arnstadt-83_1_T-4.xml
A  example_scenarios/DEU_Arnstadt-96_1_T-1.xml
A  example_scenarios/DEU_Aschaffenburg-16_5_T-1.xml
A  example_scenarios/DEU_Aschaffenburg-16_6_T-1.xml
A  example_scenarios/DEU_Aschaffenburg-7_9_T-1.xml
A  example_scenarios/ESP_Vigo-27_16_T-1_P2250.xml
D  example_scenarios/ZAM_Over-1_1_dynamic_1vehicle_10m-s.xml
D  example_scenarios/ZAM_Over-1_1_dynamic_1vehicle_15m-s.xml
D  example_scenarios/ZAM_Over-1_1_dynamic_1vehicle_5m-s.xml
D  example_scenarios/ZAM_Tjunction-1_23_T-1.xml
D  example_scenarios/ZAM_Tjunction-1_24_T-1.xml
D  example_scenarios/ZAM_Tjunction-1_27_T-1.xml
D  example_scenarios/ZAM_Tjunction-1_36_T-1.xml
D  example_scenarios/ZAM_Tjunction-1_42_T-1.xml
A  example_scenarios/ZAM_Zip-1_37_T-1_P6667.xml
M  ml_planner/planner/cost_functions.py
M  ml_planner/planner/ml_planner.py
M  ml_planner/planner/planner_utils.py
M  ml_planner/simulation_interfaces/commonroad/commonroad_interface.py
M  ml_planner/simulation_interfaces/commonroad/configurations/simulation.yaml
M  ml_planner/simulation_interfaces/commonroad/utils/predictions.py
M  scripts/run_cr_simulation.py
A  scripts/run_cr_simulation_batch.py
A  scripts/run_cr_simulation_batch_simple.py
```

## Setup

Add the submodules once (repo assembler):

```bash
git submodule add https://github.com/TUM-AVS/Frenetix-Motion-Planner Frenetix-Motion-Planner
git -C Frenetix-Motion-Planner checkout fc6e7708ba384f991c2a0a07b7300ab52276c70f
git submodule add https://github.com/TUM-AVS/RBFN-Motion-Primitives RBFN-Motion-Primitives
git -C RBFN-Motion-Primitives checkout 223c11ddcb3436a28666d86e7b7f4a23833d35d6
```

Then, and on every fresh clone:

```bash
git submodule update --init
./setup/apply_patches.sh
```

`apply_patches.sh` is idempotent: it detects an already patched checkout with
`git apply --reverse --check` and skips it.

Verification: both patches were applied to clean checkouts of the base
commits and the result compared with `diff -r` against the tracked vendored
content. Both trees are identical.

License note: both upstreams are LGPL-3.0. Shipping them as submodules plus
patches keeps upstream authorship intact and makes the local modifications
explicit, as the license requires.
