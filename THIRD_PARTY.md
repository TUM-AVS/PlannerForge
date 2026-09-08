# Third-party software

PlannerForge's own code is MIT-licensed (see LICENSE). It orchestrates the
following third-party components, which keep their own licenses.

## Motion planners (git submodules, LGPL-3.0)

Both planners are integrated as git submodules pinned to an upstream commit,
with small local modifications shipped as patch files in `setup/patches/`
(applied by `setup/apply_patches.sh`). The patches are provided under the
same LGPL-3.0 terms as the planners they modify. See
`setup/patches/UPSTREAM.md` for the pinned upstream commits and the exact
list of modified files.

- **Frenetix-Motion-Planner** — LGPL-3.0. Invoked as a subprocess
  (`main_batch.py`); PlannerForge edits its
  `configurations/frenetix_motion_planner/cost.yaml` and reads its
  `logs/score_overview.csv`.
- **RBFN-Motion-Primitives (MP-RBFN)** — LGPL-3.0. Invoked as a subprocess
  (`scripts/run_cr_simulation_batch.py`).

## CommonRoad / SUMO toolchain

Installed from PyPI into the `cr37` conda env (see `environment-cr37.yml`):
commonroad-io, commonroad-scenario-designer, commonroad-drivability-checker,
commonroad-route-planner, commonroad-vehicle-models, sumocr, Eclipse SUMO,
OSMnx, and related packages. Two local patches to installed files
(`setup/id_mapper_patch.diff`, `setup/converter_patch.diff`) are documented
in `setup/install.md`.

The scenario corpus in `CollectedScenarios/` consists of scenarios in the
CommonRoad format; scenario credits are embedded in each XML's metadata.
