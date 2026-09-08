# Installation

PlannerForge uses three isolated Python environments because the LLM/UI stack
and the CommonRoad/SUMO simulation stack have conflicting dependencies. The
app runs in the main environment and calls the others as subprocesses.

| Environment | Python | Purpose |
|---|---|---|
| app venv | 3.12 | Gradio UI, LangChain, ChromaDB, scenario retrieval |
| `cr37` (conda) | 3.10 | CommonRoad ⇄ SUMO conversion, simulation, rendering, OSM pipeline |
| planner envs | 3.11 | Frenetix-Motion-Planner venv; MP-RBFN has its own env |

## 1. App environment (Python 3.12)

```bash
git clone --recurse-submodules <this-repo>
cd PlannerForge
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

If you cloned without `--recurse-submodules`, run
`git submodule update --init` and then apply the planner patches:

```bash
bash setup/apply_patches.sh
```

## 2. CommonRoad/SUMO environment (`cr37`, Python 3.10)

```bash
conda env create -f environment-cr37.yml
```

The app invokes it as `conda run -n cr37 python ...`. If you name the env
differently, set `CR_CONDA_ENV` and `OSM_CONDA_ENV` in `.env`.

### Required patches

Two small patches must be applied inside the installed packages of the env:

```bash
conda activate cr37
# 1) sumocr id mapper
cd "$CONDA_PREFIX/lib/python3.10/site-packages/sumocr/interface"
patch -p1 < <repo>/setup/id_mapper_patch.diff
# 2) scenario-designer CR→SUMO converter
cd "$CONDA_PREFIX/lib/python3.10/site-packages/crdesigner/map_conversion/sumo_map/cr2sumo"
patch -p1 < <repo>/setup/converter_patch.diff
```

(The second patch is equivalent to setting `traffic_from_trajectories=True`
in `create_sumo_files()` in `converter.py`.)

## 3. Frenetix-Motion-Planner (submodule)

System dependencies for the C++ core:

```bash
sudo apt-get install libeigen3-dev libboost-all-dev libomp-dev python3.11-full python3.11-dev
```

Then, inside the submodule:

```bash
cd Frenetix-Motion-Planner
python3.11 -m venv venv && source venv/bin/activate
pip install .
```

The planner uses PyTorch; if the default wheel does not match your hardware,
reinstall the fitting `torch` build. The app finds the planner via the venv at
`Frenetix-Motion-Planner/venv/` (see `process_engine.py`).

## 4. MP-RBFN (submodule, own environment)

MP-RBFN runs in its own environment (do not fold it into `cr37`). Create a
dedicated env, install the submodule into it, and point `MPRBFN_PYTHON` in
`.env` at that env's interpreter:

```bash
conda create -n mprbfn python=3.11 -y
conda run -n mprbfn pip install -e RBFN-Motion-Primitives
echo "MPRBFN_PYTHON=$(conda run -n mprbfn which python)" >> .env
```

## 5. Environment variables

```bash
cp .env.example .env
```

Fill in `CONDA_BIN`, one LLM provider (a hosted API key, or a local Ollama
model with `DEFAULT_MODE=ollama`).

## 6. Scenario database

The scenario corpus lives in a Hugging Face dataset,
[`Yuan-avs/PlannerForge-Scenarios`](https://huggingface.co/datasets/Yuan-avs/PlannerForge-Scenarios),
rather than in this repository. `add_scenarios.py` downloads the 582 CommonRoad
XMLs into `CollectedScenarios/` and indexes them into ChromaDB in one step:

```bash
python add_scenarios.py
```

Re-runs reuse whatever is already in `CollectedScenarios/`, so this is cheap to
repeat. If the download fails, log in with `huggingface-cli login` (the dataset
must be readable by your account) or copy the XMLs into that directory by hand —
the rest of the pipeline only cares that the files are there.

This builds the persistent `chroma/` store used by scenario selection. The
released selection ground truth was computed against all 585 scenarios, so do
not index a subset.

## 7. Run

```bash
python interface.py
```

The Gradio UI starts locally; pick a scenario by describing it, modify it,
run a motion planner on it, and analyse the results from the chat.
