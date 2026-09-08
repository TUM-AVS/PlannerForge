"""Wrapper that runs the vendored `osm_pipeline.run` CLI inside the conda
env named by `OSM_CONDA_ENV` in config.py (defaults to "cr37" — PlannerForge's
existing CommonRoad / SUMO env).

Mirrors the pattern used by [convert_to_sumo_subprocess.py] and other
process_engine subprocess calls: build the `conda run -n <env> python -m
osm_pipeline.run <cmd> ...` command, then stream stdout/stderr line by
line so the Gradio Generate tab can live-tail progress.

Public entry points return *generators* that yield each output line, then
finally yield a `("RESULT", paths_dict)` tuple. The caller collects every
line for the UI log and reads `paths_dict` for the final artefact list.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Generator

# `PlannerForge/` (the parent of this commonroad_interface package) must be on
# PYTHONPATH so the subprocess can `import osm_pipeline.run` and
# `import config` (the vendored tools import from `config`).
_LLM_MOST_ROOT = Path(__file__).resolve().parent.parent


def _conda_bin() -> str:
    return os.getenv("CONDA_BIN", "conda")


def _conda_env() -> str:
    # Mirrors config.OSM_CONDA_ENV without importing config here — this
    # module is imported from the main PlannerForge process where config sits
    # at the working-dir root; falling back via os.getenv keeps the call
    # site identical whether or not config was loaded yet.
    return os.getenv("OSM_CONDA_ENV", "cr37").strip() or "cr37"


def _stream(cmd: list[str]) -> Generator[str, None, int]:
    """Run `cmd`, yield each merged stdout/stderr line, return exit code."""
    env = os.environ.copy()
    # Prepend PlannerForge/ so the child's `from config import ...` resolves
    # to PlannerForge/config.py (which holds the OSM constants) rather than
    # whatever happens to be on the cr37 site-packages.
    py_path = str(_LLM_MOST_ROOT)
    env["PYTHONPATH"] = (
        py_path + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH") else py_path
    )
    # Force the child's stdout/stderr unbuffered. When stdout is a pipe (not a
    # TTY) Python block-buffers it, so print() output sits in an ~8KB buffer and
    # only flushes at exit — making the Gradio log appear all-at-once at the end
    # instead of streaming line by line. (conda run already gets
    # --no-capture-output; this is the remaining buffer.)
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge into one stream — easier to tail
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
        cwd=str(_LLM_MOST_ROOT),
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.stdout.close()
    return proc.wait()


def _parse_labelled_paths(lines: list[str]) -> dict[str, str]:
    """Parse `Label → /path` lines emitted by osm_pipeline.run into a dict."""
    out: dict[str, str] = {}
    for ln in lines:
        if "→" in ln:
            label, _, value = ln.partition("→")
            out[label.strip()] = value.strip()
    return out


def run_stage1(
    bbox: tuple[float, float, float, float],
    classes: list[str],
    density: str = "medium",
    sim_seconds: float = 20.0,
    vehicle_types: list[str] | None = None,
    simulate_cr: bool = True,
    record_cr_gif: bool = True,
) -> Generator[str | tuple[str, dict], None, None]:
    """Run OSM fetch + SUMO/CR convert + simulate. Yields each stdout
    line, then a final `("RESULT", {label: path})` tuple.

    `bbox` is (south, west, north, east). `classes` are CLASS_GROUPS keys
    ("roads", "link", "all") or raw OSM highway= values."""
    south, west, north, east = bbox
    cmd = [
        _conda_bin(), "run", "--no-capture-output", "-n", _conda_env(),
        "python", "-m", "osm_pipeline.run", "stage1",
        "--south", f"{south}",
        "--west", f"{west}",
        "--north", f"{north}",
        "--east", f"{east}",
        "--classes", ",".join(classes or ["all"]),
        "--density", density,
        "--sim-seconds", f"{sim_seconds}",
        "--vehicle-types", ",".join(vehicle_types or ["car"]),
    ]
    if not simulate_cr:
        cmd.append("--no-simulate-cr")
    if not record_cr_gif:
        cmd.append("--no-record-cr-gif")

    lines: list[str] = []
    gen = _stream(cmd)
    try:
        while True:
            line = next(gen)
            lines.append(line)
            yield line
    except StopIteration as stop:
        exit_code = stop.value
    if exit_code != 0:
        yield f"=== stage1 FAILED (exit {exit_code}) ==="
    yield ("RESULT", _parse_labelled_paths(lines))


def run_synthesize(
    sim_xml: Path,
    ego_id: int,
    goal_lanelet_id: int | None = None,
    goal_position: str = "middle",
    goal_offset_steps: int = 50,
    goal_length_m: float = 6.0,
    goal_width_m: float = 2.0,
    out_path: Path | None = None,
) -> Generator[str | tuple[str, dict], None, None]:
    """Build a CR PlanningProblem on top of a sim XML. Same generator
    contract as run_stage1."""
    cmd = [
        _conda_bin(), "run", "--no-capture-output", "-n", _conda_env(),
        "python", "-m", "osm_pipeline.run", "synthesize",
        "--sim-xml", str(sim_xml),
        "--ego-id", str(ego_id),
        "--goal-position", goal_position,
        "--goal-offset-steps", str(goal_offset_steps),
        "--goal-length-m", str(goal_length_m),
        "--goal-width-m", str(goal_width_m),
    ]
    if goal_lanelet_id is not None:
        cmd.extend(["--goal-lanelet-id", str(goal_lanelet_id)])
    if out_path is not None:
        cmd.extend(["--out-path", str(out_path)])

    lines: list[str] = []
    gen = _stream(cmd)
    try:
        while True:
            line = next(gen)
            lines.append(line)
            yield line
    except StopIteration as stop:
        exit_code = stop.value
    if exit_code != 0:
        yield f"=== synthesize FAILED (exit {exit_code}) ==="
    yield ("RESULT", _parse_labelled_paths(lines))


def list_egos(sim_xml: Path, limit: int = 30) -> list[int]:
    """Synchronous helper — small enough that we don't bother streaming.
    Returns the dynamic-obstacle IDs in the sim XML in their natural order."""
    import json
    cmd = [
        _conda_bin(), "run", "--no-capture-output", "-n", _conda_env(),
        "python", "-m", "osm_pipeline.run", "list-egos",
        "--sim-xml", str(sim_xml),
        "--limit", str(limit),
    ]
    env = os.environ.copy()
    py_path = str(_LLM_MOST_ROOT)
    env["PYTHONPATH"] = (
        py_path + os.pathsep + env["PYTHONPATH"]
        if env.get("PYTHONPATH") else py_path
    )
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_LLM_MOST_ROOT), env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + "\n" + proc.stderr)
        raise RuntimeError(
            f"list-egos failed (exit {proc.returncode}): {proc.stderr[-500:]}")
    for line in proc.stdout.splitlines():
        if line.startswith("EGOS_JSON "):
            return json.loads(line[len("EGOS_JSON "):])
    return []
