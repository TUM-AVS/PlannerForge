"""
Convert an OSM XML file to SUMO .net.xml via `netconvert`.

This is structurally identical to xodr_convert but writes SUMO's native net
format instead of OpenDRIVE. The two conversions share the same SUMO_HOME
override so typemaps resolve correctly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from config import DATA_NET
from .xodr_convert import NetconvertMissing, _derive_sumo_home


def _require_netconvert() -> str:
    path = shutil.which("netconvert")
    if not path:
        raise NetconvertMissing(
            "`netconvert` not found on PATH. "
            "Install SUMO: `pip install eclipse-sumo`."
        )
    return path


def convert(osm_path: Path, out_path: Path | None = None) -> Path:
    """Run netconvert on `osm_path`, write SUMO .net.xml to `out_path`."""
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(osm_path)
    netconvert = _require_netconvert()

    if out_path is None:
        out_path = DATA_NET / (osm_path.stem + ".net.xml")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        netconvert,
        "--osm-files", str(osm_path),
        "--output-file", str(out_path),
        "--geometry.remove",
        "--roundabouts.guess",
        "--ramps.guess",
        "--junctions.join",
        # Drop disconnected fragments left behind when the bbox crops
        # roads in half — keep only the largest connected component so
        # all three sims show a single, navigable network.
        "--keep-edges.components", "1",
        # Traffic-light handling — keep OSM-reported signals:
        "--tls.guess-signals",              # OSM highway=traffic_signals → SUMO TLS
        "--tls.guess",                      # add guessed TLS at big intersections
        "--tls.join",                       # merge adjacent controllers
        "--tls.default-type", "actuated",   # actuated (not fixed) default logic
    ]
    env = os.environ.copy()
    sumo_home = _derive_sumo_home()
    if sumo_home:
        env["SUMO_HOME"] = sumo_home
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"netconvert (net.xml) failed (exit {proc.returncode}):\n"
            f"stderr: {proc.stderr[-1000:]}"
        )
    if not out_path.exists() or out_path.stat().st_size < 200:
        raise RuntimeError(f"netconvert produced empty output: {out_path}")
    return out_path
