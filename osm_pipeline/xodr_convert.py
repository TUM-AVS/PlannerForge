"""
Convert an OSM XML file to OpenDRIVE (.xodr) via SUMO's `netconvert`.

Install SUMO: `sudo apt install sumo sumo-tools`
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from config import DATA_XODR


class NetconvertMissing(RuntimeError):
    pass


def _require_netconvert() -> str:
    path = shutil.which("netconvert")
    if not path:
        raise NetconvertMissing(
            "`netconvert` not found on PATH. "
            "Install SUMO: `pip install eclipse-sumo` (ships the binary) "
            "or `sudo apt install sumo sumo-tools`."
        )
    return path


def _derive_sumo_home() -> str | None:
    """Locate the SUMO data dir shipped by eclipse-sumo, ignoring any stale
    SUMO_HOME the user may have exported in their shell profile."""
    try:
        import sumo  # noqa
        candidate = Path(sumo.__file__).parent
        if (candidate / "data" / "typemap" / "osmNetconvert.typ.xml").exists():
            return str(candidate)
    except ImportError:
        pass
    return None


def convert(osm_or_net_path: Path,
            out_path: Path | None = None) -> Path:
    """Run netconvert and write a `.xodr`. Accepts either a SUMO `.net.xml`
    (preferred — guarantees the XODR network matches SUMO bit-for-bit) or
    a raw `.osm`. The dispatcher in [llm_wrapper.py] passes the .net.xml
    so all three Stage 1 simulations share an identical road network."""
    osm_or_net_path = Path(osm_or_net_path)
    if not osm_or_net_path.exists():
        raise FileNotFoundError(osm_or_net_path)
    netconvert = _require_netconvert()

    is_net = osm_or_net_path.suffix.lower() == ".xml"
    if out_path is None:
        # Strip `.net` from `<stem>.net.xml` so the XODR keeps the OSM stem.
        stem = osm_or_net_path.stem
        if is_net and stem.endswith(".net"):
            stem = stem[: -len(".net")]
        out_path = DATA_XODR / (stem + ".xodr")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_net:
        cmd = [
            netconvert,
            "--sumo-net-file", str(osm_or_net_path),
            "--opendrive-output", str(out_path),
        ]
    else:
        cmd = [
            netconvert,
            "--osm-files", str(osm_or_net_path),
            "--opendrive-output", str(out_path),
            "--geometry.remove",
            "--roundabouts.guess",
            "--ramps.guess",
            "--junctions.join",
            "--tls.guess-signals",
            "--tls.guess",
            "--tls.join",
            "--tls.default-type", "actuated",
        ]
    env = os.environ.copy()
    sumo_home = _derive_sumo_home()
    if sumo_home:
        env["SUMO_HOME"] = sumo_home
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"netconvert failed (exit {proc.returncode}):\n"
            f"stdout: {proc.stdout[-1000:]}\n"
            f"stderr: {proc.stderr[-1000:]}"
        )
    if not out_path.exists() or out_path.stat().st_size < 200:
        raise RuntimeError(f"netconvert produced empty output: {out_path}")
    head = out_path.read_text(errors="ignore")[:20000]
    if "<OpenDRIVE" not in head:
        raise RuntimeError(f"Output does not look like OpenDRIVE:\n{head[:500]}")
    return out_path
