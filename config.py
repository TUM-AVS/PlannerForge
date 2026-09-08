import os
from pathlib import Path


class SessionConfig:
    USE_REPAIR_MODULE = True

    @classmethod
    def set_use_repair_module(cls, value: bool):
        cls.USE_REPAIR_MODULE = value  # Fixed: was USE_SPECIAL_MODULE

    @classmethod
    def get_use_repair_module(cls) -> bool:
        return cls.USE_REPAIR_MODULE  # Fixed: was USE_SPECIAL_MODULE


# ── OSM scenario-generation pipeline ─────────────────────────────────────
# Vendored from osm/config.py. Imported by every module in osm_pipeline/
# via `from config import DATA_RAW, ...`. Data lives under PlannerForge/data/osm/
# so the generation outputs do not collide with anything else.

_HERE = Path(__file__).parent
DATA_OSM_ROOT = _HERE / "data" / "osm"
DATA_RAW = DATA_OSM_ROOT / "raw"
DATA_XODR = DATA_OSM_ROOT / "xodr"
DATA_NET = DATA_OSM_ROOT / "net"
DATA_CR = DATA_OSM_ROOT / "cr"
DATA_BEV = DATA_OSM_ROOT / "bev"
DATA_BEV_OSM = DATA_BEV / "osm"
DATA_BEV_SUMO = DATA_BEV / "sumo"
DATA_BEV_CR = DATA_BEV / "cr"

for _d in (DATA_RAW, DATA_XODR, DATA_NET, DATA_CR,
           DATA_BEV, DATA_BEV_OSM, DATA_BEV_SUMO, DATA_BEV_CR):
    _d.mkdir(parents=True, exist_ok=True)

# OSM road-class grouping — matches osm/config.py so vendored tools resolve
# the same expansions. The "Generate" tab exposes these keys as checkboxes.
CLASS_GROUPS = {
    "roads": [
        "motorway", "trunk",
        "primary", "secondary", "tertiary",
        "residential", "unclassified", "living_street",
        "service", "track",
    ],
    "link": [
        "motorway_link", "trunk_link",
        "primary_link", "secondary_link", "tertiary_link",
    ],
}
CLASS_GROUPS["all"] = sorted({v for lst in CLASS_GROUPS.values() for v in lst})


def resolve_classes(names: list[str]) -> list[str]:
    """Expand group names into raw OSM highway= values. Unknown names pass
    through so callers can also supply raw OSM values directly."""
    out: set[str] = set()
    for n in names or []:
        n = n.strip().lower()
        if n in CLASS_GROUPS:
            out.update(CLASS_GROUPS[n])
        else:
            out.add(n)
    return sorted(out)


# Caps applied during OSM fetch. Raise via .env if a study area legitimately
# exceeds the default.
MAX_BBOX_KM2 = float(os.getenv("MAX_BBOX_KM2", "100"))
MAX_XODR_MB = float(os.getenv("MAX_XODR_MB", "50"))

# Conda env name to run osm_pipeline.run inside. Defaults to cr37 (PlannerForge's
# existing CommonRoad/SUMO env). Override with OSM_CONDA_ENV in .env if you
# need a dedicated env for osmnx/crdesigner.
OSM_CONDA_ENV = os.getenv("OSM_CONDA_ENV", "cr37").strip() or "cr37"
