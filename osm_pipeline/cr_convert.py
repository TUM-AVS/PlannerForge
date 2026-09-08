"""
Convert an OpenDRIVE `.xodr` to a CommonRoad scenario (.xml) via
`crdesigner.map_conversion.map_conversion_interface.opendrive_to_commonroad`.

We deliberately go through the XODR (which CARLA already loads) so the
SUMO simulation, the OpenDRIVE map in CARLA, and the CommonRoad scenario
all come from the same canonical road network. crdesigner's
`sumo_to_commonroad` is broken upstream (calls into the OpenDRIVE
parser with the wrong type), so we don't use that path. A legacy
`osm_to_commonroad` fallback is kept for callers that only have raw OSM.
"""
from __future__ import annotations

from pathlib import Path

from config import DATA_CR


def convert(input_path: Path,
            out_path: Path | None = None,
            accept_service: bool = True,
            accept_unclassified: bool = True) -> Path:
    """Convert a `.xodr` (preferred) or `.osm` to CommonRoad."""
    from crdesigner.map_conversion.map_conversion_interface import (
        osm_to_commonroad, opendrive_to_commonroad)
    from crdesigner.common.config.osm_config import osm_config
    from commonroad.common.file_writer import (
        CommonRoadFileWriter, OverwriteExistingFile)
    from commonroad.scenario.scenario import Tag
    from commonroad.planning.planning_problem import PlanningProblemSet

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    is_xodr = input_path.suffix.lower() == ".xodr"
    if out_path is None:
        out_path = DATA_CR / (input_path.stem + ".xml")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if is_xodr:
        # XODR → CR: same OpenDRIVE that CARLA loads, so SUMO/CARLA/CR
        # all agree on the road network. crdesigner's odr2cr parser
        # crashes when signal IDs are non-integer (netconvert emits
        # IDs like "cluster_..._4"), so we strip <signals> blocks into
        # a sanitised copy before parsing. Traffic lights are still
        # honoured by SUMO during simulate(); CR doesn't drive them.
        import tempfile
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(input_path))
        for road in tree.getroot().findall("road"):
            for tag in ("signals", "objects"):
                el = road.find(tag)
                if el is not None:
                    road.remove(el)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".xodr",
                                         delete=False) as tmp:
            tree.write(tmp.name, xml_declaration=True, encoding="utf-8")
            sanitised = tmp.name
        try:
            scenario = opendrive_to_commonroad(sanitised)
        finally:
            try:
                Path(sanitised).unlink()
            except Exception:
                pass
    else:
        # Legacy OSM → CR fallback. Expand the highway whitelist so the
        # CR output covers the same drivable roads that SUMO/CARLA see.
        saved = dict(osm_config.ACCEPTED_HIGHWAYS_MAINLAYER)
        try:
            for k in list(osm_config.ACCEPTED_HIGHWAYS_MAINLAYER):
                osm_config.ACCEPTED_HIGHWAYS_MAINLAYER[k] = True
            if not accept_service:
                osm_config.ACCEPTED_HIGHWAYS_MAINLAYER["service"] = False
            if not accept_unclassified:
                osm_config.ACCEPTED_HIGHWAYS_MAINLAYER["unclassified"] = False
            for k in ("path", "footway", "cycleway"):
                if k in osm_config.ACCEPTED_HIGHWAYS_MAINLAYER:
                    osm_config.ACCEPTED_HIGHWAYS_MAINLAYER[k] = False
            scenario = osm_to_commonroad(str(input_path))
        finally:
            osm_config.ACCEPTED_HIGHWAYS_MAINLAYER = saved

    writer = CommonRoadFileWriter(
        scenario=scenario,
        planning_problem_set=PlanningProblemSet(),
        author="osm-carla-chat",
        affiliation="",
        source=("converted from OpenDRIVE via crdesigner" if is_xodr
                else "converted from OSM via crdesigner"),
        tags={Tag.URBAN},
    )
    writer.write_to_file(str(out_path), OverwriteExistingFile.ALWAYS)
    return out_path
