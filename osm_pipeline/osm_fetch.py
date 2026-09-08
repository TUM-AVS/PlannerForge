"""
Fetch road-network data from OpenStreetMap and save as a .osm XML file.

Primary path: Overpass API (raw .osm XML, exactly what SUMO netconvert wants).
Geocoding: Nominatim via osmnx when a city name is provided.
"""
from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import requests

from config import DATA_RAW, resolve_classes


class BBoxTooLarge(ValueError):
    pass


def _bbox_area_km2(bbox: tuple[float, float, float, float]) -> float:
    """Rough area of a lat/lon bbox in km² (equirectangular approximation)."""
    south, west, north, east = bbox
    mean_lat_rad = math.radians((south + north) / 2.0)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * math.cos(mean_lat_rad)
    return (north - south) * km_per_deg_lat * (east - west) * km_per_deg_lon

# Ordered fastest/most-reliable first; the fetcher falls through on failure.
# overpass-api.de is the canonical instance but is frequently overloaded
# (504/connection-refused), so it is not tried first.
OVERPASS_URLS = [
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "osm-carla-chat/0.1 (research)"


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return s.strip("_") or "region"


def _geocode(location: str) -> tuple[float, float, float, float]:
    """Return (south, west, north, east) bbox for a place name."""
    r = requests.get(
        NOMINATIM_URL,
        params={"q": location, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    r.raise_for_status()
    results = r.json()
    if not results:
        raise ValueError(f"Nominatim found no result for: {location!r}")
    bb = results[0]["boundingbox"]  # [south, north, west, east] as strings
    south, north, west, east = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
    return south, west, north, east


def _build_query(bbox: tuple[float, float, float, float],
                 highway_values: list[str]) -> str:
    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"
    if not highway_values:
        way_filter = "way[highway]"
    else:
        regex = "|".join(re.escape(v) for v in highway_values)
        way_filter = f'way[highway~"^({regex})$"]'
    return (
        f"[out:xml][timeout:180];\n"
        f"(\n"
        f"  {way_filter}({bbox_str});\n"
        f");\n"
        f"(._;>;);\n"
        f"out body;\n"
    )


def fetch(location: str | None = None,
          classes: list[str] | None = None,
          bbox: tuple[float, float, float, float] | None = None,
          out_path: Path | None = None) -> Path:
    """Download an OSM XML file matching the given location + highway classes."""
    if bbox is None:
        if not location:
            raise ValueError("Provide either `location` or `bbox`.")
        bbox = _geocode(location)

    max_km2 = float(os.getenv("MAX_BBOX_KM2", "100"))
    area = _bbox_area_km2(bbox)
    if area > max_km2:
        raise BBoxTooLarge(
            f"requested bbox is {area:.0f} km² — exceeds MAX_BBOX_KM2={max_km2:.0f}. "
            f"Narrow the location (e.g. a district or address instead of a whole city), "
            f"or raise MAX_BBOX_KM2 in .env if you really want this."
        )

    hw_values = resolve_classes(classes or ["all"])
    query = _build_query(bbox, hw_values)

    label = _slugify(location) if location else "bbox"
    cls_tag = _slugify("_".join(classes)) if classes else "all"
    # Stem timestamp must be unique per fetch — earlier minute-resolution
    # timestamps caused same-minute fetches to overwrite each other. We
    # now use second resolution and append _N if a same-second collision
    # would still occur.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_path is None:
        candidate = DATA_RAW / f"{label}_{cls_tag}_{stamp}.osm"
        n = 0
        while candidate.exists():
            n += 1
            candidate = DATA_RAW / f"{label}_{cls_tag}_{stamp}_{n}.osm"
        out_path = candidate
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xml_bytes = _overpass_fetch_with_fallback(query)
    out_path.write_bytes(xml_bytes)
    # polite delay to avoid hammering Overpass
    time.sleep(1.0)
    return out_path


def _overpass_fetch_with_fallback(query: str) -> bytes:
    """Try each Overpass mirror in order, with one retry per mirror. Raises
    the final error if all mirrors fail."""
    last_err: Exception | None = None
    for url in OVERPASS_URLS:
        for attempt in (0, 1):
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers={"User-Agent": USER_AGENT},
                    timeout=(15, 120),   # (connect, read): fail dead mirrors fast
                )
                if resp.status_code in (429, 503, 504):
                    last_err = requests.HTTPError(
                        f"{resp.status_code} from {url}")
                    time.sleep(2 * (attempt + 1))
                    continue
                resp.raise_for_status()
                xml_bytes = resp.content
                if b"<osm" not in xml_bytes[:2000]:
                    raise RuntimeError(
                        f"Overpass response from {url} did not look like OSM XML.")
                return xml_bytes
            except (requests.HTTPError, requests.ConnectionError,
                    requests.Timeout) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
                continue
    raise RuntimeError(
        f"all Overpass mirrors failed; last error: {last_err!r}")
