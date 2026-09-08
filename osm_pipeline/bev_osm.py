"""
Render a BEV PNG of an OSM file.

Default path: stitch OpenStreetMap raster tiles (same source as the Leaflet
map in the UI) into a single cropped PNG. The result includes buildings,
parks, and land-use — the line-only osmnx rendering showed road
centerlines only.

Fallback: if no bbox is provided or tile fetching fails, draw the street
graph with osmnx.plot_graph. That path needs no internet but looks
"weird" (line art).
"""
from __future__ import annotations

import io
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import DATA_BEV_OSM

# Public OSM tile usage policy requires a descriptive User-Agent and no
# bulk downloading. Stage 1 renders request ~30 tiles per run; well within
# fair use. Ref: https://operations.osmfoundation.org/policies/tiles/
_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
_USER_AGENT = "osm-carla-chatbot/1.0 (self-hosted; +research)"
_TILE_PX = 256

# If tile-stitching falls back to the line drawing on the most recent
# `render()` call, the reason is stored here so the Stage-1 wrapper can
# surface it in the result log. `None` means the tile path succeeded.
LAST_DEGRADE: str | None = None


def _lonlat_to_tile_xy(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(rad) + 1.0 / math.cos(rad)) / math.pi) / 2.0 * n
    return x, y


def _pick_zoom(bbox: tuple[float, float, float, float],
               target_px: int) -> int:
    """Pick the lowest zoom ∈ [10, 19] at which the bbox's longitudinal
    span meets or exceeds `target_px`. Returns 19 as fallback for very
    small bboxes — the caller upscales the crop afterward."""
    south, west, north, east = bbox
    for z in range(10, 20):
        x_w, _ = _lonlat_to_tile_xy(west, (south + north) / 2, z)
        x_e, _ = _lonlat_to_tile_xy(east, (south + north) / 2, z)
        if abs(x_e - x_w) * _TILE_PX >= target_px:
            return z
    return 19


class TileFetchError(RuntimeError):
    """Raised when too many tiles fail for the stitch to be usable."""


def _fetch_tile(session, z: int, tx: int, ty: int):
    """Fetch a single tile. Two attempts with 0.5 s backoff on any
    requests.RequestException / non-200 response."""
    from PIL import Image
    last_exc = None
    for attempt in range(2):
        try:
            r = session.get(_TILE_URL.format(z=z, x=tx, y=ty), timeout=10)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            last_exc = e
            if attempt == 0:
                time.sleep(0.5)
    raise last_exc  # surface the final failure to the orchestrator


def _stitch_tiles(bbox: tuple[float, float, float, float],
                  out_path: Path, target_px: int) -> Path:
    import requests
    from PIL import Image

    south, west, north, east = bbox
    z = _pick_zoom(bbox, target_px)

    # Fractional tile coords for the bbox corners. Origin (0,0) of tile
    # space is top-left (north-west); x grows east, y grows south.
    x_nw, y_nw = _lonlat_to_tile_xy(west, north, z)
    x_se, y_se = _lonlat_to_tile_xy(east, south, z)
    x_min, x_max = int(math.floor(x_nw)), int(math.floor(x_se))
    y_min, y_max = int(math.floor(y_nw)), int(math.floor(y_se))

    canvas = Image.new("RGB",
                       ((x_max - x_min + 1) * _TILE_PX,
                        (y_max - y_min + 1) * _TILE_PX),
                       "white")
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    # Parallelize tile fetches — network-bound, GIL-friendly. ~30 tiles
    # drop from ~9 s sequential to ~1.5 s.
    tile_coords = [(tx, ty)
                   for tx in range(x_min, x_max + 1)
                   for ty in range(y_min, y_max + 1)]
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_tile, session, z, tx, ty): (tx, ty)
                   for tx, ty in tile_coords}
        for fut in as_completed(futures):
            tx, ty = futures[fut]
            try:
                tile = fut.result()
            except Exception:
                failures += 1
                continue
            canvas.paste(tile, ((tx - x_min) * _TILE_PX,
                                (ty - y_min) * _TILE_PX))

    if failures > 0.1 * len(tile_coords):
        raise TileFetchError(
            f"tile fetch failed ({failures}/{len(tile_coords)}); "
            f"falling back to line drawing")

    # Crop from the tile grid to exactly the bbox.
    left   = (x_nw - x_min) * _TILE_PX
    right  = (x_se - x_min) * _TILE_PX
    top    = (y_nw - y_min) * _TILE_PX
    bottom = (y_se - y_min) * _TILE_PX
    crop = canvas.crop((int(left), int(top),
                        int(math.ceil(right)), int(math.ceil(bottom))))

    # For small bboxes even z=19 can leave the crop under target_px; scale
    # the longer side up to target_px with Lanczos so the UI panel matches
    # the line-drawing panel in apparent resolution.
    longest = max(crop.size)
    if longest < target_px:
        scale = target_px / longest
        new_size = (int(crop.size[0] * scale), int(crop.size[1] * scale))
        crop = crop.resize(new_size, Image.Resampling.LANCZOS)

    crop.save(out_path)
    return out_path


def _render_graph_fallback(osm_path: Path, out_path: Path, size_px: int,
                           bg: str, edge_color: str,
                           bbox: tuple[float, float, float, float] | None) -> Path:
    import osmnx as ox
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    graph = ox.graph_from_xml(str(osm_path), simplify=True, retain_all=True)
    dpi = 200

    if bbox is not None:
        # Build a figure whose canvas aspect matches the bbox in
        # Web-Mercator-equivalent proportions (lon span shrinks by
        # cos(lat) at mid-latitude, same adjustment as the tile map).
        south, west, north, east = bbox
        mid_lat = math.radians((south + north) / 2)
        bbox_w = (east - west) * math.cos(mid_lat)
        bbox_h = (north - south)
        if bbox_w >= bbox_h:
            fig_w_px, fig_h_px = size_px, int(size_px * bbox_h / bbox_w)
        else:
            fig_w_px, fig_h_px = int(size_px * bbox_w / bbox_h), size_px
        fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi),
                         dpi=dpi, facecolor=bg)
        ax = fig.add_axes([0, 0, 1, 1])           # fill entire canvas
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        ax.set_aspect(1 / math.cos(mid_lat))
        ax.set_axis_off()
        ox.plot_graph(
            graph, ax=ax, save=False, show=False, close=False,
            bgcolor=bg, edge_color=edge_color, edge_linewidth=0.6,
            node_size=0,
        )
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        # No `bbox_inches="tight"` — the axes already fill the canvas
        # exactly, so saving the whole figure gives us the requested bbox.
        fig.savefig(out_path, dpi=dpi, pad_inches=0, facecolor=bg)
    else:
        figsize = (size_px / dpi, size_px / dpi)
        fig, ax = ox.plot_graph(
            graph, save=False, show=False, close=False,
            bgcolor=bg, edge_color=edge_color, edge_linewidth=0.6,
            node_size=0, figsize=figsize, dpi=dpi,
        )
        ax.set_aspect("equal")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                    pad_inches=0, facecolor=bg)
    plt.close(fig)
    return out_path


def render(osm_path: Path,
           out_path: Path | None = None,
           size_px: int = 2048,
           bg: str = "white",
           edge_color: str = "black",
           bbox: tuple[float, float, float, float] | None = None) -> Path:
    """Render `osm_path`'s bbox as a BEV PNG using OSM raster tiles.

    If `bbox` is provided we fetch/stitch tiles (buildings + parks + roads).
    Otherwise — or if the fetch errors out — fall back to osmnx line art.
    """
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(osm_path)

    if out_path is None:
        out_path = DATA_BEV_OSM / (osm_path.stem + ".png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    global LAST_DEGRADE
    LAST_DEGRADE = None
    if bbox is not None:
        try:
            return _stitch_tiles(bbox, out_path, target_px=size_px)
        except TileFetchError as e:
            LAST_DEGRADE = str(e)
        except Exception as e:
            # Internet issue, rate-limit, or unexpected tile response:
            # degrade gracefully to the line rendering rather than
            # failing Stage 1 outright.
            LAST_DEGRADE = f"{type(e).__name__}: {e}"
    return _render_graph_fallback(osm_path, out_path, size_px, bg,
                                  edge_color, bbox)


def render_roads(osm_path: Path,
                 out_path: Path | None = None,
                 size_px: int = 2048,
                 bg: str = "white",
                 edge_color: str = "black",
                 bbox: tuple[float, float, float, float] | None = None) -> Path:
    """Render the road graph (line drawing) of `osm_path`. Separate from
    `render()` so Stage 1 can surface both a tile-map view (with buildings)
    and a road-only BEV — useful when the line art is what the downstream
    SUMO/CR/CARLA pipeline actually consumes."""
    osm_path = Path(osm_path)
    if not osm_path.exists():
        raise FileNotFoundError(osm_path)
    if out_path is None:
        out_path = DATA_BEV_OSM / (osm_path.stem + "_roads.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _render_graph_fallback(osm_path, out_path, size_px, bg,
                                  edge_color, bbox)
