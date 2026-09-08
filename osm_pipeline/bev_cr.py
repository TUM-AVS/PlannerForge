"""
Render a CommonRoad scenario to a top-down PNG.

Styling (renderer, draw parameters, fonts, figure geometry) comes from the
repo-root `cr_draw_style` module, which is also what the database and
modified-scenario GIFs use — so a generated scenario is drawn with the same
lanelet-ID font, vehicle-ID font, traffic signs and colours as every other
picture in the app. Do not add local draw-parameter tweaks here; change
`cr_draw_style` instead.
"""
from __future__ import annotations

from pathlib import Path

from config import DATA_BEV_CR


def render(cr_path: Path,
           out_path: Path | None = None,
           size_px: int = 2000,
           timestep: int = 0) -> Path:
    """Render `cr_path` (CommonRoad .xml) as a top-down PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from commonroad.common.file_reader import CommonRoadFileReader

    from cr_draw_style import (BBOX_INCHES, FIGSIZE, PAD_INCHES, CustomRenderer,
                               apply_rcparams, build_draw_params, install_icons)

    cr_path = Path(cr_path)
    if not cr_path.exists():
        raise FileNotFoundError(cr_path)

    if out_path is None:
        out_path = DATA_BEV_CR / (cr_path.stem + ".png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scenario, pps = CommonRoadFileReader(str(cr_path)).open()

    apply_rcparams()
    install_icons()
    params = build_draw_params(timestep)

    # size_px is an output-resolution knob only: label sizes are in points, so
    # the look is fixed by FIGSIZE and unaffected by the DPI chosen here.
    dpi = size_px / FIGSIZE[0]
    fig = plt.figure(figsize=FIGSIZE, dpi=dpi, facecolor="white")
    rnd = CustomRenderer(ax=fig.gca())
    scenario.draw(rnd, draw_params=params)
    if pps is not None and pps.planning_problem_dict:
        try:
            for pp in pps.planning_problem_dict.values():
                pp.draw(rnd)
        except Exception:
            pass
    rnd.render()
    ax = fig.gca()
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches=BBOX_INCHES,
                pad_inches=PAD_INCHES, facecolor="white")
    plt.close(fig)
    return out_path
