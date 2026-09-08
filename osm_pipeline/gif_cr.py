"""
Render a CommonRoad scenario's full recorded trajectory as an animated GIF.

Assumes the input .xml has been populated with DynamicObstacle trajectories
(e.g. produced by osm_pipeline.cr_simulate.simulate).

Styling comes from the repo-root `cr_draw_style` module — the same one the
database and modified-scenario GIFs use — so a generated scenario animates
with identical lanelet-ID font, vehicle-ID font, traffic signs and colours.
Do not add local draw-parameter tweaks here; change `cr_draw_style` instead.
"""
from __future__ import annotations

import io
from pathlib import Path

from config import DATA_BEV_CR


def record(cr_path: Path,
           out_gif: Path | None = None,
           n_frames: int = 40,
           size_px: int = 2400,
           fps: int | None = None) -> Path:
    """Render `cr_path` as an animated GIF by sampling `n_frames` timesteps
    uniformly across the scenario's full trajectory horizon.

    `fps` overrides the frame rate. Left at None, the frame duration is
    derived from the sampling stride so the animation plays at the same
    apparent speed as the database/modified GIFs (which show every timestep
    at cr_draw_style.FRAME_DURATION_MS milliseconds per frame).

    Returns the GIF path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio
    import numpy as np
    from commonroad.common.file_reader import CommonRoadFileReader

    from cr_draw_style import (BBOX_INCHES, FIGSIZE, FRAME_DURATION_MS, PAD_INCHES,
                               CustomRenderer, apply_rcparams, build_draw_params,
                               install_icons)

    cr_path = Path(cr_path)
    if not cr_path.exists():
        raise FileNotFoundError(cr_path)

    if out_gif is None:
        out_gif = DATA_BEV_CR / f"{cr_path.stem}.gif"
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    scenario, pps = CommonRoadFileReader(str(cr_path)).open()

    # Determine the timestep horizon from the longest obstacle trajectory.
    max_t = 0
    for obs in scenario.dynamic_obstacles:
        pred = getattr(obs, "prediction", None)
        if pred is not None:
            last = getattr(pred, "final_time_step", None)
            if last is not None and last > max_t:
                max_t = int(last)
    if max_t <= 0:
        raise RuntimeError(
            f"{cr_path.name} has no dynamic obstacle trajectories — "
            f"run osm_pipeline.cr_simulate.simulate() first.")

    stride = max(1, max_t // max(1, n_frames))
    timesteps = list(range(0, max_t + 1, stride))[:n_frames]

    # Hold the view fixed across frames: the lanelet-network bounding box, the
    # same extent main_without_ego records in its coordinate-mapping metadata.
    # Without this, matplotlib refits the axes to whichever obstacles happen to
    # be visible and the scenario jitters (and every frame comes back a
    # slightly different size).
    centers = np.array([v for lanelet in scenario.lanelet_network.lanelets
                        for v in lanelet.center_vertices])
    lo, hi = centers.min(axis=0), centers.max(axis=0)
    plot_limits = [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])]

    apply_rcparams()
    install_icons()

    # size_px is an output-resolution knob only: label sizes are in points, so
    # the look is fixed by FIGSIZE and unaffected by the DPI chosen here.
    dpi = size_px / FIGSIZE[0]
    frames: list = []
    # Reuse a single figure across timesteps for speed.
    fig = plt.figure(figsize=FIGSIZE, dpi=dpi, facecolor="white")
    ax = fig.gca()
    try:
        for t in timesteps:
            ax.clear()
            params = build_draw_params(t)
            rnd = CustomRenderer(ax=ax, plot_limits=plot_limits)
            scenario.draw(rnd, draw_params=params)
            if pps is not None and pps.planning_problem_dict:
                try:
                    for pp in pps.planning_problem_dict.values():
                        pp.draw(rnd)
                except Exception:
                    pass
            rnd.render()
            ax.set_aspect("equal")
            ax.axis("off")
            # Same crop as the database/modified frames, so the scenario fills
            # the frame the same way and the labels read at the same size.
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches=BBOX_INCHES,
                        pad_inches=PAD_INCHES, facecolor="white")
            buf.seek(0)
            frames.append(imageio.imread(buf)[..., :3])
    finally:
        plt.close(fig)

    # The tight bbox is stable now that plot_limits are fixed, but a stray
    # artist outside the axes can still shift it by a pixel, and
    # imageio.mimsave requires identical shapes — pad to the bbox max.
    hs = [f.shape[0] for f in frames]
    ws = [f.shape[1] for f in frames]
    if hs and ws and (len(set(hs)) > 1 or len(set(ws)) > 1):
        H, W = max(hs), max(ws)
        padded = []
        for f in frames:
            h_, w_, c = f.shape
            if h_ == H and w_ == W:
                padded.append(f)
            else:
                pad = np.full((H, W, c), 255, dtype=f.dtype)
                pad[:h_, :w_] = f
                padded.append(pad)
        frames = padded

    # Each frame advances `stride` timesteps, so hold it `stride` times as long
    # as a single-timestep frame to match the database GIFs' playback speed.
    kwargs = {"fps": fps} if fps is not None else {
        "duration": stride * FRAME_DURATION_MS}
    # subrectangles=True keeps the GIF small by only re-encoding changed
    # regions frame-to-frame; palettesize=256 keeps the color palette at
    # the format maximum (cleaner anti-aliased edges).
    imageio.mimsave(out_gif, frames, loop=0, subrectangles=True,
                    palettesize=256, **kwargs)
    return out_gif
