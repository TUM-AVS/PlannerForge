"""
Render a BEV (top-down) PNG of a SUMO .net.xml by plotting its edges.

Uses `sumolib.net.readNet`. sumolib lives inside the eclipse-sumo package at
<sumo>/tools/sumolib and isn't directly importable, so we inject the tools
dir into sys.path at call time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from config import DATA_BEV_SUMO


def _ensure_sumolib():
    try:
        import sumolib  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import sumo
        tools_dir = os.path.join(os.path.dirname(sumo.__file__), "tools")
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
    except ImportError as e:
        raise RuntimeError(
            "eclipse-sumo not installed — `pip install eclipse-sumo`"
        ) from e
    import sumolib  # noqa: F401  retry with updated sys.path


def record_gif_from_cr_sim(net_path: Path,
                           cr_sim_xml: Path,
                           out_gif: Path | None = None,
                           n_frames: int = 200,
                           fps: int = 10,
                           size_px: int = 1600,
                           edge_color: str = "#444",
                           vehicle_color: str = "#2255e3",
                           bg: str = "white") -> Path:
    """Render a SUMO-style GIF using the CR sim trajectories on the
    SUMO `.net.xml` background. Equivalent in content to a sumo-gui
    replay screenshot loop, but generated entirely with matplotlib —
    no async screenshot waits, no risk of sumo-gui timing issues.

    Each GIF frame renders the SUMO net edges (gray) plus a colored
    rectangle per vehicle at its CR-recorded position/orientation."""
    _ensure_sumolib()
    import sumolib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle as MplRect
    from matplotlib.transforms import Affine2D
    import numpy as np
    import imageio.v2 as imageio
    from commonroad.common.file_reader import CommonRoadFileReader

    net_path = Path(net_path)
    cr_sim_xml = Path(cr_sim_xml)
    if not net_path.exists():
        raise FileNotFoundError(net_path)
    if not cr_sim_xml.exists():
        raise FileNotFoundError(cr_sim_xml)
    if out_gif is None:
        out_gif = DATA_BEV_SUMO / (cr_sim_xml.stem + ".gif")
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    net = sumolib.net.readNet(str(net_path))
    edges = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    for edge in net.getEdges():
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        edges.append((xs, ys))
        xmin = min(xmin, *xs); xmax = max(xmax, *xs)
        ymin = min(ymin, *ys); ymax = max(ymax, *ys)
    pad = 20.0
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    sc, _ = CommonRoadFileReader(str(cr_sim_xml)).open()
    obstacles = list(sc.dynamic_obstacles)
    max_t = 0
    for o in obstacles:
        pred = getattr(o, "prediction", None)
        ft = int(getattr(pred, "final_time_step", 0) or 0)
        if ft > max_t:
            max_t = ft

    n = min(int(n_frames), max_t + 1)
    if n <= 0:
        raise RuntimeError("CR sim has no trajectory frames to render")

    dpi = 150
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi),
                           dpi=dpi, facecolor=bg)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")

    # Pre-draw the static SUMO network once.
    for xs, ys in edges:
        ax.plot(xs, ys, color=edge_color, linewidth=0.7)

    frame_imgs = []
    for t in range(n):
        # Remove previous vehicle patches; keep edges.
        for p in list(ax.patches):
            p.remove()
        for o in obstacles:
            st = o.state_at_time(t)
            if st is None:
                continue
            x = float(st.position[0]); y = float(st.position[1])
            yaw = float(getattr(st, "orientation", 0.0) or 0.0)
            try:
                length = float(o.obstacle_shape.length)
                width  = float(o.obstacle_shape.width)
            except Exception:
                length, width = 4.5, 2.0
            rect = MplRect((-length / 2, -width / 2), length, width,
                           color=vehicle_color, alpha=0.85, ec="black",
                           linewidth=0.3)
            tr = (Affine2D().rotate(yaw).translate(x, y) +
                  ax.transData)
            rect.set_transform(tr)
            ax.add_patch(rect)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(),
                            dtype=np.uint8).reshape(h, w, 4)
        frame_imgs.append(buf[:, :, :3].copy())
    plt.close(fig)

    imageio.mimsave(out_gif, frame_imgs, fps=fps, loop=0,
                    subrectangles=True, palettesize=256)
    return out_gif


def render(net_path: Path,
           out_path: Path | None = None,
           size_px: int = 2048,
           bg: str = "white",
           edge_color: str = "black",
           line_width: float = 0.6) -> Path:
    """Render `net_path` (.net.xml) as a top-down PNG. Returns the output path."""
    _ensure_sumolib()
    import sumolib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    net_path = Path(net_path)
    if not net_path.exists():
        raise FileNotFoundError(net_path)

    if out_path is None:
        out_path = DATA_BEV_SUMO / (net_path.stem.replace(".net", "") + ".png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    net = sumolib.net.readNet(str(net_path))

    dpi = 200
    fig, ax = plt.subplots(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax.set_facecolor(bg)
    fig.set_facecolor(bg)

    for edge in net.getEdges():
        shape = edge.getShape()
        if len(shape) < 2:
            continue
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        ax.plot(xs, ys, color=edge_color, linewidth=line_width)

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor=bg)
    plt.close(fig)
    return out_path
