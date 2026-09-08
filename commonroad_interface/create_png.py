import os
import sys

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from commonroad.common.file_reader import CommonRoadFileReader
from pathlib import Path

# Styling is shared with the scenario GIFs (see cr_draw_style) so the preview
# PNG and the animation of the same scenario look the same. This module is run
# as a subprocess script, so make sure the repo root is importable.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cr_draw_style import (BBOX_INCHES, FIGSIZE, PAD_INCHES, CustomRenderer,
                           apply_rcparams, build_draw_params, install_icons)

def visualize_png(file_path: str, target_name: str, output_folder: str) -> str:
    """
    Visualizes a CommonRoad scenario and planning problem, then saves the figure as a PNG.

    Parameters:
        file_path (str): Path to the CommonRoad XML scenario file.
        target_name (str): Desired base name of the output image file (without extension).
        output_folder (str): Folder where the output image will be saved.

    Returns:
        str: Path to the saved PNG image.
    """
    print(f"\n📄 Loading scenario file: {file_path}")

    try:
        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()
        print(f"✅ Scenario and planning problems loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load scenario: {e}")
        raise

    # Set up the plot
    print("🖌️  Initializing renderer and draw parameters...")
    apply_rcparams()
    install_icons()
    fig = plt.figure(figsize=FIGSIZE)
    rnd = CustomRenderer(ax=fig.gca())
    draw_params = build_draw_params(0)
    print("🎨 Draw parameters set from cr_draw_style")

    try:
        print("🧭 Drawing scenario and planning problem set...")
        scenario.draw(rnd, draw_params=draw_params)
        planning_problem_set.draw(rnd)
        rnd.render()
        print("✅ Rendering complete")
    except Exception as e:
        print(f"❌ Failed to render scenario: {e}")
        raise

    # Save the figure
    try:
        save_path = Path(output_folder) / f"{target_name}.png"
        plt.savefig(save_path, bbox_inches=BBOX_INCHES, pad_inches=PAD_INCHES)
        print(f"💾 Saved visualization to: {save_path}")
        return str(save_path)
    except Exception as e:
        print(f"❌ Failed to save visualization: {e}")
        raise


# visualize("<repo>/Scenarios/USA_US101-23_1_T-1/Original/USA_US101-23_1_T-1.xml", "USA_US101-23_1_T-1", "<repo>/Scenarios/USA_US101-23_1_T-1/Original/")