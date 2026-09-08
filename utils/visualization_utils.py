import os

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.visualization.mp_renderer import MPDrawParams
import numpy as np
from pathlib import Path


def visualize(file_path: str) -> str:
    scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

    plt.figure(figsize=(50, 20)) # Originally 25, 10
    rnd = MPRenderer()
    draw_params = MPDrawParams()

    # Testing visualization possibilities
    draw_params.dynamic_obstacle.show_label = True
    draw_params.dynamic_obstacle.draw_icon = True
    draw_params.dynamic_obstacle.draw_shape = True

    scenario.draw(rnd, draw_params=draw_params)
    planning_problem_set.draw(rnd)
    rnd.render(show=True)

    scenario_name = Path(file_path).name.removesuffix(".xml")

    # Ensure the directory exists
    image_db = Path.cwd() / "data" / "images"
    # image_db.mkdir(parents=True, exist_ok=True)

    # Save the figure
    save_path = image_db / f"{scenario_name}.png"
    plt.savefig(save_path)

    return str(save_path)


# Stores the plots of a folder of unaltered CR scenarios to ImageDB/
# Saved with the .xml.png identifier; double ending may look ugly, but simplifies string handling for overall program
from pathlib import Path
import os

def visualize_folder(folder_path: str):
    """Visualize all files in a folder with debug output and error handling."""

    print(f"🖼️  Visualizing folder: {folder_path}")

    folder = Path(folder_path)

    if not folder.exists():
        print(f"❌ Folder does not exist: {folder_path}")
        return

    try:
        files = os.listdir(folder)
        if not files:
            print(f"⚠️  No files found in folder: {folder_path}")
            return

        for file in files:
            full_path = folder / file
            try:
                print(f"🔍 Visualizing: {full_path}")
                visualize(str(full_path))
            except Exception as e:
                print(f"❌ Failed to visualize {full_path}: {e}")

    except Exception as e:
        print(f"❌ Error reading folder {folder_path}: {e}")


# visualize("<repo>/data/raw_scenarios/DEU_Damme-17_1_T-1.xml")
# visualize_folder("<repo>/data/raw_scenarios")