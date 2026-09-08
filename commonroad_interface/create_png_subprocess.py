# This file starts a subprocess for creating a PNG in a specific conda environment
# Use the cr37 environment for this

import argparse
from pathlib import Path
from create_png import visualize_png

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Path to the .xml file to be visualized")
    parser.add_argument("--target-name", required=True, help="Name of the generated PNG image (without suffix)")
    parser.add_argument("--output-folder", required=True, help="Path to the folder where the image should be stored (typically Scenarios/ScenarioName/Original)")

    args = parser.parse_args()
    input_file = args.input_file
    target_name = args.target_name
    output_folder = args.output_folder

    visualize_png(input_file, target_name, output_folder)

if __name__ == "__main__":
    main()

# python create_png_subprocess.py --input-file <repo>/Scenarios/ZAM_Tjunction-1_55_T-1/Original/ZAM_Tjunction-1_55_T-1.xml --target-name ZAM_Tjunction-1_55_T-1 --output-folder <repo>/Scenarios/ZAM_Tjunction-1_55_T-1/Original