# This file starts a subprocess in a specific conda environment for the scenario conversion
# This process is called, if a scenario GIF is requested by the user, but the folder of the scenario does not exist (it has not been converted to a CR/SUMO project folder yet). It creates the folder in the specified output directory (sumo-root).

import argparse
from pathlib import Path
from convert_to_sumo import convert_cr_to_sumo_scenario

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Path to the input CommonRoad XML file")
    parser.add_argument("--target-name", required=True, help="Name of the resulting simulation")
    parser.add_argument("--sumo-root", required=True, help="SUMO directory")

    args = parser.parse_args()
    input_file = Path(args.input_file).resolve()
    target_name = str(args.target_name)
    sumo_root = Path(args.sumo_root).resolve()

    convert_cr_to_sumo_scenario(str(input_file), target_name, str(sumo_root))

if __name__ == "__main__":
    main()
