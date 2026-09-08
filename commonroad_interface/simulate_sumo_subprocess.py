# This file starts a subprocess for running a simulation and creating a GIF in a specific conda environment

import argparse
from pathlib import Path
from simulate_sumo import run_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", required=True, help="Path to the folder where the SUMO configuration files are located")
    parser.add_argument("--preserve_trajectories", action="store_true", 
                       help="Preserve exact trajectories for unmodified vehicles")

    args = parser.parse_args()
    folder_path = Path(args.folder_path)

    run_simulation(folder_path, folder_path, preserve_trajectories=args.preserve_trajectories)

if __name__ == "__main__":
    main()
