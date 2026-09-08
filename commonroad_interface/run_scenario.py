# run_scenario.py
import argparse
from pathlib import Path
from convert_to_sumo import convert_cr_to_sumo_scenario
from simulate_sumo import run_simulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Path to the input CommonRoad XML file")
    parser.add_argument("--sumo-root", required=True, help="Directory where the SUMO scenario folder will be created")

    args = parser.parse_args()
    input_file = Path(args.input_file).resolve()
    sumo_root = Path(args.sumo_root).resolve()

    scenario_name = input_file.stem
    output_dir = sumo_root / scenario_name

    # Create SUMO config from CR
    if not output_dir.exists():
        print(f"[run_scenario.py] Creating SUMO config for {scenario_name}...")
        convert_cr_to_sumo_scenario(str(input_file), str(sumo_root))

    gif_path = output_dir / f"{scenario_name}_without_ego.gif"
    if not gif_path.exists():
        print(f"[run_scenario.py] Running simulation to generate GIF...")
        run_simulation(str(output_dir), str(output_dir))

    print(f"[run_scenario.py] Simulation complete. GIF saved at: {gif_path}")

if __name__ == "__main__":
    main()
