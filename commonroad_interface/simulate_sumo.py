#!/usr/bin/env python3
"""
Manual Simulation Runner for CommonRoad Scenarios

This script runs the simulation manually when SUMO Python bindings are available.
Usage: python run_simulation.py <scenario_folder>

This file is only concerned with simulation without the ego vehicle, which requires SUMO in the backend. Simulation using Frenetix or a custom motion planner is implemented elsewhere, since the environments have different requirements. Additionally, it leads to a cleaner logic - this script is for simulating the trajectories of other cars with the help of SUMO, while a motion planner deals with solving the scenario from these created files.
"""

import os
import sys
import argparse
from pathlib import Path


def run_simulation(scenario_folder, output_directory, preserve_trajectories=False):
    """
    Run simulation for a given scenario folder
    
    Args:
        scenario_folder: Path to the scenario folder
        output_directory: Path to save output files
        preserve_trajectories: If True, preserve exact trajectories for unmodified vehicles
    """

    print(f"🎮 Running simulation for: {scenario_folder}")

    try:
        # Add parent directory to path to import commonroad_interface
        parent_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(parent_dir))

        # Import simulation functions
        from commonroad_interface.simulation.simulations import simulate_without_ego
        from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
        from commonroad.scenario.scenario import Tag

        # Demo attributes for saving the simulated scenarios
        author = 'CommonRoad Converter'
        affiliation = 'Technical University of Munich, Germany'
        source = 'Converted from CommonRoad to SUMO'
        tags = {Tag.URBAN}

        # Get scenario name from folder
        scenario_name = Path(scenario_folder).name

        # Create videos directory
        # videos_dir = os.path.join(scenario_folder, "videos")
        # videos_dir = Path("<repo>")
        videos_dir = Path(output_directory)
        # Adjust this directory to use relative path from current working directory
        # videos_dir = Path("<repo>/before_plots")
        os.makedirs(videos_dir, exist_ok=True)

        # NEW: Detect if this is a modified scenario and find original
        original_scenario_path = None
        if preserve_trajectories:
            scenario_path = Path(scenario_folder)
            # Check if this is a Modified scenario
            # Pattern: Scenarios/DEU_XXX/Modified/DEU_XXX_PXXXX/
            if "Modified" in scenario_path.parts:
                # Extract base scenario name (DEU_XXX)
                try:
                    scenario_parts = list(scenario_path.parts)
                    modified_idx = scenario_parts.index("Modified")
                    if modified_idx >= 2:
                        # Go up to Scenarios/DEU_XXX/
                        base_path = Path(*scenario_parts[:modified_idx])
                        # Preserve trajectories from the IMMEDIATE PARENT — the
                        # scenario this modification was applied to — NOT the root
                        # original. Strip the last "_<suffix>" to get the parent
                        # name, then resolve it as a chained modification under
                        # Modified/ first, falling back to the root under Original/.
                        # E.g.  DEU_Weimar-71_1_T-4_P5694        -> Original/DEU_Weimar-71_1_T-4
                        #       DEU_Weimar-13_1_T-4_2217_P3401    -> Modified/DEU_Weimar-13_1_T-4_2217
                        # Using the ROOT as the baseline would re-flag EVERY earlier
                        # modification (reroutes, repairs, ...) as "changed" and
                        # re-simulate the whole scene on every step — so removing a
                        # single car would disturb all the others.
                        full_scenario_name = scenario_parts[modified_idx + 1]
                        parent_name = '_'.join(full_scenario_name.split('_')[:-1])
                        parent_candidates = [
                            base_path / "Modified" / parent_name / f"{parent_name}.cr.xml",
                            base_path / "Original" / f"{parent_name}.cr.xml",
                        ]
                        original_scenario_path = next(
                            (p for p in parent_candidates if p.exists()), None)

                        if original_scenario_path is not None:
                            print(f"🔍 Preserving trajectories from parent: {original_scenario_path}")
                        else:
                            print(f"⚠️  Parent scenario not found (tried: "
                                  f"{[str(p) for p in parent_candidates]})")
                            print(f"   Trajectory preservation disabled")
                            preserve_trajectories = False
                            original_scenario_path = None
                except Exception as e:
                    print(f"⚠️  Could not determine original scenario path: {e}")
                    print(f"   Trajectory preservation disabled")
                    preserve_trajectories = False
                    original_scenario_path = None

        # Run simulation without ego vehicle
        if preserve_trajectories:
            print("🔒 Running simulation with trajectory preservation...")
        else:
            print("🎮 Running simulation without ego vehicle...")
        
        # Simulate without ego should be adjusted to return labeled plot
        scenario_without_ego, pps = simulate_without_ego(
            interactive_scenario_path=scenario_folder,
            output_folder_path=str(videos_dir),
            create_video=False,
            preserve_trajectories=preserve_trajectories,
            original_scenario_path=str(original_scenario_path) if original_scenario_path else None
        )

        # 🔁 Rename generated video files
        # scenario_name = videos_dir.name

        # Define expected extensions
        #for ext in ['.gif', '.mp4']:
        #    # Find the first file with this extension (assuming only one output of each type)
        #    for file in videos_dir.glob(f"*.{ext.lstrip('.')}"):
        #        new_name = videos_dir / f"{scenario_name}_without_ego{ext}"
        #        file.rename(new_name)
        #        print(f"✅ Renamed {file.name} → {new_name.name}")

        # Write simulated scenario to CommonRoad XML file
        # This is important, because the created file here will be used for the motion planner
        # Stores the created file as .xml and not .cr.xml --> allows for distinction
        fw = CommonRoadFileWriter(scenario_without_ego, pps, author, affiliation, source, tags)
        # simulated_xml_path = os.path.join(output_directory, f"{scenario_name}_without_ego.xml")
        simulated_xml_path = os.path.join(output_directory, f"{scenario_name}.xml")
        fw.write_to_file(simulated_xml_path, OverwriteExistingFile.ALWAYS)

        # print(f"✅ Saved simulated scenario: {simulated_xml_path}")
        print(f"   📊 Scenario with {len(scenario_without_ego.dynamic_obstacles)} dynamic obstacles")
        print(f"   🎯 Planning problem with {len(pps.planning_problem_dict)} planning problems")
        # print(f"   🎬 Video saved to: {videos_dir}/")

        return True

    except ImportError as e:
        if "libsumo" in str(e):
            print(f"❌ SUMO Python bindings not available")
            print("💡 Install SUMO Python bindings:")
            print("   pip install sumo")
            print("   # or build from source with Python bindings")
        else:
            print(f"❌ Import error: {e}")
        return False

    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        raise Exception(e)
        return False

# Do not use this for now - If directly used after conversion, scenarios will still contain disappearing obstacles -> The middle step of generating full trajectories has not been done yet
def run_simulations_for_all_scenarios(base_input_folder: str):
    """
    Runs simulation on all scenario folders in the base input folder.
    Each folder should contain a converted CommonRoad scenario.
    The created GIF and simulated .xml file will be stored in the folder of each converted scenario.
    Typically, you would use this to manually supply GIFs to the already converted scenarios in the data/run_scenarios folder.

    Parameters:
        base_input_folder (str): Path to the parent folder containing scenario subfolders.

    There are multiple reasons why an already converted scenario might fail here. The most likely one is that the scenario does not actually contain any dynamic obstacles, in which case the simulation is unable to run.
    """
    print(f"📁 Starting batch simulation from folder: {base_input_folder}")

    total = 0
    success = 0
    failed = []

    # Each subfolder is expected to be a scenario folder
    for entry in os.scandir(base_input_folder):
        if entry.is_dir():
            scenario_folder = entry.path
            total += 1
            print(f"\n🎮 [{total}] Running simulation for: {entry.name}")

            try:
                ok = run_simulation(scenario_folder, scenario_folder)
                if ok:
                    success += 1
                    print(f"✅ Simulation complete for: {entry.name}")
                else:
                    failed.append(entry.name)
                    print(f"❌ Simulation failed for: {entry.name}")
            except Exception as e:
                failed.append(entry.name)
                print(f"❌ Unexpected error while simulating {entry.name}: {e}")

    # Summary
    print("\n📊 Simulation Summary:")
    print(f"   ✅ Successful simulations: {success}/{total}")
    if failed:
        print(f"   ❌ Failed simulations ({len(failed)}):")
        for f in failed:
            print(f"      - {f}")
    else:
        print("   🎉 All simulations completed successfully!")

# run_simulation("<repo>/data/run_scenarios/DEU_Damme-17_1_T-1", "<repo>/data/run_scenarios/DEU_Damme-17_1_T-1")
# run_simulations_for_all_scenarios("<repo>/data/run_scenarios")