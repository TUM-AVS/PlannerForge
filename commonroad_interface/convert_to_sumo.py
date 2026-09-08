import os
from lxml import etree

from commonroad.scenario.scenario import Tag
from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader

import sys
import os
import pickle
import argparse
from pathlib import Path
import uuid

# Add the commonroad-scenario-designer to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "commonroad-scenario-designer"))

try:
    from commonroad_sumo.cr2sumo import (
        CR2SumoMapConverter, 
        CR2SumoMapConverterConfig,
        UnsafeResimulationTrafficGenerator
    )
    from commonroad_sumo import SumoTrafficGenerationMode
    from commonroad.scenario.traffic_sign import SupportedTrafficSignCountry
    from sumocr.sumo_config.default import DefaultConfig

    print("✅ CommonRoad SUMO modules loaded successfully")
except ImportError as e:
    print(f"❌ Error importing CommonRoad SUMO: {e}")
    print("💡 Make sure to activate the conda environment first:")
    print("   conda activate cr37")
    print("💡 Also check that commonroad-sumo is installed")
    sys.exit(1)


def create_simulation_config(scenario_name: str,
                             simulation_steps: int = 300,
                             dt: float = 0.1,
                             delta_steps: int = 2,
                             country_id: SupportedTrafficSignCountry = SupportedTrafficSignCountry.ZAMUNDA,
                             with_sumo_gui: bool = False,
                             output_dir: str = None) -> Path:
    """
    Create simulation_config.p file for CommonRoad Interactive Scenarios

    Args:
        scenario_name: Name of the scenario
        simulation_steps: Number of simulation steps (default: 300)
        dt: Simulation time step in seconds (default: 0.1)
        delta_steps: SUMO sub-steps per simulation step (default: 2)
        country_id: Traffic sign country (default: ZAMUNDA)
        with_sumo_gui: Whether to use SUMO GUI (default: False)
        output_dir: Output directory (default: current directory)

    Returns:
        Path to created simulation_config.p file
    """

    print(f"🔧 Creating simulation config for: {scenario_name}")

    try:
        # Create DefaultConfig object (required by sumocr simulation code)
        conf = DefaultConfig()
        conf.scenario_name = scenario_name
        conf.simulation_steps = simulation_steps
        conf.dt = dt
        conf.delta_steps = delta_steps
        conf.with_sumo_gui = True  # Enable GUI mode
        conf.country_id = country_id
        conf.presimulation_steps = 0
        
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = Path(".")

        output_path.mkdir(parents=True, exist_ok=True)
        config_file = output_path / "simulation_config.p"

        with open(config_file, "wb") as f:
            pickle.dump(conf, f)

        print(f"✅ Created simulation config: {config_file}")
        print(f"   📊 Scenario: {conf.scenario_name}")
        print(f"   🎮 Simulation steps: {conf.simulation_steps}")
        print(f"   ⏱️  Time step: {conf.dt}s")
        print(f"   🚦 Delta steps: {conf.delta_steps}")

        return config_file

    except Exception as e:
        print(f"❌ Error creating config file: {e}")
        return None


# The following code takes in a CR scenario file and creates all required SUMO files except for the .p file
# Currently from_trajectories = True in the crdesigner package of the cr37 conda environment

import os
from pathlib import Path
from commonroad.scenario.scenario import Scenario  # Example import, adjust if needed
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader
import xml.etree.ElementTree as ET


def fix_traffic_light_timing(scenario: Scenario, output_folder: str, target_name: str):
    """
    Fix traffic light timing synchronization between CommonRoad and SUMO.
    
    CRITICAL FIX: commonroad-sumo converter creates traffic lights that are ~10x too slow!
    It doesn't properly convert durations from timesteps to seconds.
    
    CommonRoad: 235 timesteps × 0.1s = 23.5 seconds cycle
    SUMO (wrong): 220 seconds cycle (should be ~23s!)
    
    This function scales down SUMO traffic light durations to match CommonRoad timing.
    
    Args:
        scenario: CommonRoad scenario object
        output_folder: Path to folder containing generated SUMO files
        target_name: Name of the scenario
    """
    net_file = os.path.join(output_folder, f"{target_name}.net.xml")
    cr_xml_file = os.path.join(output_folder, f"{target_name}.cr.xml")
    
    if not os.path.exists(net_file) or not os.path.exists(cr_xml_file):
        print(f"   ⚠️  Warning: Required files not found, skipping traffic light sync")
        return
    
    # Extract traffic light cycles from CommonRoad XML
    traffic_lights = {}
    
    try:
        cr_tree = ET.parse(cr_xml_file)
        cr_root = cr_tree.getroot()
        
        for tl in cr_root.iter():
            if tl.tag.endswith('trafficLight') or tl.tag == 'trafficLight':
                tl_id = tl.get('id')
                if not tl_id:
                    continue
                
                phases = []
                time_offset = 0
                
                for cycle in tl:
                    if cycle.tag.endswith('cycle') or cycle.tag == 'cycle':
                        for elem in cycle:
                            if elem.tag.endswith('timeOffset'):
                                time_offset = int(elem.text) * 0.1  # timesteps → seconds
                            elif elem.tag.endswith('cycleElement'):
                                duration = color = None
                                for sub in elem:
                                    if sub.tag.endswith('duration'):
                                        duration = int(sub.text) * 0.1  # timesteps → seconds
                                    elif sub.tag.endswith('color'):
                                        color = sub.text
                                if duration and color:
                                    phases.append((color, duration))
                
                if phases:
                    total_duration = sum(d for _, d in phases)
                    traffic_lights[tl_id] = {
                        'offset': time_offset,
                        'phases': phases,
                        'cycle_duration': total_duration
                    }
                    print(f"   📍 TL {tl_id}: cycle={total_duration:.1f}s, offset={time_offset:.1f}s")
    
    except Exception as e:
        print(f"   ⚠️  Could not parse traffic lights: {e}")
        return
    
    if not traffic_lights:
        print(f"   ℹ️  No traffic lights in scenario")
        return
    
    # Parse SUMO net.xml
    tree = ET.parse(net_file)
    root = tree.getroot()
    tl_logics = root.findall('.//tlLogic')
    
    if not tl_logics:
        return
    
    # Fix SUMO traffic light durations (scale down from incorrect 10x timing)
    sample_tl = next(iter(traffic_lights.values()))
    target_duration = sample_tl['cycle_duration']
    
    # CRITICAL FIX: Choose the offset that best matches CommonRoad traffic light states
    # When CommonRoad has multiple traffic lights with different offsets at a junction,
    # SUMO merges them into one controller. We need to pick the offset that results in
    # correct states for all links at t=0.
    #
    # Strategy: Use the maximum offset, as it often represents the offset that puts
    # the perpendicular traffic (which had min offset) in the RED phase at t=0
    all_offsets = [tl['offset'] for tl in traffic_lights.values()]
    chosen_offset = max(all_offsets) if all_offsets else 0
    
    print(f"   📊 CommonRoad traffic light offsets: {[f'{o:.1f}s' for o in all_offsets]}")
    print(f"   ✅ Chose offset: {chosen_offset:.1f}s (max offset for proper phase alignment)")
    
    for tl_logic in tl_logics:
        junction_id = tl_logic.get('id')
        current_duration = sum(int(phase.get('duration')) for phase in tl_logic.findall('phase'))
        
        # If SUMO cycle is >5x longer than CommonRoad, fix it
        if current_duration > target_duration * 5:
            scale_factor = target_duration / current_duration
            print(f"   🔧 Junction {junction_id}: Fixing 10x timing issue")
            print(f"      Was: {current_duration}s → Now: {target_duration:.1f}s")
            
            for phase in tl_logic.findall('phase'):
                old_dur = int(phase.get('duration'))
                new_dur = max(1, int(round(old_dur * scale_factor)))
                phase.set('duration', str(new_dur))
        
        # Set the chosen offset
        current_offset = float(tl_logic.get('offset', '0'))
        if abs(chosen_offset - current_offset) > 0.5:  # Only update if significantly different
            tl_logic.set('offset', str(int(round(chosen_offset))))
            print(f"   🔧 Junction {junction_id}: Updated offset {current_offset:.1f}s → {chosen_offset:.1f}s")
    
    tree.write(net_file, encoding='UTF-8', xml_declaration=True)
    print(f"   ✅ Traffic lights synchronized with CommonRoad timing")


def convert_cr_to_sumo_scenario(scenario_path: str, target_name: str, collection_folder: str):
    """
    Converts a CommonRoad scenario to a SUMO scenario, including the simulation config (.p file),
    and stores the output in a subdirectory named after the scenario (without .xml).

    Parameters:
        scenario_path (str): Full path to the .xml scenario file. Example format: <repo>/DEU_Damme-17_1_T-1.xml
        target_name (str): Resulting name of the conversion. Example format: DEU_Damme-17_1_T-1_M4444 (when converting a non-modified, raw scenario, this will simply be the file name without suffix)
        collection_folder (str): Path to the folder where the converted scenario should be stored.

    With the above example, the converted files would be stored in collection_folder/target_name
    """
    print(f"🔄 Starting conversion of scenario: {scenario_path}")
    # TODO: add check so that a .cr.xml file can also be processed as usual --> should work now

    # Extract the scenario name without the .xml extension
    # scenario_name = Path(scenario_path).stem.removesuffix(".xml")
    # print(f"📛 Scenario name: {scenario_name}")

    # Create the output subdirectory
    # output_folder = os.path.join(collection_folder, scenario_name)
    # os.makedirs(output_folder, exist_ok=True)
    # print(f"📁 Created output folder: {output_folder}")

    # TODO: rm if rm target_name
    output_folder = collection_folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Created output folder: {output_folder}")

    # Copy the original .xml file to the new folder with a .cr.xml suffix
    # TODO: move this part maybe elsewhere and use it for CR modification; also: after simulation, replace this file with the simulated result (maybe?)
    # cr_xml_path = os.path.join(output_folder, scenario_name + ".cr.xml")
    # print(f"📄 Copying scenario file to: {cr_xml_path}")
    # with open(scenario_path, "r") as src, open(cr_xml_path, "w") as dst:
    #     dst.write(src.read())
    # print(f"✅ Scenario file copied and renamed to .cr.xml")

    # TODO: rm if rm target_name
    cr_xml_path = os.path.join(output_folder, target_name + ".cr.xml")
    print(f"📄 Copying scenario file to: {cr_xml_path}")
    with open(scenario_path, "r") as src, open(cr_xml_path, "w") as dst:
        dst.write(src.read())
    print(f"✅ Scenario file copied and renamed to .cr.xml")

    # Load the scenario using CommonRoadFileReader
    print(f"📥 Loading scenario from: {cr_xml_path}")
    scenario, planning_problem = CommonRoadFileReader(cr_xml_path).open()
    print(f"✅ Scenario and planning problems loaded")
    
    # Store original scenario ID
    original_scenario_id = str(scenario.scenario_id)
    print(f"   Original scenario ID: {original_scenario_id}")
    
    # Temporarily update scenario ID for SUMO file naming
    scenario.scenario_id = target_name
    print(f"   Updated scenario ID for SUMO conversion: {target_name}")

    # Convert to SUMO format
    print(f"🔁 Converting scenario to SUMO format...")
    # Create SUMO config and converter with new API
    config = CR2SumoMapConverterConfig(
        highway_mode=True,
        country_id=SupportedTrafficSignCountry.ZAMUNDA
    )
    converter = CR2SumoMapConverter(scenario, config)
    # Create SUMO files (net.xml, sumo.cfg, etc.)
    sumo_project = converter.create_sumo_files(output_folder=Path(output_folder), cleanup_tmp_files=True)
    if sumo_project:
        print(f"✅ Scenario converted to SUMO (network)")
        print(f"   SUMO project created at: {output_folder}")
    else:
        print(f"❌ SUMO conversion failed")
        return
    
    # Fix traffic light timing synchronization
    print(f"🚦 Synchronizing traffic light timing with CommonRoad...")
    fix_traffic_light_timing(scenario, output_folder, target_name)
    print(f"✅ Traffic light timing synchronized")
    
    # Generate traffic from CommonRoad trajectories
    print(f"🚗 Generating traffic from CommonRoad trajectories...")
    traffic_generator = UnsafeResimulationTrafficGenerator()
    success = traffic_generator.generate_traffic(scenario, sumo_project)
    if success:
        print(f"✅ Traffic generated successfully")
        print(f"   Traffic files created in: {output_folder}")
    else:
        print(f"❌ Traffic generation failed")
        return
    
    # Restore original scenario ID in the .cr.xml file for planner compatibility
    scenario.scenario_id = original_scenario_id
    print(f"   Restored original scenario ID: {original_scenario_id}")
    
    # Re-save the .cr.xml file with original scenario ID
    from commonroad.common.file_writer import CommonRoadFileWriter, OverwriteExistingFile
    fw = CommonRoadFileWriter(scenario, planning_problem, "PlannerForge", "TUM", "SUMO Conversion")
    fw.write_to_file(cr_xml_path, OverwriteExistingFile.ALWAYS)
    print(f"   ✅ Saved .cr.xml with original scenario ID for planner")

    # Create the simulation config including the .p file (required for simulation)
    print(f"🛠️  Generating simulation configuration file...")
    config_path = create_simulation_config(target_name, output_dir=output_folder)
    if config_path:
        print(f"✅ Simulation config created at: {config_path}")
    else:
        print(f"❌ Failed to create simulation config")

# TODO: may need to be adjusted to above method and target name
def convert_all_cr_scenarios_in_folder(input_folder: str, output_folder: str):
    """
    Converts all CommonRoad .xml scenarios in a given input folder to SUMO format.
    Each scenario will be processed individually, and any failures will be logged without stopping the entire process.

    Parameters:
        input_folder (str): Folder containing CommonRoad .xml scenario files.
        output_folder (str): Destination folder to store converted scenarios.
    """
    print(f"📁 Starting batch conversion of scenarios from: {input_folder}")
    print(f"📤 Output folder for converted scenarios: {output_folder}")

    # Make sure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Counter for summary
    total = 0
    success = 0
    failed = []

    # Iterate over all XML files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".xml"):
            total += 1
            scenario_path = os.path.join(input_folder, file)
            print(f"\n🔄 [{total}] Converting: {file}")
            target_name = Path(scenario_path).name.removesuffix(".xml")

            try:
                convert_cr_to_sumo_scenario(scenario_path, target_name, output_folder)
                success += 1
                print(f"✅ Successfully converted: {file}")
            except Exception as e:
                print(f"❌ Failed to convert: {file}")
                print(f"   🧨 Error: {e}")
                failed.append(file)

    # Summary
    print("\n📊 Conversion Summary:")
    print(f"   ✅ Successful conversions: {success}/{total}")
    if failed:
        print(f"   ❌ Failed scenarios ({len(failed)}):")
        for f in failed:
            print(f"      - {f}")
    else:
        print("   🎉 All scenarios converted successfully!")

# convert_cr_to_sumo_scenario("<repo>/DEU_Damme-17_1_T-1.xml", "DEU_Damme-17_1_T-1_M4444", "<repo>/converted_scenarios")
# convert_all_cr_scenarios_in_folder("<repo>/data/raw_scenarios", "<repo>/data/run_scenarios")
