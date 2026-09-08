"""
SUMO simulation specific helper methods
"""

__author__ = "Peter Kocsis, Edmond Irani Liu"
__copyright__ = "TUM Cyber-Physical System Group"
__credits__ = []
__version__ = "0.1"
__maintainer__ = "Edmond Irani Liu"
__email__ = "edmond.irani@tum.de"
__status__ = "Integration"

import copy
import os
import pickle
from enum import unique, Enum
from math import sin, cos
from typing import Tuple, Dict, Optional

import numpy as np
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.common.solution import Solution
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.sumo_simulation import SumoSimulation
from sumocr.scenario.scenario_wrapper import ScenarioWrapper
from sumocr.sumo_config.default import DefaultConfig
from sumocr.sumo_docker.interface.docker_interface import SumoInterface
from sumocr.visualization.video import create_video


@unique
class SimulationOption(Enum):
    WITHOUT_EGO = "_without_ego"
    MOTION_PLANNER = "_planner"
    SOLUTION = "_solution"


def simulate_scenario(mode: SimulationOption,
                      conf: DefaultConfig,
                      scenario_wrapper: ScenarioWrapper,
                      scenario_path: str,
                      num_of_steps: int = None,
                      planning_problem_set: PlanningProblemSet = None,
                      solution: Solution = None,
                      use_sumo_manager: bool = False) -> Tuple[Scenario, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with specified mode

    :param mode: 0 = without ego, 1 = with plugged in planner, 2 = with solution trajectory
    :param conf: config of the simulation
    :param scenario_wrapper: scenario wrapper used by the Simulator
    :param scenario_path: path to the interactive scenario folder
    :param num_of_steps: number of steps to simulate
    :param planning_problem_set: planning problem set of the scenario
    :param solution: solution to the planning problem
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :return: simulated scenario and dictionary with items {planning_problem_id: EgoVehicle}
    """

    scenario_wrapper.get_rou_file()  # applies corrections to route file if necessary

    if num_of_steps is None:
        num_of_steps = conf.simulation_steps

    sumo_interface = None
    if use_sumo_manager:
        sumo_interface = SumoInterface(use_docker=True)
        sumo_sim = sumo_interface.start_simulator()

        sumo_sim.send_sumo_scenario(conf.scenario_name,
                                    scenario_path)
    else:
        sumo_sim = SumoSimulation()

    # initialize simulation
    sumo_sim.initialize(conf, scenario_wrapper, None)

    if mode is SimulationOption.WITHOUT_EGO:
        # simulation without ego vehicle
        for step in range(num_of_steps):
            # set to dummy simulation
            sumo_sim.dummy_ego_simulation = True
            sumo_sim.simulate_step()

    elif mode is SimulationOption.MOTION_PLANNER:
        # simulation with plugged in planner

        def run_simulation():
            ego_vehicles = sumo_sim.ego_vehicles
            for step in range(num_of_steps):
                if use_sumo_manager:
                    ego_vehicles = sumo_sim.ego_vehicles

                # retrieve the CommonRoad scenario at the current time step, e.g. as an input for a prediction module
                current_scenario = sumo_sim.commonroad_scenario_at_time_step(sumo_sim.current_time_step)
                for idx, ego_vehicle in enumerate(ego_vehicles.values()):
                    # retrieve the current state of the ego vehicle
                    state_current_ego = ego_vehicle.current_state

                    # ====== plug in your motion planner here
                    # example motion planner which decelerates to full stop
                    next_state = copy.deepcopy(state_current_ego)
                    next_state.steering_angle = 0.0
                    a = -4.0
                    dt = 0.1
                    if next_state.velocity > 0:
                        v = next_state.velocity
                        x, y = next_state.position
                        o = next_state.orientation

                        next_state.position = np.array([x + v * cos(o) * dt, y + v * sin(o) * dt])
                        next_state.velocity += a * dt
                    # ====== end of motion planner

                    # update the ego vehicle with new trajectory with only 1 state for the current step
                    next_state.time_step = 1
                    trajectory_ego = [next_state]
                    ego_vehicle.set_planned_trajectory(trajectory_ego)

                if use_sumo_manager:
                    # set the modified ego vehicles to synchronize in case of using sumo_docker
                    sumo_sim.ego_vehicles = ego_vehicles

                sumo_sim.simulate_step()

        run_simulation()

    elif mode is SimulationOption.SOLUTION:
        # simulation with given solution trajectory

        def run_simulation():
            ego_vehicles = sumo_sim.ego_vehicles

            for time_step in range(num_of_steps):
                if use_sumo_manager:
                    ego_vehicles = sumo_sim.ego_vehicles
                for idx_ego, ego_vehicle in enumerate(ego_vehicles.values()):
                    # update the ego vehicles with solution trajectories
                    trajectory_solution = solution.planning_problem_solutions[idx_ego].trajectory
                    next_state = copy.deepcopy(trajectory_solution.state_list[time_step])

                    next_state.time_step = 1
                    trajectory_ego = [next_state]
                    ego_vehicle.set_planned_trajectory(trajectory_ego)

                if use_sumo_manager:
                    # set the modified ego vehicles to synchronize in case of using SUMO Manager
                    sumo_sim.ego_vehicles = ego_vehicles

                sumo_sim.simulate_step()

        check_trajectories(solution, planning_problem_set, conf)
        run_simulation()

    # retrieve the simulated scenario in CR format
    simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()

    # stop the simulation
    sumo_sim.stop()
    ego_vehicles = {list(planning_problem_set.planning_problem_dict.keys())[0]:
                        ego_v for _, ego_v in sumo_sim.ego_vehicles.items()}

    if use_sumo_manager:
        sumo_interface.stop_simulator()

    return simulated_scenario, ego_vehicles


def simulate_without_ego(interactive_scenario_path: str,
                         output_folder_path: str = None,
                         create_video: bool = False,
                         use_sumo_manager: bool = False,
                         num_of_steps=None,
                         preserve_trajectories: bool = False,
                         original_scenario_path: str = None) -> Tuple[Scenario, PlanningProblemSet]:
    """
    Simulates an interactive scenario without ego vehicle

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param num_of_steps: max. number of simulated time steps
    :param preserve_trajectories: if True, preserve original trajectories for unmodified vehicles
    :param original_scenario_path: path to original .cr.xml with exact trajectories (required if preserve_trajectories=True)
    :return: Tuple of the simulated scenario and the planning problem set
    """
    # TODO: potentially remove the conf logic, but seems to contain important information
    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    num_of_steps = conf.simulation_steps if num_of_steps is None else num_of_steps
    
    # NEW: Trajectory preservation mode
    if preserve_trajectories and original_scenario_path:
        print(f"🔒 Trajectory preservation mode enabled")
        simulated_scenario_without_ego = simulate_with_trajectory_preservation(
            interactive_scenario_path,
            original_scenario_path,
            conf,
            scenario_wrapper,
            num_of_steps
        )
        simulated_scenario_without_ego.scenario_id = scenario.scenario_id
    else:
        # Original behavior: full SUMO re-simulation
        simulated_scenario_without_ego, _ = simulate_scenario(SimulationOption.WITHOUT_EGO, conf,
                                                              scenario_wrapper,
                                                              interactive_scenario_path,
                                                              num_of_steps=num_of_steps,
                                                              planning_problem_set=planning_problem_set,
                                                              solution=None,
                                                              use_sumo_manager=use_sumo_manager)
        simulated_scenario_without_ego.scenario_id = scenario.scenario_id

    # Change parameters here for visualization -> may not be possible to adjust video parameters
    if create_video:
        create_video_for_simulation(simulated_scenario_without_ego, output_folder_path, planning_problem_set,
                                    {}, SimulationOption.WITHOUT_EGO.value)

    return simulated_scenario_without_ego, planning_problem_set


def simulate_with_trajectory_preservation(interactive_scenario_path: str,
                                         original_scenario_path: str,
                                         conf,
                                         scenario_wrapper,
                                         num_of_steps: int) -> Scenario:
    """
    Simulate while preserving original trajectories for unmodified vehicles.
    
    This creates a hybrid scenario where:
    - Unmodified vehicles follow their EXACT original CommonRoad trajectories
    - Modified vehicles and nearby vehicles are simulated with SUMO
    - Far-away vehicles preserve their original trajectories
    
    Args:
        interactive_scenario_path: Path to modified scenario with SUMO files
        original_scenario_path: Path to original .cr.xml with exact trajectories
        conf: Simulation configuration
        scenario_wrapper: Scenario wrapper for SUMO
        num_of_steps: Number of simulation steps
        
    Returns:
        Hybrid scenario with preserved trajectories + simulated modified/nearby vehicles
    """
    import xml.etree.ElementTree as ET
    from copy import deepcopy
    from collections import defaultdict
    
    print(f"📂 Loading original scenario: {original_scenario_path}")
    original_scenario, _ = CommonRoadFileReader(original_scenario_path).open()
    
    # Helper function to parse route file and extract vehicle info
    def parse_route_file(route_path):
        """Parse route file and return dict of vehicle_id -> {route, vtype_attrs}"""
        vehicle_info = {}
        try:
            tree = ET.parse(route_path)
            root = tree.getroot()
            
            # Parse vTypes
            vtypes = {}
            for vtype in root.findall('vType'):
                vtype_id = vtype.get('id')
                vtypes[vtype_id] = dict(vtype.attrib)
            
            # Parse vehicles
            for vehicle in root.findall('vehicle'):
                vehicle_id = int(vehicle.get('id'))
                route_elem = vehicle.find('route')
                route_edges = route_elem.get('edges', '') if route_elem is not None else ''
                
                # Get vType attributes (either inline or from reference)
                vtype_id = vehicle.get('type')
                if vtype_id and vtype_id in vtypes:
                    vtype_attrs = vtypes[vtype_id]
                else:
                    vtype_attrs = {}
                
                vehicle_info[vehicle_id] = {
                    'route': route_edges,
                    'vtype': vtype_id,
                    'vtype_attrs': vtype_attrs,
                    'vehicle_attrs': dict(vehicle.attrib)
                }
        except Exception as e:
            print(f"⚠️  Error parsing route file {route_path}: {e}")
        
        return vehicle_info
    
    # Parse original route file (from Original folder)
    original_route_path = os.path.join(os.path.dirname(original_scenario_path), 
                                       os.path.basename(original_scenario_path).replace('.cr.xml', '.vehicles.rou.xml'))
    original_routes = parse_route_file(original_route_path)
    
    # Parse modified route file
    modified_routes_path = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.vehicles.rou.xml")
    print(f"📂 Loading modified routes: {modified_routes_path}")
    modified_routes = parse_route_file(modified_routes_path)
    
    # Build the set of edges that actually exist in the modified net. The
    # CommonRoad->SUMO conversion sometimes leaves vehicle routes referencing
    # edges that do not exist, and the modification pipeline REPAIRS those routes
    # (dropping the dangling edges) so SUMO can run at all. When we decide below
    # which vehicles were "modified", a pure repair like this must NOT count as a
    # user change — otherwise every repaired-but-untouched vehicle gets needlessly
    # re-simulated (and SUMO may drop it), disturbing the whole scene even when
    # the user only removed ONE car. Comparing routes by their NET-VALID edges
    # makes a pure repair invisible while still catching genuine reroutes.
    net_edges = set()
    try:
        _net_root = ET.parse(os.path.join(interactive_scenario_path,
                                          f"{conf.scenario_name}.net.xml")).getroot()
        net_edges = {e.get('id') for e in _net_root.findall('edge')
                     if e.get('function') != 'internal'}
    except Exception as e:
        print(f"⚠️  Could not read net edges for route normalisation: {e}")

    def _net_valid_route(route_str):
        edges = (route_str or "").split()
        return [e for e in edges if e in net_edges] if net_edges else edges

    modified_vehicle_ids = set(modified_routes.keys())
    original_vehicle_ids = {obs.obstacle_id for obs in original_scenario.dynamic_obstacles}
    
    print(f"   Original scenario had {len(original_vehicle_ids)} vehicles")
    print(f"   Modified scenario has {len(modified_vehicle_ids)} vehicles")
    
    # Identify changes
    removed_ids = original_vehicle_ids - modified_vehicle_ids
    added_ids = modified_vehicle_ids - original_vehicle_ids
    potentially_preserved_ids = original_vehicle_ids & modified_vehicle_ids
    
    # NEW: Detect which vehicles have modified routes or behavior
    route_modified_ids = set()
    behavior_modified_ids = set()
    
    for vid in potentially_preserved_ids:
        if vid in original_routes and vid in modified_routes:
            orig = original_routes[vid]
            mod = modified_routes[vid]
            
            # Check if route changed — compare by NET-VALID edges so a pure
            # route repair (dropping a non-existent edge) is NOT mistaken for a
            # user reroute. A genuine reroute changes the valid-edge sequence too.
            if _net_valid_route(orig['route']) != _net_valid_route(mod['route']):
                route_modified_ids.add(vid)
            
            # Check if behavior parameters changed (vType attributes)
            # Key behavior attributes: accel, decel, sigma, lcStrategic, etc.
            behavior_attrs = ['accel', 'decel', 'sigma', 'lcStrategic', 'lcSpeedGain', 
                            'lcCooperative', 'speedFactor', 'lcImpatience', 'impatience']
            
            for attr in behavior_attrs:
                orig_val = orig['vtype_attrs'].get(attr)
                mod_val = mod['vtype_attrs'].get(attr)
                if orig_val != mod_val and (orig_val is not None or mod_val is not None):
                    behavior_modified_ids.add(vid)
                    break
    
    modified_ids = route_modified_ids | behavior_modified_ids
    truly_preserved_ids = potentially_preserved_ids - modified_ids
    
    print(f"   🗑️  Removed: {len(removed_ids)} vehicles")
    print(f"   ➕ Added: {len(added_ids)} vehicles")
    print(f"   🔄 Route modified: {len(route_modified_ids)} vehicles")
    print(f"   ⚙️  Behavior modified: {len(behavior_modified_ids)} vehicles")
    print(f"   ✅ Initially preserved: {len(truly_preserved_ids)} vehicles with exact original trajectories")
    
    # ============================================================================
    # PROXIMITY DETECTION DISABLED - Only re-simulate explicitly modified vehicles
    # ============================================================================
    # 
    # NOTE: Proximity detection has been disabled to ensure ONLY explicitly
    # modified vehicles (route changes or behavior changes) are re-simulated.
    # All other vehicles preserve their exact original trajectories.
    #
    # This was done per user requirement: "we want only selected car will be 
    # changed, the other just follow the original preserved trajectory"
    # ============================================================================
    
    # DISABLED: Old proximity detection logic
    # def detect_nearby_vehicles(...): ...
    
    # No nearby vehicles - only explicitly modified ones will be re-simulated
    nearby_ids = set()
    
    print(f"\n   🚫 Proximity detection DISABLED - only explicitly modified vehicles will be re-simulated")
    print(f"   ✅ Final truly preserved: {len(truly_preserved_ids)} vehicles (all unmodified vehicles)")
    print(f"   🔄 Nearby vehicles to re-simulate: {len(nearby_ids)} vehicles (none - proximity disabled)")
    
    # If there are added, modified, or nearby vehicles, we need to run SUMO simulation
    needs_simulation_ids = added_ids | modified_ids | nearby_ids
    
    if len(needs_simulation_ids) > 0:
        print(f"\n🎮 Running SUMO simulation for {len(needs_simulation_ids)} vehicle(s)...")
        print(f"   - {len(added_ids)} added")
        print(f"   - {len(route_modified_ids)} with modified routes")
        print(f"   - {len(behavior_modified_ids)} with modified behavior")
        print(f"   - {len(nearby_ids)} nearby vehicles (proximity detection disabled)")
        
        # Determine simulation duration - use original scenario's duration
        # to ensure preserved vehicles' trajectories align
        max_original_timestep = 0
        for obs in original_scenario.dynamic_obstacles:
            if hasattr(obs.prediction, 'trajectory') and obs.prediction.trajectory:
                last_state = obs.prediction.trajectory.state_list[-1]
                max_original_timestep = max(max_original_timestep, last_state.time_step)
        
        simulation_steps = min(num_of_steps, max_original_timestep + 1)
        print(f"   ⏱️  Limiting simulation to {simulation_steps} steps (original scenario duration)")
        
        # Load the modified scenario file for SUMO simulation
        modified_scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
        modified_scenario, planning_problem_set = CommonRoadFileReader(modified_scenario_file).open()
        
        # Update scenario wrapper with modified scenario
        scenario_wrapper.initial_scenario = modified_scenario
        
        # Run SUMO simulation on the full modified scenario
        try:
            # Import required for simulation
            from sumocr.interface.sumo_simulation import SumoSimulation
            
            sumo_sim = SumoSimulation()
            sumo_sim.initialize(conf, scenario_wrapper, None)
            
            # Run simulation without ego (limited to original scenario duration)
            for step in range(simulation_steps):
                sumo_sim.dummy_ego_simulation = True
                sumo_sim.simulate_step()
            
            # Get the simulated scenario
            simulated_scenario = sumo_sim.commonroad_scenarios_all_time_steps()
            sumo_sim.stop()
            
            print(f"   ✅ SUMO simulation complete: {len(simulated_scenario.dynamic_obstacles)} vehicles")
            
            # Debug: Check vehicle IDs in simulated scenario
            simulated_ids = {obs.obstacle_id for obs in simulated_scenario.dynamic_obstacles}
            print(f"   🔍 DEBUG: Simulated vehicle IDs: {sorted(list(simulated_ids))[:10]}...")
            print(f"   🔍 DEBUG: Need to extract IDs: {sorted(list(needs_simulation_ids))[:10]}...")
            print(f"   🔍 DEBUG: Overlap: {len(simulated_ids & needs_simulation_ids)} vehicles")
            
            # Extract vehicles that need simulation (added + modified + nearby)
            simulated_vehicles = []
            for obstacle in simulated_scenario.dynamic_obstacles:
                if obstacle.obstacle_id in needs_simulation_ids:
                    simulated_vehicles.append(obstacle)
            
            print(f"   ✅ Extracted {len(simulated_vehicles)} simulated vehicle(s):")
            print(f"      - {len([v for v in simulated_vehicles if v.obstacle_id in added_ids])} added")
            print(f"      - {len([v for v in simulated_vehicles if v.obstacle_id in modified_ids])} modified")
            print(f"      - {len([v for v in simulated_vehicles if v.obstacle_id in nearby_ids])} nearby")
            
        except Exception as e:
            print(f"   ⚠️  Warning: SUMO simulation failed: {e}")
            print(f"   Continuing with preserved trajectories only")
            simulated_vehicles = []
    else:
        simulated_vehicles = []
    
    # Create hybrid scenario: Start with preserved original trajectories
    result_scenario = deepcopy(original_scenario)
    
    print(f"\n   🔧 Building hybrid scenario:")
    print(f"      Starting with {len(result_scenario.dynamic_obstacles)} vehicles from original")
    
    # Remove vehicles the user explicitly deleted, plus modified/nearby ones that
    # were SUCCESSFULLY re-simulated (their simulated version is added back below).
    # If a modified/nearby vehicle could NOT be re-simulated — e.g. the SUMO route
    # build failed for some *other* vehicle and took the whole simulation down —
    # keep its ORIGINAL trajectory so it never silently disappears.
    simulated_present_ids = {v.obstacle_id for v in simulated_vehicles}
    obstacles_to_remove = []
    fallback_kept = []
    for obstacle in result_scenario.dynamic_obstacles:
        oid = obstacle.obstacle_id
        if oid in removed_ids:
            obstacles_to_remove.append(obstacle)            # user asked to delete it
        elif oid in modified_ids or oid in nearby_ids:
            if oid in simulated_present_ids:
                obstacles_to_remove.append(obstacle)        # replaced by simulated version
            else:
                fallback_kept.append(oid)                   # re-sim failed -> keep original

    for obstacle in obstacles_to_remove:
        result_scenario.remove_obstacle(obstacle)

    if fallback_kept:
        print(f"      ⚠️  Could not re-simulate vehicle(s) {sorted(fallback_kept)} — keeping "
              f"their ORIGINAL trajectories (the requested change was NOT applied to them; "
              f"enable the repair module to fix the broken routes and re-apply).")

    print(f"      After removing deleted/modified/nearby: {len(result_scenario.dynamic_obstacles)} vehicles")
    
    # Add all simulated vehicles (added + modified + nearby)
    print(f"      Adding {len(simulated_vehicles)} simulated vehicles...")
    for simulated_vehicle in simulated_vehicles:
        print(f"         Adding vehicle {simulated_vehicle.obstacle_id}")
        result_scenario.add_objects(simulated_vehicle)
    
    print(f"      After adding simulated: {len(result_scenario.dynamic_obstacles)} vehicles")
    
    # Verify all expected vehicles are present
    final_ids = {obs.obstacle_id for obs in result_scenario.dynamic_obstacles}
    print(f"      Final vehicle IDs: {sorted(list(final_ids))[:10]}..." if len(final_ids) > 10 else f"      Final vehicle IDs: {sorted(list(final_ids))}")
    
    print(f"\n   📊 Final hybrid scenario:")
    print(f"      - {len(truly_preserved_ids)} preserved vehicles (exact original trajectories) ✅")
    if len(added_ids) > 0:
        print(f"      - {len(added_ids)} added vehicles (SUMO simulated) ✅")
    if len(route_modified_ids) > 0:
        print(f"      - {len(route_modified_ids)} route-modified vehicles (SUMO simulated) ✅")
    if len(behavior_modified_ids) > 0:
        print(f"      - {len(behavior_modified_ids)} behavior-modified vehicles (SUMO simulated) ✅")
    if len(nearby_ids) > 0:
        print(f"      - {len(nearby_ids)} nearby vehicles (proximity detection disabled) ✅")
    print(f"      - {len(result_scenario.dynamic_obstacles)} total vehicles ✅")
    
    return result_scenario


def simulate_with_solution(interactive_scenario_path: str,
                           output_folder_path: str = None,
                           solution: Solution = None,
                           create_video: bool = False,
                           use_sumo_manager: bool = False,
                           create_ego_obstacle: bool = False) \
        -> Tuple[Scenario, PlanningProblemSet, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with a given solution

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param solution: solution to the planning problem
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param create_ego_obstacle: indicates whether to create obstacles as the ego vehicles
    :return: Tuple of the simulated scenario and the planning problem set
    """
    if not isinstance(solution, Solution):
        raise Exception("Solution to the planning problem is not given.")

    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initialize(conf.scenario_name, sumo_cfg_file, scenario_file)
    scenario_with_solution, ego_vehicles = simulate_scenario(SimulationOption.SOLUTION, conf,
                                                             scenario_wrapper,
                                                             interactive_scenario_path,
                                                             num_of_steps=conf.simulation_steps,
                                                             planning_problem_set=planning_problem_set,
                                                             solution=solution,
                                                             use_sumo_manager=use_sumo_manager)
    scenario_with_solution.scenario_id = scenario.scenario_id

    if create_video:
        create_video_for_simulation(scenario_with_solution, output_folder_path, planning_problem_set,
                                    ego_vehicles, SimulationOption.SOLUTION.value)

    if create_ego_obstacle:
        for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            scenario_with_solution.add_objects(obstacle_ego)

    return scenario_with_solution, planning_problem_set, ego_vehicles


def simulate_with_planner(interactive_scenario_path: str,
                          output_folder_path: str = None,
                          create_video: bool = False,
                          use_sumo_manager: bool = False,
                          create_ego_obstacle: bool = False) \
        -> Tuple[Scenario, PlanningProblemSet, Dict[int, EgoVehicle]]:
    """
    Simulates an interactive scenario with a plugged in motion planner

    :param interactive_scenario_path: path to the interactive scenario folder
    :param output_folder_path: path to the output folder
    :param create_video: indicates whether to create a mp4 of the simulated scenario
    :param use_sumo_manager: indicates whether to use the SUMO Manager
    :param create_ego_obstacle: indicates whether to create obstacles from the planned trajectories as the ego vehicles
    :return: Tuple of the simulated scenario, planning problem set, and list of ego vehicles
    """
    conf = load_sumo_configuration(interactive_scenario_path)
    scenario_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.cr.xml")
    scenario, planning_problem_set = CommonRoadFileReader(scenario_file).open()

    scenario_wrapper = ScenarioWrapper()
    scenario_wrapper.sumo_cfg_file = os.path.join(interactive_scenario_path, f"{conf.scenario_name}.sumo.cfg")
    scenario_wrapper.initial_scenario = scenario

    scenario_with_planner, ego_vehicles = simulate_scenario(SimulationOption.MOTION_PLANNER, conf,
                                                            scenario_wrapper,
                                                            interactive_scenario_path,
                                                            num_of_steps=conf.simulation_steps,
                                                            planning_problem_set=planning_problem_set,
                                                            use_sumo_manager=use_sumo_manager)
    scenario_with_planner.scenario_id = scenario.scenario_id

    if create_video:
        create_video_for_simulation(scenario_with_planner, output_folder_path, planning_problem_set,
                                    ego_vehicles, SimulationOption.MOTION_PLANNER.value)

    if create_ego_obstacle:
        for pp_id, planning_problem in planning_problem_set.planning_problem_dict.items():
            obstacle_ego = ego_vehicles[pp_id].get_dynamic_obstacle()
            scenario_with_planner.add_objects(obstacle_ego)

    return scenario_with_planner, planning_problem_set, ego_vehicles


def load_sumo_configuration(interactive_scenario_path: str) -> DefaultConfig:
    with open(os.path.join(interactive_scenario_path, "simulation_config.p"), "rb") as input_file:
        conf = pickle.load(input_file)

    return conf


def check_trajectories(solution: Solution, pps: PlanningProblemSet, config: DefaultConfig):
    assert len(set(solution.planning_problem_ids) - set(pps.planning_problem_dict.keys())) == 0, \
        f"Provided solution trajectories with IDs {solution.planning_problem_ids} don't match " \
        f"planning problem IDs{list(pps.planning_problem_dict.keys())}"

    for s in solution.planning_problem_solutions:
        if s.trajectory.final_state.time_step < config.simulation_steps:
            raise ValueError(f"The simulation requires {config.simulation_steps} "
                             f"states, but the solution only provides"
                             f"{s.trajectory.final_state.time_step} time steps!")


def create_video_for_simulation(scenario_with_planner: Scenario, output_folder_path: str,
                                planning_problem_set: PlanningProblemSet,
                                ego_vehicles: Optional[Dict[int, EgoVehicle]],
                                suffix: str, follow_ego: bool = True):
    """Creates the mp4 animation for the simulation result."""
    if not output_folder_path:
        print("Output folder not specified, skipping mp4 generation.")
        return

    # create mp4 animation
    create_video(scenario_with_planner,
                 output_folder_path,
                 planning_problem_set=planning_problem_set,
                 trajectory_pred=ego_vehicles,
                 follow_ego=follow_ego,
                 suffix=suffix
                 )

    # create gif
    create_video(scenario_with_planner,
                 output_folder_path,
                 planning_problem_set=planning_problem_set,
                 trajectory_pred=ego_vehicles,
                 follow_ego=follow_ego,
                 suffix=suffix,
                 file_type="gif"
                 )