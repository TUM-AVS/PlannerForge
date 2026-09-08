import copy
from typing import Dict

from IPython import display
from commonroad.common.solution import PlanningProblemSolution, Solution, CommonRoadSolutionWriter, VehicleType, \
    VehicleModel, CostFunction
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import Trajectory
from commonroad.visualization.mp_renderer import MPRenderer
from sumocr.interface.ego_vehicle import EgoVehicle
from commonroad.visualization.draw_params import MPDrawParams


def visualize_scenario_with_trajectory(scenario: Scenario,
                                       planning_problem_set: PlanningProblemSet,
                                       ego_vehicles: Dict[int, EgoVehicle] = None,
                                       discrete_time_step: bool = False,
                                       num_time_steps: int = None) -> None:
    if ego_vehicles is not None:
        ego_vehicles = [e.get_dynamic_obstacle() for _, e in ego_vehicles.items()]
    if not num_time_steps:
        if ego_vehicles:
            num_time_steps = ego_vehicles[0].prediction.final_time_step
        else:
            num_time_steps = 50

    # visualize scenario
    for i in range(0, num_time_steps):
        if not discrete_time_step:
            display.clear_output(wait=True)
        rnd = MPRenderer()
        draw_params = MPDrawParams()
        draw_params.time_begin = i
        scenario.draw(rnd, draw_params=draw_params)
        planning_problem_set.draw(rnd)

        if ego_vehicles:
            draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "green"
            rnd.draw_list(ego_vehicles,
                          draw_params=draw_params)

        rnd.render(show=True)


def save_solution(scenario: Scenario, planning_problem_set: PlanningProblemSet, ego_vehicles: Dict[int, EgoVehicle],
                  vehicle_type: VehicleType,
                  vehicle_model: VehicleModel,
                  cost_function: CostFunction,
                  output_path: str = './', overwrite: bool = False):
    """Saves the given trajectory as a solution to the planning problem"""

    # create solution object for benchmark
    pps = []
    for pp_id, ego_vehicle in ego_vehicles.items():
        assert pp_id in planning_problem_set.planning_problem_dict
        state_initial = copy.deepcopy(planning_problem_set.planning_problem_dict[pp_id].initial_state)
        set_attributes_state_initial = set(state_initial.attributes)
        list_states_trajectory_full = [state_initial]

        # set missing attributes to correctly construct solution file
        for state in ego_vehicle.driven_trajectory.trajectory.state_list:
            set_attributes_state = set(state.attributes)

            set_attributes_in_state_extra = set_attributes_state.difference(set_attributes_state_initial)
            if set_attributes_in_state_extra:
                for attribute in set_attributes_in_state_extra:
                    setattr(state_initial, attribute, 0)

            set_attributes_in_state_initial_extra = set_attributes_state_initial.difference(set_attributes_state)
            if set_attributes_in_state_initial_extra:
                for attribute in set_attributes_in_state_initial_extra:
                    setattr(state, attribute, 0)

            list_states_trajectory_full.append(state)

        trajectory_full = Trajectory(initial_time_step=0, state_list=list_states_trajectory_full)
        pps.append(PlanningProblemSolution(planning_problem_id=pp_id,
                                           vehicle_type=vehicle_type,
                                           vehicle_model=vehicle_model,
                                           cost_function=cost_function,
                                           trajectory=trajectory_full))

    solution = Solution(scenario.scenario_id, pps)

    # write solution
    csw = CommonRoadSolutionWriter(solution)
    csw.write_to_file(output_path=output_path, overwrite=overwrite)
    print("Trajectory saved to solution file.")
