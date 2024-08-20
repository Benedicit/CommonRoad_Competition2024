import warnings

from commonroad_route_planner.route_planner import RoutePlanner
from commonroad_route_planner.utility.visualization import visualize_route

import SMP.batch_processing.helper_functions as hf
from SMP.batch_processing.process_scenario import debug_scenario
import SMP.batch_processing.timeout_config



def show_route():
    SMP.batch_processing.timeout_config.use_sequential_processing = True

    warnings.filterwarnings("ignore")

    configuration, logger, scenario_loader, def_automaton, result_dict = hf.init_processing(
        "no_logger_here", for_multi_processing=False)
    for idx, scenario_id in enumerate(scenario_loader.scenario_ids):
        scenario, planning_problem_set = scenario_loader.load_scenario(scenario_id)
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
        route_planner = RoutePlanner(scenario.lanelet_network, planning_problem, scenario)
        candidate_holder = route_planner.plan_routes()
        route = candidate_holder.retrieve_shortest_route()
        visualize_route(route, scenario, planning_problem, save_img=False, draw_route_lanelets=True, draw_reference_path=True)

if __name__ == '__main__':
    show_route()