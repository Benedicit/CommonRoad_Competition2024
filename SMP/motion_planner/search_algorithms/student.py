import numpy as np
from commonroad_route_planner.route_planner import RoutePlanner

from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import GreedyBestFirstSearch


class StudentMotionPlanner(GreedyBestFirstSearch):
    """
    Motion planner implementation by myself (Benedikt).
    It's based on GreedyBestFirstSearch.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)
        self.goal_time_step = self.planningProblem.goal.state_list[0].time_step
        route_planner = RoutePlanner(self.lanelet_network, self.planningProblem, self.scenario)
        candidate_holder = route_planner.plan_routes()

        self.route = candidate_holder.retrieve_shortetest_route_with_least_lane_changes()
        self.reference_path_points = [(point[0], point[1]) for point in self.route.reference_path]
    def evaluation_function(self, node_current: PriorityNode) -> float:
        """
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)
            # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)
        """
        node_current.priority = self.heuristic_function(node_current=node_current)
        return node_current.priority
    def heuristic_function(self, node_current: PriorityNode) -> float:
        ########################################################################
        # Hint:                                                                #
        #   Use the State of the current node and the information from the     #
        #   planning problem, as well as from the scenario.                    #
        #   Some helper functions for your convenience can be found in         #
        #   ./search_algorithms/base_class.py                             #
        ########################################################################
        current_state = node_current.list_paths[-1][-1]
        current_path = node_current.list_paths[-1]

        velocity = node_current.list_paths[-1][-1].velocity

        if np.isclose(velocity, 0):
            return np.inf
        path_efficiency = self.calc_path_efficiency(current_path)
        if self.position_desired is None:
            route_splice = self.route.get_route_slice_from_position(current_state.position[0],
                                                                    current_state.position[1])

            distance = self.calc_distance_to_nearest_point(route_splice.reference_path, current_state.position)
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step + (distance / velocity) * 10 + path_efficiency * 0.25

        if self.reached_goal(current_path):
            return 0.0
        current_distance = self.calc_heuristic_distance(current_state, 0)

        if current_distance < 0.2:
            current_distance *= 0.001
        angle_to_goal = self.calc_angle_to_goal(current_state)
        orientation_offset = self.calc_orientation_diff(angle_to_goal, current_state.orientation)

        length_path = len(current_path)
        if length_path > 1 and current_distance > self.calc_heuristic_distance(current_path[length_path-2], 0) + 10:
            return np.inf
        if current_distance is None:
            return np.inf
        if self.distance_initial is not None and self.distance_initial < current_distance:
            return np.inf
        weights = np.zeros(7)
        weights[0] = 1 # Distance
        weights[1] = 1 # velocity
        weights[2] = 0.5 # orientation
        weights[3] = 5 # distance route
        weights[4] = 0.5 # distance travelled
        weights[5] = 1 # route
        weights[6] = 0.2 # trajectory efficiency
        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(current_path)
        if cost_lanelet is None or final_lanelet_id[0] is None:
            return np.inf
        #if self.dict_lanelets_costs[final_lanelet_id[0]] > self.dict_lanelets_costs[start_lanelet_id[0]]:
         #   return np.in
        if self.dict_lanelets_costs[final_lanelet_id[0]] == -1:
            return np.inf

        route_splice = self.route.get_route_slice_from_position(current_state.position[0], current_state.position[1])

        #if final_lanelet_id[0] not in self.route.lanelet_ids:
         #   return np.inf

        distance = self.calc_distance_to_nearest_point(route_splice.reference_path, current_state.position)
        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start +
                           self.planningProblem.goal.state_list[0].velocity.end) / 2
            dist_vel = abs(current_state.velocity - v_mean_goal)
        else:
            dist_vel = 0

        cost = weights[0] * (current_distance / velocity) + \
               weights[1] * dist_vel + \
               weights[2] * orientation_offset + \
               weights[3] * (distance / velocity) + \
               weights[4] * abs(self.distance_initial - cost_lanelet) + \
               weights[5] * 0.0 + \
               weights[6] * path_efficiency
        return cost