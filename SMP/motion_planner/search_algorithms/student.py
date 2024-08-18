import math

import numpy as np
from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch, GreedyBestFirstSearch


class StudentMotionPlanner(GreedyBestFirstSearch):
    """
    Motion planner implementation by students.
    Note that you may inherit from any given motion planner as you wish, or come up with your own planner.
    Here as an example, the planner is inherited from the GreedyBestFirstSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig):
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)
        self.goal_time_step = self.planningProblem.goal.state_list[0].time_step

    def evaluation_function(self, node_current: PriorityNode) -> float:
        ########################################################################
        # todo: Implement your own evaluation function here.                   #
        ########################################################################
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
        # todo: Implement your own heuristic cost calculation here.            #
        # Hint:                                                                #
        #   Use the State of the current node and the information from the     #
        #   planning problem, as well as from the scenario.                    #
        #   Some helper functions for your convenience can be found in         #
        #   ./search_algorithms/base_class.py                             #
        ########################################################################
        current_state = node_current.list_paths[-1][-1]
        current_path = node_current.list_paths[-1]
        if self.position_desired is None:
            return self.time_desired.start - node_current.list_paths[-1][-1].time_step
        if self.reached_goal(current_path):
            return 0.0
        current_distance = self.calc_heuristic_distance(current_state, 0)

        if current_distance < 0.2:
            current_distance *= 0.001
        angle_to_goal = self.calc_angle_to_goal(current_state)
        orientation_offset = self.calc_orientation_diff(angle_to_goal, current_state.orientation)
        #time_length = abs(current_state.time_step - self.goal_time_step)
        velocity = node_current.list_paths[-1][-1].velocity

        if np.isclose(velocity, 0):
            return np.inf
        length_path = len(current_path)
        if length_path > 1 and current_distance > self.calc_heuristic_distance(current_path[length_path-2], 0) + 20:
            return np.inf
        if current_distance is None:
            return np.inf
        if self.distance_initial is not None and self.distance_initial < current_distance:
            return np.inf
        weights = np.zeros(7)
        lane_cost = 40
        weights[0] = 1 # Distance
        weights[1] = 0.25 # velocity
        weights[2] = 0.25 # orientation
        weights[3] = 0.0 # time difference
        weights[4] = 0.5 #0.1 # lanelet id
        weights[5] = 0.0 # obstacles
        weights[6] = 0.25 # trajectory efficiency

        cost_lanelet, final_lanelet_id, start_lanelet_id = self.calc_heuristic_lanelet(current_path)
        if cost_lanelet is None or final_lanelet_id[0] is None:
            return np.inf
        #obstacle_cost = self.calc_dist_to_closest_obstacle(final_lanelet_id[0], current_state.position, current_state.time_step)
        path_efficiency = self.calc_path_efficiency(current_path)
        #if self.dict_lanelets_costs[final_lanelet_id[0]] > self.dict_lanelets_costs[start_lanelet_id[0]]:
         #   return np.inf

        if self.is_goal_in_lane(final_lanelet_id[0]):
            lane_cost = 0.1
        if self.dict_lanelets_costs[final_lanelet_id[0]] == -1:
            return np.inf
        #if self.is_goal_in_lane(final_lanelet_id[0]):
         #   weights[5] /= 2
        #if current_state.time_step.length > self.goal_time_step.length:
            #time_length *= 10
        if hasattr(self.planningProblem.goal.state_list[0], 'velocity'):
            v_mean_goal = (self.planningProblem.goal.state_list[0].velocity.start +
                           self.planningProblem.goal.state_list[0].velocity.end) / 2
            dist_vel = abs(current_state.velocity - v_mean_goal)
        else:
            dist_vel = 0
        cost = weights[0] * (current_distance / velocity) + \
               weights[1] * dist_vel + \
               weights[2] * orientation_offset + \
               weights[3] * 0 + \
               weights[4] * lane_cost + \
               weights[5] * 0 + \
               weights[6] * path_efficiency
        return cost

