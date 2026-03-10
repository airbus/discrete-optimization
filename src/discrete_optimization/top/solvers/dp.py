#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import math
from typing import Any

import didppy as dp
import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.top.problem import TeamOrienteeringProblem, VrpSolution
from discrete_optimization.vrp.utils import compute_length_matrix


class DpTopSolver(DpSolver):
    problem: TeamOrienteeringProblem

    def __init__(
        self,
        problem: TeamOrienteeringProblem,
        params_objective_function: ParamsObjectiveFunction | None = None,
        **kwargs,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        _, self.distance = compute_length_matrix(self.problem)
        self.distance[self.problem.start_indexes[0], self.problem.end_indexes[0]] = 0
        self.distance[self.problem.end_indexes[0], self.problem.start_indexes[0]] = 0
        self.transitions_dict = {}

    def init_model(self, scaling: float = 1, **kwargs: Any) -> None:
        self.model = dp.Model(maximize=True)
        distance = (np.ceil(scaling * self.distance)).astype(int)
        reward = [int(c.reward) for c in self.problem.customers]
        reward_table = self.model.add_int_table(reward)
        nodes = self.model.add_object_type(self.problem.customer_count)
        vehicle = self.model.add_object_type(self.problem.vehicle_count)
        nodes_to_visit = [
            i
            for i in range(self.problem.customer_count)
            if i not in self.problem.start_indexes and i not in self.problem.end_indexes
        ]
        distance_to_end = [
            [
                distance[k, self.problem.end_indexes[j]]
                for k in range(self.problem.customer_count)
            ]
            for j in range(self.problem.vehicle_count)
        ]
        distance_to_end = [
            distance[k, self.problem.end_indexes[0]]
            for k in range(self.problem.customer_count)
        ]
        distance_to_end_table = self.model.add_int_table(distance_to_end)
        distance_from_start = [
            distance[self.problem.start_indexes[0], k]
            for k in range(self.problem.customer_count)
        ]
        distance_from_start_table = self.model.add_int_table(distance_from_start)
        distance_to_and_back = [
            [
                distance[i, j] + distance[j, self.problem.end_indexes[0]]
                for j in range(self.problem.customer_count)
            ]
            for i in range(self.problem.customer_count)
        ]
        distance_to_and_back_table = self.model.add_int_table(distance_to_and_back)
        unvisited = self.model.add_set_var(object_type=nodes, target=nodes_to_visit)
        cur_node = self.model.add_element_var(
            object_type=nodes, target=self.problem.start_indexes[0]
        )
        cur_vehicle = self.model.add_element_resource_var(
            object_type=vehicle, target=0, less_is_better=True
        )
        distance_table = self.model.add_int_table(
            [
                [distance[i, j] for j in range(distance.shape[1])]
                for i in range(distance.shape[0])
            ]
        )
        time = self.model.add_int_resource_var(target=0, less_is_better=True)
        max_time = int(math.floor(scaling * self.problem.max_length_tours))
        self.model.add_state_constr(time + distance_to_end_table[cur_node] <= max_time)
        # self.model.add_state_constr(time<=max_time)
        for i in nodes_to_visit:
            tr = dp.Transition(
                name=f"visit_{i}_same_vehicle",
                cost=100 * reward[i]
                - distance_table[cur_node, i]
                + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(i)),
                    (cur_node, i),
                    (time, time + distance_table[cur_node, i]),
                ],
                preconditions=[
                    unvisited.contains(i),
                    time + distance_to_and_back_table[cur_node, i] <= max_time,
                ],
            )
            self.transitions_dict[f"visit_{i}_same_vehicle"] = (i, "same")
            self.model.add_transition(tr)
            # tr = dp.Transition(name=f"visit_{i}_change_vehicle",
            #                    cost=1000*reward[i]-distance_to_end_table[cur_node]
            #                         -distance_from_start[i]
            #                         + dp.IntExpr.state_cost(),
            #                    effects=[(unvisited, unvisited.remove(i)),
            #                             (cur_node, i),
            #                             (time, distance_from_start[i]),
            #                             (cur_vehicle, cur_vehicle+1)],
            #                    preconditions=[unvisited.contains(i),
            #                                   cur_vehicle < self.problem.vehicle_count-1,
            #                                   # distance_from_start_table[i]<=max_time
            #                                   ])
            # self.model.add_transition(tr)
            # self.transitions_dict[f"visit_{i}_change_vehicle"] = (i, "next")
        start_nodes = self.model.add_element_table(self.problem.start_indexes)
        tr_change_vehicle = dp.Transition(
            name=f"next_vehicle",
            cost=-distance_to_end_table[cur_node] + dp.IntExpr.state_cost(),
            effects=[
                (time, 0),
                (cur_node, start_nodes[cur_vehicle + 1]),
                (cur_vehicle, cur_vehicle + 1),
            ],
            preconditions=[
                cur_vehicle < self.problem.vehicle_count - 1,
                time + distance_to_and_back_table.min(cur_node, unvisited) > max_time,
            ],
        )
        self.model.add_transition(tr_change_vehicle, forced=True)
        self.transitions_dict[f"next_vehicle"] = (None, "next-vehicle")
        finish_ = self.model.add_int_var(0)
        finish = dp.Transition(
            name=f"finish",
            cost=dp.IntExpr.state_cost(),
            effects=[(finish_, 1)],
            preconditions=[
                (cur_vehicle == self.problem.vehicle_count - 1),
                (time + distance_to_and_back_table.min(cur_node, unvisited) > max_time)
                | unvisited.is_empty(),
            ],
        )
        self.transitions_dict[f"finish"] = (None, "finish")
        self.model.add_transition(finish, forced=True)
        self.model.add_base_case([(finish_ == 1)])
        # self.model.add_dual_bound(1000*reward_table[unvisited])

    def retrieve_solution(self, sol: dp.Solution) -> VrpSolution:
        paths = []
        cur_path = []
        for tr in sol.transitions:
            name = tr.name
            descr = self.transitions_dict[name]
            if descr[1] == "same":
                cur_path.append(descr[0])
            if descr[1] == "next":
                paths.append(cur_path)
                cur_path = [descr[0]]
            if descr[1] == "next-vehicle":
                paths.append(cur_path)
                cur_path = []
            if descr[1] == "finish":
                paths.append(cur_path)
                break
        if len(paths) < self.problem.vehicle_count:
            paths.extend([[] for _ in range(self.problem.vehicle_count - len(paths))])
        return VrpSolution(
            problem=self.problem,
            list_paths=paths,
            list_start_index=self.problem.start_indexes,
            list_end_index=self.problem.end_indexes,
        )

        pass


# #  Copyright (c) 2026 AIRBUS and its affiliates.
# #  This source code is licensed under the MIT license found in the
# #  LICENSE file in the root directory of this source tree.
#
# import math
# from typing import Any, List, Tuple
# import didppy as dp
# import numpy as np
#
# from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
# from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
# from discrete_optimization.top.problem import TeamOrienteeringProblem, VrpSolution
# from discrete_optimization.vrp.utils import compute_length_matrix
#
#
# class DpTopSolver(DpSolver):
#     problem: TeamOrienteeringProblem
#
#     def __init__(self, problem: TeamOrienteeringProblem,
#                  params_objective_function: ParamsObjectiveFunction | None = None,
#                  **kwargs):
#         super().__init__(problem, params_objective_function, **kwargs)
#         # Compute distance matrix and ensure depot-to-depot distance is handled if needed
#         _, self.distance = compute_length_matrix(self.problem)
#         # Map transition names to logic for reconstruction
#         self.transitions_mapping = {}
#
#     def init_model(self, scaling: float = 1, **kwargs: Any) -> None:
#         # 1. Initialize Model (Maximize Reward)
#         self.model = dp.Model(maximize=True, float_cost=False)
#
#         # 2. Preprocessing Data
#         # Scale distances to integers for DP efficiency
#         dist_matrix = (np.ceil(scaling * self.distance)).astype(int)
#         rewards = [int(c.reward) for c in self.problem.customers]
#         max_time = int(math.floor(scaling * self.problem.max_length_tours))
#
#         num_customers = self.problem.customer_count
#         num_vehicles = self.problem.vehicle_count
#
#         # Identify customer nodes (excluding start/end depots)
#         # Note: We assume start/end indexes are depots and shouldn't be in the 'unvisited' set
#         # that we try to collect rewards from.
#         depots = set(self.problem.start_indexes + self.problem.end_indexes)
#         customer_indices = [i for i in range(num_customers) if i not in depots]
#
#         # 3. Define State Variables
#         # Set of unvisited customers
#         node_obj = self.model.add_object_type(num_customers)
#         unvisited = self.model.add_set_var(object_type=node_obj, target=customer_indices)
#
#         # Current location (node index)
#         current_node = self.model.add_element_var(object_type=node_obj,
#                                                   target=self.problem.start_indexes[0])
#
#         # Current Vehicle Index
#         vehicle_obj = self.model.add_object_type(num_vehicles + 1)  # +1 to represent "Done" state
#         current_vehicle = self.model.add_element_var(object_type=vehicle_obj, target=0)
#
#         # Time consumed by current vehicle
#         current_time = self.model.add_int_resource_var(target=0, less_is_better=True)
#
#         # 4. Define Tables for DIDPPY access
#         dist_table = self.model.add_int_table(dist_matrix)
#         reward_table = self.model.add_int_table(rewards)
#
#         # Table for End Depots per vehicle (to check return feasibility)
#         # If all vehicles share the same end depot logic, this simplifies,
#         # but we support specific end depots per vehicle.
#         end_depots_list = self.problem.end_indexes
#         # If the number of end_indexes < num_vehicles (unlikely in this struct), handle it.
#         # We create a table: vehicle_index -> end_depot_node_index
#         end_depot_table = self.model.add_int_table(end_depots_list + [end_depots_list[-1]])  # Pad for safety
#
#         start_depots_list = self.problem.start_indexes
#         start_depot_table = self.model.add_int_table(start_depots_list + [start_depots_list[-1]])
#
#         # 5. Transitions
#
#         # A. Visit Customer Transition
#         # Try to visit every customer i from the current node
#         for i in customer_indices:
#             name = f"visit_{i}"
#
#             # Cost to add to objective = Reward of customer i
#             cost_expr = reward_table[i] + dp.IntExpr.state_cost()
#
#             # Distance from current node to i
#             dist_to_i = dist_table[current_node, i]
#             # Distance from i to the current vehicle's end depot
#             dist_return = dist_table[i, end_depot_table[current_vehicle]]
#
#             # Preconditions:
#             # 1. Customer must be unvisited
#             # 2. We must have enough time to reach i AND return to the depot afterwards
#             preconditions = [
#                 unvisited.contains(i),
#                 current_time + dist_to_i + dist_return <= max_time
#             ]
#
#             # Effects:
#             effects = [
#                 (unvisited, unvisited.remove(i)),
#                 (current_node, i),
#                 (current_time, current_time + dist_to_i)
#             ]
#
#             transition = dp.Transition(
#                 name=name,
#                 cost=cost_expr,
#                 effects=effects,
#                 preconditions=preconditions
#             )
#             self.model.add_transition(transition)
#             self.transitions_mapping[name] = {"type": "visit", "node": i}
#
#         # B. Next Vehicle / Finish Transition
#         # Move to the next vehicle (or finish if it's the last one)
#         # This represents "Returning to depot" and "Starting next vehicle"
#         next_veh_name = "next_vehicle"
#
#         # Preconditions:
#         # Only allowed if we haven't exhausted all vehicles
#         preconditions_next = [current_vehicle < num_vehicles]
#
#         # Effects:
#         # Increment vehicle index
#         # Reset time to 0
#         # Set location to the start depot of the *next* vehicle
#         effects_next = [
#             (current_vehicle, current_vehicle + 1),
#             (current_time, 0),
#             (current_node, start_depot_table[current_vehicle + 1])
#         ]
#
#         transition_next = dp.Transition(
#             name=next_veh_name,
#             cost=dp.IntExpr.state_cost(),  # No reward gained for switching
#             effects=effects_next,
#             preconditions=preconditions_next
#         )
#         self.model.add_transition(transition_next)
#         self.transitions_mapping[next_veh_name] = {"type": "next_vehicle"}
#
#         # 6. Base Case
#         # The problem is "solved" (traversal complete) when we have used all vehicles.
#         # current_vehicle reaches num_vehicles
#         self.model.add_base_case([current_vehicle == num_vehicles])
#
#         # 7. Dual Bound (Heuristic)
#         # An upper bound on the remaining reward is the sum of rewards of all unvisited nodes.
#         # This ignores the time constraint but provides a valid admissible heuristic for pruning.
#         self.model.add_dual_bound(reward_table[unvisited])
#
#     def retrieve_solution(self, sol: dp.Solution) -> VrpSolution:
#         paths: List[List[int]] = [[] for _ in range(self.problem.vehicle_count)]
#         current_vehicle_idx = 0
#
#         for transition in sol.transitions:
#             t_info = self.transitions_mapping.get(transition.name)
#             if not t_info:
#                 continue
#
#             if t_info["type"] == "visit":
#                 if current_vehicle_idx < self.problem.vehicle_count:
#                     paths[current_vehicle_idx].append(t_info["node"])
#
#             elif t_info["type"] == "next_vehicle":
#                 current_vehicle_idx += 1
#
#         return VrpSolution(
#             problem=self.problem,
#             list_paths=paths,
#             list_start_index=self.problem.start_indexes,
#             list_end_index=self.problem.end_indexes
#         )
