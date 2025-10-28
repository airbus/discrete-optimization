#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import re
from typing import Any, List

import didppy as dp

from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution

logger = logging.getLogger(__name__)


class DpVrptwSolver(DpSolver):
    """
    Dynamic Programming solver for the VRPTW.

    This model combines the state variables from the VRP (load, vehicle index)
    and the TSPTW (time) to solve the VRPTW.

    State variables:
    - unvisited (Set): Set of customers not yet served.
    - location (Element): The current location of the active vehicle.
    - load (FloatResource): The current load of the active vehicle.
    - time (FloatResource): The current time (service start time) at the current location.
    - vehicles_ (Element): The index of the active vehicle (from 0 to m-1).
    """

    hyperparameters = [
        CategoricalHyperparameter(
            name="resource_var_load", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="resource_var_time", choices=[True, False], default=False
        ),
    ]
    problem: VRPTWProblem
    transitions: dict
    scaling: int

    def init_model(
        self, scaling: int = 1000, cost_per_vehicle: int = 10**6, **kwargs: Any
    ) -> None:
        """
        DP model for VRPTW.
        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        self.scaling = scaling
        n = self.problem.nb_nodes
        m = self.problem.nb_vehicles
        depot = self.problem.depot_node
        vehicle_capacity = self.problem.vehicle_capacity

        # We need a large cost to penalize using a new vehicle
        # This should be larger than the max possible total distance
        cost_per_vehicle = cost_per_vehicle

        model = dp.Model(float_cost=False)
        # --- Object Types ---
        customer = model.add_object_type(number=n)
        vehicle_obj = model.add_object_type(number=m)
        # --- Data Tables ---
        demand = model.add_int_table([int(d) for d in self.problem.demands])
        distance = model.add_int_table(
            (scaling * self.problem.distance_matrix).astype(int)
        )
        service = model.add_int_table(
            [int(scaling * x) for x in self.problem.service_times]
        )
        release = model.add_int_table(
            [int(scaling * x[0]) for x in self.problem.time_windows]
        )
        deadline = model.add_int_table(
            [int(scaling * x[1]) for x in self.problem.time_windows]
        )

        # --- State Variables ---
        unvisited = model.add_set_var(
            object_type=customer, target=self.problem.customers
        )
        location = model.add_element_var(object_type=customer, target=depot)
        if kwargs["resource_var_load"]:
            load = model.add_int_resource_var(target=0, less_is_better=False)
        else:
            load = model.add_int_var(target=0)
        if kwargs["resource_var_time"]:
            time = model.add_int_resource_var(
                target=int(scaling * self.problem.time_windows[depot][0]),
                less_is_better=True,
            )
        else:
            time = model.add_int_var(
                target=int(scaling * self.problem.time_windows[depot][0]),
            )
        vehicles_ = model.add_element_var(object_type=vehicle_obj, target=0)
        self.transitions = {}
        # --- Transitions ---
        for j in self.problem.customers:
            # --- Transition 1: Visit customer j with the *current* vehicle ---
            # Time logic:
            # arrival_j = time_at_i + service_at_i + travel_i_j
            # start_j = max(arrival_j, release_j)
            start_j = dp.max(
                time + service[location] + distance[location, j], release[j]
            )
            visit_j = dp.Transition(
                name=f"visit {j}",
                cost=distance[location, j] + dp.IntExpr.state_cost(),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, load + demand[j]),
                    (time, start_j),
                ],
                preconditions=[
                    unvisited.contains(j),
                    load + demand[j] <= vehicle_capacity,
                    start_j <= deadline[j],
                ],
            )
            model.add_transition(visit_j)
            self.transitions[("visit", "current", j)] = visit_j

            # --- Transition 2: Visit customer j with a *new* vehicle ---

            # Time logic for a new vehicle starting from depot:
            # arrival_j = time_at_depot + service_at_depot + travel_depot_j
            # (service_at_depot is 0)
            start_j_new_vehicle = dp.max(
                release[depot] + distance[depot, j], release[j]
            )

            visit_j_new_vehicle = dp.Transition(
                name=f"visit {j} with a new vehicle",
                cost=(
                    distance[location, depot]  # Cost to return to depot
                    + distance[depot, j]  # Cost to go to j
                    + cost_per_vehicle  # Penalty for new vehicle
                    + dp.IntExpr.state_cost()
                ),
                effects=[
                    (unvisited, unvisited.remove(j)),
                    (location, j),
                    (load, demand[j]),  # Reset load for new vehicle
                    (time, start_j_new_vehicle),  # Reset time for new vehicle
                    (vehicles_, vehicles_ + 1),  # Increment vehicle index
                ],
                preconditions=[
                    unvisited.contains(j),
                    vehicles_ < m - 1,  # Must have a new vehicle available
                    demand[j] <= vehicle_capacity,
                    start_j_new_vehicle <= deadline[j],
                ],
            )
            model.add_transition(visit_j_new_vehicle)
            self.transitions[("visit", "next_vehicle", j)] = visit_j_new_vehicle

        # --- Transition 3: Return to depot (end of all routes) ---

        # Time logic:
        # arrival_at_depot = time_at_last_cust + service_at_last_cust + travel_last_cust_depot
        arrival_depot = time + service[location] + distance[location, depot]

        return_to_depot = dp.Transition(
            name="return",
            cost=distance[location, depot] + cost_per_vehicle + dp.IntExpr.state_cost(),
            # Add the initial vehicle.
            effects=[(location, depot), (time, arrival_depot)],
            preconditions=[
                unvisited.is_empty(),
                location != depot,
                # arrival_depot <= deadline[depot]
                # Check final depot TW
            ],
        )
        model.add_transition(return_to_depot)
        self.transitions["return"] = return_to_depot

        # --- Base Case ---
        model.add_base_case([unvisited.is_empty(), location == depot])

        # --- State Constraints (Pruning) ---

        # 1. Capacity Pruning:
        # Total remaining demand must be <= capacity left on this vehicle
        # + capacity of all remaining unused vehicles.
        # remaining_capacity_current_vehicle = vehicle_capacity - load
        # (m - 1) is max index, vehicles_ is current. (m - 1 - vehicles_) is remaining.
        # remaining_capacity_other_vehicles = (m - 1 - vehicles_) * vehicle_capacity
        # total_remaining_capacity = (
        #        remaining_capacity_current_vehicle + remaining_capacity_other_vehicles
        # )
        # model.add_state_constr(total_remaining_capacity >= demand[unvisited])

        # 2. Time Window Pruning:
        # For every unvisited customer j, it must still be possible
        # to visit them from the current location.
        # if False:
        #     for j in self.problem.customers:
        #         model.add_state_constr(
        #             ~unvisited.contains(j)
        #             | (
        #                     dp.max(time + service[location] + distance[location, j], release[j])
        #                     <= deadline[j]
        #             )
        #         )

        # --- Dual Bounds (Heuristics) ---

        needed_vehicle = model.add_int_state_fun(
            demand[unvisited] // self.problem.vehicle_capacity + 1
        )

        # 1. Min distance to all unvisited nodes
        min_distance_to = model.add_int_table(
            [
                min(
                    int(scaling * self.problem.distance_matrix[k, j])
                    for k in range(n)
                    if k != j
                )
                if j != depot
                else 0
                for j in range(n)
            ]
        )
        model.add_dual_bound(
            needed_vehicle * cost_per_vehicle
            + min_distance_to[unvisited]
            + (location != depot).if_then_else(min_distance_to[depot], 0)
        )

        # 2. Min distance from all unvisited nodes
        min_distance_from = model.add_int_table(
            [
                min(
                    int(scaling * self.problem.distance_matrix[j, k])
                    for k in range(n)
                    if k != j
                )
                if j != depot
                else 0
                for j in range(n)
            ]
        )
        model.add_dual_bound(
            needed_vehicle * vehicle_capacity
            + min_distance_from[unvisited]
            + (location != depot).if_then_else(min_distance_from[location], 0)
        )
        self.variables = {"time": time}
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> VRPTWSolution:
        """
        Reconstructs the VRPTWSolution from the sequence of DP transitions.
        This logic is adapted from DpVrpSolver.
        """
        list_paths: List[List[int]] = [[] for _ in range(self.problem.nb_vehicles)]

        def extract_visit_number(text):
            match = re.search(r"visit\s(\d+)", text, re.IGNORECASE)
            if match:
                return int(match.group(1))
            return None

        cur_vehicle_index = 0
        for t in sol.transitions:
            name = t.name
            if "with a new vehicle" in name:
                cur_vehicle_index += 1
            if "return" in name:
                break

            customer_node = extract_visit_number(name)
            if customer_node is not None:
                list_paths[cur_vehicle_index].append(customer_node)

        logger.info(f"DP Solution Cost (includes vehicle penalties): {sol.cost}")

        # Filter out empty routes
        final_routes = [route for route in list_paths if route]

        return VRPTWSolution(
            problem=self.problem,
            routes=final_routes,
        )

    def set_warm_start(self, solution: VRPTWSolution) -> None:
        """
        Provides a warm start hint to the DP solver from an existing solution.
        Converts a VRPTWSolution into a list of didppy.Transition objects.
        """
        if self.model is None:
            self.init_model()  # Ensure transitions are populated

        initial_solution = []

        # Filter out any potential empty routes from the warmstart solution
        non_empty_routes = [route for route in solution.routes if route]
        state = self.model.target_state
        for route_idx, route in enumerate(non_empty_routes):
            is_first_customer_of_route = True
            prev_customer = None
            for customer_node in route:
                if is_first_customer_of_route:
                    is_first_customer_of_route = False
                    if route_idx == 0:
                        # First customer of the first route: use "current" vehicle
                        # (The DP state starts with vehicle 0)
                        transition_key = ("visit", "current", customer_node)
                    else:
                        # First customer of a new route: use "next_vehicle"
                        transition_key = ("visit", "next_vehicle", customer_node)
                else:
                    # Subsequent customers in a route: use "current" vehicle
                    transition_key = ("visit", "current", customer_node)

                if transition_key in self.transitions:
                    initial_solution.append(self.transitions[transition_key])
                    applicable = initial_solution[-1].is_applicable(state, self.model)
                    if not applicable:
                        print("Time, : ", state[self.variables["time"]])
                        if prev_customer is not None:
                            print(
                                state[self.variables["time"]]
                                + int(
                                    self.scaling
                                    * self.problem.service_times[prev_customer]
                                )
                                + int(
                                    self.scaling
                                    * self.problem.distance_matrix[
                                        prev_customer, customer_node
                                    ]
                                )
                            )
                            print("vs ", self.problem.time_windows[customer_node])
                        print(transition_key, "not feasible")
                    state = initial_solution[-1].apply(state, self.model)
                else:
                    logger.warning(
                        f"Warmstart: Transition {transition_key} not found. Skipping."
                    )
                prev_customer = customer_node

        if "return" in self.transitions:
            initial_solution.append(self.transitions["return"])
            applicable = initial_solution[-1].is_applicable(state, self.model)
            if not applicable:
                print("return", "not feasible")
            state = initial_solution[-1].apply(state, self.model)
        else:
            logger.warning(
                f"Warmstart: 'return' transition not found. Solution may be incomplete."
            )

        self.initial_solution = initial_solution
