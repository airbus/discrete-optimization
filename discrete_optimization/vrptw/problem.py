#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import Dict, List, Optional, Tuple

import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)


class VRPTWSolution(Solution):
    """
    Solution class for the VRPTW problem.

    Attributes:
        problem (VRPTWProblem): The problem instance.
        routes (List[List[int]]):
            List of routes. Each route is a list of customer node indices.
            The depot (start/end) is implicit and not included in these lists.
        arrival_times (Dict[int, List[float]]):
            Maps vehicle index to a list of arrival times at customer nodes in its route.
        start_service_times (Dict[int, List[float]]):
            Maps vehicle index to a list of service start times at customer nodes.
        route_loads (List[float]): Total demand for each route.
        route_distances (List[float]): Total distance for each route.

        # Evaluated metrics
        total_distance (float): Sum of distances of all routes.
        nb_vehicles_used (int): Number of routes used.
        tw_violation (float): Total violation of time windows (sum of lateness).
        capacity_violation (float): Total violation of vehicle capacities.
    """

    def __init__(
        self,
        problem: "VRPTWProblem",
        routes: Optional[List[List[int]]] = None,
    ):
        self.problem = problem
        self.routes = routes if routes is not None else []

        # Detailed schedule (filled by evaluation)
        self.arrival_times: Dict[int, List[float]] = {}
        self.start_service_times: Dict[int, List[float]] = {}
        self.route_loads: List[float] = []
        self.route_distances: List[float] = []

        # Main objective and penalty values (filled by evaluation)
        self.total_distance: float = 0.0
        self.nb_vehicles_used: int = 0
        self.tw_violation: float = 0.0
        self.capacity_violation: float = 0.0

    def copy(self) -> "VRPTWSolution":
        sol = VRPTWSolution(
            problem=self.problem,
            routes=[list(r) for r in self.routes],
        )
        # Copy evaluated data
        sol.arrival_times = {k: list(v) for k, v in self.arrival_times.items()}
        sol.start_service_times = {
            k: list(v) for k, v in self.start_service_times.items()
        }
        sol.route_loads = list(self.route_loads)
        sol.route_distances = list(self.route_distances)
        sol.total_distance = self.total_distance
        sol.nb_vehicles_used = self.nb_vehicles_used
        sol.tw_violation = self.tw_violation
        sol.capacity_violation = self.capacity_violation
        return sol

    def lazy_copy(self) -> "VRPTWSolution":
        # Routes are mutable, so we must copy them.
        return self.copy()

    def change_problem(self, new_problem: Problem) -> None:
        if not isinstance(new_problem, VRPTWProblem):
            raise ValueError("new_problem must be a VRPTWProblem instance.")
        self.problem = new_problem
        # Invalidate evaluated metrics
        self.arrival_times = {}
        self.start_service_times = {}
        self.route_loads = []
        self.route_distances = []
        self.total_distance = 0.0
        self.nb_vehicles_used = 0
        self.tw_violation = 0.0
        self.capacity_violation = 0.0

    def __str__(self) -> str:
        route_str = "\n".join(
            f"  Vehicle {i + 1}: Depot -> {' -> '.join(map(str, route))} -> Depot"
            for i, route in enumerate(self.routes)
        )
        return (
            f"VRPTW Solution:\n"
            f"Total Distance: {self.total_distance:.2f}\n"
            f"Vehicles Used: {self.nb_vehicles_used}\n"
            f"TW Violation: {self.tw_violation:.2f}\n"
            f"Capacity Violation: {self.capacity_violation:.2f}\n"
            f"Routes:\n{route_str}"
        )


class VRPTWProblem(Problem):
    """
    Vehicle Routing Problem with Time Windows (VRPTW) Problem class.

    This model includes:
    - Multiple vehicles with a shared capacity.
    - A single depot.
    - Customers with demand.
    - Customers with time windows (ready time, due date).
    - Customers with service times.
    - Objectives: 1) Minimize number of vehicles, 2) Minimize total distance.
    """

    def __init__(
        self,
        nb_vehicles: int,
        vehicle_capacity: float,
        nb_nodes: int,
        distance_matrix: np.ndarray,
        time_windows: List[Tuple[int, int]],
        service_times: List[float],
        demands: List[float],
        depot_node: int = 0,
    ):
        self.nb_vehicles = nb_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.nb_nodes = nb_nodes
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.service_times = service_times
        self.demands = demands
        self.depot_node = depot_node
        self.customers = sorted(
            [i for i in range(self.nb_nodes) if i != self.depot_node]
        )
        self.nb_customers = len(self.customers)

    def get_attribute_register(self) -> EncodingRegister:
        # VRPTW is complex to encode with simple registers.
        # We'll rely on the Solution object itself for evaluation.
        return EncodingRegister(
            {
                "routes": {
                    "name": "routes",
                    "type": [],
                }
            }
        )

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                # Primary objective
                "nb_vehicles_used": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=100000.0
                ),
                # Secondary objective
                "total_distance": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE, default_weight=1.0
                ),
                # Penalties
                "tw_violation": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=-1000.0
                ),
                "capacity_violation": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=-1000.0
                ),
            },
        )

    def evaluate(self, solution: VRPTWSolution) -> Dict[str, float]:
        """
        Evaluates a VRPTWSolution.
        Calculates distances, time window violations, and capacity violations.
        """
        solution.total_distance = 0.0
        solution.nb_vehicles_used = len(solution.routes)
        solution.tw_violation = 0.0
        solution.capacity_violation = 0.0

        solution.arrival_times = {}
        solution.start_service_times = {}
        solution.route_loads = []
        solution.route_distances = []

        depot_ready = self.time_windows[self.depot_node][0]
        depot_due = self.time_windows[self.depot_node][1]

        for v_idx, route in enumerate(solution.routes):
            if not route:
                continue

            current_time = depot_ready
            current_load = 0.0
            current_dist = 0.0
            last_node = self.depot_node

            route_arrivals = []
            route_starts = []

            # Travel to first customer
            first_customer = route[0]
            dist = self.distance_matrix[last_node, first_customer]
            current_dist += dist
            arrival_time = current_time + dist

            # Service at first customer
            ready, due = self.time_windows[first_customer]
            service = self.service_times[first_customer]

            start_service_time = max(arrival_time, ready)
            current_time = start_service_time + service
            current_load += self.demands[first_customer]

            solution.tw_violation += max(0, start_service_time - due)
            route_arrivals.append(arrival_time)
            route_starts.append(start_service_time)
            last_node = first_customer

            # Travel to subsequent customers
            for customer in route[1:]:
                dist = self.distance_matrix[last_node, customer]
                current_dist += dist
                arrival_time = current_time + dist

                ready, due = self.time_windows[customer]
                service = self.service_times[customer]

                start_service_time = max(arrival_time, ready)
                current_time = start_service_time + service
                current_load += self.demands[customer]

                solution.tw_violation += max(0, start_service_time - due)
                route_arrivals.append(arrival_time)
                route_starts.append(start_service_time)
                last_node = customer

            # Travel back to depot
            dist_to_depot = self.distance_matrix[last_node, self.depot_node]
            current_dist += dist_to_depot
            arrival_back_at_depot = current_time + dist_to_depot

            solution.tw_violation += max(0, arrival_back_at_depot - depot_due)

            # Store route-level stats
            solution.capacity_violation += max(0, current_load - self.vehicle_capacity)
            solution.total_distance += current_dist
            solution.arrival_times[v_idx] = route_arrivals
            solution.start_service_times[v_idx] = route_starts
            solution.route_loads.append(current_load)
            solution.route_distances.append(current_dist)

        return {
            "nb_vehicles_used": solution.nb_vehicles_used,
            "total_distance": solution.total_distance,
            "tw_violation": -solution.tw_violation,
            "capacity_violation": -solution.capacity_violation,
        }

    def satisfy(self, solution: VRPTWSolution) -> bool:
        # Evaluate if not already done
        if solution.total_distance == 0.0 and solution.nb_vehicles_used == 0:
            self.evaluate(solution)

        return (
            solution.tw_violation == 0
            and solution.capacity_violation == 0
            and solution.nb_vehicles_used <= self.nb_vehicles
        )

    def get_dummy_solution(self) -> VRPTWSolution:
        """Returns a dummy solution (one vehicle per customer)."""
        routes = [[c] for c in self.customers]
        return VRPTWSolution(problem=self, routes=routes)

    def get_solution_type(self) -> type:
        return VRPTWSolution
