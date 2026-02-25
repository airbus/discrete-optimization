#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from discrete_optimization.generic_tasks_tools.allocation import (
    AllocationProblem,
    AllocationSolution,
)
from discrete_optimization.generic_tasks_tools.scheduling import (
    SchedulingProblem,
    SchedulingSolution,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    TypeObjective,
)

Task = int
UnaryResource = int


class VRPTWSolution(SchedulingSolution[Task], AllocationSolution[Task, UnaryResource]):
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

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        return task in self.routes[unary_resource]

    problem: VRPTWProblem

    def __init__(
        self,
        problem: VRPTWProblem,
        routes: Optional[List[List[int]]] = None,
        scaling: float = 1.0,
    ):
        super().__init__(problem=problem)
        self.routes = routes if routes is not None else []
        self.scaling = scaling
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
        super().change_problem(new_problem=new_problem)
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

    def get_end_time(self, task: Task) -> int:
        if getattr(self, "_times", None) is not None:
            t = getattr(self, "_times")[task]
            return t
        if len(self.arrival_times) == 0:
            self.problem.evaluate(self)
        for v in range(len(self.routes)):
            for j in range(len(self.routes[v])):
                if self.routes[v][j] == task:
                    return int(self.scaling * (self.start_service_times[v][j])) + int(
                        self.scaling * self.problem.service_times[task]
                    )
        if task == self.problem.depot_node:
            return 0
        return None

    def get_start_time(self, task: Task) -> int:
        return self.get_end_time(task)


class VRPTWProblem(SchedulingProblem[Task], AllocationProblem[Task, UnaryResource]):
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

    def get_makespan_upper_bound(self) -> int:
        return round(1000 ** self.time_windows[self.depot_node][1])

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

    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return list(range(self.nb_vehicles))

    @property
    def tasks_list(self) -> list[Task]:
        return self.customers

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

    def evaluate(self, variable: VRPTWSolution) -> Dict[str, float]:
        """
        Evaluates a VRPTWSolution.
        Calculates distances, time window violations, and capacity violations.
        """
        variable.total_distance = 0.0
        variable.nb_vehicles_used = len([r for r in variable.routes if len(r) > 0])
        variable.tw_violation = 0.0
        variable.capacity_violation = 0.0

        variable.arrival_times = {}
        variable.start_service_times = {}
        variable.route_loads = []
        variable.route_distances = []

        depot_ready = self.time_windows[self.depot_node][0]
        depot_due = self.time_windows[self.depot_node][1]

        for v_idx, route in enumerate(variable.routes):
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
            variable.tw_violation += max(0, start_service_time - due)
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
                variable.tw_violation += max(0, start_service_time - due)
                route_arrivals.append(arrival_time)
                route_starts.append(start_service_time)
                last_node = customer
            # Travel back to depot
            dist_to_depot = self.distance_matrix[last_node, self.depot_node]
            current_dist += dist_to_depot
            arrival_back_at_depot = current_time + dist_to_depot
            variable.tw_violation += max(0, arrival_back_at_depot - depot_due)
            # Store route-level stats
            variable.capacity_violation += max(0, current_load - self.vehicle_capacity)
            variable.total_distance += current_dist
            variable.arrival_times[v_idx] = route_arrivals
            variable.start_service_times[v_idx] = route_starts
            variable.route_loads.append(current_load)
            variable.route_distances.append(current_dist)

        return {
            "nb_vehicles_used": variable.nb_vehicles_used,
            "total_distance": variable.total_distance,
            "tw_violation": -variable.tw_violation,
            "capacity_violation": -variable.capacity_violation,
        }

    def satisfy(self, variable: VRPTWSolution) -> bool:
        # Evaluate if not already done
        if variable.total_distance == 0.0 and variable.nb_vehicles_used == 0:
            self.evaluate(variable)

        return (
            variable.tw_violation == 0
            and variable.capacity_violation == 0
            and variable.nb_vehicles_used <= self.nb_vehicles
        )

    def get_dummy_solution(self) -> VRPTWSolution:
        """Returns a dummy solution (one vehicle per customer)."""
        routes = [[c] for c in self.customers]
        return VRPTWSolution(problem=self, routes=routes)

    def get_solution_type(self) -> type:
        return VRPTWSolution
