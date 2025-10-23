#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from typing import Any, Dict, List, Optional

from ortools.sat.python.cp_model import CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import StatusSolver, WarmstartMixin
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution

logger = logging.getLogger(__name__)


class CpSatVRPTWSolver(OrtoolsCpSatSolver, WarmstartMixin):
    """
    CP-SAT solver for the Vehicle Routing Problem with Time Windows (VRPTW).

    This solver uses a path-based formulation with time and load dimensions
    to find an optimal solution. It aims to minimize the number of vehicles
    used, and then the total distance.

    Attributes:
        problem (VRPTWProblem): The VRPTW problem instance.
        variables (Dict[str, Any]): Stores CP-SAT model variables.
    """

    problem: VRPTWProblem

    def __init__(
        self,
        problem: VRPTWProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.variables: Dict[str, Any] = {}
        self.scaling_factor = None  # Scale distances and times to use integers

    def init_model(self, scaling: int, cost_per_vehicle: int, **kwargs: Any) -> None:
        """Initialise the CP-SAT model."""
        super().init_model(**kwargs)
        self.scaling_factor = scaling
        if self.cp_model is None:
            raise RuntimeError(
                "self.cp_model must not be None after super().init_model()."
            )

        n = self.problem.nb_nodes
        k = self.problem.nb_vehicles
        depot = self.problem.depot_node

        # --- Create variables ---

        # x_arc[i, j]: BoolVar, true if arc (i, j) is used by any vehicle
        x_arc = {
            (i, j): self.cp_model.NewBoolVar(f"x_{i},{j}")
            for i in range(n)
            for j in range(n)
            if i != j
        }

        # vehicle_used[v]: BoolVar, true if vehicle v is used
        vehicle_used = [self.cp_model.NewBoolVar(f"vehicle_used_{v}") for v in range(k)]

        # Time dimension variables: t_time[i] is the time service *starts* at node i
        # We must scale times to use integers
        t_time = [
            self.cp_model.NewIntVar(
                lb=int(self.scaling_factor * self.problem.time_windows[i][0]),
                ub=int(self.scaling_factor * self.problem.time_windows[i][1]),
                name=f"t_{i}",
            )
            for i in range(n)
        ]

        # Load dimension variables: load[i] is the cumulative load *after* service at node i
        load = [
            self.cp_model.NewIntVar(
                lb=int(self.problem.demands[i]),
                ub=int(self.problem.vehicle_capacity),
                name=f"load_{i}",
            )
            for i in range(n)
        ]

        self.variables = {
            "x_arc": x_arc,
            "t_time": t_time,
            "load": load,
            "vehicle_used": vehicle_used,
        }

        # --- Add constraints ---

        # 1. Degree constraints
        # Each customer is visited exactly once
        for i in self.problem.customers:
            # One successor
            self.cp_model.Add(sum(x_arc[i, j] for j in range(n) if i != j) == 1)
            # One predecessor
            self.cp_model.Add(sum(x_arc[j, i] for j in range(n) if i != j) == 1)

        # 2. Depot constraints
        # Number of routes leaving depot <= k
        self.cp_model.Add(sum(x_arc[depot, j] for j in self.problem.customers) <= k)
        # Number of routes returning to depot <= k
        self.cp_model.Add(sum(x_arc[i, depot] for i in self.problem.customers) <= k)
        # Same number of routes leave and return
        self.cp_model.Add(
            sum(x_arc[depot, j] for j in self.problem.customers)
            == sum(x_arc[i, depot] for i in self.problem.customers)
        )
        # Link to vehicle_used variables
        self.cp_model.Add(
            sum(x_arc[depot, j] for j in self.problem.customers) == sum(vehicle_used)
        )
        # No arc from depot to depot
        # self.cp_model.Add(x_arc[depot, depot] == 0)

        # 3. Time dimension constraints
        self.cp_model.Add(
            t_time[depot]
            == int(self.scaling_factor * self.problem.time_windows[depot][0])
        )
        t_time_return_depot = self.cp_model.NewIntVar(
            lb=int(self.scaling_factor * self.problem.time_windows[depot][0]),
            ub=int(self.scaling_factor * self.problem.time_windows[depot][1]),
            name="time_return_depot",
        )
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if j == depot:
                    var_ = t_time_return_depot
                else:
                    var_ = t_time[j]
                # Scaled travel time and service time
                travel_time = int(
                    self.scaling_factor * self.problem.distance_matrix[i, j]
                )
                service_time = int(self.scaling_factor * self.problem.service_times[i])
                # If arc (i,j) is taken, time at j must be after time at i + service + travel
                self.cp_model.Add(
                    var_ >= t_time[i] + service_time + travel_time
                ).OnlyEnforceIf(x_arc[i, j])

        # 4. Load dimension constraints
        self.cp_model.Add(load[depot] == 0)

        for i in range(n):
            for j in self.problem.customers:  # Customers only
                if i == j:
                    continue

                demand_j = int(self.problem.demands[j])

                # If arc (i,j) is taken, load at j = load at i + demand at j
                # (load[i] is load *after* service at i)
                self.cp_model.Add(load[j] == load[i] + demand_j).OnlyEnforceIf(
                    x_arc[i, j]
                )

                # Special case: arc from depot (i=depot)
                if i == depot:
                    self.cp_model.Add(load[j] == demand_j).OnlyEnforceIf(
                        x_arc[depot, j]
                    )
        scaled_distances = {
            (i, j): int(self.scaling_factor * self.problem.distance_matrix[i, j])
            for i in range(n)
            for j in range(n)
        }
        # Total distance
        total_distance = self.cp_model.NewIntVar(0, 10**10, "total_distance")
        self.cp_model.Add(
            total_distance
            == sum(x_arc[i, j] * scaled_distances[i, j] for i, j in x_arc)
        )
        # Number of vehicles
        nb_vehicles = self.cp_model.NewIntVar(0, k, "nb_vehicles")
        self.cp_model.Add(nb_vehicles == sum(vehicle_used))
        # Lexicographic objective: 1. Min vehicles, 2. Min distance
        # We use a large constant to separate the objectives
        self.cp_model.Minimize(nb_vehicles * cost_per_vehicle + total_distance)
        logger.info("CP-SAT model initialized.")

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> VRPTWSolution:
        """
        Build a VRPTWSolution from the CP-SAT solver's callback.
        """
        x_arc = self.variables["x_arc"]
        logger.info(f"Obj : {cpsolvercb.objective_value}")
        # for i,j in x_arc:
        #    val = cpsolvercb.Value(x_arc[(i, j)])
        #    if val == 1:
        #        print("Flow ", (i, j))
        depot = self.problem.depot_node
        n = self.problem.nb_nodes
        routes = []

        # Find all successors of the depot
        starters = []
        for j in self.problem.customers:
            if cpsolvercb.Value(x_arc[depot, j]):
                starters.append(j)

        visited = set()

        # Reconstruct each route
        for start_node in starters:
            if start_node in visited:
                continue

            route = [start_node]
            visited.add(start_node)
            current_node = start_node

            while True:
                # Find the successor
                successor_found = False
                for j in range(n):
                    if current_node == j:
                        continue
                    # print("Value :", cpsolvercb.Value(x_arc[current_node, j]), j)
                    if (current_node, j) in x_arc and cpsolvercb.Value(
                        x_arc[current_node, j]
                    ):
                        if j == depot:
                            # Route finished
                            current_node = depot
                            successor_found = True
                            break
                        else:
                            # Next customer
                            route.append(j)
                            visited.add(j)
                            current_node = j
                            successor_found = True
                            break
                # print(route, current_node, depot)
                if not successor_found or current_node == depot:
                    break  # Should not happen unless route ends at depot
            routes.append(route)
        return VRPTWSolution(problem=self.problem, routes=routes)

    def set_warm_start(self, solution: VRPTWSolution) -> None:
        """
        Provides a warm start hint to the CP-SAT solver from an existing solution.
        """
        if self.cp_model is None:
            self.init_model()
        if self.cp_model is None:
            raise RuntimeError(
                "self.cp_model must not be None after self.init_model()."
            )

        self.cp_model.ClearHints()
        logger.info("Setting warm start from solution.")

        x_arc = self.variables["x_arc"]
        depot = self.problem.depot_node

        # Hint arc variables
        for route in solution.routes:
            if not route:
                continue

            # Arc from depot
            self.cp_model.AddHint(x_arc[depot, route[0]], 1)

            # Arcs between customers
            for i in range(len(route) - 1):
                u, v = route[i], route[i + 1]
                self.cp_model.AddHint(x_arc[u, v], 1)

            # Arc back to depot
            self.cp_model.AddHint(x_arc[route[-1], depot], 1)

        # Hint time variables (if evaluated)
        if solution.start_service_times:
            t_time = self.variables["t_time"]
            for v_idx, route in enumerate(solution.routes):
                starts = solution.start_service_times[v_idx]
                for cust_idx, node_id in enumerate(route):
                    self.cp_model.AddHint(
                        t_time[node_id], int(self.scaling_factor * starts[cust_idx])
                    )
