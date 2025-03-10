#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import print_function

import logging
from enum import Enum
from typing import Any, Optional

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.solvers import VrpSolver
from discrete_optimization.vrp.utils import build_graph

logger = logging.getLogger(__name__)


class FirstSolutionStrategy(Enum):
    SAVINGS = 0
    PATH_MOST_CONSTRAINED_ARC = 1


class LocalSearchMetaheuristic(Enum):
    GUIDED_LOCAL_SEARCH = 0
    SIMULATED_ANNEALING = 1


first_solution_map = {
    FirstSolutionStrategy.SAVINGS: routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC: routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC,
}
metaheuristic_map = {
    LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH: routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    LocalSearchMetaheuristic.SIMULATED_ANNEALING: routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
}

status_description = {
    val: key
    for key, val in pywrapcp.RoutingModel.__dict__.items()
    if key.startswith("ROUTING_")
}
"""Mapping from status integer to description string.

This maps the integer returned by routing_model.status to the corresponding string.

We use the attributes of RoutingModel to construct the dictionary.
https://developers.google.com/optimization/routing/routing_options#search_status

"""


class BasicRoutingMonitor:
    def __init__(self, do_solver: "OrtoolsVrpSolver"):
        self.model = do_solver.routing
        self.problem = do_solver.problem
        self.do_solver = do_solver
        self._counter = 0
        self._best_objective = np.inf
        self._counter_limit = 10000000
        self.nb_solutions = 0

    def __call__(self):
        logger.info(
            f"New solution found : --Cur objective : {self.model.CostVar().Max()}"
        )
        return True


class OrtoolsVrpSolver(VrpSolver):
    manager: Optional[pywrapcp.RoutingIndexManager] = None

    def init_model(self, **kwargs: Any) -> None:
        first_solution_strategy = kwargs.get(
            "first_solution_strategy", FirstSolutionStrategy.SAVINGS
        )
        local_search_metaheuristic = kwargs.get(
            "local_search_metaheuristic", LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        first_solution_strategy = first_solution_map[first_solution_strategy]
        local_search_metaheuristic = metaheuristic_map[local_search_metaheuristic]
        G, matrix_distance = build_graph(self.problem)
        matrix_distance_int = np.array(10**5 * matrix_distance, dtype=np.int_)
        demands = [
            self.problem.customers[i].demand for i in range(self.problem.customer_count)
        ]
        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            self.problem.customer_count,
            self.problem.vehicle_count,
            self.problem.start_indexes,
            self.problem.end_indexes,
        )
        routing = pywrapcp.RoutingModel(manager)
        # Create and register a transit callback.

        def distance_callback(from_index: int, to_index: int) -> int:
            """Returns the distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return matrix_distance_int[from_node, to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Capacity constraint.
        def demand_callback(from_index: int) -> float:
            """Returns the demand of the node."""
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            self.problem.vehicle_capacities,  # vehicle maximum capacities
            True,  # start cumul to zero
            "Capacity",
        )
        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = first_solution_strategy
        search_parameters.local_search_metaheuristic = local_search_metaheuristic
        # Solve the problem.
        self.manager = manager
        self.routing = routing
        self.search_parameters = search_parameters
        logger.info("Initialized ...")

    def retrieve(self, solution: pywrapcp.Assignment) -> VrpSolution:
        if self.manager is None:
            raise RuntimeError(
                "self.manager should be not None when calling self.retrieve()."
            )
        vehicle_tours: list[list[int]] = []
        vehicle_tours_all: list[list[int]] = []
        vehicle_count: int = self.problem.vehicle_count
        objective = 0.0
        route_distance = 0.0
        for vehicle_id in range(vehicle_count):
            vehicle_tours.append([])
            vehicle_tours_all.append([])
            index = self.routing.Start(vehicle_id)
            plan_output = f"Route for vehicle {vehicle_id}:\n"
            route_load = 0.0
            cnt = 0
            while not self.routing.IsEnd(index):
                node_index = self.manager.IndexToNode(index)
                if cnt != 0:
                    vehicle_tours[-1] += [node_index]
                vehicle_tours_all[-1] += [node_index]
                cnt += 1
                route_load += self.problem.customers[node_index].demand
                plan_output += f" {node_index} Load({route_load}) -> "
                previous_index = index
                index = solution.Value(self.routing.NextVar(index))
                route_distance += self.routing.GetArcCostForVehicle(
                    previous_index, index, vehicle_id
                )
                objective += self.problem.evaluate_function_indexes(
                    node_index, self.manager.IndexToNode(index)
                )
            vehicle_tours_all[-1] += [self.manager.IndexToNode(index)]
        logger.debug(f"Route distance : {route_distance}")
        logger.debug(f"Vehicle tours : {vehicle_tours}")
        logger.info(f"Objective : {objective}")
        logger.debug(f"Vehicle tours all : {vehicle_tours_all}")
        variable_vrp = VrpSolution(
            problem=self.problem,
            list_start_index=self.problem.start_indexes,
            list_end_index=self.problem.end_indexes,
            list_paths=vehicle_tours,
            length=None,
            lengths=None,
            capacities=None,
        )
        return variable_vrp

    def solve(self, time_limit: Optional[int] = 100, **kwargs: Any) -> ResultStorage:
        if self.manager is None:
            self.init_model(**kwargs)
            if self.manager is None:
                raise RuntimeError(
                    "self.manager should be not None after self.init_model() being called."
                )
        if time_limit is not None:
            self.search_parameters.time_limit.seconds = time_limit
        if kwargs.get("verbose", False):
            callback = BasicRoutingMonitor(do_solver=self)
            self.routing.AddAtSolutionCallback(callback)
        solution: pywrapcp.Assignment = self.routing.SolveWithParameters(
            self.search_parameters
        )
        variable_vrp = self.retrieve(solution)
        fit = self.aggreg_from_sol(variable_vrp)
        return self.create_result_storage(
            [(variable_vrp, fit)],
        )
