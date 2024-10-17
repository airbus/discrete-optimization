"""Simple travelling salesman problem between cities."""


#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import print_function

import logging
from typing import Any, Optional

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsp.solvers import TspSolver
from discrete_optimization.tsp.utils import build_matrice_distance

logger = logging.getLogger(__name__)


class ORtoolsTspSolver(TspSolver):
    def __init__(
        self,
        problem: TspProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):

        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.node_count = self.problem.node_count
        self.list_points = self.problem.list_points
        self.start_index = self.problem.start_index
        self.end_index = self.problem.end_index
        self.routing: pywrapcp.RoutingModel = None
        self.manager: pywrapcp.RoutingIndexManager = None
        self.search_parameters = None

    def init_model(self, **kwargs: Any) -> None:
        # Create the routing index manager.

        if self.node_count < 1000:
            matrix = build_matrice_distance(
                self.node_count,
                method=self.problem.evaluate_function_indexes,
            )
            distance_matrix = 10**6 * matrix.astype(np.int_)

            def distance_callback(from_index: int, to_index: int) -> int:
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return distance_matrix[from_node, to_node]

        else:

            def distance_callback(from_index: int, to_index: int) -> int:
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                return int(
                    10**6
                    * self.problem.evaluate_function_indexes(
                        self.list_points[from_node], self.list_points[to_node]
                    )
                )

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            self.node_count, 1, [self.start_index], [self.end_index]
        )
        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc.
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = 30
        self.manager = manager
        self.routing = routing
        self.search_parameters = search_parameters

    def solve(self, time_limit: Optional[int] = 100, **kwargs: Any) -> ResultStorage:
        """Prints solution on console."""
        if self.routing is None:
            self.init_model(**kwargs)
        self.search_parameters.time_limit.seconds = int(time_limit)
        solution = self.routing.SolveWithParameters(self.search_parameters)
        logger.debug(f"Objective: {solution.ObjectiveValue()} miles")
        index = self.routing.Start(0)
        index_real = self.manager.IndexToNode(index)
        sol = [index_real]
        route_distance = 0
        while not self.routing.IsEnd(index):
            previous_index = index
            index = solution.Value(self.routing.NextVar(index))
            index_real = self.manager.IndexToNode(index)
            sol += [index_real]
            route_distance += self.routing.GetArcCostForVehicle(
                previous_index, index, 0
            )
        variableTsp = TspSolution(
            problem=self.problem,
            start_index=self.problem.start_index,
            end_index=self.problem.end_index,
            permutation=sol[1:-1],
            lengths=None,
            length=None,
        )
        fitness = self.aggreg_from_sol(variableTsp)
        return self.create_result_storage(
            [(variableTsp, fitness)],
        )
