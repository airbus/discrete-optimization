"""Simple travelling salesman problem between cities."""


#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import print_function

import logging
from typing import Any, Optional

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.tsp.common_tools_tsp import build_matrice_distance
from discrete_optimization.tsp.solver.tsp_solver import SolverTSP
from discrete_optimization.tsp.tsp_model import SolutionTSP, TSPModel

logger = logging.getLogger(__name__)


class TSP_ORtools(SolverTSP):
    def __init__(
        self,
        tsp_model: TSPModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):

        SolverTSP.__init__(self, tsp_model=tsp_model)
        self.node_count = self.tsp_model.node_count
        self.list_points = self.tsp_model.list_points
        self.start_index = self.tsp_model.start_index
        self.end_index = self.tsp_model.end_index
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.tsp_model, params_objective_function=params_objective_function
        )

    def init_model(self, **kwargs: Any) -> None:
        # Create the routing index manager.

        if self.node_count < 1000:
            matrix = build_matrice_distance(
                self.node_count,
                method=self.tsp_model.evaluate_function_indexes,
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
                    * self.tsp_model.evaluate_function_indexes(
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

    def solve(self, **kwargs: Any) -> ResultStorage:
        """Prints solution on console."""
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
        variableTSP = SolutionTSP(
            problem=self.tsp_model,
            start_index=self.tsp_model.start_index,
            end_index=self.tsp_model.end_index,
            permutation=sol[1:-1],
            lengths=None,
            length=None,
        )
        fitness = self.aggreg_sol(variableTSP)
        return ResultStorage(
            list_solution_fits=[(variableTSP, fitness)],
            mode_optim=self.params_objective_function.sense_function,
        )
