#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from ortools.constraint_solver import (
    pywrapcp,
    routing_parameters_pb2,
)
from ortools.util.optional_boolean_pb2 import BOOL_FALSE, BOOL_TRUE

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import ResultStorage, SolverDO
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop

# Attempt to import Enums from the existing GPDP ortools_routing module
# to maintain consistency as requested.
from discrete_optimization.gpdp.solvers.ortools_routing import (
    FirstSolutionStrategy,
    LocalSearchMetaheuristic,
)
from discrete_optimization.top.problem import TeamOrienteeringProblem
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.utils import compute_length_matrix

logger = logging.getLogger(__name__)


class OrtoolsTopSolver(SolverDO):
    problem: TeamOrienteeringProblem
    manager: Optional[pywrapcp.RoutingIndexManager] = None
    routing: Optional[pywrapcp.RoutingModel] = None

    def __init__(
        self,
        problem: TeamOrienteeringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        # Precompute distance matrix (float)
        _, self.distance = compute_length_matrix(self.problem)
        self.search_parameters = None

    def init_model(self, scaling: float = 1, **kwargs: Any) -> None:
        num_locations = len(self.problem.customers)
        num_vehicles = self.problem.vehicle_count
        self.manager = pywrapcp.RoutingIndexManager(
            num_locations,
            num_vehicles,
            self.problem.start_indexes,
            self.problem.end_indexes,
        )
        self.routing = pywrapcp.RoutingModel(self.manager)
        # Pre-scale distance matrix to integers
        matrix_distance_int = np.array(scaling * self.distance, dtype=np.int64)

        def distance_callback(from_index: int, to_index: int) -> int:
            from_node = self.manager.IndexToNode(from_index)
            to_node = self.manager.IndexToNode(to_index)
            return matrix_distance_int[from_node, to_node]

        transit_callback_index = self.routing.RegisterTransitCallback(distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 5. Add Distance Dimension (Max Length Constraint)
        dimension_name = "Distance"
        self.routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            int(self.problem.max_length_tours * scaling),  # vehicle max travel distance
            True,  # start cumul to zero
            dimension_name,
        )

        # 6. Define Rewards (Disjunctions) for TOP
        # In TOP, nodes can be skipped. We use disjunctions.
        # Penalty should be high enough to encourage visiting.
        penalty_scaler = kwargs.get("penalty_scaler", 1000)
        for node_idx in range(num_locations):
            # Skip Start/End depots
            if (
                node_idx in self.problem.start_indexes
                or node_idx in self.problem.end_indexes
            ):
                continue

            reward = self.problem.customers[node_idx].reward
            if reward > 0:
                # We penalize skipping a node based on its reward.
                penalty = int(reward * scaling * penalty_scaler)
                self.routing.AddDisjunction(
                    [self.manager.NodeToIndex(node_idx)], penalty
                )

        # 7. Build Search Parameters
        self.search_parameters = self.build_search_parameters(**kwargs)
        logger.info("Initialized OR-Tools model for TOP.")

    def build_search_parameters(
        self, **kwargs: Any
    ) -> routing_parameters_pb2.RoutingSearchParameters:
        # Retrieve strategies from kwargs or defaults
        first_solution_strategy = kwargs.get(
            "first_solution_strategy", FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        local_search_metaheuristic = kwargs.get(
            "local_search_metaheuristic", LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        time_limit = kwargs.get("time_limit", 30)
        use_lns = kwargs.get("use_lns", True)
        use_cp = kwargs.get("use_cp", True)
        use_cp_sat = kwargs.get("use_cp_sat", False)
        verbose = kwargs.get("verbose", False)

        # Convert Enums to integers if necessary
        if hasattr(first_solution_strategy, "value"):
            first_solution_strategy = first_solution_strategy.value
        if hasattr(local_search_metaheuristic, "value"):
            local_search_metaheuristic = local_search_metaheuristic.value

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.log_search = verbose
        search_parameters.first_solution_strategy = first_solution_strategy
        search_parameters.local_search_metaheuristic = local_search_metaheuristic
        search_parameters.time_limit.seconds = time_limit

        # Configure LNS and CP settings
        if use_lns:
            search_parameters.local_search_operators.use_path_lns = BOOL_TRUE
            search_parameters.local_search_operators.use_inactive_lns = BOOL_TRUE
        search_parameters.use_cp = BOOL_TRUE if use_cp else BOOL_FALSE
        search_parameters.use_cp_sat = BOOL_TRUE if use_cp_sat else BOOL_FALSE
        return search_parameters

    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: int = 100,
        **kwargs: Any,
    ) -> ResultStorage:
        if self.manager is None:
            self.init_model(**kwargs)
        self.search_parameters.time_limit.seconds = time_limit
        # 1. Setup Callback/Monitor
        callbacks_list = CallbackList(callbacks=callbacks)
        monitor = TopRoutingMonitor(self, callback=callbacks_list)
        self.routing.AddSearchMonitor(monitor)
        # 2. Solve
        try:
            self.routing.SolveWithParameters(self.search_parameters)
        except SolveEarlyStop as e:
            logger.info(e)
        return monitor.res


class TopRoutingMonitor(pywrapcp.SearchMonitor):
    """
    Monitor to retrieve intermediate solutions and handle callbacks
    (logging, early stopping) for the Team Orienteering Problem.
    """

    def __init__(self, do_solver: OrtoolsTopSolver, callback: Callback):
        super().__init__(do_solver.routing.solver())
        self.do_solver = do_solver
        self.model = do_solver.routing
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0
        self._best_objective = float("inf")

    def AtSolution(self) -> bool:
        """Called by OR-Tools whenever a valid solution is found."""
        current_obj = self.model.CostVar().Max()
        logger.info(f"Solution found: Objective {current_obj}")
        # Update best objective tracking
        if current_obj < self._best_objective:
            self._best_objective = current_obj
            self.retrieve_current_solution()
        else:
            # Optional: Retrieve periodically even if not strictly better
            # to capture diversity or if using different criteria
            if self.nb_solutions % 100 == 0:
                self.retrieve_current_solution()
        self.nb_solutions += 1
        # Check user defined callbacks (e.g. timeout, user interruption)
        stopping = self.callback.on_step_end(
            step=self.nb_solutions, res=self.res, solver=self.do_solver
        )

        if stopping:
            raise SolveEarlyStop("Solver stopped by user callback.")

        return not stopping

    def EnterSearch(self):
        self.callback.on_solve_start(solver=self.do_solver)

    def ExitSearch(self):
        self.callback.on_solve_end(res=self.res, solver=self.do_solver)

    def retrieve_current_solution(self) -> None:
        """Converts the current OR-Tools assignment to a VrpSolution."""
        vehicle_count = self.do_solver.problem.vehicle_count
        vehicle_tours: list[list[int]] = []

        for vehicle_id in range(vehicle_count):
            path = []
            index = self.model.Start(vehicle_id)

            # Note: VrpSolution expects the path strictly BETWEEN start and end.
            # We skip the start node for the list, but we must traverse it.

            # Move to next
            index = self.model.NextVar(index).Value()

            while not self.model.IsEnd(index):
                node_index = self.do_solver.manager.IndexToNode(index)
                path.append(node_index)
                index = self.model.NextVar(index).Value()

            vehicle_tours.append(path)

        # Create VrpSolution (which is compatible with TOP evaluation)
        # Note: TOP problem usually defines start/end indexes in the problem class
        variable_vrp = VrpSolution(
            problem=self.do_solver.problem,
            list_start_index=self.do_solver.problem.start_indexes,
            list_end_index=self.do_solver.problem.end_indexes,
            list_paths=vehicle_tours,
            length=None,
            lengths=None,
            capacities=None,
        )

        # Evaluate to get the fitness (fitness is usually MINIMIZATION in DO)
        # For TOP, fitness might be -reward or similar aggregation
        fit = self.do_solver.aggreg_from_sol(variable_vrp)
        self.res.append((variable_vrp, fit))
