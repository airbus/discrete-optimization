"""Greedy solvers for coloring problem : binding from networkx library methods."""

#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Optional

import networkx as nx

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


strategies = [
    "largest_first",
    "random_sequential",
    "smallest_last",
    "independent_set",
    "connected_sequential_dfs",
    "connected_sequential_bfs",
    "connected_sequential",
    "saturation_largest_first",
    "DSATUR",
]


class NXGreedyColoringMethod(Enum):
    largest_first = "largest_first"
    random_sequential = "random_sequential"
    smallest_last = "smallest_last"
    independent_set = "independent_set"
    connected_sequential_dfs = "connected_sequential_dfs"
    connected_sequential_bfs = "connected_sequential_bfs"
    connected_sequential = "connected_sequential"
    saturation_largest_first = "saturation_largest_first"
    dsatur = "DSATUR"
    best = "best"


class GreedyColoring(SolverColoring):
    """Binded solver of networkx heuristics for coloring problem."""

    def __init__(
        self,
        coloring_model: ColoringProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        SolverColoring.__init__(self, coloring_model=coloring_model)
        self.nx_graph = self.coloring_model.graph.to_networkx()
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.coloring_model,
            params_objective_function=params_objective_function,
        )

    def solve(self, **kwargs: Any) -> ResultStorage:
        """Run the greedy solver for the given problem.

        Keyword Args:
            strategy (NXGreedyColoringMethod) : one of the method used by networkx to compute coloring solution,
                                                or use NXGreedyColoringMethod.best to run each of them and return
                                                the best result.
            verbose (bool)


        Returns:
            results (ResultStorage) : storage of solution found by the greedy solver.

        """
        greedy_strategy: NXGreedyColoringMethod = kwargs.get(
            "strategy", NXGreedyColoringMethod.best
        )
        strategy_name = greedy_strategy.name
        if strategy_name == "best":
            strategies_to_test = strategies
        else:
            strategies_to_test = [strategy_name]
        best_solution = None
        best_nb_color = float("inf")
        for strategy in strategies_to_test:
            try:
                colors = nx.algorithms.coloring.greedy_color(
                    self.nx_graph, strategy=strategy, interchange=False
                )
                sorted_nodes = sorted(list(colors.keys()))
                number_colors = len(set(list(colors.values())))
                raw_solution = [colors[i] for i in sorted_nodes]
                logger.debug(f"{strategy} : number colors : {number_colors}")
                if number_colors < best_nb_color:
                    best_solution = raw_solution
                    best_nb_color = number_colors
            except Exception as e:
                logger.info(f"Failed strategy : {strategy} {e}")
        logger.debug(f"best : {best_nb_color}")
        solution = ColoringSolution(
            self.coloring_model, colors=best_solution, nb_color=None
        )
        solution = solution.to_reformated_solution()
        fit = self.aggreg_sol(solution)
        logger.debug(f"Solution found : {solution}")
        return ResultStorage(
            list_solution_fits=[(solution, fit)],
            best_solution=solution,
            mode_optim=self.params_objective_function.sense_function,
        )
