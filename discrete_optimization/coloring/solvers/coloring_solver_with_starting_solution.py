#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any

from discrete_optimization.coloring.coloring_model import (
    ColoringProblem,
    ColoringSolution,
)
from discrete_optimization.coloring.solvers.coloring_solver import SolverColoring
from discrete_optimization.coloring.solvers.greedy_coloring import (
    GreedyColoring,
    NXGreedyColoringMethod,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
    EnumHyperparameter,
)

logger = logging.getLogger(__name__)


class SolverColoringWithStartingSolution(SolverColoring):
    hyperparameters = [
        CategoricalHyperparameter("greedy_start", choices=[True], default=True),
        EnumHyperparameter(
            "greedy_method",
            enum=NXGreedyColoringMethod,
            default=NXGreedyColoringMethod.best,
        ),
    ]

    def get_starting_solution(self, **kwargs: Any) -> ColoringSolution:
        """Used by the init_model method to provide a greedy first solution

        Keyword Args:
            greedy_start (bool): use heuristics (based on networkx) to compute starting solution, otherwise the
                          dummy method is used.
            verbose (bool): verbose option.

        Returns (ColoringSolution): a starting coloring solution that can be used by lns.

        """
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        greedy_start = kwargs["greedy_start"]
        params_objective_function = kwargs.get("params_objective_function", None)
        if greedy_start:
            logger.info("Computing greedy solution")
            greedy_solver = GreedyColoring(
                self.problem,
                params_objective_function=params_objective_function,
            )
            result_store = greedy_solver.solve(strategy=kwargs["greedy_method"])
            solution = result_store.get_best_solution()
            if solution is None:
                raise RuntimeError(
                    "greedy_solver.solve().get_best_solution() " "should not be None."
                )
            if not isinstance(solution, ColoringSolution):
                raise RuntimeError(
                    "greedy_solver.solve().get_best_solution() "
                    "should be a ColoringSolution."
                )
        else:
            logger.info("Get dummy solution")
            solution = self.problem.get_dummy_solution()
        return solution
