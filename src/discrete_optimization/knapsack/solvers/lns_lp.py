#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from collections.abc import Iterable
from typing import Any

from discrete_optimization.generic_tools.lns_mip import (
    OrtoolsMathOptConstraintHandler,
)
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers.greedy import (
    ResultStorage,
)
from discrete_optimization.knapsack.solvers.lp import MathOptKnapsackSolver


class MathOptKnapsackConstraintHandler(OrtoolsMathOptConstraintHandler):
    def __init__(self, problem: KnapsackProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: MathOptKnapsackSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Add constraints to the internal model of a solver based on previous solutions

        Args:
            solver: solver whose internal model is updated
            result_storage: all results so far
            result_storage_last_iteration: results from last LNS iteration only
            **kwargs:

        Returns:
            list of added constraints

        """
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution = self.extract_best_solution_from_last_iteration(
            result_storage=result_storage,
            result_storage_last_iteration=result_storage_last_iteration,
        )
        if current_solution is None:
            raise ValueError("result_storage.get_best_solution() should not be None.")
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() should be a KnapsackSolution."
            )
        solver.set_warm_start(current_solution)

        x_var = solver.variable_decision["x"]
        lns_constraint = []
        for c in range(self.problem.nb_items):
            if c in subpart_item:
                lns_constraint.append(
                    solver.add_linear_constraint(
                        x_var[c] == current_solution.list_taken[c], name=str(c)
                    )
                )
        return lns_constraint
