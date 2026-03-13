#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Iterable

from minizinc import Instance

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    FloatHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp_mzn import MznConstraintHandler
from discrete_optimization.generic_tools.mzn_tools import MinizincCpSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers.cp_mzn import Cp2KnapsackSolver


class KnapsackMznConstraintHandler(MznConstraintHandler):
    hyperparameters = [
        FloatHyperparameter(name="fraction_to_fix", default=0.9, low=0.0, high=1.0),
    ]

    def __init__(self, problem: KnapsackProblem, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        solver: MinizincCpSolver,
        result_storage: ResultStorage,
        result_storage_last_iteration: ResultStorage,
        child_instance: Instance,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Add constraints to the internal model of a solver based on previous solutions

        Args:
            solver: solver whose internal model is updated
            result_storage: all results so far
            result_storage_last_iteration: results from last LNS iteration only
            child_instance: minizinc instance where to include the constraints
            **kwargs:

        Returns:
            empty list, not used

        """
        if not isinstance(solver, Cp2KnapsackSolver):
            raise ValueError("cp_solver must a CPKnapsackMZN2 for this constraint.")
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
        if not isinstance(current_solution, KnapsackSolution):
            raise ValueError(
                "result_storage.get_best_solution() should be a KnapsackSolution."
            )
        list_strings = []
        for item in subpart_item:
            list_strings += [
                "constraint taken["
                + str(item + 1)
                + "] == "
                + str(current_solution.list_taken[item])
                + ";\n"
            ]
            child_instance.add_string(list_strings[-1])
        return list_strings
