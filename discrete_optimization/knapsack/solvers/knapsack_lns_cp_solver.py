#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Iterable, Optional

from minizinc import Instance

from discrete_optimization.generic_tools.lns_cp import ConstraintHandler
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN2
from discrete_optimization.knapsack.solvers.greedy_solvers import ResultStorage


class ConstraintHandlerKnapsack(ConstraintHandler):
    def __init__(self, problem: KnapsackModel, fraction_to_fix: float = 0.9):
        self.problem = problem
        self.fraction_to_fix = fraction_to_fix
        self.iter = 0

    def adding_constraint_from_results_store(
        self,
        cp_solver: CPKnapsackMZN2,
        child_instance: Instance,
        result_storage: ResultStorage,
        last_result_store: Optional[ResultStorage] = None,
    ) -> Iterable[Any]:
        subpart_item = set(
            random.sample(
                range(self.problem.nb_items),
                int(self.fraction_to_fix * self.problem.nb_items),
            )
        )
        current_solution: KnapsackSolution = result_storage.get_best_solution_fit()[0]
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

    def remove_constraints_from_previous_iteration(
        self,
        cp_solver: CPKnapsackMZN2,
        child_instance,
        previous_constraints: Iterable[Any],
    ):
        pass
