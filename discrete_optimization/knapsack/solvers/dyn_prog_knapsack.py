#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import time
from typing import Any, List, Optional

import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers.knapsack_solver import SolverKnapsack

logger = logging.getLogger(__name__)


class KnapsackDynProg(SolverKnapsack):
    hyperparameters = [
        CategoricalHyperparameter(
            name="greedy_start", default=False, choices=[True, False]
        ),
    ]

    def __init__(
        self,
        problem: KnapsackModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.nb_items = self.problem.nb_items
        capacity = int(self.problem.max_capacity)
        if capacity != self.problem.max_capacity:
            logger.warning(
                "knapsack_model.max_capacity should be an integer for dynamic programming. "
                "Coercing it to an integer for the solver."
            )
        self.capacity: int = capacity
        self.table = np.zeros((self.nb_items + 1, self.capacity + 1))

    def solve(self, **kwargs: Any) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        start_by_most_promising = kwargs["greedy_start"]
        max_items = kwargs.get("max_items", self.problem.nb_items + 1)
        max_items = min(self.problem.nb_items + 1, max_items)
        max_time_seconds = kwargs.get("max_time_seconds", None)
        if max_time_seconds is None:
            do_time = False
        else:
            do_time = True
        if start_by_most_promising:
            promising = sorted(
                [
                    l
                    for l in self.problem.list_items
                    if l.weight <= self.problem.max_capacity
                ],
                key=lambda x: x.value / x.weight,
                reverse=True,
            )
            index_promising = [l.index for l in promising]
        else:
            index_promising = [l.index for l in self.problem.list_items]
        if do_time:
            t_start = time.time()
        cur_indexes = (0, 0)
        for nb_item in range(1, len(index_promising) + 1):
            if do_time:
                if time.time() - t_start > max_time_seconds:
                    break
            index_item = self.problem.index_to_index_list[index_promising[nb_item - 1]]
            for capacity in range(self.capacity + 1):
                weight = int(
                    self.problem.list_items[index_item].weight
                )  # weight should be an integer for this algo
                value = self.problem.list_items[index_item].value
                if weight > capacity:
                    self.table[nb_item, capacity] = self.table[nb_item - 1, capacity]
                    continue
                self.table[nb_item, capacity] = max(
                    self.table[nb_item - 1, capacity],
                    self.table[nb_item - 1, capacity - weight] + value,
                )
                if capacity == self.capacity:
                    cur_indexes = (nb_item, capacity)
                    logger.debug(f"Cur obj : {self.table[nb_item, capacity]}")
        taken = [0] * self.nb_items
        weight = 0
        value = 0
        cur_value = self.table[cur_indexes[0], cur_indexes[1]]
        while cur_indexes[0] != 0:
            value_left = self.table[cur_indexes[0] - 1, cur_indexes[1]]
            if cur_value != value_left:
                index_item = self.problem.index_to_index_list[
                    index_promising[cur_indexes[0] - 1]
                ]
                taken[index_item] = 1
                value += self.problem.list_items[index_item].value
                weight += int(self.problem.list_items[index_item].weight)
                cur_indexes = (
                    cur_indexes[0] - 1,
                    cur_indexes[1] - int(self.problem.list_items[index_item].weight),
                )
            else:
                cur_indexes = (cur_indexes[0] - 1, cur_indexes[1])
            cur_value = self.table[cur_indexes[0], cur_indexes[1]]
        sol = KnapsackSolution(
            problem=self.problem, value=value, weight=weight, list_taken=taken
        )
        fit = self.aggreg_from_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            best_solution=sol,
            mode_optim=self.params_objective_function.sense_function,
        )
