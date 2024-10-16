#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import time
from typing import Any, Optional

import didppy as dp
import numpy as np

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    CategoricalHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.knapsack.problem import KnapsackProblem, KnapsackSolution
from discrete_optimization.knapsack.solvers import KnapsackSolver

logger = logging.getLogger(__name__)


class DpKnapsackSolver(KnapsackSolver, DpSolver, WarmstartMixin):
    hyperparameters = DpSolver.hyperparameters + [
        CategoricalHyperparameter(
            name="float_cost", choices=[True, False], default=False
        ),
        CategoricalHyperparameter(
            name="dual_bound",
            choices=[True, False],
            default=True,
            depends_on=("float_cost", True),
        ),
    ]
    model: dp.Model
    transitions: dict

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["float_cost"]:
            self.init_model_float(**kwargs)
        else:
            self.init_model_int(**kwargs)

    def init_model_int(self, **kwargs):
        """Adapted from https://didppy.readthedocs.io/en/stable/quickstart.html"""
        n = self.problem.nb_items
        weights = [self.problem.list_items[i].weight for i in range(n)]
        profits = [self.problem.list_items[i].value for i in range(n)]
        capacity = self.problem.max_capacity
        model = dp.Model(maximize=True, float_cost=False)
        item = model.add_object_type(number=self.problem.nb_items)
        r = model.add_int_var(target=capacity)
        i = model.add_element_var(object_type=item, target=0)

        w = model.add_int_table(weights)
        p = model.add_int_table(profits)
        pack = dp.Transition(
            name="pack",
            cost=p[i] + dp.IntExpr.state_cost(),
            effects=[(r, r - w[i]), (i, i + 1)],
            preconditions=[i < n, r >= w[i]],
        )
        model.add_transition(pack)
        self.transitions = {"pack": pack}

        ignore = dp.Transition(
            name="ignore",
            cost=dp.IntExpr.state_cost(),
            effects=[(i, i + 1)],
            preconditions=[i < n],
        )
        self.transitions["ignore"] = ignore
        model.add_transition(ignore)
        model.add_base_case([i == n])
        self.model = model

    def init_model_float(self, **kwargs):
        n = self.problem.nb_items
        weights = [self.problem.list_items[i].weight for i in range(n)]
        profits = [self.problem.list_items[i].value for i in range(n)]
        capacity = self.problem.max_capacity
        model = dp.Model(maximize=True, float_cost=True)
        item = model.add_object_type(number=self.problem.nb_items)
        r = model.add_float_var(target=capacity)
        i = model.add_element_var(object_type=item, target=0)

        w = model.add_float_table(weights)
        p = model.add_float_table(profits)
        undone = model.add_set_var(
            object_type=item, target=range(self.problem.nb_items)
        )
        pack = dp.Transition(
            name="pack",
            cost=p[i] + dp.FloatExpr.state_cost(),
            effects=[(r, r - w[i]), (i, i + 1), (undone, undone.remove(i))],
            preconditions=[i < n, r >= w[i]],
        )
        model.add_transition(pack)
        self.transitions = {"pack": pack}

        ignore = dp.Transition(
            name="ignore",
            cost=dp.FloatExpr.state_cost(),
            effects=[(i, i + 1), (undone, undone.remove(i))],
            preconditions=[i < n],
        )
        self.transitions["ignore"] = ignore
        model.add_transition(ignore)
        model.add_base_case([i == n])
        if kwargs["dual_bound"]:
            value_per_weight = [
                profits[i] / weights[i] for i in range(self.problem.nb_items)
            ]
            value_pw = model.add_float_table(value_per_weight)
            model.add_dual_bound(p[undone])
            model.add_dual_bound(
                (undone.is_empty()).if_then_else(0, value_pw.max(undone) * r)
            )
        self.model = model

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        taken = [0 for _ in range(self.problem.nb_items)]
        for i, t in enumerate(sol.transitions):
            if t.name == "pack":
                taken[i] = 1
                logger.debug(f"pack {i}")
        return KnapsackSolution(problem=self.problem, list_taken=taken)

    def set_warm_start(self, solution: KnapsackSolution) -> None:
        transition = []
        for i in range(len(solution.list_taken)):
            if solution.list_taken[i] == 1:
                transition.append(self.transitions["pack"])
            else:
                transition.append(self.transitions["ignore"])
        self.initial_solution = transition


class ExactDpKnapsackSolver(KnapsackSolver):
    hyperparameters = [
        CategoricalHyperparameter(
            name="greedy_start", default=False, choices=[True, False]
        ),
    ]

    def __init__(
        self,
        problem: KnapsackProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.nb_items = self.problem.nb_items
        capacity = int(self.problem.max_capacity)
        if capacity != self.problem.max_capacity:
            logger.warning(
                "knapsack_problem.max_capacity should be an integer for dynamic programming. "
                "Coercing it to an integer for the solver."
            )
        self.capacity: int = capacity
        self.table = np.zeros((self.nb_items + 1, self.capacity + 1))

    def solve(self, **kwargs: Any) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        start_by_most_promising = kwargs["greedy_start"]
        max_items = kwargs.get("max_items", self.problem.nb_items + 1)
        max_items = min(self.problem.nb_items + 1, max_items)
        max_time_seconds = kwargs.get("time_limit", None)
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
        return self.create_result_storage(
            [(sol, fit)],
        )
