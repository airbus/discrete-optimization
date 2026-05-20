#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any, Optional

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_solver import ResultStorage
from discrete_optimization.knapsack.problem import (
    Item,
    KnapsackProblem,
    KnapsackSolution,
)
from discrete_optimization.knapsack.solvers import KnapsackSolver


def compute_density(knapsack_problem: KnapsackProblem) -> list[Item]:
    dd = sorted(
        [
            l
            for l in knapsack_problem.list_items
            if l.weight <= knapsack_problem.max_capacity
        ],
        key=lambda x: x.value / x.weight,
        reverse=True,
    )
    return dd


def compute_density_and_penalty(knapsack_problem: KnapsackProblem) -> list[Item]:
    dd = sorted(
        [
            l
            for l in knapsack_problem.list_items
            if l.weight <= knapsack_problem.max_capacity
        ],
        key=lambda x: x.value / x.weight - x.weight,
        reverse=True,
    )
    return dd


def greedy_using_queue(
    knapsack_problem: KnapsackProblem,
    method_queue: Optional[Callable[[KnapsackProblem], list[Item]]] = None,
) -> KnapsackSolution:
    if method_queue is None:
        method_queue = compute_density
    value = 0.0
    weight = 0.0
    taken = [0] * knapsack_problem.nb_items
    sorted_per_density = method_queue(knapsack_problem)
    for i in range(len(sorted_per_density)):
        if sorted_per_density[i].weight + weight <= knapsack_problem.max_capacity:
            taken[knapsack_problem.index_to_index_list[sorted_per_density[i].index]] = 1
            value += sorted_per_density[i].value
            weight += sorted_per_density[i].weight
        else:
            continue
    return KnapsackSolution(
        problem=knapsack_problem, value=value, weight=weight, list_taken=taken
    )


def best_of_greedy(knapsack_problem: KnapsackProblem) -> KnapsackSolution:
    result1 = greedy_using_queue(knapsack_problem, compute_density)
    result2 = greedy_using_queue(knapsack_problem, compute_density_and_penalty)
    if result1.value is None or result2.value is None:
        raise RuntimeError(
            "result1.value and result2.value should not be None at this point."
        )
    return result1 if result1.value > result2.value else result2


class GreedyBestKnapsackSolver(KnapsackSolver):
    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        res = best_of_greedy(self.problem)
        fit = self.aggreg_from_sol(res)
        result = self.create_result_storage(
            [(res, fit)],
        )
        cb.on_solve_end(result, self)
        return result


class GreedyDummyKnapsackSolver(KnapsackSolver):
    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        cb = CallbackList(callbacks)
        cb.on_solve_start(self)
        sol = KnapsackSolution(
            problem=self.problem,
            value=0,
            weight=0,
            list_taken=[0] * self.problem.nb_items,
        )
        fit = self.aggreg_from_sol(sol)
        result = self.create_result_storage(
            [(sol, fit)],
        )
        cb.on_solve_end(result, self)
        return result
