#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from collections.abc import MutableSequence
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    TupleFitness,
    build_aggreg_function_and_params_objective,
    get_default_objective_setup,
)

fitness_class = Union[float, TupleFitness]


class ResultStorage(MutableSequence):
    """Storage for solver results.

    ResultStorage inherits from MutableSequence so you can
    - iterate over it (will iterate over tuples (sol, fit)
    - append directly to it (a tuple (sol, fit))
    - pop, extend, ...

    """

    list_solution_fits: list[tuple[Solution, fitness_class]]

    def __init__(
        self,
        mode_optim: ModeOptim,
        list_solution_fits: Optional[list[tuple[Solution, fitness_class]]] = None,
    ):
        if list_solution_fits is None:
            self.list_solution_fits = []
        else:
            self.list_solution_fits = list_solution_fits
        self.mode_optim = mode_optim
        self.maximize = mode_optim == ModeOptim.MAXIMIZATION

    def __add__(self, other: ResultStorage) -> ResultStorage:
        return merge_results_storage(self, other)

    def __getitem__(self, index) -> tuple[Solution, fitness_class]:
        return self.list_solution_fits[index]

    def __len__(self) -> int:
        return len(self.list_solution_fits)

    def __setitem__(self, index: int, value: tuple[Solution, fitness_class]):
        self.list_solution_fits[index] = value

    def __delitem__(self, index: int) -> None:
        del self.list_solution_fits[index]

    def insert(self, index, value: tuple[Solution, fitness_class]) -> None:
        self.list_solution_fits.insert(index, value)

    def get_best_solution_fit(
        self, satisfying: Optional[Problem] = None
    ) -> Union[tuple[Solution, fitness_class], tuple[None, None]]:
        if len(self.list_solution_fits) == 0:
            return None, None
        if satisfying is None:
            f = max if self.maximize else min
            return f(self.list_solution_fits, key=lambda x: x[1])
        else:
            sorted_solution_fits = sorted(
                self.list_solution_fits, key=lambda x: x[1], reverse=self.maximize
            )
            for sol, fit in sorted_solution_fits:
                if satisfying.satisfy(sol):
                    return sol, fit
            return None, None

    def get_last_best_solution(self) -> tuple[Solution, fitness_class]:
        f = max if self.maximize else min
        best = f(self.list_solution_fits, key=lambda x: x[1])[1]
        sol = max(
            [
                i
                for i in range(len(self.list_solution_fits))
                if self.list_solution_fits[i][1] == best
            ]
        )
        return self.list_solution_fits[sol]

    def get_random_best_solution(self) -> tuple[Solution, fitness_class]:
        f = max if self.maximize else min
        best = f(self.list_solution_fits, key=lambda x: x[1])[1]
        sol = random.choice(
            [
                i
                for i in range(len(self.list_solution_fits))
                if self.list_solution_fits[i][1] == best
            ]
        )
        return self.list_solution_fits[sol]

    def get_random_solution(self) -> tuple[Solution, fitness_class]:
        s = [
            l
            for l in self.list_solution_fits
            if l[1] != self.get_best_solution_fit()[1]
        ]
        if len(s) > 0:
            return random.choice(s)
        else:
            return random.choice(self.list_solution_fits)

    def get_best_solution(self) -> Optional[Solution]:
        f = max if self.maximize else min
        if len(self.list_solution_fits) == 0:
            return None
        return f(self.list_solution_fits, key=lambda x: x[1])[0]

    def get_n_best_solution(
        self, n_solutions: int
    ) -> list[tuple[Solution, fitness_class]]:
        f = max if self.maximize else min
        n = min(n_solutions, len(self.list_solution_fits))
        l = sorted(self.list_solution_fits, key=lambda x: x[1])[:n]
        return l

    def remove_duplicate_solutions(self, var_name: str) -> None:
        index_to_remove = []
        for i in range(len(self.list_solution_fits) - 1):
            sol1 = getattr(self.list_solution_fits[i][0], var_name)
            for j in range(i + 1, len(self.list_solution_fits)):
                sol2 = getattr(self.list_solution_fits[j][0], var_name)
                all_similar = True
                for k in range(len(sol1)):
                    if sol1[k] != sol2[k]:
                        all_similar = False
                        break
                if all_similar:
                    if j not in index_to_remove:
                        index_to_remove.append(j)

        self.list_solution_fits = [
            self.list_solution_fits[i]
            for i in range(len(self.list_solution_fits))
            if i not in index_to_remove
        ]


def merge_results_storage(
    result_1: ResultStorage, result_2: ResultStorage
) -> ResultStorage:
    if result_1.mode_optim != result_2.mode_optim:
        raise ValueError("Cannot merge result_storages with different mode_optim")
    return ResultStorage(
        mode_optim=result_1.mode_optim,
        list_solution_fits=result_1.list_solution_fits + result_2.list_solution_fits,
    )


def from_solutions_to_result_storage(
    list_solution: list[Solution],
    problem: Problem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
) -> ResultStorage:
    if params_objective_function is None:
        params_objective_function = get_default_objective_setup(problem)
    mode_optim = params_objective_function.sense_function
    (
        aggreg_from_sol,
        aggreg_from_dict,
        params_objective_function,
    ) = build_aggreg_function_and_params_objective(
        problem=problem, params_objective_function=params_objective_function
    )
    list_solution_fit: list[tuple[Solution, fitness_class]] = []
    for s in list_solution:
        list_solution_fit += [(s, aggreg_from_sol(s))]
    return ResultStorage(mode_optim=mode_optim, list_solution_fits=list_solution_fit)


def result_storage_to_pareto_front(
    result_storage: ResultStorage, problem: Optional[Problem] = None
) -> "ParetoFront":
    list_solution_fits = result_storage.list_solution_fits
    if problem is not None:
        list_solution_fits = [
            (solution, problem.evaluate_mobj(solution))
            for solution, _ in list_solution_fits
        ]
    pf = ParetoFront(
        list_solution_fits=list_solution_fits,
        mode_optim=result_storage.mode_optim,
    )
    pf.finalize()
    return pf


class ParetoFront(ResultStorage):
    def __init__(
        self,
        list_solution_fits: list[tuple[Solution, fitness_class]],
        mode_optim: ModeOptim = ModeOptim.MAXIMIZATION,
    ):
        super().__init__(mode_optim=mode_optim, list_solution_fits=list_solution_fits)
        self.paretos: list[tuple[Solution, TupleFitness]] = []

    def add_point(self, solution: Solution, tuple_fitness: TupleFitness) -> None:
        if self.maximize:
            if all(tuple_fitness >= t[1] for t in self.paretos):
                self.paretos += [(solution, tuple_fitness)]
                pp = []
                for p in self.paretos:
                    if p[1] < tuple_fitness:
                        continue
                    else:
                        pp += [p]
                self.paretos = pp
        if not self.maximize:
            if all(tuple_fitness <= t[1] for t in self.paretos):
                self.paretos += [(solution, tuple_fitness)]
                pp = []
                for p in self.paretos:
                    if p[1] > tuple_fitness:
                        continue
                    else:
                        pp += [p]
                self.paretos = pp

    def len_pareto_front(self) -> int:
        return len(self.paretos)

    def finalize(self) -> None:
        self.paretos = []
        for s, t in self.list_solution_fits:
            if not isinstance(t, TupleFitness):
                raise RuntimeError(
                    "self.list_solution_fits must be a list of tuple[Solution, TupleFitness] "
                    "for a Pareto front."
                )
            self.add_point(solution=s, tuple_fitness=t)

    def compute_extreme_points(self) -> list[tuple[Solution, TupleFitness]]:
        function_used = max if self.maximize else min
        number_fitness = self.list_solution_fits[0][1].size  # type: ignore
        extreme_points: list[tuple[Solution, TupleFitness]] = []
        for i in range(number_fitness):
            extr: tuple[Solution, TupleFitness] = function_used(self.paretos, key=lambda x: x[1].vector_fitness[i])  # type: ignore
            extreme_points += [extr]
        return extreme_points


def plot_storage_2d(
    result_storage: ResultStorage,
    name_axis: list[str],
    ax: Any = None,
    color: str = "r",
) -> None:
    if ax is None:
        fig, ax = plt.subplots(1)
    # Specify for mypy that we should be in the multiobjective case
    list_solution_fits = cast(
        list[tuple[Solution, TupleFitness]], result_storage.list_solution_fits
    )
    ax.scatter(
        x=[p[1].vector_fitness[0] for p in list_solution_fits],
        y=[p[1].vector_fitness[1] for p in list_solution_fits],
        color=color,
    )
    ax.set_xlabel(name_axis[0])
    ax.set_ylabel(name_axis[1])


def plot_pareto_2d(
    pareto_front: ParetoFront, name_axis: list[str], ax: Any = None, color: str = "b"
) -> Any:
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.scatter(
        x=[p[1].vector_fitness[0] for p in pareto_front.paretos],
        y=[p[1].vector_fitness[1] for p in pareto_front.paretos],
        color=color,
    )
    ax.set_xlabel(name_axis[0])
    ax.set_ylabel(name_axis[1])


def plot_fitness(
    result_storage: ResultStorage, ax: Any = None, color: str = "b", title: str = ""
) -> Any:
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.set_title(title)
    ax.plot([x[1] for x in result_storage.list_solution_fits], color=color)
    ax.set_xlabel("Solution number")
    ax.set_ylabel("Fitness")
