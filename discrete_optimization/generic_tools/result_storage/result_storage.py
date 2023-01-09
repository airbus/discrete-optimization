#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from heapq import heapify, heappush, heappushpop, nlargest, nsmallest
from typing import Any, Dict, List, Optional, Tuple, Union, cast

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


class ResultStorage:
    list_solution_fits: List[Tuple[Solution, fitness_class]]
    best_solution: Optional[Solution]
    map_solutions: Dict[Solution, fitness_class]

    def __init__(
        self,
        list_solution_fits: List[Tuple[Solution, fitness_class]],
        best_solution: Optional[Solution] = None,
        mode_optim: ModeOptim = ModeOptim.MAXIMIZATION,
        limit_store: bool = True,
        nb_best_store: int = 1000,
    ):
        self.list_solution_fits = list_solution_fits
        self.best_solution = best_solution
        self.mode_optim = mode_optim
        self.maximize = mode_optim == ModeOptim.MAXIMIZATION
        self.size_heap = 0
        self.heap: List[fitness_class] = []
        self.limit_store = limit_store
        self.nb_best_score = nb_best_store
        self.map_solutions = {}
        for i in range(len(self.list_solution_fits)):
            if self.list_solution_fits[i][0] not in self.map_solutions:
                self.map_solutions[
                    self.list_solution_fits[i][0]
                ] = self.list_solution_fits[i][1]
                heappush(self.heap, self.list_solution_fits[i][1])
                self.size_heap += 1
        if self.size_heap >= self.nb_best_score and self.limit_store:
            self.heap = (
                nsmallest(self.nb_best_score, self.heap)
                if not self.maximize
                else nlargest(self.nb_best_score, self.heap)
            )
            heapify(self.heap)
            self.size_heap = self.nb_best_score
        if len(self.heap) > 0:
            self.min = min(self.heap)
            self.max = max(self.heap)
            if self.best_solution is None:
                f = min if not self.maximize else max
                self.best_solution = f(self.list_solution_fits, key=lambda x: x[1])[0]

    def add_solution(self, solution: Solution, fitness: fitness_class) -> None:
        self.list_solution_fits += [(solution, fitness)]
        if solution not in self.map_solutions:
            self.map_solutions[solution] = fitness
            self.list_solution_fits += [(solution, fitness)]
        if (
            self.maximize
            and fitness > self.max
            or (not self.maximize and fitness < self.min)
        ):
            self.best_solution = solution
        if (
            self.maximize
            and fitness >= self.min
            or (not self.maximize and fitness <= self.max)
        ):
            if self.size_heap >= self.nb_best_score and self.limit_store:
                heappushpop(self.heap, fitness)
                self.min = min(fitness, self.min)
                self.max = max(fitness, self.max)
            else:
                heappush(self.heap, fitness)
                self.size_heap += 1
                self.min = min(fitness, self.min)
                self.max = max(fitness, self.max)

    def finalize(self) -> None:
        self.heap = sorted(self.heap, reverse=self.maximize)

    def get_best_solution_fit(
        self,
    ) -> Union[Tuple[Solution, fitness_class], Tuple[None, None]]:
        if len(self.list_solution_fits) == 0:
            return None, None
        f = max if self.maximize else min
        return f(self.list_solution_fits, key=lambda x: x[1])

    def get_last_best_solution(self) -> Tuple[Solution, fitness_class]:
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

    def get_random_best_solution(self) -> Tuple[Solution, fitness_class]:
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

    def get_random_solution(self) -> Tuple[Solution, fitness_class]:
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
    ) -> List[Tuple[Solution, fitness_class]]:
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
    return ResultStorage(
        result_1.list_solution_fits + result_2.list_solution_fits,
        mode_optim=result_1.mode_optim,
    )


def from_solutions_to_result_storage(
    list_solution: List[Solution],
    problem: Problem,
    params_objective_function: Optional[ParamsObjectiveFunction] = None,
) -> ResultStorage:
    if params_objective_function is None:
        params_objective_function = get_default_objective_setup(problem)
    mode_optim = params_objective_function.sense_function
    (
        aggreg_sol,
        aggreg_dict,
        params_objective_function,
    ) = build_aggreg_function_and_params_objective(
        problem=problem, params_objective_function=params_objective_function
    )
    list_solution_fit: List[Tuple[Solution, fitness_class]] = []
    for s in list_solution:
        list_solution_fit += [(s, aggreg_sol(s))]
    return ResultStorage(
        list_solution_fits=list_solution_fit,
        mode_optim=mode_optim,
    )


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
        best_solution=None,
        mode_optim=result_storage.mode_optim,
        limit_store=result_storage.limit_store,
        nb_best_store=result_storage.nb_best_score,
    )
    pf.finalize()
    return pf


class ParetoFront(ResultStorage):
    def __init__(
        self,
        list_solution_fits: List[Tuple[Solution, fitness_class]],
        best_solution: Optional[Solution],
        mode_optim: ModeOptim = ModeOptim.MAXIMIZATION,
        limit_store: bool = True,
        nb_best_store: int = 1000,
    ):
        super().__init__(
            list_solution_fits=list_solution_fits,
            best_solution=best_solution,
            mode_optim=mode_optim,
            limit_store=limit_store,
            nb_best_store=nb_best_store,
        )
        self.paretos: List[Tuple[Solution, TupleFitness]] = []

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
        super().finalize()
        self.paretos = []
        for s, t in self.list_solution_fits:
            if not isinstance(t, TupleFitness):
                raise RuntimeError(
                    "self.list_solution_fits must be a list of tuple[Solution, TupleFitness] "
                    "for a Pareto front."
                )
            self.add_point(solution=s, tuple_fitness=t)

    def compute_extreme_points(self) -> List[Tuple[Solution, TupleFitness]]:
        function_used = max if self.maximize else min
        number_fitness = self.list_solution_fits[0][1].size  # type: ignore
        extreme_points: List[Tuple[Solution, TupleFitness]] = []
        for i in range(number_fitness):
            extr: Tuple[Solution, TupleFitness] = function_used(self.paretos, key=lambda x: x[1].vector_fitness[i])  # type: ignore
            extreme_points += [extr]
        return extreme_points


def plot_storage_2d(
    result_storage: ResultStorage,
    name_axis: List[str],
    ax: Any = None,
    color: str = "r",
) -> None:
    if ax is None:
        fig, ax = plt.subplots(1)
    # Specify for mypy that we should be in the multiobjective case
    list_solution_fits = cast(
        List[Tuple[Solution, TupleFitness]], result_storage.list_solution_fits
    )
    ax.scatter(
        x=[p[1].vector_fitness[0] for p in list_solution_fits],
        y=[p[1].vector_fitness[1] for p in list_solution_fits],
        color=color,
    )
    ax.set_xlabel(name_axis[0])
    ax.set_ylabel(name_axis[1])


def plot_pareto_2d(
    pareto_front: ParetoFront, name_axis: List[str], ax: Any = None, color: str = "b"
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
