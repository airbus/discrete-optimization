#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Tuple

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
    fitness_class,
)


class ModeMutation(Enum):
    MUTATE = 0
    MUTATE_AND_EVALUATE = 1


class RestartHandler:
    solution_restart: Solution
    solution_best: Solution
    best_fitness: fitness_class

    def __init__(self) -> None:
        self.nb_iteration = 0
        self.nb_iteration_no_local_improve = 0
        self.nb_iteration_no_global_improve = 0

    def update(
        self,
        nv: Solution,
        fitness: fitness_class,
        improved_global: bool,
        improved_local: bool,
    ) -> None:
        self.nb_iteration += 1
        if improved_global:
            self.nb_iteration_no_global_improve = 0
            self.best_fitness = fitness
            self.solution_best = nv.copy()
        else:
            self.nb_iteration_no_global_improve += 1
        if improved_local:
            self.nb_iteration_no_local_improve = 0
        else:
            self.nb_iteration_no_local_improve += 1

    def restart(
        self, cur_solution: Solution, cur_objective: fitness_class
    ) -> Tuple[Solution, fitness_class]:
        return cur_solution, cur_objective


class RestartHandlerLimit(RestartHandler):
    def __init__(
        self,
        nb_iteration_no_improvement: int,
        cur_solution: Solution,
        cur_objective: fitness_class,
    ):
        RestartHandler.__init__(self)
        self.nb_iteration_no_improvement = nb_iteration_no_improvement
        self.solution_best = cur_solution.copy()
        self.best_fitness = cur_objective

    def restart(
        self, cur_solution: Solution, cur_objective: fitness_class
    ) -> Tuple[Solution, fitness_class]:
        if (
            self.nb_iteration_no_global_improve > self.nb_iteration_no_improvement
            or self.nb_iteration_no_local_improve > self.nb_iteration_no_improvement
        ):
            self.nb_iteration_no_global_improve = 0
            self.nb_iteration_no_local_improve = 0
            return self.solution_best.copy(), self.best_fitness
        else:
            return cur_solution, cur_objective


class ResultLS(ResultStorage):
    def __init__(
        self,
        result_storage: ResultStorage,
        best_solution: Solution,
        best_objective: fitness_class,
    ):
        self.result_storage = result_storage
        self.list_solution_fits = result_storage.list_solution_fits
        self.maximize = result_storage.maximize
        self.limit_store = result_storage.limit_store
        self.nb_best_score = result_storage.nb_best_score
        self.map_solutions = result_storage.map_solutions
        self.heap = result_storage.heap
        self.min = result_storage.min
        self.max = result_storage.max
        self.best_solution = best_solution
        self.best_objective = best_objective
