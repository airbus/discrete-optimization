#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from enum import Enum

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
    ) -> tuple[Solution, fitness_class]:
        return cur_solution, cur_objective


class RestartHandlerLimit(RestartHandler):
    def __init__(
        self,
        nb_iteration_no_improvement: int,
    ):
        RestartHandler.__init__(self)
        self.nb_iteration_no_improvement = nb_iteration_no_improvement

    def restart(
        self, cur_solution: Solution, cur_objective: fitness_class
    ) -> tuple[Solution, fitness_class]:
        if (
            self.nb_iteration_no_global_improve > self.nb_iteration_no_improvement
            or self.nb_iteration_no_local_improve > self.nb_iteration_no_improvement
        ):
            self.nb_iteration_no_global_improve = 0
            self.nb_iteration_no_local_improve = 0
            return self.solution_best.copy(), self.best_fitness
        else:
            return cur_solution, cur_objective
