#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import pickle
import random
import time
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    Solution,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandler,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class TemperatureScheduling:
    nb_iteration: int
    restart_handler: RestartHandler
    temperature: float

    @abstractmethod
    def next_temperature(self) -> float:
        ...


class SimulatedAnnealing:
    def __init__(
        self,
        evaluator: Problem,
        mutator: Mutation,
        restart_handler: RestartHandler,
        temperature_handler: TemperatureScheduling,
        mode_mutation: ModeMutation,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        store_solution: bool = False,
        nb_solutions: int = 1000,
    ):
        self.evaluator = evaluator
        self.mutator = mutator
        self.restart_handler = restart_handler
        self.temperature_handler = temperature_handler
        self.mode_mutation = mode_mutation
        self.aggreg_from_solution: Callable[[Solution], float]
        self.aggreg_from_dict_values: Callable[[Dict[str, float]], float]
        (
            self.aggreg_from_solution,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(  # type: ignore
            evaluator, params_objective_function=params_objective_function
        )
        if (
            self.params_objective_function.objective_handling
            == ObjectiveHandling.MULTI_OBJ
        ):
            raise NotImplementedError(
                "SimulatedAnnealing is not implemented for multi objective optimization."
            )
        self.mode_optim = self.params_objective_function.sense_function
        self.store_solution = store_solution
        self.nb_solutions = nb_solutions

    def solve(
        self,
        initial_variable: Solution,
        nb_iteration_max: int,
        max_time_seconds: Optional[int] = None,
        pickle_result: bool = False,
        pickle_name: str = "debug",
        **kwargs: Any,
    ) -> ResultStorage:
        init_time = time.time()
        objective = self.aggreg_from_dict_values(
            self.evaluator.evaluate(initial_variable)
        )
        cur_variable = initial_variable.copy()
        cur_best_variable = initial_variable.copy()
        cur_objective = objective
        cur_best_objective = objective
        if self.store_solution:
            store = ResultStorage(
                list_solution_fits=[(initial_variable, objective)],
                best_solution=initial_variable.copy(),
                limit_store=True,
                mode_optim=self.params_objective_function.sense_function,
                nb_best_store=1000,
            )
        else:
            store = ResultStorage(
                list_solution_fits=[(initial_variable, objective)],
                best_solution=initial_variable.copy(),
                limit_store=True,
                mode_optim=self.params_objective_function.sense_function,
                nb_best_store=1,
            )
        self.restart_handler.best_fitness = objective
        iteration = 0
        while iteration < nb_iteration_max:
            local_improvement = False
            global_improvement = False
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_dict_values(self.evaluator.evaluate(nv))
            else:  # self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective_dict_values = self.mutator.mutate_and_compute_obj(
                    cur_variable
                )
                objective = self.aggreg_from_dict_values(objective_dict_values)
            logger.debug(
                f"{iteration} / {nb_iteration_max} {objective} {cur_objective}"
            )
            if self.mode_optim == ModeOptim.MINIMIZATION and objective < cur_objective:
                accept = True
                local_improvement = True
                global_improvement = objective < cur_best_objective
            elif (
                self.mode_optim == ModeOptim.MAXIMIZATION and objective > cur_objective
            ):
                accept = True
                local_improvement = True
                global_improvement = objective > cur_best_objective
            else:
                r = random.random()
                fac = 1 if self.mode_optim == ModeOptim.MAXIMIZATION else -1
                p = np.exp(
                    fac
                    * (objective - cur_objective)
                    / self.temperature_handler.temperature
                )
                accept = p > r
            if accept:
                cur_objective = objective
                cur_variable = nv
                logger.debug(f"iter accepted {iteration}")
                logger.debug(f"acceptance {objective}")
            else:
                cur_variable = move.backtrack_local_move(nv)
            if self.store_solution:
                store.add_solution(nv.copy(), objective)
            if global_improvement:
                logger.debug(f"iter {iteration}")
                logger.debug(f"new obj {objective} better than {cur_best_objective}")
                cur_best_objective = objective
                cur_best_variable = cur_variable.copy()
                if not self.store_solution:
                    store.add_solution(cur_variable.copy(), objective)
            self.temperature_handler.next_temperature()
            # Update the temperature
            self.restart_handler.update(
                nv, objective, global_improvement, local_improvement
            )
            # Update info in restart handler
            cur_variable, cur_objective = self.restart_handler.restart(  # type: ignore
                cur_variable, cur_objective
            )
            # possibly restart somewhere
            iteration += 1
            if pickle_result and iteration % 20000 == 0:
                pickle.dump(cur_best_variable, open(pickle_name + ".pk", "wb"))
            if max_time_seconds is not None and iteration % 1000 == 0:
                if time.time() - init_time > max_time_seconds:
                    break
        store.finalize()
        return store


class TemperatureSchedulingFactor(TemperatureScheduling):
    def __init__(
        self,
        temperature: float,
        restart_handler: RestartHandler,
        coefficient: float = 0.99,
    ):
        self.temperature = temperature
        self.restart_handler = restart_handler
        self.coefficient = coefficient

    def next_temperature(self) -> float:
        self.temperature *= self.coefficient
        return self.temperature
