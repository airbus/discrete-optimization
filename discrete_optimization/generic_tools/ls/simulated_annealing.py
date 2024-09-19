#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import random
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_mutation import Mutation
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
    Problem,
    Solution,
)
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
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


class SimulatedAnnealing(SolverDO, WarmstartMixin):
    aggreg_from_sol: Callable[[Solution], float]
    aggreg_from_dict: Callable[[dict[str, float]], float]

    initial_solution: Optional[Solution] = None
    """Initial solution used for warm start."""

    def __init__(
        self,
        problem: Problem,
        mutator: Mutation,
        restart_handler: RestartHandler,
        temperature_handler: TemperatureScheduling,
        mode_mutation: ModeMutation,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        store_solution: bool = False,
        **kwargs,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function
        )
        self.mutator = mutator
        self.restart_handler = restart_handler
        self.temperature_handler = temperature_handler
        self.mode_mutation = mode_mutation
        if (
            self.params_objective_function.objective_handling
            == ObjectiveHandling.MULTI_OBJ
        ):
            raise NotImplementedError(
                "SimulatedAnnealing is not implemented for multi objective optimization."
            )
        self.mode_optim = self.params_objective_function.sense_function
        self.store_solution = store_solution

    def set_warm_start(self, solution: Solution) -> None:
        """Make the solver warm start from the given solution.

        Will be ignored if arg `initial_variable` is set and not None in call to `solve()`.

        """
        self.initial_solution = solution

    def solve(
        self,
        nb_iteration_max: int,
        initial_variable: Optional[Solution] = None,
        callbacks: Optional[list[Callback]] = None,
        **kwargs: Any,
    ) -> ResultStorage:
        callbacks_list = CallbackList(callbacks=callbacks)

        if initial_variable is None:
            if self.initial_solution is None:
                raise ValueError(
                    "initial_variable cannot be None if self.initial_solution is None.\n"
                    "Use set_warm_start() to define it."
                )
            else:
                initial_variable = self.initial_solution

        objective = self.aggreg_from_dict(self.problem.evaluate(initial_variable))
        cur_variable = initial_variable.copy()
        cur_objective = objective
        cur_best_objective = objective
        store = self.create_result_storage(
            [(initial_variable, objective)],
        )
        self.restart_handler.best_fitness = objective
        self.restart_handler.solution_best = initial_variable.copy()
        iteration = 0
        # start of solve callback
        callbacks_list.on_solve_start(solver=self)

        while iteration < nb_iteration_max:
            local_improvement = False
            global_improvement = False
            if self.mode_mutation == ModeMutation.MUTATE:
                nv, move = self.mutator.mutate(cur_variable)
                objective = self.aggreg_from_dict(self.problem.evaluate(nv))
            else:  # self.mode_mutation == ModeMutation.MUTATE_AND_EVALUATE:
                nv, move, objective_dict_values = self.mutator.mutate_and_compute_obj(
                    cur_variable
                )
                objective = self.aggreg_from_dict(objective_dict_values)
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
                store.append((nv.copy(), objective))
            if global_improvement:
                logger.info(f"iter {iteration}")
                logger.info(f"new obj {objective} better than {cur_best_objective}")
                cur_best_objective = objective
                if not self.store_solution:
                    store.append((cur_variable.copy(), objective))
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

            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(
                step=iteration, res=store, solver=self
            )
            if stopping:
                break

        # end of solve callback
        callbacks_list.on_solve_end(res=store, solver=self)
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
