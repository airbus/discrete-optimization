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
    def next_temperature(self, move_accepted: bool = False) -> float:
        """Update and return the next temperature.

        Args:
            move_accepted: Whether the last move was accepted.
                          Used by threshold-based schedulers.

        Returns:
            The updated temperature
        """
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
            # Update the temperature (pass acceptance info for threshold-based schedulers)
            self.temperature_handler.next_temperature(move_accepted=accept)
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
    """Geometric cooling schedule: T ← coefficient × T every iteration.

    This scheduler cools the temperature by a constant factor at every iteration,
    regardless of acceptance rate. This is simpler but may cool too quickly compared
    to threshold-based approaches.

    Args:
        initial_temperature: Starting temperature (T0)
        restart_handler: Handler for restarts
        cooling_factor: Multiplicative factor (typically 0.95-0.9999)
                       T_new = cooling_factor × T_old
    """

    def __init__(
        self,
        initial_temperature: float,
        restart_handler: RestartHandler,
        cooling_factor: float = 0.99,
        temperature: float = None,  # Deprecated, kept for backwards compatibility
        coefficient: float = None,  # Deprecated, kept for backwards compatibility
    ):
        # Handle backwards compatibility
        if temperature is not None:
            initial_temperature = temperature
        if coefficient is not None:
            cooling_factor = coefficient

        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.restart_handler = restart_handler
        self.cooling_factor = cooling_factor

    def next_temperature(self, move_accepted: bool = False) -> float:
        """Cool temperature by constant factor (ignores move_accepted)."""
        self.temperature *= self.cooling_factor
        return self.temperature


class TemperatureSchedulingThresholdBased(TemperatureScheduling):
    """Threshold-based geometric cooling: cool only when thresholds are reached.

    This is the classic SA cooling schedule from Kirkpatrick et al. (1983) and
    used in many SA implementations. Temperature stays constant while exploring
    at each "temperature level", and only decreases when enough moves have been
    sampled or accepted.

    This approach allows more thorough exploration at each temperature level
    before cooling, often leading to better quality solutions compared to
    per-iteration cooling.

    Implementation based on:
    - Kirkpatrick et al. (1983) - "Optimization by Simulated Annealing"
    - Ceschia et al. (2017) - "Solving discrete lot-sizing and scheduling..."

    Args:
        initial_temperature: Starting temperature (T0)
        restart_handler: Handler for restarts
        cooling_factor: Multiplicative factor when cooling (alpha, typically 0.95-0.99)
        n_moves_sampled_before_cooling: Number of moves to sample before cooling (n_s)
                                       Temperature cools when this threshold is reached
        n_moves_accepted_before_cooling: Number of accepted moves before cooling (n_a)
                                        Temperature cools when this threshold is reached
                                        (if reached before n_moves_sampled)
    Note:
        Temperature cools when EITHER threshold is reached (OR condition).
        Typically n_moves_accepted < n_moves_sampled, so high acceptance early
        in search triggers faster cooling, while low acceptance later maintains
        temperature longer for continued exploration.
    """

    def __init__(
        self,
        initial_temperature: float,
        restart_handler: RestartHandler,
        cooling_factor: float = 0.99,
        n_moves_sampled_before_cooling: int = 60240,
        n_moves_accepted_before_cooling: int = 12049,
    ):
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.restart_handler = restart_handler
        self.cooling_factor = cooling_factor
        self.n_moves_sampled_before_cooling = n_moves_sampled_before_cooling
        self.n_moves_accepted_before_cooling = n_moves_accepted_before_cooling

        # Internal counters
        self._n_moves_sampled_current_level = 0
        self._n_moves_accepted_current_level = 0
        self._n_cooling_events = 0

    def next_temperature(self, move_accepted: bool = False) -> float:
        """Update temperature based on acceptance thresholds.

        Args:
            move_accepted: Whether the last move was accepted

        Returns:
            The current temperature (may or may not have changed)
        """
        # Increment counters
        self._n_moves_sampled_current_level += 1
        if move_accepted:
            self._n_moves_accepted_current_level += 1

        # Check if we should cool
        should_cool = (
            self._n_moves_sampled_current_level >= self.n_moves_sampled_before_cooling
            or self._n_moves_accepted_current_level
            >= self.n_moves_accepted_before_cooling
        )

        if should_cool:
            # Cool the temperature
            self.temperature *= self.cooling_factor
            self._n_cooling_events += 1

            # Reset counters for next temperature level
            self._n_moves_sampled_current_level = 0
            self._n_moves_accepted_current_level = 0

            logger.debug(
                f"Temperature cooled to {self.temperature:.4f} "
                f"(cooling event #{self._n_cooling_events})"
            )

        return self.temperature

    def get_statistics(self) -> dict:
        """Get statistics about the cooling schedule.

        Returns:
            Dictionary with cooling statistics
        """
        return {
            "current_temperature": self.temperature,
            "initial_temperature": self.initial_temperature,
            "n_cooling_events": self._n_cooling_events,
            "n_moves_sampled_at_current_level": self._n_moves_sampled_current_level,
            "n_moves_accepted_at_current_level": self._n_moves_accepted_current_level,
            "cooling_factor": self.cooling_factor,
        }
