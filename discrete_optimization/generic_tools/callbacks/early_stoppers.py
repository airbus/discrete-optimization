#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from datetime import datetime
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


class TimerStopper(Callback):
    """Callback to stop the optimization after a given time.

    Stops the optimization process if a limit training time has been elapsed.
    This time is checked after each `check_nb_steps` steps.

    """

    def __init__(self, total_seconds: int, check_nb_steps: int = 1):
        """

        Args:
            total_seconds: Total time in seconds allowed to solve
            check_nb_steps: Number of steps to wait before next time check

        """
        self.total_seconds = total_seconds
        self.check_nb_steps = check_nb_steps

    def on_solve_start(self, solver: SolverDO):
        self.initial_training_time = datetime.utcnow()

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        if step % self.check_nb_steps == 0:
            current_time = datetime.utcnow()
            difference = current_time - self.initial_training_time
            difference_seconds = difference.total_seconds()
            logger.debug(f"{difference_seconds} seconds elapsed since solve start.")
            if difference_seconds >= self.total_seconds:
                logger.info(f"{self.__class__.__name__} callback met its criteria")
                return True
        return False


class NbIterationStopper(Callback):
    """Callback to stop the optimization when a given number of solutions are found."""

    def __init__(self, nb_iteration_max: int):
        self.nb_iteration_max = nb_iteration_max
        self.nb_iteration = 0

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        self.nb_iteration += 1
        if self.nb_iteration >= self.nb_iteration_max:
            logger.info(
                f"{self.__class__.__name__} callback met its criteria: max number of iterations reached"
            )
            return True
        else:
            return False


class ObjectiveGapCpSatSolver(Callback):
    """
    Stop the cpsat solver according to some classical convergence criteria
    It could be done differently (playing with parameters of cpsat directly)
    """

    def __init__(
        self,
        objective_gap_rel: Optional[float] = None,
        objective_gap_abs: Optional[float] = None,
    ):
        self.objective_gap_rel = objective_gap_rel
        self.objective_gap_abs = objective_gap_abs

    def on_step_end(
        self, step: int, res: ResultStorage, solver: OrtoolsCpSatSolver
    ) -> Optional[bool]:
        best_sol = solver.clb.ObjectiveValue()
        bound = solver.clb.BestObjectiveBound()
        if self.objective_gap_abs is not None:
            if abs(bound - best_sol) <= self.objective_gap_abs:
                logger.debug(
                    f"Stopping search, absolute gap {abs(bound-best_sol)}<{self.objective_gap_abs}"
                )
                return True
        if self.objective_gap_rel is not None:
            if bound != 0:
                if abs(bound - best_sol) / abs(bound) <= self.objective_gap_rel:
                    logger.debug(
                        f"Stopping search, relative gap {abs(bound-best_sol)/abs(bound)}<{self.objective_gap_rel}"
                    )
                    return True
