#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from datetime import datetime
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import BoundsProviderMixin, SolverDO
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


class ObjectiveGapStopper(Callback):
    """Stop the solver according to some classical convergence criteria: relative and absolute gap.

    It assumes that the solver is able to provide the current best value and bound for the internal objective.

    """

    def __init__(
        self,
        objective_gap_rel: Optional[float] = None,
        objective_gap_abs: Optional[float] = None,
    ):
        self.objective_gap_rel = objective_gap_rel
        self.objective_gap_abs = objective_gap_abs

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        if not isinstance(solver, BoundsProviderMixin):
            raise ValueError(
                "The ObjectiveGapStopper can be applied only to a solver deriving from BoundsProviderMixin."
            )
        abs_gap = None
        if self.objective_gap_abs is not None:
            abs_gap = solver.get_current_absolute_gap()
            if abs_gap is not None:
                if abs_gap <= self.objective_gap_abs:
                    logger.debug(
                        f"Stopping search, absolute gap {abs_gap} <= {self.objective_gap_abs}"
                    )
                    return True
        if self.objective_gap_rel is not None:
            bound = solver.get_current_best_internal_objective_bound()
            if bound is not None and bound != 0:
                if self.objective_gap_abs is None:
                    abs_gap = solver.get_current_absolute_gap()
                if abs_gap is not None:  # could be still None (e.g. mathopt + cp-sat)
                    rel_gap = abs_gap / abs(bound)
                    if rel_gap <= self.objective_gap_rel:
                        logger.debug(
                            f"Stopping search, relative gap {rel_gap} <= {self.objective_gap_rel}"
                        )
                        return True
