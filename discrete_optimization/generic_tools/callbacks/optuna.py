#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

import logging
import time
from typing import Optional

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

logger = logging.getLogger(__name__)


try:
    import optuna
except ImportError:
    logger.warning("You should install optuna to use callbacks for optuna.")


class OptunaCallback(Callback):
    """Callback reporting intermediate values to prune unpromising trials during Optuna hyperparameters tuning.

    Adapted to single objective optimization (res.fit is a float)

    The callback report to optuna intermediate fitness with the corresponding step number
    or elapsed time since starting time.
    It also updates the user attribute used to store computing time,
    so that pruned or failed trials will still have the user attribute updated.
    If the optuna pruner see that the trial should be pruned, raise the appropriate TrialPruned exception.

    Args:
        trial:
            A :class:`optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        optuna_report_nb_steps: report intermediate result every `optuna_report_nb_steps` steps
            when the number of iterations is high, setting this to 1 could slow too much run of a single trial
        starting_time: float representing the start time of the solving process.
            Should be the result of a call to `time.perf_counter()`.
            Default to `time.perf_counter()` called by `on_solve_start()`.
            Useful to be on par with a clock set outside the callback.
        elapsed_time_attr: key of trial user attribute used to store the elapsed time at each step
        report_time: if True, report to optuna intermediate fitness with elapsed time instead of step
        pruning: if True, use the optuna pruner to decide if we the trial should be pruned. Else never try to prune.

    """

    def __init__(
        self,
        trial: optuna.trial.Trial,
        optuna_report_nb_steps: int = 1,
        starting_time: Optional[float] = None,
        elapsed_time_attr: str = "elapsed_time",
        report_time: bool = False,
        pruning: bool = True,
        **kwargs,
    ) -> None:
        self.pruning = pruning
        self.report_time = report_time
        self.elapsed_time_attr = elapsed_time_attr
        self.report_nb_steps = optuna_report_nb_steps
        self.trial = trial
        self.starting_time = starting_time

    def on_solve_start(self, solver: SolverDO):
        if self.starting_time is None:
            self.starting_time = time.perf_counter()

    def on_step_end(
        self, step: int, res: ResultStorage, solver: SolverDO
    ) -> Optional[bool]:
        """Called at the end of an optimization step.

        Args:
            step: index of step
            res: current result storage
            solver: solvers using the callback

        Returns:
            If `True`, the optimization process is stopped, else it goes on.

        """
        if step % self.report_nb_steps == 0:
            _, fit = res.get_best_solution_fit()

            step_time = time.perf_counter() - self.starting_time
            self.trial.set_user_attr(self.elapsed_time_attr, step_time)

            # Report current score and step to Optuna's trial.
            if self.report_time:
                self.trial.report(float(fit), step=float(step_time))
            else:
                self.trial.report(float(fit), step=step)

            # Prune trial if needed
            if self.pruning:
                if self.trial.should_prune():
                    message = "Trial was pruned at step {}.".format(step)
                    raise optuna.TrialPruned(message)
