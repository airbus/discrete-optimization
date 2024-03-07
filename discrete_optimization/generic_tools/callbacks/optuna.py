#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations  # see annotations as str

import logging
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


class OptunaReportSingleFitCallback(Callback):
    """Callback to report intermediary objective values, this is mostly useful for visualisation purposes.


    Adapted to single objective optimization (res.fit is a float)

    Args:
        trial:
            A :class:`optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        optuna_report_nb_steps: report intermediate result every `optuna_report_nb_steps` steps
            when the number of iterations is high, setting this to 1 could slow too much run of a single trial

    """

    def __init__(
        self, trial: optuna.trial.Trial, optuna_report_nb_steps: int = 1, **kwargs
    ) -> None:
        self.report_nb_steps = optuna_report_nb_steps
        self.trial = trial

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
            fit = res.best_fit

            # Report current score and step to Optuna's trial.
            self.trial.report(float(fit), step=step)


class OptunaPruningSingleFitCallback(Callback):
    """Callback to prune unpromising trials during Optuna hyperparameters tuning.

    Adapted to single objective optimization (res.fit is a float)

    Args:
        trial:
            A :class:`optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        optuna_report_nb_steps: report intermediate result every `optuna_report_nb_steps` steps
            when the number of iterations is high, setting this to 1 could slow too much run of a single trial

    """

    def __init__(
        self, trial: optuna.trial.Trial, optuna_report_nb_steps: int = 1, **kwargs
    ) -> None:
        self.report_nb_steps = optuna_report_nb_steps
        self.trial = trial

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
            fit = res.best_fit

            # Report current score and step to Optuna's trial.
            self.trial.report(float(fit), step=step)

            # Prune trial if needed
            if self.trial.should_prune():
                message = "Trial was pruned at step {}.".format(step)
                raise optuna.TrialPruned(message)
