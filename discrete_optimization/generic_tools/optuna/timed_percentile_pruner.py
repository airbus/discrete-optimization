#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#
# This module is adapted from https://github.com/optuna/optuna/blob/master/optuna/pruners/_percentile.py

from __future__ import annotations

import functools
import logging
import math
from typing import KeysView, List

import numpy as np

logger = logging.getLogger(__name__)


try:
    import optuna
except ImportError:
    logger.warning("You should install optuna to use TimedPercentilePruner for optuna.")
else:
    from optuna.pruners import BasePruner
    from optuna.study._study_direction import StudyDirection
    from optuna.trial._state import TrialState

    def _get_best_intermediate_result_over_steps(
        trial: "optuna.trial.FrozenTrial", direction: StudyDirection
    ) -> float:
        values = np.asarray(list(trial.intermediate_values.values()), dtype=float)
        if direction == StudyDirection.MAXIMIZE:
            return np.nanmax(values)
        return np.nanmin(values)

    def _get_interpolated_intermediate_value(
        trial: "optuna.trial.FrozenTrial", step: float
    ) -> float:
        xp = sorted(trial.intermediate_values)
        yp = [trial.intermediate_values[xx] for xx in xp]
        if len(xp) > 0:
            return np.interp(step, xp=xp, fp=yp, left=np.nan, right=np.nan)
        else:
            return np.nan

    def _get_percentile_intermediate_result_over_trials(
        completed_trials: List["optuna.trial.FrozenTrial"],
        direction: StudyDirection,
        step: int,
        percentile: float,
        n_min_trials: int,
    ) -> float:
        if len(completed_trials) == 0:
            raise ValueError("No trials have been completed.")

        intermediate_values = [
            _get_interpolated_intermediate_value(trial=t, step=step)
            for t in completed_trials
        ]
        intermediate_values = [v for v in intermediate_values if not np.isnan(v)]

        if len(intermediate_values) < n_min_trials:
            return math.nan

        if direction == StudyDirection.MAXIMIZE:
            percentile = 100 - percentile

        return float(
            np.nanpercentile(
                np.array(intermediate_values, dtype=float),
                percentile,
            )
        )

    def _is_first_in_interval_step(
        step: int,
        intermediate_steps: KeysView[int],
        n_warmup_steps: int,
        interval_steps: int,
    ) -> bool:
        nearest_lower_pruning_step = (
            step - n_warmup_steps
        ) // interval_steps * interval_steps + n_warmup_steps
        assert nearest_lower_pruning_step >= 0

        # `intermediate_steps` may not be sorted so we must go through all elements.
        second_last_step = functools.reduce(
            lambda second_last_step, s: s
            if s > second_last_step and s != step
            else second_last_step,
            intermediate_steps,
            -1,
        )

        return second_last_step < nearest_lower_pruning_step

    class TimedPercentilePruner(BasePruner):
        """Pruner to keep the specified percentile of the trials.

        Prune if the best intermediate value is in the bottom percentile among trials at the same step.
        If no report for the exact step exits in other trials, we use an interpolated value.

        Args:
            percentile:
                Percentile which must be between 0 and 100 inclusive
                (e.g., When given 25.0, top of 25th percentile trials are kept).
            n_startup_trials:
                Pruning is disabled until the given number of trials finish in the same study.
            n_warmup_steps:
                Pruning is disabled until the trial exceeds the given number of step. Note that
                this feature assumes that ``step`` starts at zero.
            interval_steps:
                Interval in number of steps between the pruning checks, offset by the warmup steps.
                If no value has been reported at the time of a pruning check, that particular check
                will be postponed until a value is reported. Value must be at least 1.
            n_min_trials:
                Minimum number of reported trial results at a step to judge whether to prune.
                If the number of reported intermediate values from all trials at the current step
                is less than ``n_min_trials``, the trial will not be pruned. This can be used to ensure
                that a minimum number of trials are run to completion without being pruned.
        """

        def __init__(
            self,
            percentile: float,
            n_startup_trials: int = 5,
            n_warmup_steps: int = 0,
            interval_steps: int = 1,
            *,
            n_min_trials: int = 1,
        ) -> None:
            if not 0.0 <= percentile <= 100:
                raise ValueError(
                    "Percentile must be between 0 and 100 inclusive but got {}.".format(
                        percentile
                    )
                )
            if n_startup_trials < 0:
                raise ValueError(
                    "Number of startup trials cannot be negative but got {}.".format(
                        n_startup_trials
                    )
                )
            if n_warmup_steps < 0:
                raise ValueError(
                    "Number of warmup steps cannot be negative but got {}.".format(
                        n_warmup_steps
                    )
                )
            if interval_steps < 1:
                raise ValueError(
                    "Pruning interval steps must be at least 1 but got {}.".format(
                        interval_steps
                    )
                )
            if n_min_trials < 1:
                raise ValueError(
                    "Number of trials for pruning must be at least 1 but got {}.".format(
                        n_min_trials
                    )
                )

            self._percentile = percentile
            self._n_startup_trials = n_startup_trials
            self._n_warmup_steps = n_warmup_steps
            self._interval_steps = interval_steps
            self._n_min_trials = n_min_trials

        def prune(
            self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial"
        ) -> bool:
            completed_trials = study.get_trials(
                deepcopy=False, states=(TrialState.COMPLETE,)
            )
            n_trials = len(completed_trials)

            if n_trials == 0:
                return False

            if n_trials < self._n_startup_trials:
                return False

            step = trial.last_step
            if step is None:
                return False

            n_warmup_steps = self._n_warmup_steps
            if step < n_warmup_steps:
                return False

            if not _is_first_in_interval_step(
                step,
                trial.intermediate_values.keys(),
                n_warmup_steps,
                self._interval_steps,
            ):
                return False

            direction = study.direction
            best_intermediate_result = _get_best_intermediate_result_over_steps(
                trial, direction
            )
            if math.isnan(best_intermediate_result):
                return True

            p = _get_percentile_intermediate_result_over_trials(
                completed_trials, direction, step, self._percentile, self._n_min_trials
            )
            if math.isnan(p):
                return False

            if direction == StudyDirection.MAXIMIZE:
                return best_intermediate_result < p
            return best_intermediate_result > p
