#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import random
from time import sleep
from typing import Any, Optional

import numpy as np
import pytest

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import ColoringSolution
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)

try:
    import optuna
except ImportError:
    optuna_available = False
else:
    optuna_available = True
    from discrete_optimization.generic_tools.optuna.timed_percentile_pruner import (
        TimedPercentilePruner,
    )

SEED = 42


@pytest.fixture()
def random_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    return SEED


class FakeSolver(SolverDO):
    hyperparameters = [IntegerHyperparameter("param", low=1, high=100, default=1)]

    nb_colors_series = [100 - i for i in range(96)]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        param = kwargs["param"]

        res = self.create_result_storage()
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        for step, nb_colors in enumerate(self.nb_colors_series):
            solution = ColoringSolution(
                problem=self.problem, nb_color=nb_colors, nb_violations=0
            )
            fit = self.aggreg_from_dict(self.problem.evaluate(solution))
            res.append((solution, fit))
            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(step=step, res=res, solver=self)
            if stopping:
                break
            sleep(0.01 / param)

        callbacks_list.on_solve_end(res=res, solver=self)
        return res


@pytest.mark.skipif(
    not optuna_available, reason="You need Optuna to test this callback."
)
def test_optuna_timed_callback_timed_pruner(random_seed):
    # timed pruner on timestamps =>  pruning (as solver always return same fitnesses sequence, but at various speed)
    files = get_data_available()
    files = [f for f in files if "gc_70_9" in f]  # Multi mode RCPSP
    file_path = files[0]
    problem = parse_file(file_path)

    def objective(trial: optuna.trial.Trial) -> float:
        kwargs = FakeSolver.suggest_hyperparameters_with_optuna(trial=trial)
        solver = FakeSolver(problem=problem, **kwargs)
        solver.init_model(**kwargs)
        sol, fit = solver.solve(
            callbacks=[
                OptunaCallback(
                    trial=trial,
                    optuna_report_nb_steps=10,
                    report_time=True,
                    report_time_unit=0.001,
                )
            ],
            **kwargs
        ).get_best_solution_fit()
        return fit

    study = optuna.create_study(
        study_name="coloring-fake-solver",
        direction=problem.get_optuna_study_direction(),
        sampler=optuna.samplers.TPESampler(seed=random_seed),
        pruner=TimedPercentilePruner(  # intermediate values interpolated at same "step"
            percentile=2,  # 2% pruner
            n_min_trials=2,
        ),
    )
    study.optimize(objective, n_trials=40)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    assert len(pruned_trials) > 0
    assert len(complete_trials) > 0


@pytest.mark.skipif(
    not optuna_available, reason="You need Optuna to test this callback."
)
def test_optuna_step_callback_step_pruner(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_9" in f]  # Multi mode RCPSP
    file_path = files[0]
    problem = parse_file(file_path)

    def objective(trial: optuna.trial.Trial) -> float:
        kwargs = FakeSolver.suggest_hyperparameters_with_optuna(trial=trial)
        solver = FakeSolver(problem=problem, **kwargs)
        solver.init_model(**kwargs)
        sol, fit = solver.solve(
            callbacks=[
                OptunaCallback(
                    trial=trial,
                    optuna_report_nb_steps=10,
                    report_time=False,
                )
            ],
            **kwargs
        ).get_best_solution_fit()
        return fit

    study = optuna.create_study(
        study_name="coloring-fake-solver",
        direction=problem.get_optuna_study_direction(),
        sampler=optuna.samplers.TPESampler(seed=random_seed),
    )
    study.optimize(objective, n_trials=40)

    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]
    )
    assert len(pruned_trials) == 0
    assert len(complete_trials) > 0
