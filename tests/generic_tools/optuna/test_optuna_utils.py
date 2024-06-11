#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from time import sleep
from typing import Any, List, Optional

import numpy as np
import pytest

from discrete_optimization.coloring.coloring_model import ColoringSolution
from discrete_optimization.coloring.coloring_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    IntegerHyperparameter,
)
from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
    generic_optuna_experiment_multiproblem,
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
    hyperparameters = [IntegerHyperparameter("param", low=1, high=4, default=1)]

    nb_colors_series = [100 - i for i in range(96)]

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        param = kwargs["param"]

        res = ResultStorage(
            list_solution_fits=[],
            mode_optim=self.params_objective_function.sense_function,
        )
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        for step, nb_colors in enumerate(self.nb_colors_series):
            solution = ColoringSolution(
                problem=self.problem, nb_color=nb_colors, nb_violations=0
            )
            fit = self.aggreg_from_dict(self.problem.evaluate(solution))
            res.list_solution_fits.append((solution, fit))
            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(step=step, res=res, solver=self)
            if stopping:
                break

        callbacks_list.on_solve_end(res=res, solver=self)
        return res


class FakeSolver2(SolverDO):
    hyperparameters = [IntegerHyperparameter("param2", low=1, high=3, default=1)]

    nb_colors_series = [100 - i for i in range(96)]

    def solve(
        self, callbacks: Optional[List[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        param = kwargs["param2"]

        res = ResultStorage(
            list_solution_fits=[],
            mode_optim=self.params_objective_function.sense_function,
        )
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        for step, nb_colors in enumerate(self.nb_colors_series):
            solution = ColoringSolution(
                problem=self.problem, nb_color=nb_colors - param, nb_violations=0
            )
            fit = self.aggreg_from_dict(self.problem.evaluate(solution))
            res.list_solution_fits.append((solution, fit))
            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(step=step, res=res, solver=self)
            if stopping:
                break

        callbacks_list.on_solve_end(res=res, solver=self)
        return res


@pytest.mark.skipif(
    not optuna_available, reason="You need Optuna to test this callback."
)
def test_generic_optuna_experiment_monoproblem(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_9" in f]  # Multi mode RCPSP
    file_path = files[0]
    problem = parse_file(file_path)

    solvers_to_test = [FakeSolver, FakeSolver2]

    generic_optuna_experiment_monoproblem(
        problem=problem, solvers_to_test=solvers_to_test, n_trials=10
    )


@pytest.mark.skipif(
    not optuna_available, reason="You need Optuna to test this callback."
)
def test_generic_optuna_experiment_multiproblem(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_" in f]  # Multi mode RCPSP
    problems = [parse_file(file_path) for file_path in files]

    solvers_to_test = [FakeSolver, FakeSolver2]

    generic_optuna_experiment_multiproblem(
        problems=problems, solvers_to_test=solvers_to_test, n_trials=10
    )
