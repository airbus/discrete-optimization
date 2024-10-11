#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random
from typing import Any, Optional

import numpy as np
import pytest

from discrete_optimization.coloring.parser import get_data_available, parse_file
from discrete_optimization.coloring.problem import ColoringSolution
from discrete_optimization.generic_tools.callbacks.callback import (
    Callback,
    CallbackList,
)
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO, WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    IntegerHyperparameter,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
    generic_optuna_experiment_multiproblem,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
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


class FakeSolver(SolverDO, WarmstartMixin):
    hyperparameters = [IntegerHyperparameter("param", low=1, high=4, default=1)]

    nb_colors_series = [100 - i for i in range(96)]
    nb_colors_warm_start = 6

    def set_warm_start(self, solution: ColoringSolution) -> None:
        self.nb_colors_warm_start = solution.nb_color

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
                problem=self.problem,
                nb_color=nb_colors - 6 + self.nb_colors_warm_start,
                nb_violations=0,
            )
            fit = self.aggreg_from_dict(self.problem.evaluate(solution))
            res.append((solution, fit))
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
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        param = kwargs["param2"]

        res = self.create_result_storage()
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        for step, nb_colors in enumerate(self.nb_colors_series):
            solution = ColoringSolution(
                problem=self.problem, nb_color=nb_colors - param, nb_violations=0
            )
            fit = self.aggreg_from_dict(self.problem.evaluate(solution))
            res.append((solution, fit))
            # end of step callback: stopping?
            stopping = callbacks_list.on_step_end(step=step, res=res, solver=self)
            if stopping:
                break

        callbacks_list.on_solve_end(res=res, solver=self)
        return res


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_monoproblem.",
)
def test_generic_optuna_experiment_monoproblem(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_9" in f]  # Multi mode RCPSP
    file_path = files[0]
    problem = parse_file(file_path)

    solvers_to_test = [FakeSolver, FakeSolver2]

    study = generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=solvers_to_test,
        n_trials=10,
        seed=random_seed,
        check_satisfy=False,
        overwrite_study=True,
        create_another_study=False,
    )
    assert study.best_value == -2.0


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_monoproblem.",
)
def test_generic_optuna_experiment_monoproblem_sequential_metasolver():
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_9" in f]  # Multi mode RCPSP
    file_path = files[0]
    problem = parse_file(file_path)

    solvers_to_test = [SequentialMetasolver]

    suggest_optuna_kwargs_by_name_by_solver = {
        SequentialMetasolver: dict(
            subsolver_0=dict(
                choices=[FakeSolver, FakeSolver2],
                fixed_hyperparameters={"param": 1, "param2": 1},
            ),
            next_subsolvers=dict(
                choices=[FakeSolver, FakeSolver2],
                length_high=2,
                fixed_hyperparameters={"param": 1, "param2": 1},
            ),
        )
    }

    study = generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=solvers_to_test,
        check_satisfy=False,
        overwrite_study=True,
        create_another_study=False,
        suggest_optuna_kwargs_by_name_by_solver=suggest_optuna_kwargs_by_name_by_solver,
        sampler=optuna.samplers.BruteForceSampler(),
    )
    # best trial: FakeSolver2 -> FakeSolver -> FakeSolver
    assert study.best_value == -2.0
    best_trial_params = study.best_trial.params
    assert best_trial_params["SequentialMetasolver.next_subsolvers.length"] == 2
    assert best_trial_params["SequentialMetasolver.subsolver_0.cls"] == "FakeSolver2"
    assert best_trial_params["SequentialMetasolver.subsolver_1.cls"] == "FakeSolver"
    assert best_trial_params["SequentialMetasolver.subsolver_2.cls"] == "FakeSolver"

    # Should have at least 1 error due to FakeSolver2 not being a warmstartable solver
    pruned_trials = study.get_trials(
        deepcopy=False, states=[optuna.trial.TrialState.PRUNED]
    )
    assert len(pruned_trials) > 0
    trial = pruned_trials[0]
    assert (
        "Each subsolver except the first one must inherit WarmstartMixin."
        in trial.user_attrs["Error"]
    )
    assert (
        trial.params["SequentialMetasolver.subsolver_1.cls"] == "FakeSolver2"
        or trial.params["SequentialMetasolver.subsolver_2.cls"] == "FakeSolver2"
    )


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_multiproblem.",
)
def test_generic_optuna_experiment_multiproblem(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_" in f]  # Multi mode RCPSP
    problems = [parse_file(file_path) for file_path in files]

    solvers_to_test = [FakeSolver, FakeSolver2]

    study = generic_optuna_experiment_multiproblem(
        problems=problems,
        solvers_to_test=solvers_to_test,
        n_trials=10,
        seed=random_seed,
        check_satisfy=False,
        overwrite_study=True,
        create_another_study=False,
    )
    assert study.best_value == -2.0


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_multiproblem.",
)
def test_generic_optuna_experiment_multiproblem_cumulative_wilcoxon_nok(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_" in f]  # Multi mode RCPSP
    problems = [parse_file(file_path) for file_path in files]

    solvers_to_test = [FakeSolver, FakeSolver2]

    with pytest.raises(ValueError):
        generic_optuna_experiment_multiproblem(
            problems=problems,
            solvers_to_test=solvers_to_test,
            n_trials=10,
            report_cumulated_fitness=True,
            overwrite_study=True,
            create_another_study=False,
        )


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_multiproblem.",
)
def test_generic_optuna_experiment_multiproblem_cumulative(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_" in f]  # Multi mode RCPSP
    problems = [parse_file(file_path) for file_path in files]

    solvers_to_test = [FakeSolver, FakeSolver2]

    study = generic_optuna_experiment_multiproblem(
        problems=problems,
        solvers_to_test=solvers_to_test,
        n_trials=10,
        report_cumulated_fitness=True,
        pruner=optuna.pruners.MedianPruner(),
        randomize_instances=False,
        seed=random_seed,
        check_satisfy=False,
        overwrite_study=True,
        create_another_study=False,
    )
    assert study.best_value == -2.0


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_multiproblem.",
)
def test_generic_optuna_experiment_multiproblem_check_satisfy(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_" in f]  # Multi mode RCPSP
    problems = [parse_file(file_path) for file_path in files]

    solvers_to_test = [FakeSolver, FakeSolver2]

    study = generic_optuna_experiment_multiproblem(
        problems=problems,
        solvers_to_test=solvers_to_test,
        n_trials=10,
        seed=random_seed,
        check_satisfy=True,
        overwrite_study=True,
        create_another_study=False,
    )
    assert (
        len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        == 0
    )


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_monoproblem.",
)
def test_generic_optuna_experiment_monoproblem_check_satisfy(random_seed):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_9" in f]  # Multi mode RCPSP
    file_path = files[0]
    problem = parse_file(file_path)

    solvers_to_test = [FakeSolver, FakeSolver2]

    study = generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=solvers_to_test,
        n_trials=10,
        seed=random_seed,
        check_satisfy=True,
        overwrite_study=True,
        create_another_study=False,
    )
    assert (
        len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))
        == 0
    )


class FakeMetaSolver(SolverDO):
    hyperparameters = [
        SubBrickHyperparameter("subsolver", choices=[FakeSolver, FakeSolver2])
    ]

    def solve(
        self, callbacks: Optional[list[Callback]] = None, **kwargs: Any
    ) -> ResultStorage:
        subbrick = kwargs["subsolver"]
        subsolver_kwargs = dict(subbrick.kwargs)
        subsolver_kwargs.update(kwargs)
        subsolver = subbrick.cls(problem=self.problem, **subsolver_kwargs)
        subsolver.init_model(**subsolver_kwargs)
        return subsolver.solve(callbacks=callbacks, **subsolver_kwargs)


@pytest.mark.skipif(
    not optuna_available,
    reason="You need Optuna to test generic_optuna_experiment_monoproblem.",
)
def test_generic_optuna_experiment_monoproblem_metasolver_with_fixed_param_by_subsolver(
    random_seed,
):
    # classic pruner on steps => no pruning (as solver always return same fitnesses sequence)
    files = get_data_available()
    files = [f for f in files if "gc_70_9" in f]  # Multi mode RCPSP
    file_path = files[0]
    problem = parse_file(file_path)

    solvers_to_test = [FakeMetaSolver]

    suggest_optuna_kwargs_by_name_by_solver = {
        FakeMetaSolver: dict(
            subsolver=dict(
                fixed_hyperparameters_by_subbrick={
                    FakeSolver2: dict(param2=4),
                }
            )
        )
    }

    study = generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=solvers_to_test,
        n_trials=10,
        seed=random_seed,
        check_satisfy=False,
        overwrite_study=True,
        create_another_study=False,
        suggest_optuna_kwargs_by_name_by_solver=suggest_optuna_kwargs_by_name_by_solver,
    )
    assert study.best_value == -1.0
