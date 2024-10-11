#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random
from typing import Any, Optional

import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    Problem,
)
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (
    ListHyperparameter,
    SubBrick,
    SubBrickHyperparameter,
)
from discrete_optimization.generic_tools.lns_cp import (
    LnsOrtoolsCpSat,
    OrtoolsCpSatConstraintHandler,
)
from discrete_optimization.generic_tools.lns_tools import (
    ConstraintHandler,
    ConstraintHandlerMix,
    InitialSolution,
    PostProcessSolution,
    TrivialInitialSolution,
)
from discrete_optimization.generic_tools.optuna.utils import (
    generic_optuna_experiment_monoproblem,
)
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import (
    from_solutions_to_result_storage,
)
from discrete_optimization.maximum_independent_set.parser import (
    dimacs_parser_nx,
    get_data_available,
)
from discrete_optimization.maximum_independent_set.problem import (
    MisProblem,
    MisSolution,
)
from discrete_optimization.maximum_independent_set.solvers.cpsat import CpSatMisSolver
from discrete_optimization.maximum_independent_set.solvers.lns import (
    AllVarsOrtoolsCpSatMisConstraintHandler,
    DestroyOrtoolsCpSatMisConstraintHandler,
    LocalMovesOrtoolsCpSatMisConstraintHandler,
    OrtoolsCpSatMisConstraintHandler,
)

try:
    import optuna
except ImportError:
    optuna_available = False
else:
    optuna_available = True
    from optuna.trial import TrialState

SEED = 42


@pytest.fixture()
def random_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    return SEED


class LnsOrtoolsCpSatMix(LnsOrtoolsCpSat):
    """LNS around ortools-csat using a constraint handler mix.

    We use this wrapper to show how to use a ListHyperparameter
    to generate the constrainthandler from a mix.

    We also store the resulting constraint handler to make assertion on it during test.

    """

    generated_constraint_handlers: list[ConstraintHandlerMix] = []

    hyperparameters = [
        h for h in LnsOrtoolsCpSat.hyperparameters if h.name != "constraint_handler"
    ] + [
        ListHyperparameter(
            name="list_constraint_handlers",
            name_in_kwargs="list_constraint_handler_subbricks",
            hyperparameter_template=SubBrickHyperparameter(
                name="constraint_handler",
                name_in_kwargs="constraint_handler_subbrick",
                choices=[
                    AllVarsOrtoolsCpSatMisConstraintHandler,
                    OrtoolsCpSatMisConstraintHandler,
                    DestroyOrtoolsCpSatMisConstraintHandler,
                    LocalMovesOrtoolsCpSatMisConstraintHandler,
                ],
            ),
            length_low=2,
            length_high=3,
        )
    ]

    def __init__(
        self,
        problem: Problem,
        list_constraint_handlers: Optional[list[ConstraintHandler]] = None,
        subsolver: Optional[OrtoolsCpSatSolver] = None,
        initial_solution_provider: Optional[InitialSolution] = None,
        post_process_solution: Optional[PostProcessSolution] = None,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        # get the list of constraint handlers and construct the corresponding mix
        if list_constraint_handlers is None:
            list_constraint_handlers = []
        if "list_constraint_handler_subbricks" in kwargs:
            for subbrick in kwargs["list_constraint_handler_subbricks"]:
                constraint_handler = subbrick.cls(problem=problem, **subbrick.kwargs)
                list_constraint_handlers.append(constraint_handler)
        constraint_handler = ConstraintHandlerMix(
            problem=problem,
            list_constraints_handler=list_constraint_handlers,
            list_proba=[1 / len(list_constraint_handlers)]
            * len(list_constraint_handlers),
            update_proba=False,
        )
        # store the generated mix for testing purpose
        LnsOrtoolsCpSatMix.generated_constraint_handlers.append(constraint_handler)

        super().__init__(
            problem=problem,
            subsolver=subsolver,
            initial_solution_provider=initial_solution_provider,
            constraint_handler=constraint_handler,
            post_process_solution=post_process_solution,
            params_objective_function=params_objective_function,
            **kwargs,
        )


@pytest.mark.skipif(
    not optuna_available, reason="You need Optuna to test this callback."
)
def test_lns_mix(random_seed):
    small_example = [f for f in get_data_available() if "1dc.64" in f][0]
    problem: MisProblem = dimacs_parser_nx(small_example)

    solvers_to_test = [LnsOrtoolsCpSatMix]

    # fixed parameters
    params_cp = ParametersCp.default()

    subsolver = CpSatMisSolver(problem)
    subsolver.init_model()

    initial_solution_provider = TrivialInitialSolution(
        subsolver.create_result_storage(
            list_solution_fits=[(problem.get_dummy_solution(), 0.0)]
        )
    )

    kwargs_fixed_by_solver = {
        LnsOrtoolsCpSatMix: dict(
            skip_initial_solution_provider=False,
            subsolver=subsolver,
            initial_solution_provider=initial_solution_provider,
            post_process_solution=None,
            parameters_cp=params_cp,
            time_limit=10,
            time_limit_iter0=1,
            nb_iteration_lns=5,
        )
    }

    n_trials = 5
    study = generic_optuna_experiment_monoproblem(
        problem=problem,
        solvers_to_test=solvers_to_test,
        n_trials=n_trials,
        kwargs_fixed_by_solver=kwargs_fixed_by_solver,
        seed=random_seed,
        check_satisfy=False,
        callbacks=[TimerStopper(total_seconds=5)],
        create_another_study=False,
        overwrite_study=True,
    )

    completed_trials = study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    assert len(completed_trials) == n_trials

    assert len(LnsOrtoolsCpSatMix.generated_constraint_handlers) == n_trials

    represented_classes = set()
    for c in LnsOrtoolsCpSatMix.generated_constraint_handlers:
        assert len(c.list_constraints_handler) >= 2
        assert len(c.list_constraints_handler) <= 3
        assert all(
            isinstance(sub_c, OrtoolsCpSatConstraintHandler)
            for sub_c in c.list_constraints_handler
        )
        for sub_c in c.list_constraints_handler:
            represented_classes.add(sub_c.__class__)

    assert len(represented_classes) == 4
    assert AllVarsOrtoolsCpSatMisConstraintHandler in represented_classes
    assert DestroyOrtoolsCpSatMisConstraintHandler in represented_classes
