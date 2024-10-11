#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to tune hyperparameters of several solvers.

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""
import logging
from collections import defaultdict
from typing import Any

import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.cp_tools import CpSolverName, ParametersCp
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.asp import AspKnapsackSolver
from discrete_optimization.knapsack.solvers.cp_mzn import (
    Cp2KnapsackSolver,
    CpKnapsackSolver,
)
from discrete_optimization.knapsack.solvers.decomposition import (
    DecomposedKnapsackSolver,
)
from discrete_optimization.knapsack.solvers.dp import ExactDpKnapsackSolver
from discrete_optimization.knapsack.solvers.greedy import (
    GreedyBestKnapsackSolver,
    GreedyDummyKnapsackSolver,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(name)s:%(levelname)s:%(message)s"
)


seed = 42
optuna_nb_trials = 20


usecase_name = "ks_1000_0"

study_name = f"knapsack-decomposed-{usecase_name}"
storage_path = "./optuna-journal.log"  # NFS path for distributed optimization


# Solvers to test
kwargs_fixed: dict[str, Any] = dict(
    initial_solver=SubBrick(
        cls=GreedyDummyKnapsackSolver, kwargs={}
    ),  # Start from empty solution : make the experiment more interesting :)
)
kwargs_fixed_by_root_subsolver: dict[type[SolverDO], dict[str, Any]] = defaultdict(
    dict,  # default kwargs factory for unspecified solvers
    {
        Cp2KnapsackSolver: dict(time_limit=5),
        CpKnapsackSolver: dict(time_limit=5),
        AspKnapsackSolver: dict(time_limit=5),
    },
)
suggest_optuna_kwargs_by_name: dict[str, Any] = {
    "root_solver": dict(
        # limit choices to some solvers
        choices=[
            ExactDpKnapsackSolver,
            CpKnapsackSolver,
            Cp2KnapsackSolver,
            AspKnapsackSolver,
        ],
        # avoid not installed solvers
        kwargs_by_name={
            "cp_solver_name": dict(choices=[CpSolverName.CHUFFED, CpSolverName.ORTOOLS])
        },
    ),
    "nb_iteration": dict(low=5, high=20),  # limit number of iterations
    "proportion_to_remove": dict(low=0.85),
}
suggest_optuna_names = [
    name
    for name in DecomposedKnapsackSolver.get_hyperparameters_names()
    if name not in kwargs_fixed
]

# problem definition
file = [f for f in get_data_available() if usecase_name in f][0]
problem = parse_file(file)

# sense of optimization
objective_register = problem.get_objective_register()
if objective_register.objective_sense == ModeOptim.MINIMIZATION:
    direction = "minimize"
else:
    direction = "maximize"


# objective definition
def objective(trial: Trial):
    # hyperparameters to test

    # first parameter: solver choice
    solver_class = DecomposedKnapsackSolver

    # hyperparameters for the chosen solver
    suggested_hyperparameters_kwargs = solver_class.suggest_hyperparameters_with_optuna(
        trial=trial,
        names=suggest_optuna_names,
        kwargs_by_name=suggest_optuna_kwargs_by_name,
    )

    # complete root_solver_kwargs with fixed parameters
    root_solver: SubBrick = suggested_hyperparameters_kwargs["root_solver"]
    root_solver_kwargs = dict(kwargs_fixed_by_root_subsolver[root_solver.cls])
    root_solver_kwargs.update(root_solver.kwargs)
    root_solver.kwargs = root_solver_kwargs

    # use existing value if corresponding to a previous complete trial
    states_to_consider = (TrialState.COMPLETE,)
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            logger.warning(
                "Trial with same hyperparameters as a previous complete trial: returning previous fit."
            )
            return t.value

    # prune if corresponding to a previous failed trial
    states_to_consider = (TrialState.FAIL,)
    trials_to_consider = trial.study.get_trials(
        deepcopy=False, states=states_to_consider
    )
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            raise optuna.TrialPruned(
                "Pruning trial identical to a previous failed trial."
            )

    # log start of trial with chosen hyperparameters
    logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

    # construct kwargs for __init__, init_model, and solve
    kwargs = dict(kwargs_fixed)  # copy the frozen kwargs dict
    kwargs.update(suggested_hyperparameters_kwargs)

    logger.debug(f"kwargs= {kwargs}")

    # solver init
    solver = solver_class(problem=problem, **kwargs)
    solver.init_model(**kwargs)

    # solve
    sol, fit = solver.solve(
        callbacks=[
            OptunaCallback(trial=trial, **kwargs),
        ],
        **kwargs,
    ).get_best_solution_fit()

    return fit


# create study + database to store it
storage = JournalStorage(JournalFileStorage(storage_path))
try:
    optuna.delete_study(study_name=study_name, storage=storage)
except:
    pass
study = optuna.create_study(
    study_name=study_name,
    direction=direction,
    sampler=optuna.samplers.TPESampler(seed=seed),
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=optuna_nb_trials)
