#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to tune hyperparameters of several solvers.

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""
import logging
from collections import defaultdict
from typing import Any, Dict, List, Type

import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_tools.callbacks.optuna import (
    OptunaPruningSingleFitCallback,
)
from discrete_optimization.generic_tools.cp_tools import CPSolverName, ParametersCP
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN2
from discrete_optimization.knapsack.solvers.dyn_prog_knapsack import KnapsackDynProg
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest
from discrete_optimization.knapsack.solvers.knapsack_asp_solver import KnapsackASPSolver
from discrete_optimization.knapsack.solvers.knapsack_decomposition import (
    KnapsackDecomposedSolver,
)
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    GPDP,
    create_selective_tsp,
)
from discrete_optimization.pickup_vrp.solver.ortools_solver import ORToolsGPDP

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
kwargs_fixed: Dict[str, Any] = dict(
    initial_solver=GreedyBest,
    initial_solver_kwargs={},
)
params_cp = ParametersCP.default()
params_cp.time_limit = 5
kwargs_fixed_by_root_subsolver: Dict[Type[SolverDO], Dict[str, Any]] = defaultdict(
    dict,  # default kwargs factory for unspecified solvers
    {
        CPKnapsackMZN2: dict(params_cp=params_cp),
    },
)
suggest_optuna_kwargs_by_name: Dict[str, Any] = {
    "root_solver": dict(
        choices=[CPKnapsackMZN2, KnapsackDynProg]
    ),  # limit choices to 2 solvers
    "nb_iteration": dict(high=10),  # limit number of iterations
    "root_solver_kwargs": dict(
        kwargs_by_name={
            "cp_solver_name": dict(
                choices=[CPSolverName.CHUFFED, CPSolverName.CPLEX, CPSolverName.GECODE]
            )
        }
    ),  # avoid not installed solvers
}
suggest_optuna_names = [
    name
    for name in KnapsackDecomposedSolver.get_hyperparameters_names()
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
    solver_class = KnapsackDecomposedSolver

    # hyperparameters for the chosen solver
    suggested_hyperparameters_kwargs = solver_class.suggest_hyperparameters_with_optuna(
        trial=trial,
        names=suggest_optuna_names,
        kwargs_by_name=suggest_optuna_kwargs_by_name,
    )

    # complete root_solver_kwargs with fixed parameters
    root_solver = suggested_hyperparameters_kwargs["root_solver"]
    root_solver_kwargs = dict(kwargs_fixed_by_root_subsolver[root_solver])
    root_solver_kwargs.update(suggested_hyperparameters_kwargs["root_solver_kwargs"])
    suggested_hyperparameters_kwargs["root_solver_kwargs"] = root_solver_kwargs

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
            OptunaPruningSingleFitCallback(trial=trial, **kwargs),
        ],
        **kwargs,
    ).get_best_solution_fit()

    return fit


# create study + database to store it
storage = JournalStorage(JournalFileStorage(storage_path))
optuna.delete_study(study_name=study_name, storage=storage)
study = optuna.create_study(
    study_name=study_name,
    direction=direction,
    sampler=optuna.samplers.TPESampler(seed=seed),
    storage=storage,
    load_if_exists=True,
)
study.optimize(objective, n_trials=optuna_nb_trials)
