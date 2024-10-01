#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to choose a solving method and tune its hyperparameters for quantum solvers.

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""


import logging
import time
from typing import Any

import optuna
from optuna import Trial
from optuna.trial import TrialState

from discrete_optimization.coloring.coloring_model import ColoringProblem
from discrete_optimization.coloring.solvers.coloring_quantum import (
    QAOAColoringSolver_MinimizeNbColor,
    VQEColoringSolver_MinimizeNbColor,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.optuna.timed_percentile_pruner import (
    TimedPercentilePruner,
)
from discrete_optimization.generic_tools.qiskit_tools import QiskitSolver

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s:%(levelname)s:%(message)s")

seed = 42
optuna_nb_trials = 100

create_another_study = True  # avoid relaunching the same study, keep the previous ones
max_time_per_solver = 60  # max duration (s)
min_time_per_solver = 5  # min duration before pruning (s)

nodes = [(1, {}), (2, {}), (3, {}), (4, {})]
edges = [(1, 2, {}), (1, 3, {}), (2, 4, {})]
nb_max_color = 2

# we create an instance of ColoringProblem
problem = ColoringProblem(Graph(nodes=nodes, edges=edges))

modelfilename = "coloring_quantum_solvers"

suffix = f"-{time.time()}" if create_another_study else ""
study_name = f"{modelfilename}{suffix}"
storage_path = "./optuna-journal.log"  # NFS path for distributed optimization
elapsed_time_attr = "elapsed_time"  # name of the user attribute used to store duration of trials (updated during intermediate reports)


solvers: dict[str, list[tuple[type[QiskitSolver], dict[str, Any]]]] = {
    "qaoa": [
        (
            QAOAColoringSolver_MinimizeNbColor,
            {},
        ),
    ],
    "vqe": [
        (
            VQEColoringSolver_MinimizeNbColor,
            {},
        ),
    ],
}

solvers_map = {}
for key in solvers:
    for solver, param in solvers[key]:
        solvers_map[solver] = (key, param)

solvers_to_test: list[type[SolverDO]] = [s for s in solvers_map]

# we need to map the classes to a unique string, to be seen as a categorical hyperparameter by optuna
# by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}" could be used.
solvers_by_name: dict[str, type[SolverDO]] = {
    cls.__name__: cls for cls in solvers_to_test
}

# sense of optimization
objective_register = problem.get_objective_register()
if objective_register.objective_sense == ModeOptim.MINIMIZATION:
    direction = "minimize"
else:
    direction = "maximize"


def objective(trial: Trial):
    # hyperparameters to test

    # first parameter: solver choice
    solver_name: str = trial.suggest_categorical("solver", choices=solvers_by_name)
    solver_class = solvers_by_name[solver_name]

    # hyperparameters for the chosen solver
    suggested_hyperparameters_kwargs = solver_class.suggest_hyperparameters_with_optuna(
        trial=trial, prefix=solver_name + "."
    )

    # use existing value if corresponding to a previous complete trial
    """
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
    """
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

    logger.info(f"Launching trial {trial.number} with parameters: {trial.params}")

    # construct kwargs for __init__, init_model, and solve
    kwargs = {}
    kwargs.update(suggested_hyperparameters_kwargs)
    print(kwargs)

    # solver init
    solver = solver_class(problem=problem, nb_max_color=nb_max_color)
    solver.init_model()

    # init timer
    starting_time = time.perf_counter()

    # solve
    res = solver.solve(
        callbacks=[
            OptunaCallback(
                trial=trial,
                starting_time=starting_time,
                elapsed_time_attr=elapsed_time_attr,
                report_time=True,
            ),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
        ],
        **kwargs,
    )

    # store elapsed time
    elapsed_time = time.perf_counter() - starting_time
    trial.set_user_attr(elapsed_time_attr, elapsed_time)

    if len(res.list_solution_fits) != 0:
        sol, fit = res.get_best_solution_fit()
        trial.set_user_attr("satisfy", problem.satisfy(sol))
        trial.set_user_attr("color by node", sol.colors)
        trial.set_user_attr("nb_colors", sol.nb_color)
        return fit
    else:
        raise optuna.TrialPruned("Pruned because failed")


# create study + database to store it
storage = "sqlite:///example.db"
study = optuna.create_study(
    study_name=study_name,
    direction=direction,
    sampler=optuna.samplers.TPESampler(seed=seed),
    pruner=TimedPercentilePruner(  # intermediate values interpolated at same "step"
        percentile=50,  # median pruner
        n_warmup_steps=min_time_per_solver,  # no pruning during first seconds
    ),
    storage=storage,
    load_if_exists=True,
)
study.set_metric_names(["nb_color"])
study.optimize(objective, n_trials=optuna_nb_trials)
