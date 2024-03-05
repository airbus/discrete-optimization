#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to tune hyperparameters of several solvers.

Results can be viewed on optuna-dashboard with:

    optuna-dashboard optuna-journal.log

"""
import logging
from typing import Any, Dict, List, Type

import optuna
from optuna.storages import JournalFileStorage, JournalStorage
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_tools.callbacks.optuna import (
    OptunaPruningSingleFitCallback,
)
from discrete_optimization.generic_tools.do_problem import ModeOptim
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    GPDP,
    create_selective_tsp,
)
from discrete_optimization.pickup_vrp.solver.ortools_solver import ORToolsGPDP

logger = logging.getLogger(__name__)


seed = 42
optuna_nb_trials = 20


nb_nodes = 500
nb_vehicles = 1
nb_clusters = 100

study_name = f"selective-tsp-ortools-{nb_nodes}nodes-full-auto"
storage_path = "./optuna-journal.log"  # NFS path for distributed optimization


# Solvers to test
solvers_to_test: List[Type[SolverDO]] = [ORToolsGPDP]
kwargs_fixed_by_solver: Dict[Type[SolverDO], Dict[str, Any]] = {
    ORToolsGPDP: dict(
        one_visit_per_cluster=True,
        one_visit_per_node=False,
        include_time_dimension=True,
        include_demand=True,
        include_mandatory=True,
        include_pickup_and_delivery=False,
        time_limit=20,
        optuna_report_nb_steps=1,  # nb of steps between OptunaPruningSingleFitCallback reports for current best fit
    ),
}
# we need to map the classes to a unique string, to be seen as a categoricale hyperparameter by optuna
# by default, we use the class name, but if there are identical names, f"{cls.__module__}.{cls.__name__}" could be used.
solvers_by_name: Dict[str, Type[SolverDO]] = {
    cls.__name__: cls for cls in solvers_to_test
}

# problem definition
problem: GPDP = create_selective_tsp(
    nb_nodes=nb_nodes, nb_vehicles=nb_vehicles, nb_clusters=nb_clusters
)

# sense of optimization
objective_register = problem.get_objective_register()
if objective_register.objective_sense == ModeOptim.MINIMIZATION:
    direction = "minimize"
else:
    direction = "maximize"

# objective names
objs, weights = objective_register.get_list_objective_and_default_weight()


# objective definition
def objective(trial: Trial):
    # hyperparameters to test

    # first parameter: solver choice
    solver_name = trial.suggest_categorical("solver", choices=solvers_by_name)
    solver_class = solvers_by_name[solver_name]

    # hyperparameters for the chosen solver
    hyperparameters_names = solver_class.get_hyperparameters_names()
    hyperparameters_values = ORToolsGPDP.suggest_hyperparameters_values_with_optuna(
        names=hyperparameters_names,
        trial=trial,
    )

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

    # construct kwargs for __init__, init_model, and solve
    kwargs = kwargs_fixed_by_solver[solver_class]
    kwargs.update(dict(zip(hyperparameters_names, hyperparameters_values)))

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
study = optuna.create_study(
    study_name=f"selective-tsp-ortools-{nb_nodes}nodes-full-auto",
    direction=direction,
    sampler=optuna.samplers.TPESampler(seed=seed),
    storage=storage,
    load_if_exists=True,
)
study.set_metric_names(objs)
study.optimize(objective, n_trials=optuna_nb_trials)
