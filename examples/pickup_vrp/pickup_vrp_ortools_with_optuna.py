#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
"""Example using OPTUNA to tune hyperparameters.

Solver: ortools
Model: pickup_vrp selective_tsp

Results can be viewed on optuna-dashboard with:

    optuna-dashboard sqlite:///example.db

"""

import logging

import numpy as np
import optuna
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    GPDP,
    create_pickup_and_delivery,
    create_selective_tsp,
)
from discrete_optimization.pickup_vrp.gpdp import GPDPSolution
from discrete_optimization.pickup_vrp.plots.gpdp_plot_utils import plot_gpdp_solution
from discrete_optimization.pickup_vrp.solver.ortools_solver import (
    FirstSolutionStrategy,
    LocalSearchMetaheuristic,
    ORToolsGPDP,
    ParametersCost,
)

SEED = 42
nb_nodes = 500
nb_vehicles = 1
nb_clusters = 100


# logging.basicConfig(level=logging.DEBUG)


def objective(trial: Trial):
    gpdp: GPDP = create_selective_tsp(
        nb_nodes=nb_nodes, nb_vehicles=nb_vehicles, nb_clusters=nb_clusters
    )
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=["distance_max"],
        weights=[1],
        sense_function=ModeOptim.MINIMIZATION,
    )
    # hyperparameters
    first_solution_strategy_name = trial.suggest_categorical(
        "first_solution_strategy", choices=[v.name for v in FirstSolutionStrategy]
    )
    local_search_metaheuristic_name = trial.suggest_categorical(
        "local_search_metaheuristic", choices=[v.name for v in LocalSearchMetaheuristic]
    )
    local_search_metaheuristic = LocalSearchMetaheuristic[
        local_search_metaheuristic_name
    ]
    first_solution_strategy = FirstSolutionStrategy[first_solution_strategy_name]
    use_lns = trial.suggest_categorical("use_lns", [True, False])
    use_cp = trial.suggest_categorical("use_cp", [True, False])
    use_cp_sat = trial.suggest_categorical("use_cp_sat", [True, False])
    n_solutions = trial.suggest_int("n_solutions", 10, 200)

    # solver init
    solver = ORToolsGPDP(
        problem=gpdp,
        factor_multiplier_distance=1,
        factor_multiplier_time=1,
        params_objective_function=params_objective_function,
    )
    solver.init_model(
        one_visit_per_cluster=True,
        one_visit_per_node=False,
        include_time_dimension=True,
        include_demand=True,
        include_mandatory=True,
        include_pickup_and_delivery=False,
        parameters_cost=[ParametersCost(dimension_name="Distance", global_span=True)],
        local_search_metaheuristic=local_search_metaheuristic,
        first_solution_strategy=first_solution_strategy,
        time_limit=60,
        use_cp=use_cp,
        use_lns=use_lns,
        use_cp_sat=use_cp_sat,
    )

    nb_iteration_stopper = NbIterationStopper(nb_iteration_max=n_solutions)

    # solve
    sol, fit = solver.solve(callbacks=[nb_iteration_stopper]).get_best_solution_fit()
    return fit


# create study + database to store it
study = optuna.create_study(
    study_name=f"selective-tsp-ortools-{nb_nodes}nodes-v3",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    storage="sqlite:///example.db",
    load_if_exists=True,
)
study.set_metric_names(["distance"])
study.enqueue_trial({"first_solution_strategy": "SWEEP"})
study.optimize(objective, n_trials=10)


pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))
