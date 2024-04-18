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
from operator import itemgetter

import optuna
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.pickup_vrp.builders.instance_builders import (
    GPDP,
    create_selective_tsp,
)
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
print(f"Hyperparameters: {ORToolsGPDP.get_hyperparameters_names()}")
print(f"Hyperparameters: {ORToolsGPDP.get_hyperparameter('first_solution_strategy')}")


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
    hyperparameters_names = [
        "first_solution_strategy",
        "local_search_metaheuristic",
        "use_lns",
        "use_cp_sat",
    ]
    hyperparameters = ORToolsGPDP.suggest_hyperparameters_with_optuna(
        names=hyperparameters_names,
        trial=trial,
        kwargs_by_name={
            # restrict choices for `first_solution_strategy`
            "first_solution_strategy": dict(
                choices=[
                    FirstSolutionStrategy.AUTOMATIC,
                    FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
                ]
            )
        },
    )
    # extract individual hyperparameter from the hyperparameters dict
    (
        first_solution_strategy,
        local_search_metaheuristic,
        use_lns,
        use_cp_sat,
    ) = itemgetter(hyperparameters_names)(hyperparameters)

    use_cp = True
    n_solutions = trial.suggest_int("n_solutions_max", 10, 200)

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
        time_limit=20,
        use_cp=use_cp,
        use_lns=use_lns,
        use_cp_sat=use_cp_sat,
    )

    nb_iteration_stopper = NbIterationStopper(nb_iteration_max=n_solutions)

    # solve
    sol, fit = solver.solve(
        callbacks=[
            nb_iteration_stopper,
            OptunaCallback(trial=trial, optuna_report_nb_steps=1),
        ]
    ).get_best_solution_fit()
    trial.set_user_attr("n_solutions_found", nb_iteration_stopper.nb_iteration)

    return fit


# create study + database to store it
study = optuna.create_study(
    study_name=f"selective-tsp-ortools-{nb_nodes}nodes-pruning-v4-auto",
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=200),
    storage="sqlite:///example.db",
    load_if_exists=True,
)
study.set_metric_names(["distance"])
study.optimize(objective, n_trials=20)


pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))
