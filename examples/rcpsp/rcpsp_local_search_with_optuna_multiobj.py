#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example using OPTUNA to tune hyperparameters.

Solver: SA
Model: RCPSP j1201_1.sm
no pruning

pareto
 objectives:
    makespan
    nb_iteration done (nb_iteration_max)


Results can be viewed on optuna-dashboard with:

    optuna-dashboard sqlite:///example.db

"""

import numpy as np
import optuna
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ls.hill_climber import HillClimberPareto
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    PermutationMutationRCPSP,
    get_available_mutations,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ParetoFront,
    plot_pareto_2d,
    plot_storage_2d,
    result_storage_to_pareto_front,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_utils import (
    plot_resource_individual_gantt,
    plot_ressource_view,
)

# for reproducibility of the example
SEED = 42
np.random.seed(SEED)
RCPSP_FILE_NAME = "j1201_1.sm"


def objective(trial: Trial):
    files = get_data_available()
    files = [f for f in files if RCPSP_FILE_NAME in f]
    file_path = files[0]
    rcpsp_model: RCPSPModel = parse_file(file_path)
    rcpsp_model.set_fixed_modes([1 for i in range(rcpsp_model.n_jobs)])

    dummy = rcpsp_model.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_model, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_model, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRCPSP
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    objectives = ["makespan"]
    objective_weights = [1]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MINIMIZATION,
    )

    initial_temperature = trial.suggest_float(
        "initial_temperature", 0.1, 100.0, log=True
    )
    one_minus_coefficient_temperature = trial.suggest_float(
        "one_minus_coefficient_temperature", 0.00001, 0.2, log=True
    )
    trial.set_user_attr(
        "coefficient_temperature", 1.0 - one_minus_coefficient_temperature
    )
    nb_iteration_max = trial.suggest_int("nb_iteration_max", 1000, 10000)
    nb_iteration_no_improvement_max = trial.suggest_int(
        "nb_iteration_no_improvement_max", 10, nb_iteration_max
    )

    restart_handler = RestartHandlerLimit(
        nb_iteration_no_improvement=nb_iteration_no_improvement_max
    )
    temperature_handler = TemperatureSchedulingFactor(
        temperature=initial_temperature,
        restart_handler=restart_handler,
        coefficient=1.0 - one_minus_coefficient_temperature,
    )

    sa = SimulatedAnnealing(
        problem=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=restart_handler,
        temperature_handler=temperature_handler,
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
    )
    sol, fit = sa.solve(
        dummy,
        nb_iteration_max=nb_iteration_max,
    ).get_best_solution_fit()
    return fit, nb_iteration_max


# create study + database to store it
study = optuna.create_study(
    study_name=f"rcpsp-sa-simple-multiobj-{RCPSP_FILE_NAME}",
    directions=["minimize", "minimize"],
    sampler=optuna.samplers.TPESampler(seed=SEED),
    storage="sqlite:///example.db",
    load_if_exists=True,
)
study.set_metric_names(["makespan", "nb_iterations"])
study.optimize(objective, n_trials=100)


pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))
