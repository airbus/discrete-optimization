#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

"""Example using OPTUNA to tune hyperparameters.

Solver: SA
Model: RCPSP j1201_1.sm
with pruning

increase minimal nb_iteration_max

Results can be viewed on optuna-dashboard with:

    optuna-dashboard sqlite:///example.db

"""

import numpy as np
import optuna
from optuna.trial import Trial, TrialState

from discrete_optimization.generic_tools.callbacks.optuna import OptunaCallback
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ObjectiveHandling,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    ModeMutation,
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    RcpspMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_portfolio import (
    create_mutations_portfolio_from_problem,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem

# for reproducibility of the example
SEED = 42
np.random.seed(SEED)
RCPSP_FILE_NAME = "j1201_1.sm"


def objective(trial: Trial):
    files = get_data_available()
    files = [f for f in files if RCPSP_FILE_NAME in f]
    file_path = files[0]
    rcpsp_problem: RcpspProblem = parse_file(file_path)
    rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])

    dummy = rcpsp_problem.get_dummy_solution()
    mixed_mutation = create_mutations_portfolio_from_problem(
        problem=rcpsp_problem, selected_mutations={RcpspMutation}
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
        problem=rcpsp_problem,
        mutator=mixed_mutation,
        restart_handler=restart_handler,
        temperature_handler=temperature_handler,
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
    )
    sol, fit = sa.solve(
        initial_variable=dummy,
        nb_iteration_max=nb_iteration_max,
        callbacks=[OptunaCallback(trial=trial, optuna_report_nb_steps=100)],
    ).get_best_solution_fit()
    return fit


if __name__ == "__main__":
    # create study + database to store it
    study = optuna.create_study(
        study_name=f"rcpsp-sa-pruning-{RCPSP_FILE_NAME}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.HyperbandPruner(min_resource=100, max_resource=10000),
        storage="sqlite:///example.db",
        load_if_exists=True,
    )
    study.set_metric_names(["makespan"])
    study.optimize(objective, n_trials=100)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
