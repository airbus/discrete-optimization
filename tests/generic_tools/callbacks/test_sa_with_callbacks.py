#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import logging
import os
from time import sleep

import numpy as np

from discrete_optimization.generic_tools.callbacks.backup import (
    PickleBestSolutionBackup,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
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
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    PermutationMutationRCPSP,
    get_available_mutations,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file


def test_sa_with_callbacks(caplog):
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Multi mode RCPSP
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
    objective_weights = [-1]
    params_objective_function = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=objectives,
        weights=objective_weights,
        sense_function=ModeOptim.MAXIMIZATION,
    )

    initial_temperature = 0.5
    coefficient_temperature = 0.9999
    nb_iteration_max = 1000
    nb_iteration_no_improvement_max = 50

    restart_handler = RestartHandlerLimit(
        nb_iteration_no_improvement=nb_iteration_no_improvement_max
    )
    temperature_handler = TemperatureSchedulingFactor(
        temperature=initial_temperature,
        restart_handler=restart_handler,
        coefficient=coefficient_temperature,
    )

    class SleepCallback(Callback):
        def on_step_end(self, step: int, res, solver):
            print("zzz")
            sleep(1)

    nb_iteration_tracker = NbIterationTracker()
    backuper = PickleBestSolutionBackup(save_nb_steps=2)
    callbacks = [
        SleepCallback(),
        TimerStopper(total_seconds=3, check_nb_steps=5),
        nb_iteration_tracker,
        backuper,
    ]
    sa = SimulatedAnnealing(
        evaluator=rcpsp_model,
        mutator=mixed_mutation,
        restart_handler=restart_handler,
        temperature_handler=temperature_handler,
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
    )

    with caplog.at_level(logging.DEBUG):
        sa.solve(
            dummy,
            nb_iteration_max=nb_iteration_max,
            callbacks=callbacks,
        ).get_best_solution_fit()

    assert nb_iteration_tracker.nb_iteration == 5
    assert os.path.isfile(backuper.backup_path)
    assert len([line for line in caplog.text.splitlines() if "Pickling" in line]) == 2
