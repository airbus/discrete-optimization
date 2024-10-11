#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import logging
import os
import random
from time import sleep

import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.backup import (
    PickleBestSolutionBackup,
)
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
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
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    PermutationMutationRcpsp,
    get_available_mutations,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver

SEED = 42


@pytest.fixture()
def random_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    return SEED


def test_sa_warm_start(random_seed):
    files = get_data_available()
    files = [f for f in files if "j1010_1.mm" in f]  # Multi mode RCPSP
    file_path = files[0]
    rcpsp_problem: RcpspProblem = parse_file(file_path)
    rcpsp_problem.set_fixed_modes([1 for i in range(rcpsp_problem.n_jobs)])

    dummy = rcpsp_problem.get_dummy_solution()
    _, mutations = get_available_mutations(rcpsp_problem, dummy)
    list_mutation = [
        mutate[0].build(rcpsp_problem, dummy, **mutate[1])
        for mutate in mutations
        if mutate[0] == PermutationMutationRcpsp
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
    sa = SimulatedAnnealing(
        problem=rcpsp_problem,
        mutator=mixed_mutation,
        restart_handler=restart_handler,
        temperature_handler=temperature_handler,
        mode_mutation=ModeMutation.MUTATE,
        params_objective_function=params_objective_function,
        store_solution=False,
    )

    # test warm start
    start_solution: RcpspSolution = (
        PileRcpspSolver(problem=rcpsp_problem).solve().get_best_solution_fit()[0]
    )

    sa.set_warm_start(start_solution)
    res = sa.solve(
        nb_iteration_max=nb_iteration_max,
    )
    assert res[0][0].rcpsp_schedule == start_solution.rcpsp_schedule
