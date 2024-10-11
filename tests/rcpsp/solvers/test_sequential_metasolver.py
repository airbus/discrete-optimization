#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random

import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.callbacks.loggers import (
    NbIterationTracker,
    ObjectiveLogger,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.ls.local_search import (
    ModeMutation,
    RestartHandlerLimit,
)
from discrete_optimization.generic_tools.ls.simulated_annealing import (
    SimulatedAnnealing,
    TemperatureSchedulingFactor,
)
from discrete_optimization.generic_tools.mutations.mixed_mutation import (
    BasicPortfolioMutation,
)
from discrete_optimization.generic_tools.mutations.mutation_catalog import (
    get_available_mutations,
)
from discrete_optimization.generic_tools.sequential_metasolver import (
    SequentialMetasolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.lp import GurobiMultimodeRcpspSolver
from discrete_optimization.rcpsp.solvers.pile import PileRcpspSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


logging.basicConfig(level=logging.INFO)


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


def test_sequential_metasolver_rcpsp(random_seed):
    logging.basicConfig(level=logging.INFO)

    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)

    # kwargs SA
    solution = rcpsp_problem.get_dummy_solution()
    _, list_mutation = get_available_mutations(rcpsp_problem, solution)
    list_mutation = [
        mutate[0].build(rcpsp_problem, solution, **mutate[1])
        for mutate in list_mutation
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    restart_handler = RestartHandlerLimit(3000)
    temperature_handler = TemperatureSchedulingFactor(1000, restart_handler, 0.99)

    # kwargs cpsat
    parameters_cp = ParametersCp.default_cpsat()

    list_subbricks = [
        SubBrick(cls=PileRcpspSolver, kwargs=dict()),
        SubBrick(
            cls=SimulatedAnnealing,
            kwargs=dict(
                mutator=mixed_mutation,
                restart_handler=restart_handler,
                temperature_handler=temperature_handler,
                mode_mutation=ModeMutation.MUTATE,
                nb_iteration_max=5000,
            ),
        ),
        SubBrick(
            cls=CpSatRcpspSolver,
            kwargs=dict(parameters_cp=parameters_cp, time_limit=20),
        ),
    ]

    solver = SequentialMetasolver(problem=rcpsp_problem, list_subbricks=list_subbricks)
    result_storage = solver.solve(
        callbacks=[
            NbIterationTracker(step_verbosity_level=logging.INFO),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
            TimerStopper(total_seconds=30),
        ],
    )
    solution, fit = result_storage.get_best_solution_fit()
    print(solution, fit)
    assert rcpsp_problem.satisfy(solution)


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_sequential_metasolver_rcpsp_with_dynamic_kwargs(random_seed):
    logging.basicConfig(level=logging.INFO)

    files_available = get_data_available()
    file = [f for f in files_available if "j301_1.sm" in f][0]
    rcpsp_problem = parse_file(file)

    # kwargs SA
    solution = rcpsp_problem.get_dummy_solution()
    _, list_mutation = get_available_mutations(rcpsp_problem, solution)
    list_mutation = [
        mutate[0].build(rcpsp_problem, solution, **mutate[1])
        for mutate in list_mutation
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    restart_handler = RestartHandlerLimit(3000)
    temperature_handler = TemperatureSchedulingFactor(1000, restart_handler, 0.99)

    list_subbricks = [
        SubBrick(cls=PileRcpspSolver, kwargs=dict()),
        SubBrick(
            cls=SimulatedAnnealing,
            kwargs=dict(
                mutator=mixed_mutation,
                restart_handler=restart_handler,
                temperature_handler=temperature_handler,
                mode_mutation=ModeMutation.MUTATE,
                nb_iteration_max=5000,
            ),
        ),
        SubBrick(
            cls=GurobiMultimodeRcpspSolver,
            kwargs=dict(),
            kwargs_from_solution=dict(start_solution=lambda sol: sol),
        ),
    ]

    solver = SequentialMetasolver(problem=rcpsp_problem, list_subbricks=list_subbricks)
    result_storage = solver.solve(
        callbacks=[
            NbIterationTracker(step_verbosity_level=logging.INFO),
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            ),
            TimerStopper(total_seconds=30),
        ],
    )
    solution, fit = result_storage.get_best_solution_fit()
    print(solution, fit)
    assert rcpsp_problem.satisfy(solution)
