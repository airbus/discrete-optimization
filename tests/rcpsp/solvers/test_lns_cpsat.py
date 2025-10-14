#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest

from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_extractor import (
    ConstraintExtractorList,
    MultimodeConstraintExtractor,
    SchedulingConstraintExtractor,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.constraint_handler import (
    TasksConstraintHandler,
)
from discrete_optimization.generic_tasks_tools.solvers.lns_cp.neighbor_tools import (
    NeighborBuilderMix,
    NeighborBuilderSubPart,
    NeighborRandom,
    NeighborRandomAndNeighborGraph,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsOrtoolsCpSat
from discrete_optimization.generic_tools.lns_tools import TrivialInitialSolution
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize(
    "fix_primary_tasks_modes, fix_secondary_tasks_modes", [(False, True), (True, False)]
)
def test_lns(fix_primary_tasks_modes, fix_secondary_tasks_modes, random_seed):
    model = "j1010_1.mm"
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    problem = parse_file(file)
    subsolver = CpSatRcpspSolver(problem=problem)

    parameters_cp = ParametersCp.default()
    initial_res = subsolver.solve(
        parameters_cp=parameters_cp, callbacks=[NbIterationStopper(nb_iteration_max=1)]
    )
    initial_solution_provider = TrivialInitialSolution(solution=initial_res)

    constraints_extractor = ConstraintExtractorList(
        extractors=[
            SchedulingConstraintExtractor(),
            MultimodeConstraintExtractor(
                fix_primary_tasks_modes=fix_primary_tasks_modes,
                fix_secondary_tasks_modes=fix_secondary_tasks_modes,
            ),
        ]
    )

    constraint_handler = TasksConstraintHandler(
        problem=problem,
        constraints_extractor=constraints_extractor,
    )

    solver = LnsOrtoolsCpSat(
        problem=problem,
        subsolver=subsolver,
        constraint_handler=constraint_handler,
        initial_solution_provider=initial_solution_provider,
    )
    res = solver.solve(
        nb_iteration_lns=20,
        time_limit_subsolver=10,
        parameters_cp=parameters_cp,
    )
    sol = res.get_best_solution()
    problem.satisfy(sol)


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_default_constraint_handler(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    problem = parse_file(file)
    constraint_handler = TasksConstraintHandler(
        problem=problem,
    )
    assert isinstance(constraint_handler.constraints_extractor, ConstraintExtractorList)
    assert (
        MultimodeConstraintExtractor
        in [
            type(extractor)
            for extractor in constraint_handler.constraints_extractor.extractors
        ]
    ) == problem.is_multimode
    assert isinstance(constraint_handler.neighbor_builder, NeighborBuilderMix)
    assert NeighborRandomAndNeighborGraph in (
        type(builder) for builder in constraint_handler.neighbor_builder.list_neighbor
    )
