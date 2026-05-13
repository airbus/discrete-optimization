#  Copyright (c) 2026 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import random

import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.transformation.transformation_solver import (
    SubBrick,
    TransformationSolver,
)
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solution import RcpspSolution
from discrete_optimization.rcpsp.solvers.preemptive.cpsat import (
    CpSatPreemptiveRcpspSolver,
)
from discrete_optimization.rcpsp.transformations.to_multiskill import (
    RcpspToMultiskillTransformation,
)
from discrete_optimization.rcpsp.transformations.to_preemptive import (
    RcpspToPreemptiveTransformation,
)
from discrete_optimization.rcpsp_multiskill.solvers.cpsat import (
    CpSatMultiskillRcpspSolver,
)


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_via_multiskill(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = TransformationSolver(
        transformation=RcpspToMultiskillTransformation(),
        source_problem=rcpsp_problem,
        solver_brick=SubBrick(CpSatMultiskillRcpspSolver, {}),
    )
    result_storage = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=100)
    solution: RcpspSolution
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)
    assert solution.check_all_renewable_resource_capacity_constraints()


@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_via_preemptive(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = TransformationSolver(
        transformation=RcpspToPreemptiveTransformation(),
        source_problem=rcpsp_problem,
        solver_brick=SubBrick(CpSatPreemptiveRcpspSolver, {}),
    )
    result_storage = solver.solve(callbacks=[NbIterationStopper(1)], time_limit=100)
    solution: RcpspSolution
    solution, fit = result_storage.get_best_solution_fit()
    # assert rcpsp_problem.satisfy(solution)
    # assert solution.check_all_renewable_resource_capacity_constraints()
