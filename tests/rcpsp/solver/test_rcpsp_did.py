#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from itertools import product

import numpy as np
import pytest

from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solver.did_rcpsp_solver import (
    DidRCPSPModeling,
    DidRCPSPSolver,
    dp,
)


@pytest.mark.parametrize(
    "model,solver_cls",
    list(product(["j301_1.sm", "j1010_1.mm"], [dp.LNBS, dp.CABS, dp.DDLNS])),
)
def test_rcpsp_did(model, solver_cls):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = DidRCPSPSolver(problem=rcpsp_problem)
    if rcpsp_problem.is_rcpsp_multimode():
        solver.init_model(modeling=DidRCPSPModeling.TASK_MULTIMODE)
    result_storage = solver.solve(solver=solver_cls, time_limit=10)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)


@pytest.mark.parametrize(
    "model,solver_cls",
    list(product(["j301_1.sm", "j1010_1.mm"], [dp.LNBS, dp.CABS, dp.DDLNS])),
)
def test_rcpsp_did_calendar(model, solver_cls):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    for resource in rcpsp_problem.resources:
        rcpsp_problem.resources[resource] = np.array(
            rcpsp_problem.get_resource_availability_array(resource)
        )
        if resource not in rcpsp_problem.non_renewable_resources:
            rcpsp_problem.resources[resource][10:15] = 0
            rcpsp_problem.resources[resource][30:35] = 0
    rcpsp_problem.is_calendar = True
    rcpsp_problem.update_functions()
    solver = DidRCPSPSolver(problem=rcpsp_problem)
    solver.init_model(modeling=DidRCPSPModeling.TASK_MULTIMODE)
    result_storage = solver.solve(solver=solver_cls, time_limit=5)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)
