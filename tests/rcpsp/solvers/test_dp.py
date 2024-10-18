#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from itertools import product

import numpy as np
import pytest

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.dp import DpRcpspModeling, DpRcpspSolver, dp


@pytest.mark.parametrize(
    "model,solver_cls",
    list(product(["j301_1.sm", "j1010_1.mm"], [dp.LNBS, dp.CABS, dp.DDLNS])),
)
def test_rcpsp_dp(model, solver_cls):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = DpRcpspSolver(problem=rcpsp_problem)
    if rcpsp_problem.is_rcpsp_multimode():
        solver.init_model(modeling=DpRcpspModeling.TASK_MULTIMODE)
    result_storage = solver.solve(solver=solver_cls, time_limit=10)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)


@pytest.mark.parametrize(
    "model,solver_cls",
    list(product(["j301_1.sm", "j1010_1.mm"], [dp.LNBS, dp.CABS, dp.DDLNS])),
)
def test_rcpsp_dp_calendar(model, solver_cls):
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
    solver = DpRcpspSolver(problem=rcpsp_problem)
    solver.init_model(modeling=DpRcpspModeling.TASK_MULTIMODE)
    result_storage = solver.solve(solver=solver_cls, time_limit=5)
    solution, fit = result_storage.get_best_solution_fit()
    assert rcpsp_problem.satisfy(solution)


@pytest.mark.parametrize(
    "modeling",
    [
        DpRcpspModeling.TASK_AND_TIME,
        DpRcpspModeling.TASK_ORIGINAL,
        DpRcpspModeling.TASK_AVAIL_TASK_UPD,
        DpRcpspModeling.TASK_MULTIMODE,
    ],
)
def test_dp_rcpsp_ws(modeling):
    files_available = get_data_available()
    file = [f for f in files_available if "j301_2.sm" in f][0]
    rcpsp_problem = parse_file(file)
    rcpsp_problem.horizon = 60
    sol_ws = rcpsp_problem.get_dummy_solution()
    solver = DpRcpspSolver(problem=rcpsp_problem)
    solver.init_model(modeling=modeling)
    solver.set_warm_start(sol_ws)
    res = solver.solve(
        callbacks=[NbIterationStopper(nb_iteration_max=1)],
        time_limit=30,
        solver=dp.LNBS,
        retrieve_intermediate_solutions=True,
        threads=5,
    )
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
