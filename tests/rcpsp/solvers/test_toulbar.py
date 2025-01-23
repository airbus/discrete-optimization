#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import pytest

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.lns_tools import (
    BaseLns,
    TrivialInitialSolution,
    from_solutions_to_result_storage,
)
from discrete_optimization.generic_tools.toulbar_tools import to_lns_toulbar
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.toulbar import (
    RcpspConstraintHandlerToulbar,
    ToulbarMultimodeRcpspSolver,
    ToulbarRcpspSolver,
    toulbar_available,
)

logging.basicConfig(level=logging.INFO)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_toulbar_rcpsp():
    files_available = get_data_available()
    file = [f for f in files_available if "j601_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = ToulbarRcpspSolver(rcpsp_problem)
    solver.init_model()
    res = solver.solve(time_limit=5)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_toulbar_mrcpsp():
    files_available = get_data_available()
    file = [f for f in files_available if "j1010_1.mm" in f][0]
    rcpsp_problem = parse_file(file)
    solver = ToulbarMultimodeRcpspSolver(rcpsp_problem)
    solver.init_model()
    res = solver.solve(time_limit=5)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_toulbar_rcpsp_ws():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_1.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy = rcpsp_problem.get_dummy_solution()
    print(rcpsp_problem.evaluate(dummy), " dummy solution ")
    rcpsp_problem.horizon = 130
    solver = ToulbarRcpspSolver(rcpsp_problem)
    solver.init_model(ub=130)
    solver.set_warm_start(dummy)
    res = solver.solve(time_limit=5)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))


@pytest.mark.skipif(
    not toulbar_available, reason="You need Toulbar2 to test this solver."
)
def test_toulbar_rcpsp_lns():
    files_available = get_data_available()
    file = [f for f in files_available if "j1201_9.sm" in f][0]
    rcpsp_problem = parse_file(file)
    dummy = rcpsp_problem.get_dummy_solution()
    rcpsp_problem.horizon = dummy.get_start_time(rcpsp_problem.sink_task)
    solver = to_lns_toulbar(ToulbarRcpspSolver)(rcpsp_problem)
    solver.init_model()
    solver_lns = BaseLns(
        problem=rcpsp_problem,
        subsolver=solver,
        initial_solution_provider=TrivialInitialSolution(
            solution=from_solutions_to_result_storage([dummy], problem=rcpsp_problem)
        ),
        constraint_handler=RcpspConstraintHandlerToulbar(
            problem=rcpsp_problem, fraction_task=0.8
        ),
    )
    res = solver_lns.solve(
        nb_iteration_lns=2,
        time_limit_subsolver=5,
        callbacks=[
            ObjectiveLogger(
                step_verbosity_level=logging.INFO, end_verbosity_level=logging.INFO
            )
        ],
    )
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
