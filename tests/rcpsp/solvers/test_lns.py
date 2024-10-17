#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp import (
    LnsCpMznGenericRcpspSolver,
)
from discrete_optimization.generic_rcpsp_tools.solvers.lns_cp.neighbor_builder import (
    mix,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import TimerStopper
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.lns_cp import LnsCpMzn
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.problem import RcpspProblem
from discrete_optimization.rcpsp.solvers.cp_mzn import CpRcpspSolver


@pytest.mark.parametrize(
    "file_name",
    ["j1201_1.sm"],
)
def test_lns_solver(file_name):
    files_available = get_data_available()
    file = [f for f in files_available if file_name in f][0]
    rcpsp_problem: RcpspProblem = parse_file(file)
    solver = LnsCpMznGenericRcpspSolver(problem=rcpsp_problem)
    parameters_cp = ParametersCp.default()
    results = solver.solve(
        nb_iteration_lns=100,
        skip_initial_solution_provider=False,
        stop_first_iteration_if_optimal=False,
        parameters_cp=parameters_cp,
        time_limit_subsolver_iter0=5,
        time_limit_subsolver=2,
        nb_iteration_no_improvement=50,
        callbacks=[TimerStopper(total_seconds=20)],
    )
    sol, fit = results.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)


@pytest.mark.parametrize(
    "file_name",
    ["j1201_1.sm"],
)
def test_mix_constraints_handlers(file_name):
    files_available = get_data_available()
    file = [f for f in files_available if file_name in f][0]
    rcpsp_problem: RcpspProblem = parse_file(file)

    constraints_handler = mix(rcpsp_problem=rcpsp_problem)
    solver = LnsCpMzn(
        problem=rcpsp_problem,
        subsolver=CpRcpspSolver(problem=rcpsp_problem),
        constraint_handler=constraints_handler,
    )
    solver.init_model()
    res = solver.solve(
        nb_iteration_lns=5, skip_initial_solution_provider=True, time_limit_subsolver=5
    )
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_problem.satisfy(sol)
