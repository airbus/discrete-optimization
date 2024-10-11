#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.lp import GurobiMultimodeRcpspSolver

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
@pytest.mark.parametrize(
    "model",
    ["j301_1.sm", "j1010_1.mm"],
)
def test_gurobi(model):
    files_available = get_data_available()
    file = [f for f in files_available if model in f][0]
    rcpsp_problem = parse_file(file)
    solver = GurobiMultimodeRcpspSolver(problem=rcpsp_problem)

    result_storage = solver.solve()

    # test warm start
    parameters_cp = ParametersCp.default()
    parameters_cp.time_limit = 20
    start_solution = (
        CpSatRcpspSolver(problem=rcpsp_problem)
        .solve(parameters_cp=parameters_cp)
        .get_best_solution_fit()[0]
    )

    # first solution is not start_solution
    assert result_storage[0][0].rcpsp_schedule != start_solution.rcpsp_schedule

    # warm start at first solution
    solver = GurobiMultimodeRcpspSolver(problem=rcpsp_problem)
    solver.init_model()
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve()
    assert result_storage[0][0].rcpsp_schedule == start_solution.rcpsp_schedule
