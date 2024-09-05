#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solver.cpsat_solver import CPSatRCPSPSolver
from discrete_optimization.rcpsp.solver.rcpsp_lp_solver import LP_MRCPSP_GUROBI

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
    solver = LP_MRCPSP_GUROBI(problem=rcpsp_problem)

    result_storage = solver.solve()

    # test warm start
    parameters_cp = ParametersCP.default()
    parameters_cp.time_limit = 20
    start_solution = (
        CPSatRCPSPSolver(problem=rcpsp_problem)
        .solve(parameters_cp=parameters_cp)
        .get_best_solution_fit()[0]
    )

    # first solution is not start_solution
    assert result_storage[0][0].rcpsp_schedule != start_solution.rcpsp_schedule

    # warm start at first solution
    solver = LP_MRCPSP_GUROBI(problem=rcpsp_problem)
    solver.init_model()
    solver.set_warm_start(start_solution)
    # force first solution to be the hinted one
    result_storage = solver.solve()
    assert result_storage[0][0].rcpsp_schedule == start_solution.rcpsp_schedule
