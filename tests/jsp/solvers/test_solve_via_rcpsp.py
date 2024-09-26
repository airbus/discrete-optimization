#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.jsp.job_shop_parser import get_data_available, parse_file
from discrete_optimization.jsp.job_shop_utils import transform_jsp_to_rcpsp
from discrete_optimization.rcpsp.solver.cpsat_solver import CPSatRCPSPSolver


def test_cpsat_jsp_via_rcpsp():
    file_path = get_data_available()[0]
    problem = parse_file(file_path)
    rcpsp_problem = transform_jsp_to_rcpsp(problem)
    solver = CPSatRCPSPSolver(problem=rcpsp_problem)
    p = ParametersCP.default_cpsat()
    res = solver.solve(parameters_cp=p, time_limit=10)
    sol = res.get_best_solution_fit()[0]
    assert rcpsp_problem.satisfy(sol)
    print(rcpsp_problem.evaluate(sol))
