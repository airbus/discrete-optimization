#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver


def test_cpsat_jsp():
    problem = parse_file(get_data_available()[0])
    solver = CpSatJspSolver(problem=problem)
    res = solver.solve(time_limit=5)
    assert problem.satisfy(res.get_best_solution_fit()[0])
