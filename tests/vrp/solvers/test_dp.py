#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.vrp.parser import get_data_available, parse_file
from discrete_optimization.vrp.problem import VrpSolution
from discrete_optimization.vrp.solvers.dp import DpVrpSolver, dp


def test_dp_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    solver = DpVrpSolver(problem=problem)
    res = solver.solve(solver=dp.LNBS, time_limit=10)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    assert problem.satisfy(sol)
