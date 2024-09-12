#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.vrp.solver.did_vrp_solver import DidVrpSolver, dp
from discrete_optimization.vrp.vrp_model import VrpSolution
from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file


def test_did_vrp():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    solver = DidVrpSolver(problem=problem)
    res = solver.solve(solver=dp.LNBS, time_limit=10)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    assert problem.satisfy(sol)
