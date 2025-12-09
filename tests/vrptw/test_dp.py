#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.vrptw.parser import get_data_available, parse_vrptw_file
from discrete_optimization.vrptw.solvers.dp import DpVrptwSolver

logging.basicConfig(level=logging.INFO)


def test_dp():
    file = [f for f in get_data_available() if "R1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)
    solver = DpVrptwSolver(problem=problem)
    solver.init_model(
        scaling=100,
        cost_per_vehicle=100000,
        resource_var_load=False,
        resource_var_time=False,
    )
    res = solver.solve(callbacks=[], solver="CABS", time_limit=20, threads=2)
    sol = res[-1][0]
    # assert problem.satisfy(sol)
