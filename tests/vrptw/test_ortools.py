#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrptw.parser import get_data_available, parse_vrptw_file
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution
from discrete_optimization.vrptw.solvers.ortools_routing import OrtoolsVrpTwSolver

logging.basicConfig(level=logging.INFO)


def test_ortools_vrptw():
    file = [f for f in get_data_available() if "R1_2_1.TXT" in f][0]
    problem = parse_vrptw_file(file)
    solver = OrtoolsVrpTwSolver(problem=problem)
    solver.init_model(time_limit=20, cost_per_vehicle=10000000)
    res = solver.solve()
    sol = res[-1][0]
    # Can't be sure of satisfiability here
    assert sol is not None
