#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrptw.parser import parse_solomon
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution
from discrete_optimization.vrptw.solvers.ortools_routing import OrtoolsVrpTwSolver

logging.basicConfig(level=logging.INFO)


def run_ortools():
    problem = parse_solomon("C1_2_1.TXT")
    solver = OrtoolsVrpTwSolver(problem=problem)
    solver.init_model(time_limit=20, cost_per_vehicle=10000000)
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_ortools()
