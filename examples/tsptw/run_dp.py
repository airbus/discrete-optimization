#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution
from discrete_optimization.tsptw.solvers.dp import DpTspTwSolver, dp

logging.basicConfig(level=logging.INFO)


def run_dp():
    problem = parse_tsptw_file(get_data_available()[8])
    solver = DpTspTwSolver(problem=problem)
    solver.init_model()
    res = solver.solve(time_limit=10, solver=dp.CABS)
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))


if __name__ == "__main__":
    run_dp()
