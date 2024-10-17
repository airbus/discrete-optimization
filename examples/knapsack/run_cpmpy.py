#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

from cpmpy import SolverLookup

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.cpmpy import CpmpyKnapsackSolver


def run():
    logging.basicConfig(level=logging.DEBUG)
    file = [f for f in get_data_available() if "ks_60_0" in f][0]
    knapsack_problem = parse_file(file)
    a = SolverLookup.base_solvers()
    print(SolverLookup.base_solvers())
    solver = CpmpyKnapsackSolver(problem=knapsack_problem)
    solver.init_model()
    res = solver.solve(solver="ortools", time_limit=20)
    sol = res.get_best_solution()
    print(knapsack_problem.satisfy(sol))
    print(knapsack_problem.max_capacity)
    print(sol, "\n", sol)


if __name__ == "__main__":
    run()
