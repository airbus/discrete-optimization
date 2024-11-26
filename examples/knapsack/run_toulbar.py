#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.toulbar import ToulbarKnapsackSolver


def run_toulbar():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_10000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = ToulbarKnapsackSolver(problem=knapsack_problem)
    solver.init_model(vns=-4)
    res = solver.solve(time_limit=100)
    sol = res.get_best_solution()
    print(knapsack_problem.satisfy(sol))
    print(knapsack_problem.max_capacity)
    print(sol, "\n", sol)


def run_toulbar_ws():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_500_0" in f][0]
    knapsack_problem = parse_file(file)
    from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver

    solver = GreedyBestKnapsackSolver(problem=knapsack_problem)
    sol, fit = solver.solve().get_best_solution_fit()
    print(fit, " current sol")
    solver = ToulbarKnapsackSolver(problem=knapsack_problem)
    solver.init_model(vns=-4)
    solver.set_warm_start(solution=sol)
    res = solver.solve(time_limit=100)
    sol = res.get_best_solution()
    print(knapsack_problem.satisfy(sol))
    print(knapsack_problem.max_capacity)
    print(sol, "\n", sol)


if __name__ == "__main__":
    run_toulbar_ws()
