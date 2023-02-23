#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os

os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.knapsack_asp_solver import KnapsackASPSolver


def run_asp_coloring():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackASPSolver(knapsack_model, params_objective_function=None)
    solver.init_model(max_models=50)
    result_store = solver.solve(timeout_seconds=20)
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)


if __name__ == "__main__":
    run_asp_coloring()
