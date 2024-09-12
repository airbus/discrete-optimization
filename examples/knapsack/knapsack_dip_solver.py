#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.dip_knapsack_solver import DidKnapsackSolver


def run():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_10000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = DidKnapsackSolver(problem=knapsack_model)
    solver.init_model()
    res = solver.solve()
    sol = res.get_best_solution()
    print(knapsack_model.satisfy(sol))
    print(knapsack_model.max_capacity)
    print(sol, "\n", sol)


def run_optuna():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    from discrete_optimization.generic_tools.optuna.utils import (
        generic_optuna_experiment_monoproblem,
    )

    study = generic_optuna_experiment_monoproblem(
        problem=knapsack_model, solvers_to_test=[DidKnapsackSolver]
    )


if __name__ == "__main__":
    run_optuna()
