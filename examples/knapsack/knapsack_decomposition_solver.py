#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import os

os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN2
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest
from discrete_optimization.knapsack.solvers.knapsack_asp_solver import KnapsackASPSolver
from discrete_optimization.knapsack.solvers.knapsack_cpsat_solver import (
    CPSatKnapsackSolver,
)
from discrete_optimization.knapsack.solvers.knapsack_decomposition import (
    KnapsackDecomposedSolver,
)


def run_decomposed_knapsack_asp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model, params_objective_function=None
    )
    result_store = solver.solve(
        initial_solver=GreedyBest,
        root_solver=KnapsackASPSolver,
        root_solver_kwargs=dict(timeout_seconds=2),
        nb_iteration=1000,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)


def run_decomposed_knapsack_greedy():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_10000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model, params_objective_function=None
    )
    result_store = solver.solve(
        initial_solver=GreedyBest,
        root_solver=GreedyBest,
        nb_iteration=100,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)


def run_decomposed_knapsack_cp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model,
        params_objective_function=None,
    )
    params_cp = ParametersCP.default()
    params_cp.time_limit = 5
    result_store = solver.solve(
        root_solver=CPKnapsackMZN2,
        root_solver_kwargs=dict(parameters_cp=params_cp),
        nb_iteration=100,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)


def run_decomposed_knapsack_cpsat():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model,
        params_objective_function=None,
    )
    params_cp = ParametersCP.default()
    params_cp.time_limit = 5
    result_store = solver.solve(
        root_solver=CPSatKnapsackSolver,
        root_solver_kwargs=dict(parameters_cp=params_cp),
        nb_iteration=200,
        proportion_to_remove=0.85,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)
    assert knapsack_model.satisfy(solution)


if __name__ == "__main__":
    run_decomposed_knapsack_cpsat()
