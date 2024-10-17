#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.asp import AspKnapsackSolver
from discrete_optimization.knapsack.solvers.cp_mzn import Cp2KnapsackSolver
from discrete_optimization.knapsack.solvers.cpsat import CpSatKnapsackSolver
from discrete_optimization.knapsack.solvers.decomposition import (
    DecomposedKnapsackSolver,
)
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver


def run_decomposed_knapsack_asp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem, params_objective_function=None
    )
    result_store = solver.solve(
        initial_solver=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        root_solver=SubBrick(cls=AspKnapsackSolver, kwargs=dict(time_limit=2)),
        nb_iteration=1000,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)


def run_decomposed_knapsack_greedy():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_10000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem, params_objective_function=None
    )
    result_store = solver.solve(
        initial_solver=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        root_solver=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        nb_iteration=100,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)


def run_decomposed_knapsack_cp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem,
        params_objective_function=None,
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=Cp2KnapsackSolver, kwargs=dict(time_limit=5)),
        nb_iteration=100,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)


def run_decomposed_knapsack_cpsat():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem,
        params_objective_function=None,
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=CpSatKnapsackSolver, kwargs=dict(time_limit=5)),
        nb_iteration=200,
        proportion_to_remove=0.85,
    )
    solution, fit = result_store.get_best_solution_fit()
    print(solution, fit)
    assert knapsack_problem.satisfy(solution)


if __name__ == "__main__":
    run_decomposed_knapsack_cpsat()
