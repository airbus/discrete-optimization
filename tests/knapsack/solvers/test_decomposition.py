#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.asp import AspKnapsackSolver
from discrete_optimization.knapsack.solvers.cp_mzn import Cp2KnapsackSolver
from discrete_optimization.knapsack.solvers.decomposition import (
    DecomposedKnapsackSolver,
)
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver


def test_decomposed_knapsack_asp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem, params_objective_function=None
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=AspKnapsackSolver, kwargs=dict(time_limit=2)),
        nb_iteration=50,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert knapsack_problem.satisfy(solution)


def test_decomposed_knapsack_greedy():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem, params_objective_function=None
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert knapsack_problem.satisfy(solution)


def test_decomposed_knapsack_warm_start():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    start_solution = knapsack_problem.get_dummy_solution()

    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem, params_objective_function=None
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    assert result_store[0][0].list_taken != start_solution.list_taken
    solver.set_warm_start(start_solution)
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    assert result_store[0][0].list_taken == start_solution.list_taken


def test_decomposed_knapsack_cb():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem, params_objective_function=None
    )
    iteration_stopper = NbIterationStopper(nb_iteration_max=2)
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBestKnapsackSolver, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
        callbacks=[iteration_stopper],
    )
    solution, fit = result_store.get_best_solution_fit()
    assert iteration_stopper.nb_iteration == iteration_stopper.nb_iteration_max


def test_decomposed_knapsack_cp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_problem = parse_file(file)
    solver = DecomposedKnapsackSolver(
        problem=knapsack_problem,
        params_objective_function=None,
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=Cp2KnapsackSolver, kwargs=dict(time_limit=5)),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert knapsack_problem.satisfy(solution)
