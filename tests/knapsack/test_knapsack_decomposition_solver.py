#  Copyright (c) 2023 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging

from discrete_optimization.generic_tools.callbacks.early_stoppers import (
    NbIterationStopper,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.cp_solvers import CPKnapsackMZN2
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest
from discrete_optimization.knapsack.solvers.knapsack_asp_solver import KnapsackASPSolver
from discrete_optimization.knapsack.solvers.knapsack_decomposition import (
    KnapsackDecomposedSolver,
)


def test_decomposed_knapsack_asp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model, params_objective_function=None
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=KnapsackASPSolver, kwargs=dict(time_limit=2)),
        nb_iteration=50,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert knapsack_model.satisfy(solution)


def test_decomposed_knapsack_greedy():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model, params_objective_function=None
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBest, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert knapsack_model.satisfy(solution)


def test_decomposed_knapsack_warm_start():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    start_solution = knapsack_model.get_dummy_solution()

    solver = KnapsackDecomposedSolver(
        problem=knapsack_model, params_objective_function=None
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBest, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    assert result_store[0][0].list_taken != start_solution.list_taken
    solver.set_warm_start(start_solution)
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBest, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    assert result_store[0][0].list_taken == start_solution.list_taken


def test_decomposed_knapsack_cb():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model, params_objective_function=None
    )
    iteration_stopper = NbIterationStopper(nb_iteration_max=2)
    result_store = solver.solve(
        root_solver=SubBrick(cls=GreedyBest, kwargs={}),
        nb_iteration=5,
        proportion_to_remove=0.9,
        callbacks=[iteration_stopper],
    )
    solution, fit = result_store.get_best_solution_fit()
    assert iteration_stopper.nb_iteration == iteration_stopper.nb_iteration_max


def test_decomposed_knapsack_cp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "ks_1000_0" in f][0]
    knapsack_model = parse_file(file)
    solver = KnapsackDecomposedSolver(
        problem=knapsack_model,
        params_objective_function=None,
    )
    result_store = solver.solve(
        root_solver=SubBrick(cls=CPKnapsackMZN2, kwargs=dict(time_limit=5)),
        nb_iteration=5,
        proportion_to_remove=0.9,
    )
    solution, fit = result_store.get_best_solution_fit()
    assert knapsack_model.satisfy(solution)
