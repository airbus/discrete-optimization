#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import random

import numpy as np
import pytest
from ortools.math_opt.python import mathopt

from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.problem import KnapsackProblem
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver
from discrete_optimization.knapsack.solvers.lp import (
    GurobiKnapsackSolver,
    MathOptKnapsackSolver,
)
from discrete_optimization.knapsack.solvers_map import solve, solvers_map

try:
    import gurobipy
except ImportError:
    gurobi_available = False
else:
    gurobi_available = True


@pytest.fixture
def random_seed():
    random.seed(0)
    np.random.seed(0)


@pytest.mark.parametrize("knapsack_problem_file", get_data_available())
def test_load_file(knapsack_problem_file):
    knapsack_problem: KnapsackProblem = parse_file(knapsack_problem_file)
    dummy_solution = knapsack_problem.get_dummy_solution()
    assert knapsack_problem.satisfy(dummy_solution)


@pytest.mark.parametrize("solver_class", solvers_map)
def test_solvers(solver_class):
    if solver_class == GurobiKnapsackSolver and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")

    logging.basicConfig(level=logging.INFO)
    small_example = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_problem: KnapsackProblem = parse_file(small_example)
    results = solve(
        method=solver_class, problem=knapsack_problem, **solvers_map[solver_class][1]
    )
    s, f = results.get_best_solution_fit()
    logging.info(f"fitness={f}")


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_gurobi_warm_start():
    small_example = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_problem: KnapsackProblem = parse_file(small_example)

    solver = GurobiKnapsackSolver(knapsack_problem)

    result_storage = solver.solve()

    start_solution = (
        GreedyBestKnapsackSolver(problem=knapsack_problem).solve().get_best_solution()
    )

    # first solution is not start_solution
    assert result_storage[0][0].list_taken != start_solution.list_taken

    # warm start => first solution is start_solution
    solver.set_warm_start(start_solution)
    result_storage = solver.solve()
    assert result_storage[0][0].list_taken == start_solution.list_taken


def test_mathopt_warm_start():
    small_example = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_problem: KnapsackProblem = parse_file(small_example)

    solver = MathOptKnapsackSolver(knapsack_problem)
    kwargs = dict(
        mathopt_enable_output=True, mathopt_solver_type=mathopt.SolverType.GSCIP
    )
    result_storage = solver.solve(**kwargs)

    start_solution = (
        GreedyBestKnapsackSolver(problem=knapsack_problem).solve().get_best_solution()
    )

    # first solution is not start_solution
    assert result_storage[0][0].list_taken != start_solution.list_taken

    # warm start => first solution is start_solution
    solver.set_warm_start(start_solution)
    result_storage = solver.solve(**kwargs)
    assert result_storage[0][0].list_taken == start_solution.list_taken
