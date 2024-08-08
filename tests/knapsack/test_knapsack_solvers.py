#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
import platform
import random

import numpy as np
import pytest

from discrete_optimization.generic_tools.lp_tools import PymipMilpSolver
from discrete_optimization.knapsack.knapsack_model import (
    KnapsackModel,
    KnapsackSolution,
)
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.knapsack_solvers import solve, solvers_map
from discrete_optimization.knapsack.solvers.greedy_solvers import GreedyBest
from discrete_optimization.knapsack.solvers.lp_solvers import LPKnapsackGurobi

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
    knapsack_model: KnapsackModel = parse_file(knapsack_problem_file)
    dummy_solution = knapsack_model.get_dummy_solution()
    assert knapsack_model.satisfy(dummy_solution)


@pytest.mark.parametrize("solver_class", solvers_map)
def test_solvers(solver_class):
    if solver_class == LPKnapsackGurobi and not gurobi_available:
        pytest.skip("You need Gurobi to test this solver.")
    if issubclass(solver_class, PymipMilpSolver) and platform.machine() == "arm64":
        pytest.skip(
            "Python-mip has issues with cbclib on macos arm64. "
            "See https://github.com/coin-or/python-mip/issues/167"
        )

    logging.basicConfig(level=logging.INFO)
    small_example = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_model: KnapsackModel = parse_file(small_example)
    results = solve(
        method=solver_class, problem=knapsack_model, **solvers_map[solver_class][1]
    )
    s, f = results.get_best_solution_fit()
    logging.info(f"fitness={f}")


@pytest.mark.skipif(not gurobi_available, reason="You need Gurobi to test this solver.")
def test_gurobi_warm_start():
    small_example = [f for f in get_data_available() if "ks_40_0" in f][0]
    knapsack_model: KnapsackModel = parse_file(small_example)

    solver = LPKnapsackGurobi(knapsack_model)

    result_storage = solver.solve()

    start_solution = GreedyBest(problem=knapsack_model).solve().get_best_solution()

    # first solution is not start_solution
    assert result_storage[0][0].list_taken != start_solution.list_taken

    # warm start => first solution is start_solution
    solver.set_warm_start(start_solution)
    result_storage = solver.solve()
    assert result_storage[0][0].list_taken == start_solution.list_taken
