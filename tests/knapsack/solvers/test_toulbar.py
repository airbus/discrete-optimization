#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import pytest

from discrete_optimization.knapsack.parser import get_data_available, parse_file
from discrete_optimization.knapsack.solvers.toulbar import (
    ToulbarKnapsackSolver,
    toulbar_available,
)


@pytest.mark.skipif(True, reason="You need Toulbar2 to test this solver.")
def test_toulbar():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_problem = parse_file(files[0])
    solver = ToulbarKnapsackSolver(problem=knapsack_problem)
    results = solver.solve(time_limit=5)
    sol, fit = results.get_best_solution_fit()
    assert knapsack_problem.satisfy(sol)
