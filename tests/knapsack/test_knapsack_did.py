#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.knapsack.knapsack_parser import (
    get_data_available,
    parse_file,
)
from discrete_optimization.knapsack.solvers.dip_knapsack_solver import (
    DidKnapsackSolver,
    dp,
)


def test_did():
    files = [f for f in get_data_available() if "ks_60_0" in f]
    knapsack_model = parse_file(files[0])
    solver = DidKnapsackSolver(problem=knapsack_model)
    results = solver.solve(solver=dp.LNBS, time_limit=10)
    sol, fit = results.get_best_solution_fit()
    assert knapsack_model.satisfy(sol)
