#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.solvers.gpdp import GpdpBasedTspSolver


def test_gpdp():
    files = get_data_available()
    files = [f for f in files if "tsp_200_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    solver = GpdpBasedTspSolver(problem=model)
    res = solver.solve(time_limit=10)
    sol, fit = res.get_best_solution_fit()
    assert model.satisfy(sol)
