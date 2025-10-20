#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.tsptw.solvers.ortools_routing import OrtoolsTspTwSolver


def test_ortools(problem):
    solver = OrtoolsTspTwSolver(problem=problem)
    solver.init_model(
        time_limit=5,
    )
    sol = solver.solve().get_best_solution()
    assert problem.satisfy(sol)
